"""
An OpenAI-compatible server with steering, built on ``mlx_lm.server``.

Everything mlx-lm's server does (chat and text completions, streaming, batching,
prompt caching, tool calls) is kept. On top of it:

- a ``steering`` request field, e.g.
  ``"steering": [{"vector": "honesty", "alpha": -4}]``, reachable from OpenAI
  clients through ``extra_body``;
- presets: named steering configurations that clients select as the ``model``,
  for apps that only let users pick a model name. They are listed by
  ``/v1/models``.

Steering is folded into the request's adapter slot, which mlx-lm already uses
to decide which requests share a batch and which prompt-cache entries a request
may reuse. Requests with different steering therefore never share a batch or a
KV cache, while the weights stay loaded once: only the steering wrappers change
between batches.

By default requests may only name vectors the operator registered with
``--vector name=path``. ``--allow-any-vector`` lets requests pass paths and
``hf://`` ids, which makes the server read local files and download from the
Hub on a client's say-so.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
from unittest import mock

from .vector import SteeringVector, parse_vector_spec

logger = logging.getLogger(__name__)

# Prefix marking an adapter string that carries steering.
STEER_TAG = "pts-steering:"

Steering = Tuple[Tuple[str, Optional[float]], ...]


def canonical(steering) -> Steering:
    """Sorted, so equivalent configurations share batches and cache entries.

    Order never changes the result: additions sum and ablations are
    orthonormalised together.
    """
    return tuple(sorted(steering, key=lambda s: (s[0], float("-inf") if s[1] is None else s[1])))


def encode_adapter(adapter: Optional[str], steering: Steering) -> Optional[str]:
    if not steering:
        return adapter
    return STEER_TAG + json.dumps({"adapter": adapter, "steering": [list(s) for s in steering]})


def decode_adapter(value: Optional[str]) -> Tuple[Optional[str], Steering]:
    if not value or not value.startswith(STEER_TAG):
        return value, ()
    data = json.loads(value[len(STEER_TAG):])
    return data["adapter"], tuple((ref, alpha) for ref, alpha in data["steering"])


@dataclass
class Preset:
    steering: Steering
    model: Optional[str] = None  # None: the server's --model


@dataclass
class SteeringRegistry:
    """Which vectors requests may use, and the presets built from them."""

    named: Dict[str, str] = field(default_factory=dict)
    presets: Dict[str, Preset] = field(default_factory=dict)
    allow_any: bool = False
    _cache: Dict[str, SteeringVector] = field(default_factory=dict)

    def resolve(self, ref: str) -> SteeringVector:
        if ref in self.named:
            source = self.named[ref]
        elif self.allow_any:
            source = ref
        else:
            known = ", ".join(sorted(self.named)) or "none"
            raise ValueError(
                f"Unknown steering vector {ref!r}. Registered vectors: {known}. "
                "(Start the server with --allow-any-vector to accept paths.)"
            )
        if source not in self._cache:
            self._cache[source] = SteeringVector.load(source)
        return self._cache[source]

    def parse(self, value: Any) -> Steering:
        """Canonicalise a request's ``steering`` field.

        Accepts ``"name[:alpha]"``, a list of those, or a list of
        ``{"vector": name, "alpha": a}``.
        """
        if value is None:
            return ()
        items = value if isinstance(value, list) else [value]
        out = []
        for item in items:
            if isinstance(item, str):
                ref, alpha = parse_vector_spec(item)
            elif isinstance(item, dict) and isinstance(item.get("vector"), str):
                ref, alpha = item["vector"], item.get("alpha")
                if alpha is not None:
                    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
                        raise ValueError("steering alpha must be a number")
                    alpha = float(alpha)
            else:
                raise ValueError(
                    'steering must be "name[:alpha]", a list of those, or a list of '
                    '{"vector": name, "alpha": number}'
                )
            self.resolve(ref)  # fail on the request, not later in the generation thread
            out.append((ref, alpha))
        return canonical(out)


def parse_preset(text: str) -> Tuple[str, Preset]:
    """``name=vec[:alpha][,vec[:alpha]...][@model]``."""
    name, sep, rest = text.partition("=")
    if not sep or not name or not rest:
        raise ValueError(f"Bad preset {text!r}: expected name=vector[:alpha][,...][@model]")
    rest, at, model = rest.partition("@")
    specs = tuple(parse_vector_spec(s.strip()) for s in rest.split(",") if s.strip())
    return name, Preset(steering=specs, model=model or None)


def parse_named(text: str) -> Tuple[str, str]:
    """``name=path`` for --vector."""
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise ValueError(f"Bad --vector {text!r}: expected name=path")
    return name, path


# ---------------------------------------------------------------------------
# mlx-lm integration
# ---------------------------------------------------------------------------

def mlx_cli_args(server_argv: Sequence[str]) -> argparse.Namespace:
    """Parse mlx-lm's own server flags, so every one of them keeps working.

    mlx_lm.server builds its parser inside ``main()``; run it with ``run``
    stubbed out and keep the namespace it would have served with.
    """
    import mlx_lm.server as S

    captured: Dict[str, Any] = {}

    def capture(host, port, model_provider, *args, **kwargs):
        captured["args"] = model_provider.cli_args

    with mock.patch.object(S, "run", capture), mock.patch.object(
        sys, "argv", ["mlx_lm.server", *server_argv]
    ):
        S.main()
    return captured["args"]


def _provider_class():
    import mlx_lm.server as S

    from .mlx_backend import install

    class SteeringModelProvider(S.ModelProvider):
        def __init__(self, cli_args, registry: SteeringRegistry):
            super().__init__(cli_args)
            self.registry = registry
            self._base_key = None
            self._steering: Steering = ()
            self._remove = None

        def _set_steering(self, steering: Steering, model_id: str) -> None:
            if self._remove is not None:
                self._remove()
                self._remove = None
            if steering:
                vectors = [(self.registry.resolve(ref), alpha) for ref, alpha in steering]
                self._remove = install(self.model, vectors, model_id=model_id)
            self._steering = steering

        def load(self, model_path, adapter_path=None, draft_model_path=None):
            adapter_path, steering = decode_adapter(adapter_path)
            preset = self.registry.presets.get(model_path)
            if preset is not None:
                model_path = preset.model or "default_model"
                steering = canonical(preset.steering + steering)

            # Same resolution as ModelProvider.load.
            model_path = self._model_map.get(model_path, model_path)
            adapter_path = self._adapter_map.get(model_path, adapter_path)
            draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)

            base_key = (model_path, adapter_path, draft_model_path)
            if self._base_key != base_key:
                # _load replaces the model, which discards any installed wrappers.
                self._remove, self._steering = None, ()
                self._load(*base_key)
                self._base_key = base_key
            if steering != self._steering:
                self._set_steering(steering, model_id=model_path)

            # The prompt cache and batch membership are keyed on this, so it
            # must change whenever steering does.
            self.model_key = (model_path, encode_adapter(adapter_path, steering), draft_model_path)
            return self.model, self.tokenizer

    return SteeringModelProvider


def _handler_class():
    import mlx_lm.server as S

    class SteeringAPIHandler(S.APIHandler):
        @property
        def _registry(self) -> SteeringRegistry:
            return self.response_generator.model_provider.registry

        def do_POST(self):
            # mlx-lm lets a ValueError from request validation escape, which
            # drops the connection with no response. Validation runs before any
            # response is written, so turn those into a 400.
            self._validated = False
            try:
                super().do_POST()
            except ValueError as e:
                if self._validated:
                    raise
                self._set_completion_headers(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                self.wfile.flush()

        def validate_model_parameters(self):
            super().validate_model_parameters()
            # A preset's own steering is added in the provider, so a request
            # naming a preset can stack more vectors on top of it.
            steering = self._registry.parse(self.body.get("steering"))
            self.adapter = encode_adapter(self.adapter, steering)
            self._validated = True

        def handle_models_request(self):
            presets = self._registry.presets
            if not presets:
                return super().handle_models_request()
            self._set_completion_headers(200)
            self.end_headers()
            data = [
                {
                    "id": name,
                    "object": "model",
                    "created": self.created,
                    "owned_by": "pts",
                    "steering": [{"vector": r, "alpha": a} for r, a in p.steering],
                }
                for name, p in sorted(presets.items())
            ]
            self.wfile.write(json.dumps({"object": "list", "data": data}).encode())
            self.wfile.flush()

    return SteeringAPIHandler


def make_server(host: str, port: int, cli_args: argparse.Namespace, registry: SteeringRegistry):
    """Build (but do not start) the HTTP server and its generation thread.

    Returns ``(httpd, response_generator)``. Call ``httpd.serve_forever()``;
    shut down with ``httpd.shutdown()`` then ``response_generator.stop_and_join()``.
    mlx-lm's ``run`` cannot be used because it drops ``handler_class``.
    """
    import socket
    from http.server import ThreadingHTTPServer

    import mlx_lm.server as S

    provider = _provider_class()(cli_args, registry)
    response_generator = S.ResponseGenerator(provider, S.LRUPromptCache(cli_args.prompt_cache_size))
    handler = _handler_class()
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    family, _, _, _, address = next(iter(infos))

    class Server(ThreadingHTTPServer):
        address_family = family

    httpd = Server(
        address,
        lambda *a, **kw: handler(
            response_generator, *a, system_fingerprint=S.get_system_fingerprint(), **kw
        ),
    )
    return httpd, response_generator


def serve(registry: SteeringRegistry, server_argv: Sequence[str]) -> None:
    cli_args = mlx_cli_args(server_argv)
    for name, preset in registry.presets.items():
        for ref, _ in preset.steering:
            registry.resolve(ref)  # fail at startup on a bad preset
    httpd, response_generator = make_server(cli_args.host, cli_args.port, cli_args, registry)
    logger.info(
        f"Serving {cli_args.model or '(model per request)'} on {cli_args.host}:{cli_args.port} "
        f"with {len(registry.named)} vector(s) and preset(s) {sorted(registry.presets) or 'none'}"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        response_generator.stop_and_join()
