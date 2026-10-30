"""
An OpenAI-compatible server with steering, on PyTorch / transformers.

The same requests as the MLX server (``pts serve``): ``/v1/completions``,
``/v1/chat/completions`` (both with ``stream``), and ``/v1/models``; the same
``steering`` field and presets (``serving.py``). Generation is
``model.generate``.

Requests run one at a time. Each applies its own steering hooks for exactly
the duration of its generation, so differently-steered requests can never
share a forward pass or a KV cache. That costs the batching the MLX server
has; the model is loaded once either way.
"""

import json
import logging
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .serving import Steering, SteeringRegistry, canonical

logger = logging.getLogger(__name__)


class TorchSteeringServer:
    """One loaded model, the vectors requests may use, and a generation lock."""

    def __init__(self, model, tokenizer, model_id: str, registry: SteeringRegistry,
                 max_tokens: int = 512, temperature: float = 0.0):
        self.model = model
        self.tok = tokenizer
        self.model_id = model_id
        self.registry = registry
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.lock = threading.Lock()

    # -- requests -----------------------------------------------------------

    def steering_for(self, body: Dict[str, Any]) -> Steering:
        """A request's steering: its preset's (if ``model`` names one) plus its own."""
        model = body.get("model")
        steering: Steering = ()
        preset = self.registry.presets.get(model) if model else None
        if preset is not None:
            if preset.model and preset.model != self.model_id:
                raise ValueError(f"Preset {model!r} is for {preset.model}; this server runs {self.model_id}")
            steering = preset.steering
        elif model not in (None, "", "default_model", self.model_id):
            raise ValueError(f"This server runs {self.model_id}; unknown model or preset {model!r}")
        return canonical(steering + self.registry.parse(body.get("steering")))

    def prompt_ids(self, body: Dict[str, Any], chat: bool) -> List[int]:
        if chat:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **(body.get("chat_template_kwargs") or {}),
            )
            # The rendered template already carries its special tokens.
            return list(self.tok.encode(text, add_special_tokens=False))
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        return list(self.tok.encode(prompt))

    def sampling(self, body: Dict[str, Any]) -> Dict[str, Any]:
        def number(key, default, kind=float):
            value = body.get(key, default)
            if value is None:
                return default
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            return kind(value)

        max_tokens = number("max_completion_tokens", None, int) or number("max_tokens", self.max_tokens, int)
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        temperature = number("temperature", self.temperature)
        out: Dict[str, Any] = {"max_new_tokens": max_tokens}
        if temperature > 0:
            out.update(do_sample=True, temperature=temperature,
                       top_p=number("top_p", 1.0), top_k=number("top_k", 0, int))
        else:
            out.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        stop = body.get("stop")
        if stop:
            stop = [stop] if isinstance(stop, str) else stop
            if not all(isinstance(s, str) for s in stop):
                raise ValueError("stop must be a string or a list of strings")
            out.update(stop_strings=stop, tokenizer=self.tok)
        return out

    # -- generation ---------------------------------------------------------

    def _run(self, ids: List[int], steering: Steering, params: Dict[str, Any], streamer=None):
        import torch

        from .torch_backend import steer

        vectors = [(self.registry.resolve(ref), alpha) for ref, alpha in steering]
        device = next(self.model.parameters()).device
        inputs = torch.tensor([ids], device=device)
        with torch.no_grad(), steer(self.model, vectors, model_id=self.model_id):
            out = self.model.generate(
                inputs, attention_mask=torch.ones_like(inputs), streamer=streamer,
                pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id,
                **params,
            )
        return out[0, len(ids):].tolist()

    def _finish(self, new_ids: List[int], params: Dict[str, Any]) -> Tuple[str, str]:
        eos = self.tok.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        reason = "length" if len(new_ids) >= params["max_new_tokens"] else "stop"
        text = self.tok.decode([t for t in new_ids if t not in eos], skip_special_tokens=True)
        for s in params.get("stop_strings") or []:
            if s in text:
                text, reason = text[:text.index(s)], "stop"
        return text, reason

    def complete(self, ids, steering, params) -> Tuple[str, str, int]:
        with self.lock:
            new_ids = self._run(ids, steering, params)
        text, reason = self._finish(new_ids, params)
        return text, reason, len(new_ids)

    def stream(self, ids, steering, params) -> Iterator[Tuple[str, Optional[str]]]:
        """Yield ``(text_piece, None)`` as tokens arrive, then ``("", finish_reason)``."""
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        result: Dict[str, Any] = {}

        def work():
            try:
                result["ids"] = self._run(ids, steering, params, streamer)
            except Exception as e:  # surfaced to the client below
                result["error"] = e
                streamer.end()

        with self.lock:
            thread = threading.Thread(target=work, daemon=True)
            thread.start()
            sent, stops = "", params.get("stop_strings") or []
            for piece in streamer:
                full = sent + piece
                hits = [full.index(s) for s in stops if s in full]
                if hits:
                    # Send up to the stop string; generate() stops on it too.
                    if min(hits) > len(sent):
                        yield full[len(sent):min(hits)], None
                    break
                sent = full
                if piece:
                    yield piece, None
            thread.join()
        if "error" in result:
            raise result["error"]
        _, reason = self._finish(result.get("ids", []), params)
        yield "", reason


def _handler_class(server: TorchSteeringServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            logger.debug(fmt % args)

        def _json(self, status: int, payload: Dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") != "/v1/models":
                return self._json(404, {"error": f"Unknown path {self.path}"})
            created = int(time.time())
            data = [{"id": server.model_id, "object": "model", "created": created, "owned_by": "pts"}]
            data += [
                {"id": name, "object": "model", "created": created, "owned_by": "pts",
                 "steering": [{"vector": r, "alpha": a} for r, a in p.steering]}
                for name, p in sorted(server.registry.presets.items())
            ]
            self._json(200, {"object": "list", "data": data})

        def do_POST(self):
            path = self.path.rstrip("/")
            if path not in ("/v1/completions", "/v1/chat/completions"):
                return self._json(404, {"error": f"Unknown path {self.path}"})
            chat = path.endswith("chat/completions")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                steering = server.steering_for(body)
                ids = server.prompt_ids(body, chat)
                params = server.sampling(body)
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})

            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            model = body.get("model") or server.model_id
            created = int(time.time())
            kind = "chat.completion" if chat else "text_completion"

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def chunk(text, reason):
                    choice = ({"index": 0, "delta": {"content": text} if text else {}, "finish_reason": reason}
                              if chat else {"index": 0, "text": text, "finish_reason": reason})
                    msg = {"id": rid, "object": kind + ".chunk" if chat else kind, "created": created,
                           "model": model, "choices": [choice]}
                    self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                    self.wfile.flush()

                try:
                    for text, reason in server.stream(ids, steering, params):
                        chunk(text, reason)
                except Exception as e:  # headers are sent; report in-band
                    logger.exception("generation failed")
                    self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            try:
                text, reason, n = server.complete(ids, steering, params)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}
                      if chat else {"index": 0, "text": text, "finish_reason": reason})
            self._json(200, {
                "id": rid, "object": kind, "created": created, "model": model, "choices": [choice],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n},
            })

    return Handler


def make_server(host: str, port: int, server: TorchSteeringServer) -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server; call ``serve_forever()``."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    family, _, _, _, address = next(iter(infos))

    class Server(ThreadingHTTPServer):
        address_family = family
        daemon_threads = True

    return Server(address, _handler_class(server))


def serve(server: TorchSteeringServer, host: str, port: int) -> None:
    for preset in server.registry.presets.values():
        for ref, _ in preset.steering:
            server.registry.resolve(ref)  # fail at startup on a bad preset
    httpd = make_server(host, port, server)
    logger.info(
        f"Serving {server.model_id} (PyTorch) on {host}:{port} with {len(server.registry.named)} "
        f"vector(s) and preset(s) {sorted(server.registry.presets) or 'none'}"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
