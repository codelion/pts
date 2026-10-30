"""
What both steering servers share: which vectors requests may use, how a
request's ``steering`` field is parsed, and presets clients select as the
model name. Pure Python; the MLX and PyTorch servers differ only in how they
generate.

By default requests may only name vectors the operator registered with
``--vector name=path``. ``allow_any`` lets requests pass paths and ``hf://``
ids, which makes the server read local files and download from the Hub on a
client's say-so.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .vector import SteeringVector, parse_vector_spec

Steering = Tuple[Tuple[str, Optional[float]], ...]


def canonical(steering) -> Steering:
    """Sorted, so equivalent configurations share batches and cache entries.

    Order never changes the result: additions sum and ablations are
    orthonormalised together.
    """
    return tuple(sorted(steering, key=lambda s: (s[0], float("-inf") if s[1] is None else s[1])))


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
