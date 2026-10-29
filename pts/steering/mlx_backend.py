"""
Steering and residual capture for mlx-lm models.

mlx-lm has no hooks, but every model exposes its decoder blocks as a plain
Python list (``model.layers``). Swapping a block for a wrapper that edits the
block's output is therefore enough, and it works unchanged with mlx-lm's
``generate``, its server, and quantised checkpoints. The wrapper's output is
the hook point the format defines: the residual stream after block ``L``.
"""

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import build
from .vector import LayerPlan, SteeringVector, combine


def get_layers(model) -> list:
    """The model's list of decoder blocks. Mutating it changes the model."""
    for path in (("layers",), ("model", "layers"), ("language_model", "model", "layers")):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, list):
            return obj
    raise ValueError(f"Cannot find the decoder blocks of {type(model).__name__}")


def model_shape(model) -> Tuple[int, int]:
    """``(hidden_size, num_layers)``."""
    args = getattr(model, "args", None)
    hidden = getattr(args, "hidden_size", None)
    if hidden is None and isinstance(getattr(args, "text_config", None), dict):
        hidden = args.text_config.get("hidden_size")
    if hidden is None:
        raise ValueError(f"Cannot read hidden_size from {type(model).__name__}")
    return int(hidden), len(get_layers(model))


class _Holder:
    """Keeps arrays off the module tree, so they never show up as parameters."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _edit_output(out, fn):
    # A few architectures return extra values alongside the hidden state.
    if isinstance(out, tuple):
        return (fn(out[0]), *out[1:])
    return fn(out)


class _BlockWrapper(nn.Module):
    """Wraps a decoder block and looks like it.

    Model code reads attributes off its blocks (``layer.use_sliding``,
    ``layer.self_attn``...), so anything the wrapper lacks is forwarded to the
    wrapped block.
    """

    def __getattr__(self, key: str):
        try:
            return super().__getattr__(key)
        except AttributeError:
            # During __init__ the block is not set yet; behave like a plain module.
            block = dict.get(self, "block")
            if block is None:
                raise
            return getattr(block, key)


class SteeredBlock(_BlockWrapper):
    """A decoder block whose output is ablated, then shifted."""

    def __init__(self, block: nn.Module, plan: LayerPlan):
        super().__init__()
        self.block = block
        self._plan = _Holder(
            ablate=None if plan.ablate is None else mx.array(plan.ablate),
            add=None if plan.add is None else mx.array(plan.add),
        )

    def __call__(self, x, *args, **kwargs):
        return _edit_output(self.block(x, *args, **kwargs), self._steer)

    def _steer(self, h):
        p = self._plan
        if p.ablate is not None:
            u = p.ablate.astype(h.dtype)
            h = h - (h @ u.T) @ u
        if p.add is not None:
            h = h + p.add.astype(h.dtype)
        return h


class _RecordingBlock(_BlockWrapper):
    def __init__(self, block: nn.Module, sink: List[mx.array]):
        super().__init__()
        self.block = block
        self._sink = _Holder(items=sink)

    def __call__(self, x, *args, **kwargs):
        out = self.block(x, *args, **kwargs)
        self._sink.items.append(out[0] if isinstance(out, tuple) else out)
        return out


@contextmanager
def _swapped(model, wrappers: Dict[int, nn.Module]) -> Iterator[None]:
    layers = get_layers(model)
    originals = {i: layers[i] for i in wrappers}
    try:
        for i, w in wrappers.items():
            layers[i] = w
        yield
    finally:
        for i, block in originals.items():
            layers[i] = block


@contextmanager
def steer(
    model,
    vectors: Sequence[Tuple[SteeringVector, Optional[float]]],
    model_id: Optional[str] = None,
) -> Iterator[None]:
    """Apply one or more steering vectors for the duration of the block.

    ``vectors`` is a list of ``(vector, alpha)``; ``alpha=None`` uses the
    vector's ``default_alpha``. The original blocks are restored on exit, even
    if generation raises.
    """
    hidden, num_layers = model_shape(model)
    for vec, _ in vectors:
        vec.check_compatible(hidden, num_layers, model_id)
    plans = combine(vectors)
    layers = get_layers(model)
    with _swapped(model, {i: SteeredBlock(layers[i], p) for i, p in plans.items()}):
        yield


def capture(model, token_ids: Sequence[int], layers: Sequence[int]) -> Dict[int, np.ndarray]:
    """Residual stream after each requested block, ``{layer: [T, d] float32}``."""
    blocks = get_layers(model)
    sinks: Dict[int, List[mx.array]] = {i: [] for i in layers}
    with _swapped(model, {i: _RecordingBlock(blocks[i], sinks[i]) for i in layers}):
        model(mx.array(list(token_ids))[None])
    out = {}
    for i, items in sinks.items():
        h = items[0][0].astype(mx.float32)
        mx.eval(h)
        out[i] = np.array(h)
    return out


def mean_difference(
    model,
    positive: Sequence[Sequence[int]],
    negative: Sequence[Sequence[int]],
    layers: Sequence[int],
    position: int = -1,
    base_model: Optional[str] = None,
    behavior: Optional[str] = None,
) -> SteeringVector:
    """Mean-difference vector from paired examples; see ``build.mean_difference``."""
    hidden, num_layers = model_shape(model)
    return build.mean_difference(
        lambda ids, ls: capture(model, ids, ls), hidden, num_layers, positive, negative,
        layers, position=position, base_model=base_model, behavior=behavior, backend="mlx",
    )
