"""
Steering and residual capture for Hugging Face (PyTorch) models.

Forward hooks on the decoder blocks edit each block's output, which is the hook
point the format defines: the residual stream after block ``L``. Hooks stay
active through ``model.generate``, including cached decode steps, and are
removed on exit.
"""

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..latent.activations import get_layer_modules
from . import build
from .vector import LayerPlan, SteeringVector, combine


def model_shape(model) -> Tuple[int, int]:
    """``(hidden_size, num_layers)``."""
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    return int(config.hidden_size), len(get_layer_modules(model))


def _edit_output(output, fn):
    # Decoder blocks return a tensor or a tuple (hidden, *extras) depending on
    # the architecture and transformers version.
    if isinstance(output, tuple):
        return (fn(output[0]), *output[1:])
    return fn(output)


class _PlanTensors:
    """A layer plan as tensors, converted once per (device, dtype)."""

    def __init__(self, plan: LayerPlan):
        self.plan = plan
        self._cache: Dict[Tuple[torch.device, torch.dtype], Tuple] = {}

    def get(self, h: torch.Tensor):
        key = (h.device, h.dtype)
        if key not in self._cache:
            conv = lambda a: None if a is None else torch.from_numpy(a).to(h.device, h.dtype)
            self._cache[key] = (conv(self.plan.ablate), conv(self.plan.add))
        return self._cache[key]

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        ablate, add = self.get(h)
        if ablate is not None:
            h = h - (h @ ablate.T) @ ablate
        if add is not None:
            h = h + add
        return h


@contextmanager
def steer(
    model,
    vectors: Sequence[Tuple[SteeringVector, Optional[float]]],
    model_id: Optional[str] = None,
) -> Iterator[None]:
    """Apply one or more steering vectors for the duration of the block.

    ``vectors`` is a list of ``(vector, alpha)``; ``alpha=None`` uses the
    vector's ``default_alpha``. Hooks are removed on exit, even if generation
    raises.
    """
    hidden, num_layers = model_shape(model)
    for vec, _ in vectors:
        vec.check_compatible(hidden, num_layers, model_id)
    blocks = get_layer_modules(model)
    handles = []
    try:
        for i, plan in combine(vectors).items():
            edit = _PlanTensors(plan)
            handles.append(
                blocks[i].register_forward_hook(
                    lambda module, inputs, output, edit=edit: _edit_output(output, edit)
                )
            )
        yield
    finally:
        for h in handles:
            h.remove()


def capture(model, token_ids: Sequence[int], layers: Sequence[int]) -> Dict[int, np.ndarray]:
    """Residual stream after each requested block, ``{layer: [T, d] float32}``."""
    blocks = get_layer_modules(model)
    store: Dict[int, torch.Tensor] = {}
    handles = []

    def make(i):
        def hook(module, inputs, output):
            store[i] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    try:
        for i in layers:
            handles.append(blocks[i].register_forward_hook(make(i)))
        device = next(model.parameters()).device
        with torch.no_grad():
            model(torch.tensor([list(token_ids)], device=device))
    finally:
        for h in handles:
            h.remove()
    return {i: store[i][0].float().cpu().numpy() for i in layers}


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
        layers, position=position, base_model=base_model, behavior=behavior, backend="torch",
    )
