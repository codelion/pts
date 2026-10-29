"""
Steering vectors: a backend-neutral format, plus the backends that apply it.

``vector`` (format, loading, stacking) and ``build`` need only numpy and
safetensors. The backends import their framework, so they are loaded on first
use, picked by the type of the model passed in.
"""

from importlib import import_module

from .vector import (  # noqa: F401
    CONFIG_NAME,
    HOOK_POINT,
    WEIGHTS_NAME,
    LayerPlan,
    LayerVector,
    SteeringVector,
    combine,
    load_vectors,
    parse_vector_spec,
)


def backend_for(model):
    """The backend module for a model: MLX for mlx-lm models, PyTorch otherwise."""
    if type(model).__module__.split(".")[0] in ("mlx", "mlx_lm"):
        return import_module("pts.steering.mlx_backend")
    return import_module("pts.steering.torch_backend")


def steer(model, vectors, model_id=None):
    """Context manager applying ``[(vector, alpha), ...]`` to an MLX or PyTorch model."""
    return backend_for(model).steer(model, vectors, model_id=model_id)


def capture(model, token_ids, layers):
    """Residual stream after each requested block, ``{layer: [T, d] float32}``."""
    return backend_for(model).capture(model, token_ids, layers)
