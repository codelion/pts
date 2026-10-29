"""
Steering vectors: a backend-neutral format, plus the backends that apply it.

``vector`` (format, loading, stacking) needs only numpy and safetensors. The
backends import their framework, so they are loaded on first use.
"""

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
