"""
Building steering vectors from captured activations.

Backend-neutral: each backend supplies ``capture(token_ids, layers)`` returning
``{layer: [T, d] float32}``, and the arithmetic here is shared, so a vector
built on MLX and one built on PyTorch from the same weights are the same vector.
"""

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .vector import LayerVector, SteeringVector

CaptureFn = Callable[[Sequence[int], Sequence[int]], Dict[int, np.ndarray]]


def mean_difference(
    capture: CaptureFn,
    hidden_size: int,
    num_layers: int,
    positive: Sequence[Sequence[int]],
    negative: Sequence[Sequence[int]],
    layers: Sequence[int],
    position: int = -1,
    base_model: Optional[str] = None,
    behavior: Optional[str] = None,
    backend: Optional[str] = None,
) -> SteeringVector:
    """``mean(h | positive) - mean(h | negative)`` at one token position, per layer.

    Each example is a list of token ids; the residual is read at ``position``
    (the last token by default). Only running sums are kept, so memory does not
    grow with the number of examples. This is the contrastive-activation-addition
    baseline: it needs no events, only paired examples.
    """
    if not positive or not negative:
        raise ValueError("mean_difference needs at least one positive and one negative example")
    layers = list(layers)
    sums = {s: {i: np.zeros(hidden_size, dtype=np.float64) for i in layers} for s in ("pos", "neg")}
    norm_sum = {i: 0.0 for i in layers}

    for side, examples in (("pos", positive), ("neg", negative)):
        for ids in examples:
            acts = capture(ids, layers)
            for i in layers:
                h = acts[i][position].astype(np.float64)
                sums[side][i] += h
                norm_sum[i] += float(np.linalg.norm(h))

    n_pos, n_neg = len(positive), len(negative)
    entries = [
        LayerVector(
            layer=i,
            op="add",
            vector=(sums["pos"][i] / n_pos - sums["neg"][i] / n_neg).astype(np.float32),
            ref_norm=norm_sum[i] / (n_pos + n_neg),
        )
        for i in layers
    ]
    return SteeringVector(
        entries=entries,
        hidden_size=hidden_size,
        num_layers=num_layers,
        base_model=base_model,
        behavior=behavior,
        provenance={
            "method": "mean_difference",
            "position": position,
            "num_positive": n_pos,
            "num_negative": n_neg,
            "backend": backend,
        },
    )
