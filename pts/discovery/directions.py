"""
Steering directions from pivotal-token events, and the CAA baseline.

All directions are oriented so that adding them pushes toward the behavior
(the matching answer); a negative alpha pushes away from it.

- ``matched_pairs``: at each event's prefix, the residual after the pivotal
  token minus the residual after the model's most likely *other* token, signed
  by the event's ``prob_delta``. Same prefix, one token different, so the
  difference isolates what that token did.
- ``signed_mean``: mean residual after tokens that raised P(behavior) minus
  after tokens that lowered it. Needs events of both signs.
- ``caa``: contrastive activation addition, the baseline. Residual after the
  matching answer label minus after the other label, at the direct-answer
  prompt, over both label orders.

Only emitted token events carry a ``prob_delta``; latent events never do and
are ignored here.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..events import EVENT_TOKEN, CausalReasoningEvent
from .evals import ChoiceItem
from .scoring import ChoiceScorer

CaptureFn = Callable[[Sequence[int], Sequence[int]], Dict[int, np.ndarray]]


def _token_events(events: Sequence[CausalReasoningEvent]) -> List[CausalReasoningEvent]:
    """Emitted token events with a delta, excluding tokens that just state the answer."""
    return [
        e for e in events
        if e.event_type == EVENT_TOKEN and e.prob_delta is not None and e.token_id is not None
        and not e.metadata.get("states_answer", False)
    ]


def _context_ids(scorer: ChoiceScorer, event: CausalReasoningEvent) -> List[int]:
    # event.context already contains its special tokens.
    return list(scorer.tok.encode(event.context, add_special_tokens=False))


def _last(capture: CaptureFn, ids: Sequence[int], layers: Sequence[int]) -> Dict[int, np.ndarray]:
    acts = capture(ids, layers)
    return {i: acts[i][-1].astype(np.float64) for i in layers}


def matched_pairs(
    scorer: ChoiceScorer,
    capture: CaptureFn,
    events: Sequence[CausalReasoningEvent],
    layers: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """``(vectors, mean residual norms, pairs used)``."""
    import mlx.core as mx

    sums = {i: np.zeros(0) for i in layers}
    norms = {i: 0.0 for i in layers}
    n = 0
    for e in _token_events(events):
        ctx = _context_ids(scorer, e)
        logits = scorer.model(mx.array(ctx)[None])[0, -1].astype(mx.float32)
        order = np.argsort(-np.array(logits))
        alt = next(int(t) for t in order[:5] if int(t) != e.token_id)
        chosen = _last(capture, ctx + [e.token_id], layers)
        other = _last(capture, ctx + [alt], layers)
        sign = 1.0 if e.prob_delta > 0 else -1.0
        for i in layers:
            d = sign * (chosen[i] - other[i])
            sums[i] = d if sums[i].size == 0 else sums[i] + d
            norms[i] += float(np.linalg.norm(chosen[i]))
        n += 1
    if n == 0:
        raise ValueError("No token events to build matched-pair directions from")
    return {i: sums[i] / n for i in layers}, {i: norms[i] / n for i in layers}, n


def signed_mean(
    scorer: ChoiceScorer,
    capture: CaptureFn,
    events: Sequence[CausalReasoningEvent],
    layers: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    pos = {i: [] for i in layers}
    neg = {i: [] for i in layers}
    for e in _token_events(events):
        h = _last(capture, _context_ids(scorer, e) + [e.token_id], layers)
        for i in layers:
            (pos if e.prob_delta > 0 else neg)[i].append(h[i])
    if not pos[layers[0]] or not neg[layers[0]]:
        raise ValueError("signed_mean needs events that raise and events that lower P(behavior)")
    vecs = {i: np.mean(pos[i], axis=0) - np.mean(neg[i], axis=0) for i in layers}
    norms = {i: float(np.mean([np.linalg.norm(h) for h in pos[i] + neg[i]])) for i in layers}
    return vecs, norms, len(pos[layers[0]]) + len(neg[layers[0]])


def caa(
    scorer: ChoiceScorer,
    capture: CaptureFn,
    items: Sequence[ChoiceItem],
    layers: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """Contrastive activation addition over both label orders of each item."""
    import mlx.core as mx

    sums = {i: np.zeros(0) for i in layers}
    norms = {i: 0.0 for i in layers}
    n = 0
    for base in items:
        for item in filter(None, (base, base.swapped())):
            prompt = scorer.prompt_ids(item, cot=False)
            p = mx.softmax(scorer.model(mx.array(prompt)[None])[0, -1].astype(mx.float32))
            p = np.array(p)

            def best(label: str) -> int:
                ids = np.array(scorer.label_ids(label))
                return int(ids[np.argmax(p[ids])])

            yes = _last(capture, prompt + [best(item.matching)], layers)
            no = _last(capture, prompt + [best(item.not_matching)], layers)
            for i in layers:
                d = yes[i] - no[i]
                sums[i] = d if sums[i].size == 0 else sums[i] + d
                norms[i] += float(np.linalg.norm(yes[i]))
            n += 1
    if n == 0:
        raise ValueError("No items to build a CAA direction from")
    return {i: sums[i] / n for i in layers}, {i: norms[i] / n for i in layers}, n


def to_vector(
    vecs: Dict[int, np.ndarray],
    norms: Dict[int, float],
    hidden_size: int,
    num_layers: int,
    method: str,
    count: int,
    base_model: Optional[str] = None,
    behavior: Optional[str] = None,
):
    from ..steering.vector import LayerVector, SteeringVector

    return SteeringVector(
        entries=[LayerVector(i, "add", vecs[i].astype(np.float32), ref_norm=norms[i]) for i in sorted(vecs)],
        hidden_size=hidden_size,
        num_layers=num_layers,
        base_model=base_model,
        behavior=behavior,
        provenance={"method": method, "count": count, "backend": "mlx"},
    )
