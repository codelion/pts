"""
Steering directions from pivotal-token events, and the CAA baseline.

All directions are oriented so that adding them pushes toward the behavior
(the matching answer); a negative alpha pushes away from it.

- ``matched_pairs``: at each event's prefix, the residual after the pivotal
  token minus the residual after the model's most likely *other* token, signed
  by the event's ``prob_delta``. Same prefix, one token different, so the
  difference isolates what that token did.
- ``signed_mean``: mean residual after tokens that raised P(behavior) minus
  after tokens that lowered it, centered per question so prompt content
  cannot leak in. Needs events of both signs.
- Controls, built the same way as ``matched_pairs`` so they differ only in
  the thing under test: ``sign_shuffled_pairs`` (same tokens, permuted signs)
  and ``random_position_pairs`` (random positions in the same traces, signed
  by the trace's overall direction).
- ``caa``: contrastive activation addition, the baseline. Residual after the
  matching answer label minus after the other label, at the direct-answer
  prompt, over both label orders.

Only emitted token events carry a ``prob_delta``; latent events never do and
are ignored here.
"""

from dataclasses import dataclass
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


@dataclass
class PairSpec:
    """One matched pair: at ``context_ids``, ``token_id`` vs the model's next-likeliest token."""

    context_ids: List[int]
    token_id: int
    sign: float          # +1 if the token pushes toward the behavior


def _pair_vectors(
    scorer: ChoiceScorer,
    capture: CaptureFn,
    specs: Sequence[PairSpec],
    layers: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """Mean of ``sign * (h(ctx + token) - h(ctx + alternative))``; the shared construction.

    The real method and its controls all go through here, so they differ only
    in which positions and signs they are given.
    """
    import mlx.core as mx

    if not specs:
        raise ValueError("No token events to build matched-pair directions from")
    sums = {i: None for i in layers}
    norms = {i: 0.0 for i in layers}
    for spec in specs:
        logits = scorer.model(mx.array(spec.context_ids)[None])[0, -1].astype(mx.float32)
        order = np.argsort(-np.array(logits))
        alt = next(int(t) for t in order[:5] if int(t) != spec.token_id)
        chosen = _last(capture, spec.context_ids + [spec.token_id], layers)
        other = _last(capture, spec.context_ids + [alt], layers)
        for i in layers:
            d = spec.sign * (chosen[i] - other[i])
            sums[i] = d if sums[i] is None else sums[i] + d
            norms[i] += float(np.linalg.norm(chosen[i]))
    n = len(specs)
    return {i: sums[i] / n for i in layers}, {i: norms[i] / n for i in layers}, n


def _event_specs(scorer: ChoiceScorer, events: Sequence[CausalReasoningEvent]) -> List[PairSpec]:
    return [
        PairSpec(_context_ids(scorer, e), e.token_id, 1.0 if e.prob_delta > 0 else -1.0)
        for e in _token_events(events)
    ]


def matched_pairs(
    scorer: ChoiceScorer,
    capture: CaptureFn,
    events: Sequence[CausalReasoningEvent],
    layers: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """The PTS direction: matched pairs at pivotal tokens, signed by their measured delta."""
    return _pair_vectors(scorer, capture, _event_specs(scorer, events), layers)


def sign_shuffled_pairs(
    scorer: ChoiceScorer,
    capture: CaptureFn,
    events: Sequence[CausalReasoningEvent],
    layers: Sequence[int],
    seed: int = 0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """Control: the same pivotal tokens with their signs permuted.

    Keeps the number of positive and negative signs. If this steers as well as
    ``matched_pairs``, the measured deltas carry no information the vector uses.
    """
    specs = _event_specs(scorer, events)
    signs = [s.sign for s in specs]
    np.random.default_rng(seed).shuffle(signs)
    return _pair_vectors(
        scorer, capture, [PairSpec(s.context_ids, s.token_id, g) for s, g in zip(specs, signs)], layers
    )


def random_position_pairs(
    scorer: ChoiceScorer,
    capture: CaptureFn,
    events: Sequence[CausalReasoningEvent],
    traces: Sequence[dict],
    layers: Sequence[int],
    seed: int = 0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """Control: one random position per pivotal token, in the same trace.

    Signed by the direction the whole trace moved. If this steers as well as
    ``matched_pairs``, finding the specific pivotal token adds nothing beyond
    knowing which way the trace went.
    """
    rng = np.random.default_rng(seed)
    by_trace = {(t["item_id"], t["trace_index"]): t for t in traces}
    used: Dict[Tuple[str, int], set] = {}
    for e in events:
        if e.event_type == EVENT_TOKEN and "response_position" in e.metadata:
            key = (e.dataset_item_id, e.metadata["trace_index"])
            used.setdefault(key, set()).add(e.metadata["response_position"])

    specs = []
    for e in _token_events(events):
        key = (e.dataset_item_id, e.metadata.get("trace_index"))
        trace = by_trace.get(key)
        if trace is None:
            continue
        response = trace["response_ids"]
        answer = set()
        if hasattr(scorer, "label_ids"):
            for label in ("A", "B", "Yes", "No"):
                try:
                    answer |= set(np.array(scorer.label_ids(label)).tolist())
                except ValueError:
                    pass
        options = [
            t for t in range(len(response))
            if t not in used.get(key, set()) and response[t] not in answer
        ]
        if not options:
            continue
        t = int(rng.choice(options))
        sign = 1.0 if trace["p_end"] > trace["p_start"] else -1.0
        specs.append(PairSpec(list(trace["prompt_ids"]) + list(response[:t]), int(response[t]), sign))
    return _pair_vectors(scorer, capture, specs, layers)


def signed_mean(
    scorer: ChoiceScorer,
    capture: CaptureFn,
    events: Sequence[CausalReasoningEvent],
    layers: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """Mean residual after behavior-raising tokens minus after lowering ones,
    centered per question.

    Without centering, positive and negative events come from different
    questions, and the difference can encode what distinguishes those prompts
    (e.g. the persona) instead of anything the tokens did. Subtracting each
    question's own mean removes that; questions with a single event therefore
    contribute nothing.
    """
    rows = []
    for e in _token_events(events):
        h = _last(capture, _context_ids(scorer, e) + [e.token_id], layers)
        rows.append((e.dataset_item_id, 1.0 if e.prob_delta > 0 else -1.0, h))
    if not any(s > 0 for _, s, _ in rows) or not any(s < 0 for _, s, _ in rows):
        raise ValueError("signed_mean needs events that raise and events that lower P(behavior)")
    by_q: Dict[str, List[int]] = {}
    for k, (q, _, _) in enumerate(rows):
        by_q.setdefault(q, []).append(k)
    vecs, norms = {}, {}
    for i in layers:
        centered = [None] * len(rows)
        for ks in by_q.values():
            mean = np.mean([rows[k][2][i] for k in ks], axis=0)
            for k in ks:
                centered[k] = rows[k][2][i] - mean
        pos = [c for c, (_, s, _) in zip(centered, rows) if s > 0]
        neg = [c for c, (_, s, _) in zip(centered, rows) if s < 0]
        vecs[i] = np.mean(pos, axis=0) - np.mean(neg, axis=0)
        norms[i] = float(np.mean([np.linalg.norm(r[2][i]) for r in rows]))
    if all(np.abs(v).max() == 0 for v in vecs.values()):
        raise ValueError("signed_mean: no question has more than one event, nothing left after centering")
    return vecs, norms, len(rows)


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
