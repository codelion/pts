"""
Does a steering vector change the behavior, and what does it break?

Every steered measurement reports three numbers:

- ``p_match``: P(success), the fraction of sampled completions the oracle
  accepts, with the same estimator the search uses.
- ``answered``: the fraction of completions with a recognisable answer. A
  vector that stops the model answering would otherwise look like it reduced
  the behavior; a floor on this catches it.
- ``kl``: mean KL(unsteered || steered) of next-token distributions on neutral
  text, the cost of the vector to everything else.

Layer and strength are chosen on the selection split; only the test split is
reported, alongside the same measurements for baselines and controls. Effects
are paired per question and carry bootstrap intervals, because test splits
are small.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .evals import ChoiceItem
from .scoring import ChoiceScorer

NEUTRAL_TEXTS = (
    "The river flows through the valley and into the sea, carrying sediment from the mountains.",
    "To bake bread, mix flour, water, salt and yeast, then let the dough rise for several hours.",
    "The committee will meet on Tuesday to review the budget and discuss next year's plans.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "She opened the window, and the sound of traffic and distant birds filled the small room.",
    "A prime number has exactly two divisors: one and itself. The first few are 2, 3, 5 and 7.",
    "The museum's new exhibit features pottery, textiles and tools from several ancient cultures.",
    "Regular exercise, enough sleep and a balanced diet all contribute to long-term health.",
)


@dataclass
class Score:
    p_match: float
    answered: float
    kl: float = 0.0
    per_item: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        return {"p_match": self.p_match, "answered": self.answered, "kl": self.kl}


def bootstrap_ci(values: Sequence[float], n: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap interval for the mean of ``values``."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return (float(lo), float(hi))


def paired_effect(steered: Score, base: Score, seed: int = 0) -> Dict[str, float]:
    """Mean per-question change in P(behavior), with a bootstrap interval."""
    diffs = np.array(steered.per_item) - np.array(base.per_item)
    lo, hi = bootstrap_ci(diffs, seed=seed)
    return {"delta": float(diffs.mean()), "ci_low": lo, "ci_high": hi}


def _neutral_logprobs(scorer: ChoiceScorer) -> List[np.ndarray]:
    import mlx.core as mx

    out = []
    for text in NEUTRAL_TEXTS:
        ids = list(scorer.tok.encode(text))
        logits = scorer.model(mx.array(ids)[None])[0].astype(mx.float32)
        out.append(np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True)))
    return out


class Evaluator:
    """Scores items with and without steering by sampling ``samples``
    completions per question, all questions in one batch."""

    def __init__(self, scorer: ChoiceScorer, model_id: Optional[str] = None,
                 samples: int = 16, max_tokens: int = 512):
        self.scorer = scorer
        self.model_id = model_id
        self.samples = samples
        self.max_tokens = max_tokens
        self._neutral = _neutral_logprobs(scorer)

    def _measure(self, items: Sequence[ChoiceItem], kl: bool) -> Score:
        requests = [(it, self.scorer.prompt_ids(it), []) for it in items]
        est = self.scorer.estimate_many(requests, self.samples, self.max_tokens)
        score = Score(
            p_match=float(np.mean([e.p for e in est])),
            answered=float(np.mean([e.answered for e in est])),
            per_item=[e.p for e in est],
        )
        if kl:
            steered = _neutral_logprobs(self.scorer)
            score.kl = float(np.mean([
                (np.exp(a) * (a - b)).sum(-1).mean() for a, b in zip(self._neutral, steered)
            ]))
        return score

    def score(self, items: Sequence[ChoiceItem], steering=(), prompt: Optional[str] = None) -> Score:
        """P(behavior) on ``items``, optionally steered and/or with an added instruction."""
        from contextlib import nullcontext

        from ..steering import steer

        previous = self.scorer.extra_instruction
        self.scorer.extra_instruction = prompt
        try:
            ctx = steer(self.scorer.model, steering, model_id=self.model_id) if steering else nullcontext()
            with ctx:
                return self._measure(items, kl=bool(steering))
        finally:
            self.scorer.extra_instruction = previous


def single_layer(vec, layer: int, scale: float = 1.0):
    """The part of a SteeringVector at one layer, as its own vector."""
    from ..steering.vector import LayerVector, SteeringVector

    e = next(e for e in vec.entries if e.layer == layer and e.op == "add")
    return SteeringVector(
        [LayerVector(layer, "add", e.vector * scale, e.ref_norm)], vec.hidden_size, vec.num_layers,
        base_model=vec.base_model, behavior=vec.behavior, provenance=dict(vec.provenance),
    )


def random_like(vec, layer: int, seed: int = 0):
    """A random direction with the same norm as ``vec`` at ``layer``: the null baseline."""
    from ..steering.vector import LayerVector, SteeringVector

    e = next(e for e in vec.entries if e.layer == layer and e.op == "add")
    r = np.random.default_rng(seed).normal(size=e.vector.shape)
    r = r / np.linalg.norm(r) * np.linalg.norm(e.vector)
    return SteeringVector(
        [LayerVector(layer, "add", r.astype(np.float32), e.ref_norm)], vec.hidden_size, vec.num_layers,
        base_model=vec.base_model, provenance={"method": "random", "seed": seed},
    )


def sweep(
    evaluator: Evaluator,
    vec,
    items: Sequence[ChoiceItem],
    layers: Sequence[int],
    alphas: Sequence[float],
) -> List[Dict[str, float]]:
    rows = []
    for layer in layers:
        one = single_layer(vec, layer)
        for alpha in alphas:
            s = evaluator.score(items, [(one, alpha)])
            rows.append({"layer": layer, "alpha": alpha, **s.as_dict()})
    return rows


def choose(
    rows: Sequence[Dict[str, float]],
    base: Score,
    direction: int,
    min_answered_ratio: float = 0.9,
    max_kl: float = 0.5,
) -> Optional[Dict[str, float]]:
    """The setting that moves P(behavior) furthest in ``direction`` without damage.

    Damage means answering less than ``min_answered_ratio`` times as often as
    unsteered, or a mean neutral-text KL above ``max_kl`` nats. Returns None if
    nothing both passes and moves P(behavior) the right way.
    """
    ok = [
        r for r in rows
        if r["answered"] >= min_answered_ratio * base.answered and r["kl"] <= max_kl
        and direction * (r["p_match"] - base.p_match) > 0
    ]
    return max(ok, key=lambda r: direction * (r["p_match"] - base.p_match)) if ok else None
