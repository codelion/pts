"""
Pivotal-token search for a behavior, localised by sampled estimates.

For each question the model reasons before answering. A trace can only contain
a pivotal token if its final answer differs from where the question started,
and a trace's final answer is read for free from its own KV cache. So:

1. ``P(start)``: P(eventual answer matches | no reasoning yet), by sampling.
2. Sample traces; read each one's final answer. Traces whose answer did not
   move by ``prob_threshold`` are dropped without further sampling.
3. In the rest, bisect: estimate P at the midpoint of any interval whose ends
   differ by ``prob_threshold`` and recurse, down to single tokens. Every
   estimate is cached, and each trace has an estimate budget.

Each event's ``prob_before`` / ``prob_after`` are sampled estimates of the
eventual answer at that prefix (``metadata["estimator"]``). A forced
"answer now" screen was tried first and dropped: on Qwen2.5-1.5B sycophancy,
positions it picked had the same confirmed effect as random positions
(mean |dP| 0.054 vs 0.050, correlation 0.03).

Bisection can miss a rise and a fall that cancel inside one interval, as in
the original token PTS.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

from ..events import CausalReasoningEvent, make_token_event
from .evals import ChoiceItem
from .scoring import ChoiceScorer

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    traces_per_item: int = 4
    max_tokens: int = 300
    temperature: float = 0.7
    # Questions whose direct P(behavior) is outside this band are skipped: when
    # the answer is already certain, no single token can move it.
    min_direct: float = 0.1
    max_direct: float = 0.9
    samples: int = 32               # per estimate; standard error ~0.45/sqrt(n) for split answers
    prob_threshold: float = 0.2
    max_estimates_per_trace: int = 24


def search_item(
    scorer: ChoiceScorer,
    item: ChoiceItem,
    config: SearchConfig,
    model_id: str,
    behavior: str,
) -> Iterator[CausalReasoningEvent]:
    """Pivotal-token events for one question."""
    import mlx.core as mx

    direct = scorer.direct(item)
    if not config.min_direct <= direct.p_match <= config.max_direct:
        logger.debug(f"{item.item_id}: direct P={direct.p_match:.2f} outside band, skipped")
        return

    prompt = scorer.prompt_ids(item, cot=True)
    estimates: Dict[Tuple[int, ...], Tuple[float, float, float]] = {}

    def estimate(prefix: Sequence[int]) -> Tuple[float, float, float]:
        key = tuple(prefix)
        if key not in estimates:
            estimates[key] = scorer.estimate(
                item, prompt, prefix, config.samples, config.max_tokens, config.temperature,
            )
        return estimates[key]

    import numpy as np

    answer_tokens = set(np.array(scorer.label_ids(item.matching)).tolist()) | set(
        np.array(scorer.label_ids(item.not_matching)).tolist()
    )
    start = estimate([])
    traces, caches = scorer.sample(
        [prompt] * config.traces_per_item, config.max_tokens, config.temperature, return_caches=True
    )
    cue = mx.array(scorer.cue_ids(item))[None]

    for trace_index, (response, cache) in enumerate(zip(traces, caches)):
        end = scorer._dist(scorer.model(cue, cache=cache)[0, -1], item)
        scorer.cost.cue_passes += 1
        if len(response) < 2 or abs(end.p_match - start[0]) < config.prob_threshold:
            continue
        # The final answer of a finished trace is known exactly; it needs no sampling.
        estimates[tuple(response)] = (end.p_match, 0.0, end.mass)
        budget = [config.max_estimates_per_trace]
        found: List[int] = []

        def bisect(lo: int, hi: int) -> None:
            p_lo, p_hi = estimate(response[:lo])[0], estimate(response[:hi])[0]
            if abs(p_hi - p_lo) < config.prob_threshold:
                return
            if hi - lo == 1:
                found.append(lo)
                return
            if budget[0] <= 0:
                return
            budget[0] -= 1
            mid = (lo + hi) // 2
            bisect(lo, mid)
            bisect(mid, hi)

        bisect(0, len(response))
        if budget[0] <= 0:
            logger.debug(f"{item.item_id} trace {trace_index}: estimate budget exhausted")

        for t in found:
            before, se_before, mass_before = estimate(response[:t])
            after, se_after, mass_after = estimate(response[: t + 1])
            token_id = response[t]
            yield make_token_event(
                query=item.question,
                context=scorer.tok.decode(prompt + response[:t]),
                token=scorer.tok.decode([token_id]),
                token_id=token_id,
                prob_before=before,
                prob_after=after,
                model_id=model_id,
                task_type=behavior,
                dataset_id=item.source,
                dataset_item_id=item.item_id,
                position=len(prompt) + t,
                search_method="token_pts_sampled_bisection",
                metadata={
                    "behavior": behavior,
                    "matching_answer": item.matching,
                    "estimator": "sampled_answer_distribution",
                    "samples": config.samples,
                    "se_before": se_before,
                    "se_after": se_after,
                    "mass_before": mass_before,
                    "mass_after": mass_after,
                    "p_start": start[0],
                    "p_trace_end": end.p_match,
                    "direct_p": direct.p_match,
                    "trace_index": trace_index,
                    "response_position": t,
                    # Writing the answer down is pivotal by definition and says
                    # nothing about why; direction building skips these.
                    "states_answer": token_id in answer_tokens,
                },
            )


def search(
    scorer: ChoiceScorer,
    items: Sequence[ChoiceItem],
    config: SearchConfig,
    model_id: str,
    behavior: str,
    storage=None,
    show_progress: bool = True,
) -> List[CausalReasoningEvent]:
    from tqdm import tqdm

    events: List[CausalReasoningEvent] = []
    for item in tqdm(items, desc=f"Searching {behavior}", disable=not show_progress):
        for event in search_item(scorer, item, config, model_id, behavior):
            events.append(event)
            if storage is not None:
                storage.add_event(event)
    logger.info(
        f"{len(events)} pivotal tokens from {len(items)} questions; "
        f"cost {scorer.cost.as_dict()}; config {asdict(config)}"
    )
    return events
