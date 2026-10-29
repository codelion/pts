"""
Pivotal-token search for a behavior: token PTS with an oracle-judged behavior.

The algorithm and estimator are token PTS's (``pts/searchers/token.py``):

1. ``P(start)``: P(success | prompt), by sampling completions and asking the
   oracle. Questions outside ``[min_start, max_start]`` are skipped: when the
   outcome is already settled, no token can move it.
2. Sample traces. A finished trace's outcome is known exactly (the oracle's
   verdict on it), so a trace that ends where it started needs no bisection.
3. Bisect the rest: estimate P at the midpoint of any interval whose ends
   differ by ``prob_threshold`` and recurse, down to single tokens. Every
   estimate is cached.

Bisection only ever asks whether two points differ by ``prob_threshold``. So
estimates start with ``min_samples`` and a comparison draws more, in rounds,
only while it is undecided: until the difference is ``confidence_z`` standard
errors above or below the threshold, or both sides have ``samples``. Clear
cases, the common ones once the answer is written, cost a fifth of a fixed
50-sample estimate. ``min_samples = samples`` is token PTS exactly.

Bisection can miss a rise and a fall that cancel inside one interval, as in
the original token PTS.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import math

from ..events import CausalReasoningEvent, make_token_event
from .evals import ChoiceItem, reads_as_label
from .scoring import ChoiceScorer, Estimate

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    traces_per_item: int = 10
    max_tokens: int = 512
    min_start: float = 0.2
    max_start: float = 0.8
    samples: int = 50               # most samples per estimate, as in token PTS
    min_samples: int = 10           # first round; more only while a comparison is undecided
    confidence_z: float = 2.0
    prob_threshold: float = 0.2
    max_estimates_per_trace: int = 0    # 0: no limit, as in token PTS


def search_item(
    scorer: ChoiceScorer,
    item: ChoiceItem,
    config: SearchConfig,
    model_id: str,
    behavior: str,
    traces_out: Optional[List[dict]] = None,
) -> Iterator[CausalReasoningEvent]:
    """Pivotal-token events for one question.

    Every trace whose outcome moved is appended to ``traces_out`` (when
    given), so controls can be built from the same traces the events came from.
    """
    prompt = scorer.prompt_ids(item)
    estimates: Dict[Tuple[int, ...], Estimate] = {}

    def ensure(prefixes: Sequence[Tuple[int, ...]], n: int) -> None:
        """Bring each prefix's estimate up to ``n`` samples, in one batch."""
        need: Dict[int, List[Tuple[int, ...]]] = {}
        for x in dict.fromkeys(prefixes):
            e = estimates.get(x, Estimate())
            if not e.exact and e.n < n:
                need.setdefault(n - e.n, []).append(x)
        for k, group in need.items():
            fresh = scorer.estimate_many([(item, prompt, list(x)) for x in group], k, config.max_tokens)
            for x, e in zip(group, fresh):
                estimates[x] = estimates.get(x, Estimate()) + e

    def var(e: Estimate) -> float:
        if e.exact:
            return 0.0
        q = (e.successes + 1) / (e.n + 2)   # keeps 0/n and n/n from claiming certainty
        return q * (1 - q) / e.n

    def differs(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
        """Whether P(success) at prefixes ``a`` and ``b`` differ by ``prob_threshold``."""
        ensure([a, b], config.min_samples)
        while True:
            ea, eb = estimates[a], estimates[b]
            gap = abs(eb.p - ea.p) - config.prob_threshold
            margin = config.confidence_z * math.sqrt(var(ea) + var(eb))
            if abs(gap) > margin:
                return gap > 0
            grow = [x for x in (a, b) if not estimates[x].exact and estimates[x].n < config.samples]
            if not grow:
                return gap >= 0
            ensure(grow, min(config.samples, max(estimates[x].n for x in grow) + config.min_samples))

    ensure([()], config.samples)
    start = estimates[()]
    if not config.min_start <= start.p <= config.max_start:
        logger.debug(f"{item.item_id}: P(start)={start.p:.2f} outside band, skipped")
        return
    traces = scorer.sample([prompt] * config.traces_per_item, config.max_tokens)

    for trace_index, response in enumerate(traces):
        success, answered = scorer.judge(item, response)
        end = Estimate(int(success), int(answered), 1, exact=True)
        estimates[tuple(response)] = end
        if len(response) < 2 or not differs((), tuple(response)):
            continue
        if traces_out is not None:
            traces_out.append({
                "item_id": item.item_id, "trace_index": trace_index,
                "prompt_ids": list(prompt), "response_ids": list(response),
                "p_start": start.p, "p_end": end.p,
            })
        budget = [config.max_estimates_per_trace or float("inf")]
        found: List[int] = []

        def bisect(lo: int, hi: int) -> None:
            if not differs(tuple(response[:lo]), tuple(response[:hi])):
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
            before, after = estimates[tuple(response[:t])], estimates[tuple(response[: t + 1])]
            token_id = response[t]
            token = scorer.tok.decode([token_id])
            yield make_token_event(
                query=item.question,
                context=scorer.tok.decode(prompt + response[:t]),
                token=token,
                token_id=token_id,
                prob_before=before.p,
                prob_after=after.p,
                model_id=model_id,
                task_type=behavior,
                dataset_id=item.source,
                dataset_item_id=item.item_id,
                position=len(prompt) + t,
                search_method="token_pts_bisection",
                metadata={
                    "behavior": behavior,
                    "matching_answer": item.matching,
                    "samples_before": before.n,
                    "samples_after": after.n,
                    "se_before": before.se,
                    "se_after": after.se,
                    "answered_before": before.answered,
                    "answered_after": after.answered,
                    "p_start": start.p,
                    "p_trace_end": end.p,
                    "trace_index": trace_index,
                    "response_position": t,
                    # Writing the answer down is pivotal by definition and says
                    # nothing about why; direction building skips these.
                    "states_answer": any(reads_as_label(token, c) for c in item.choices),
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
    traces_out: Optional[List[dict]] = None,
) -> List[CausalReasoningEvent]:
    from tqdm import tqdm

    events: List[CausalReasoningEvent] = []
    for item in tqdm(items, desc=f"Searching {behavior}", disable=not show_progress):
        for event in search_item(scorer, item, config, model_id, behavior, traces_out):
            events.append(event)
            if storage is not None:
                storage.add_event(event)
    logger.info(
        f"{len(events)} pivotal tokens from {len(items)} questions; "
        f"cost {scorer.cost.as_dict()}; config {asdict(config)}"
    )
    return events
