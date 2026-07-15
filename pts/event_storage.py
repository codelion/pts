"""
Storage for unified causal reasoning events (PTS).

``EventStorage`` holds ``CausalReasoningEvent`` records and reads/writes JSONL.
It accepts legacy pivotal-token and legacy thought-anchor records on load and migrates
them on the way in, so an old dataset opens without a separate migration step.
"""

import os
import json
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional

from .events import (
    CausalReasoningEvent,
    EVENT_LATENT,
    EVENT_SENTENCE,
    EVENT_TOKEN,
    from_any_record,
)

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """Coerce numpy/torch scalars and arrays into JSON.

    The verification pass puts torch/numpy floats into event metadata
    (``attention_entropy``, ``attention_focus_score``). Previously ``save()``
    caught the resulting TypeError *per event*, logged it, and carried on -- so
    a run would drop events on the floor, write a short file, and exit 0
    reporting success. Converting is right; silently losing data is not, and a
    genuinely unserializable value should still raise.
    """
    for attr in ("item", "tolist"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except (ValueError, TypeError):
                continue
    raise TypeError(
        f"Cannot serialize {type(obj).__name__} in an event. Convert it to a "
        "plain Python value before storing it in metadata."
    )


class EventStorage:
    """A collection of causal reasoning events backed by a JSONL file."""

    def __init__(self, filepath: Optional[str] = None, autoload: bool = True):
        self.events: List[CausalReasoningEvent] = []
        self.filepath = filepath
        self._index: Dict[str, CausalReasoningEvent] = {}

        if filepath and autoload and os.path.exists(filepath):
            self.load(filepath)

    # -- mutation ---------------------------------------------------------

    def add_event(self, event: Any) -> Optional[CausalReasoningEvent]:
        """Add an event. Accepts PTS events, legacy records, or legacy dataclasses.

        Adding an event whose ``event_id`` is already present is a no-op. The
        v1 searchers wrote to storage *and* let the CLI write the same object
        again, which duplicated every record; de-duplicating here makes that
        double-write harmless rather than silently doubling the dataset.
        """
        try:
            evt = from_any_record(event)
        except (ValueError, TypeError) as e:
            logger.error(f"Could not add event to storage: {e}")
            return None

        if evt.event_id in self._index:
            return self._index[evt.event_id]

        self.events.append(evt)
        self._index[evt.event_id] = evt
        return evt

    def add_events(self, events: List[Any]) -> None:
        for event in events:
            self.add_event(event)

    def get(self, event_id: str) -> Optional[CausalReasoningEvent]:
        return self._index.get(event_id)

    # -- persistence ------------------------------------------------------

    def save(self, filepath: Optional[str] = None) -> None:
        filepath = filepath or self.filepath
        if not filepath:
            raise ValueError("No filepath specified for saving")

        directory = os.path.dirname(os.path.abspath(filepath))
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w") as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict(), default=_json_default) + "\n")

        logger.info(f"Saved {len(self.events)} events to {filepath}")
        self.filepath = filepath

    def load(self, filepath: Optional[str] = None) -> None:
        filepath = filepath or self.filepath
        if not filepath:
            raise ValueError("No filepath specified for loading")
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return

        self.events = []
        self._index = {}

        migrated = 0
        skipped = 0
        with open(filepath, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"{filepath}:{lineno}: invalid JSON, skipping")
                    skipped += 1
                    continue

                try:
                    evt = from_any_record(record)
                except (ValueError, TypeError) as e:
                    logger.warning(f"{filepath}:{lineno}: {e}")
                    skipped += 1
                    continue

                if "migrated" in evt.tags:
                    migrated += 1
                if evt.event_id not in self._index:
                    self.events.append(evt)
                    self._index[evt.event_id] = evt

        msg = f"Loaded {len(self.events)} events from {filepath}"
        if migrated:
            msg += f" ({migrated} migrated from the legacy format)"
        if skipped:
            msg += f", {skipped} skipped"
        logger.info(msg)
        self.filepath = filepath

    # -- querying ---------------------------------------------------------

    def filter(
        self,
        criteria: Optional[Dict[str, Any]] = None,
        event_type: Optional[str] = None,
        granularity: Optional[str] = None,
        category: Optional[str] = None,
        is_positive: Optional[bool] = None,
        min_prob_delta: Optional[float] = None,
        min_readout_score: Optional[float] = None,
        layer_range: Optional[tuple] = None,
        custom_filter: Optional[Callable[[CausalReasoningEvent], bool]] = None,
    ) -> "EventStorage":
        """Filter events.

        There is deliberately no single ``min_score``. An emitted event's score
        is a probability delta and a latent event's score is a readout
        probability over the vocabulary; one threshold applied to both looks
        principled and is meaningless. A 0.5 floor keeps the banal meta-token
        ``" the"`` (readout 0.92) and discards a pivotal token worth +0.45.

        So the two scales get their own thresholds: ``min_prob_delta`` for
        emitted events, ``min_readout_score`` for latent ones. Each applies only
        to the scale it belongs to and leaves the other untouched.
        """
        result = EventStorage()

        for evt in self.events:
            if event_type is not None and evt.event_type != event_type:
                continue
            if granularity is not None and evt.granularity != granularity:
                continue
            if category is not None and evt.category != category:
                continue
            if is_positive is not None and evt.is_positive is not is_positive:
                continue
            if min_prob_delta is not None and evt.event_type != EVENT_LATENT:
                if evt.prob_delta is None or abs(evt.prob_delta) < min_prob_delta:
                    continue
            if min_readout_score is not None and evt.event_type == EVENT_LATENT:
                if evt.score < min_readout_score:
                    continue
            if layer_range is not None:
                lo, hi = layer_range
                if evt.layer is None or not (lo <= evt.layer <= hi):
                    continue
            if criteria:
                if any(getattr(evt, k, None) != v for k, v in criteria.items()):
                    continue
            if custom_filter and not custom_filter(evt):
                continue

            result.add_event(evt)

        return result

    def by_event_type(self, event_type: str) -> List[CausalReasoningEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def by_granularity(self, granularity: str) -> List[CausalReasoningEvent]:
        return [e for e in self.events if e.granularity == granularity]

    def by_query(self, query: str) -> List[CausalReasoningEvent]:
        return [e for e in self.events if e.query == query]

    def most_important(self, n: int = 10) -> List[CausalReasoningEvent]:
        """The emitted events with the largest effect on success probability.

        Latent events are excluded, not ranked below: their score is a readout
        probability, so ranking them alongside probability deltas puts the
        model's most *probable* filler tokens at the top of a list labelled
        "most important". Use ``most_surfaced()`` for latent events.
        """
        emitted = [e for e in self.events if e.event_type != EVENT_LATENT]
        return sorted(emitted, key=lambda e: e.score, reverse=True)[:n]

    def most_surfaced(self, n: int = 10) -> List[CausalReasoningEvent]:
        """The latent events the lens surfaced most strongly.

        This is a readout ranking, not an importance ranking. A high score means
        the lens is confident the model is pushing toward that token, not that
        the token matters causally.
        """
        latent = self.by_event_type(EVENT_LATENT)
        return sorted(latent, key=lambda e: e.score, reverse=True)[:n]

    def queries(self) -> List[str]:
        seen = []
        for e in self.events:
            if e.query not in seen:
                seen.append(e.query)
        return seen

    def summary(self) -> Dict[str, Any]:
        """Per-scale statistics.

        There is no aggregate mean or max over ``score``: averaging a
        probability delta with a vocabulary readout probability produces a
        number that means nothing. The two scales are reported separately.
        """
        if not self.events:
            return {"total_events": 0}

        by_type: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        for e in self.events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            by_category[e.category or "unknown"] = by_category.get(e.category or "unknown", 0) + 1

        emitted = [e for e in self.events if e.prob_delta is not None]
        latent = self.by_event_type(EVENT_LATENT)

        summary = {
            "total_events": len(self.events),
            "latent_events": len(latent),
            "token_events": len(self.by_event_type(EVENT_TOKEN)),
            "sentence_events": len(self.by_event_type(EVENT_SENTENCE)),
            "event_type_distribution": by_type,
            "category_distribution": by_category,
            "positive_events": sum(1 for e in emitted if e.is_positive),
            "negative_events": sum(1 for e in emitted if e.is_positive is False),
            "num_queries": len(self.queries()),
            "num_links": sum(len(e.linked_event_ids) for e in self.events),
            # Emitted only: these are probability deltas.
            "average_abs_prob_delta": (
                sum(abs(e.prob_delta) for e in emitted) / len(emitted) if emitted else None
            ),
            "max_abs_prob_delta": (
                max(abs(e.prob_delta) for e in emitted) if emitted else None
            ),
            # Latent only: these are readout probabilities, on a different scale.
            "average_readout_score": (
                sum(e.score for e in latent) / len(latent) if latent else None
            ),
            "max_readout_score": max((e.score for e in latent), default=None),
        }
        # the legacy format's get_anchor_summary reads these names; they mean emitted-only.
        summary["average_score"] = summary["average_abs_prob_delta"]
        summary["max_score"] = summary["max_abs_prob_delta"]
        return summary

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> CausalReasoningEvent:
        return self.events[index]

    def __iter__(self) -> Iterator[CausalReasoningEvent]:
        return iter(self.events)


class TokenStorage(EventStorage):
    """Backwards-compatible alias for the v1 ``TokenStorage`` API.

    legacy code holds dicts in ``.tokens`` and calls ``add_token``. Both keep
    working: ``.tokens`` is a live view rendering events back to dicts.
    """

    def add_token(self, token: Any) -> Optional[CausalReasoningEvent]:
        return self.add_event(token)

    def add_tokens(self, tokens: List[Any]) -> None:
        self.add_events(tokens)

    @property
    def tokens(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]
