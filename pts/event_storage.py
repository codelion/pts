"""
Storage for unified causal reasoning events (PTS v2).

``EventStorage`` holds ``CausalReasoningEvent`` records and reads/writes JSONL.
It accepts v1 pivotal-token and v1 thought-anchor records on load and migrates
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
        """Add an event. Accepts v2 events, v1 records, or legacy dataclasses.

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

        written = 0
        with open(filepath, "w") as f:
            for i, event in enumerate(self.events):
                try:
                    f.write(json.dumps(event.to_dict()) + "\n")
                    written += 1
                except (TypeError, ValueError) as e:
                    logger.error(f"Skipping unserializable event {i} ({event.event_id}): {e}")

        logger.info(f"Saved {written} events to {filepath}")
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
            msg += f" ({migrated} migrated from v1)"
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
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        min_prob_delta: Optional[float] = None,
        layer_range: Optional[tuple] = None,
        custom_filter: Optional[Callable[[CausalReasoningEvent], bool]] = None,
    ) -> "EventStorage":
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
            if min_score is not None and evt.score < min_score:
                continue
            if max_score is not None and evt.score > max_score:
                continue
            if min_prob_delta is not None:
                if evt.prob_delta is None or abs(evt.prob_delta) < min_prob_delta:
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
        return sorted(self.events, key=lambda e: e.score, reverse=True)[:n]

    def queries(self) -> List[str]:
        seen = []
        for e in self.events:
            if e.query not in seen:
                seen.append(e.query)
        return seen

    def summary(self) -> Dict[str, Any]:
        if not self.events:
            return {"total_events": 0}

        by_type: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        for e in self.events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            by_category[e.category or "unknown"] = by_category.get(e.category or "unknown", 0) + 1

        scored = [e.score for e in self.events]
        emitted = [e for e in self.events if e.prob_delta is not None]

        return {
            "total_events": len(self.events),
            "latent_events": len(self.by_event_type(EVENT_LATENT)),
            "token_events": len(self.by_event_type(EVENT_TOKEN)),
            "sentence_events": len(self.by_event_type(EVENT_SENTENCE)),
            "event_type_distribution": by_type,
            "category_distribution": by_category,
            "positive_events": sum(1 for e in emitted if e.is_positive),
            "negative_events": sum(1 for e in emitted if e.is_positive is False),
            "average_score": sum(scored) / len(scored),
            "max_score": max(scored),
            "num_queries": len(self.queries()),
            "num_links": sum(len(e.linked_event_ids) for e in self.events),
        }

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> CausalReasoningEvent:
        return self.events[index]

    def __iter__(self) -> Iterator[CausalReasoningEvent]:
        return iter(self.events)


class TokenStorage(EventStorage):
    """Backwards-compatible alias for the v1 ``TokenStorage`` API.

    v1 code holds dicts in ``.tokens`` and calls ``add_token``. Both keep
    working: ``.tokens`` is a live view rendering events back to dicts.
    """

    def add_token(self, token: Any) -> Optional[CausalReasoningEvent]:
        return self.add_event(token)

    def add_tokens(self, tokens: List[Any]) -> None:
        self.add_events(tokens)

    @property
    def tokens(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]
