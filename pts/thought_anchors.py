"""
Compatibility shim for the v1 ``pts.thought_anchors`` module.

Thought anchors are Sentence PTS: they are one scale of the unified framework,
not a separate mechanism. The implementation now lives in
``pts.searchers.sentence``.

    from pts.thought_anchors import ThoughtAnchorSearcher   # still works

Three v1 defects were fixed in the move, and they changed results rather than
just tidying code (see ``docs/migration.md``):

* the legacy code deleted its own ``reasoning_trace`` parameter under memory pressure, which
  raised ``UnboundLocalError`` on the next anchor and -- because the CLI caught
  every exception -- silently dropped the whole example.
* the legacy format blanked already-processed sentences in place (``sentences[j] = ""``) while
  later iterations still read them, so every ``prob_delta`` computed after that
  point was measured against a blank-padded prefix. Those numbers were wrong,
  not merely noisy.
* the legacy format truncated the stored trace to 3000 characters with no marker.

Anchors found now can therefore differ from those the older code produced.
"""

from typing import Any, Optional

from .event_storage import EventStorage
from .events import CausalReasoningEvent, make_sentence_event, to_legacy_thought_anchor
from .searchers.sentence import SentencePTSSearcher, SentenceSegmenter

__all__ = [
    "ThoughtAnchor",
    "ThoughtAnchorSearcher",
    "ThoughtAnchorStorage",
    "SentenceSegmenter",
    "SentenceClassifier",
    "SentencePTSSearcher",
    "to_legacy_thought_anchor",
]


def ThoughtAnchor(
    query: str,
    sentence: str,
    sentence_id: int,
    prefix_context: str,
    prob_with_sentence: float,
    prob_without_sentence: float,
    model_id: str,
    task_type: str,
    prob_delta: Optional[float] = None,
    dataset_id: Optional[str] = None,
    dataset_item_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    sentence_category: Optional[str] = None,
    **kwargs: Any,
) -> CausalReasoningEvent:
    """v1 ``ThoughtAnchor`` constructor, returning a unified event."""
    event = make_sentence_event(
        query=query,
        context=prefix_context,
        sentence=sentence,
        sentence_id=sentence_id,
        prob_before=prob_without_sentence,
        prob_after=prob_with_sentence,
        model_id=model_id,
        task_type=task_type,
        dataset_id=dataset_id,
        dataset_item_id=dataset_item_id,
        category=sentence_category,
        metadata=kwargs or None,
    )
    if timestamp:
        event.timestamp = timestamp
    return event


class ThoughtAnchorSearcher(SentencePTSSearcher):
    """legacy name for ``SentencePTSSearcher``."""


class SentenceClassifier:
    """v1 sentence classifier, backed by the unified taxonomy.

    the legacy format had its own 8-category vocabulary. Those categories now map onto the
    unified ones (``self_checking`` -> ``verification``, and so on), so latent,
    token, and sentence events can be compared in one category space.
    """

    def classify_sentence(self, sentence: str) -> Optional[str]:
        from .classification import classify_event_label

        return classify_event_label(sentence, "sentence")


class ThoughtAnchorStorage(EventStorage):
    """legacy name for event storage, with the v1 method names."""

    def add_thought_anchor(self, anchor: Any) -> Optional[CausalReasoningEvent]:
        return self.add_event(anchor)

    def get_anchors_by_category(self, category: str):
        return [e.to_dict() for e in self.events if e.category == category]

    def get_most_important_anchors(self, n: int = 10):
        return [e.to_dict() for e in self.most_important(n)]

    def get_anchor_summary(self):
        summary = self.summary()
        if not summary.get("total_events"):
            return {"total_anchors": 0}
        return {
            "total_anchors": summary["sentence_events"],
            "positive_anchors": summary["positive_events"],
            "negative_anchors": summary["negative_events"],
            "category_distribution": summary["category_distribution"],
            "average_importance": summary["average_score"],
            "max_importance": summary["max_score"],
        }
