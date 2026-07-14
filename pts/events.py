"""
Unified causal reasoning event schema for PTS v2.

PTS searches for pivotal reasoning events at three representational scales:

    latent meta-token / workspace event   (granularity="latent")
            v
    emitted pivotal token                 (granularity="token")
            v
    sentence-level thought anchor         (granularity="sentence")
            v
    success/failure probability shift

All three are instances of one abstraction, ``CausalReasoningEvent``, scored by
a common principle::

    event_importance = outcome_with_event - outcome_without_or_altered_event

This module defines that schema, factory constructors for each scale, and
migration helpers that read v1 pivotal-token and thought-anchor records.
"""

import time
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "2.0"

# Event types
EVENT_LATENT = "latent_metatoken"
EVENT_TOKEN = "pivotal_token"
EVENT_SENTENCE = "thought_anchor"

# Granularities
GRANULARITY_LATENT = "latent"
GRANULARITY_TOKEN = "token"
GRANULARITY_SENTENCE = "sentence"

# Visibility
VISIBILITY_LATENT = "latent"
VISIBILITY_EMITTED = "emitted"

GRANULARITY_FOR_EVENT_TYPE = {
    EVENT_LATENT: GRANULARITY_LATENT,
    EVENT_TOKEN: GRANULARITY_TOKEN,
    EVENT_SENTENCE: GRANULARITY_SENTENCE,
}

VISIBILITY_FOR_EVENT_TYPE = {
    EVENT_LATENT: VISIBILITY_LATENT,
    EVENT_TOKEN: VISIBILITY_EMITTED,
    EVENT_SENTENCE: VISIBILITY_EMITTED,
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _classify(text: str, granularity: str) -> Optional[str]:
    """Assign a unified category. Imported lazily to keep this module dependency-free."""
    from .classification import classify_event_label

    return classify_event_label(text, granularity)


def map_v1_category(v1_category: Optional[str]) -> Optional[str]:
    from .classification import map_v1_category as _map

    return _map(v1_category)


def make_event_id(event_type: str, *parts: Any) -> str:
    """Build a stable event id from its identifying parts.

    Deterministic so that re-migrating the same v1 record yields the same id and
    links stay valid across re-runs. Falls back to a random suffix when no parts
    are supplied.
    """
    prefix = {
        EVENT_LATENT: "latent",
        EVENT_TOKEN: "token",
        EVENT_SENTENCE: "sentence",
    }.get(event_type, "evt")

    if not parts:
        return f"{prefix}_evt_{uuid.uuid4().hex[:12]}"

    payload = "\x1f".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_evt_{digest}"


@dataclass
class CausalReasoningEvent:
    """A single pivotal reasoning event at any representational scale.

    ``score`` is the common currency across scales. For token and sentence
    events it is ``abs(prob_delta)``. For latent events it is the readout score
    (e.g. the J-lens probability of the meta-token), which is *not* a
    probability delta and is not directly comparable to the emitted scales --
    see ``docs/latent_pts.md``.
    """

    # Identity
    event_id: str
    event_type: str              # latent_metatoken | pivotal_token | thought_anchor
    granularity: str             # latent | token | sentence
    visibility: str              # latent | emitted

    # Dataset / model
    model_id: str
    task_type: str
    query: str

    # Location / context
    context: str                 # prefix/context before the event
    label: str                   # the token, sentence, or meta-token itself

    # Scoring
    score: float

    # Method
    search_method: str

    schema_version: str = SCHEMA_VERSION

    dataset_id: Optional[str] = None
    dataset_item_id: Optional[str] = None

    position: Optional[int] = None
    token_id: Optional[int] = None
    layer: Optional[int] = None
    layer_name: Optional[str] = None

    prob_before: Optional[float] = None
    prob_after: Optional[float] = None
    prob_delta: Optional[float] = None
    is_positive: Optional[bool] = None
    confidence: Optional[float] = None

    intervention_type: Optional[str] = None
    readout_method: Optional[str] = None

    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    parent_event_id: Optional[str] = None
    linked_event_ids: List[str] = field(default_factory=list)
    precedes_event_ids: List[str] = field(default_factory=list)
    follows_event_ids: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalReasoningEvent":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        # Fields this schema version does not know about are moved into
        # `metadata` rather than dropped. The value survives; its location does
        # not -- a v2.1 writer reading back its own file will find its field
        # under metadata, not at the top level. Real metadata wins on conflict.
        extra = {k: v for k, v in data.items() if k not in known}
        if extra:
            merged = dict(extra)
            merged.update(kwargs.get("metadata") or {})
            kwargs["metadata"] = merged
        return cls(**kwargs)

    def link_to(self, other_id: str, precedes: bool = False, follows: bool = False) -> None:
        if other_id not in self.linked_event_ids:
            self.linked_event_ids.append(other_id)
        if precedes and other_id not in self.precedes_event_ids:
            self.precedes_event_ids.append(other_id)
        if follows and other_id not in self.follows_event_ids:
            self.follows_event_ids.append(other_id)


def _resolve_positivity(prob_delta: Optional[float]) -> Optional[bool]:
    if prob_delta is None:
        return None
    return prob_delta > 0


def make_token_event(
    query: str,
    context: str,
    token: str,
    token_id: Optional[int],
    prob_before: float,
    prob_after: float,
    model_id: str,
    task_type: str = "generic",
    dataset_id: Optional[str] = None,
    dataset_item_id: Optional[str] = None,
    position: Optional[int] = None,
    category: Optional[str] = None,
    search_method: str = "token_pts",
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> CausalReasoningEvent:
    """Build an emitted-pivotal-token event (Phi-4-style PTS)."""
    prob_delta = prob_after - prob_before
    meta = {"pivot_context": context, "pivot_token": token}
    if metadata:
        meta.update(metadata)

    tags = ["token_pts", "positive" if prob_delta > 0 else "negative"]

    return CausalReasoningEvent(
        event_id=make_event_id(EVENT_TOKEN, model_id, query, context, token),
        event_type=EVENT_TOKEN,
        granularity=GRANULARITY_TOKEN,
        visibility=VISIBILITY_EMITTED,
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_item_id=dataset_item_id,
        task_type=task_type,
        query=query,
        context=context,
        label=token,
        position=position,
        token_id=token_id,
        prob_before=prob_before,
        prob_after=prob_after,
        prob_delta=prob_delta,
        score=abs(prob_delta),
        is_positive=prob_delta > 0,
        search_method=search_method,
        intervention_type="append_token",
        category=category,
        tags=tags,
        metadata=meta,
        **kwargs,
    )


def make_sentence_event(
    query: str,
    context: str,
    sentence: str,
    sentence_id: int,
    prob_before: float,
    prob_after: float,
    model_id: str,
    task_type: str = "generic",
    dataset_id: Optional[str] = None,
    dataset_item_id: Optional[str] = None,
    category: Optional[str] = None,
    search_method: str = "sentence_pts",
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> CausalReasoningEvent:
    """Build a sentence-level thought-anchor event.

    ``prob_before`` is the probability without the sentence (or with an
    alternative substituted); ``prob_after`` is the probability with it.
    """
    prob_delta = prob_after - prob_before
    meta = {"sentence": sentence, "sentence_id": sentence_id, "prefix_context": context}
    if metadata:
        meta.update(metadata)

    tags = ["sentence_pts", "thought_anchor", "positive" if prob_delta > 0 else "negative"]

    return CausalReasoningEvent(
        event_id=make_event_id(EVENT_SENTENCE, model_id, query, sentence_id, sentence),
        event_type=EVENT_SENTENCE,
        granularity=GRANULARITY_SENTENCE,
        visibility=VISIBILITY_EMITTED,
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_item_id=dataset_item_id,
        task_type=task_type,
        query=query,
        context=context,
        label=sentence,
        position=sentence_id,
        prob_before=prob_before,
        prob_after=prob_after,
        prob_delta=prob_delta,
        score=abs(prob_delta),
        is_positive=prob_delta > 0,
        search_method=search_method,
        intervention_type="replace_sentence",
        category=category,
        tags=tags,
        metadata=meta,
        **kwargs,
    )


def make_latent_event(
    query: str,
    context: str,
    metatoken: str,
    token_id: Optional[int],
    score: float,
    layer: int,
    model_id: str,
    position: Optional[int] = None,
    layer_name: Optional[str] = None,
    task_type: str = "generic",
    dataset_id: Optional[str] = None,
    dataset_item_id: Optional[str] = None,
    category: Optional[str] = None,
    readout_method: str = "jlens",
    search_method: str = "latent_pts_enrichment",
    source_event_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> CausalReasoningEvent:
    """Build a latent workspace meta-token event read out of the residual stream.

    These are observational by default: no probability delta is measured, so
    ``prob_delta`` stays ``None`` and ``is_positive`` stays ``None``. ``score``
    is the readout score, not a causal effect. Interventional latent events
    (steer/ablate) set ``prob_*`` and carry ``intervention_type``.

    ``context`` and ``source_event_id`` are part of the identity, not just the
    payload. Without them, two different pivotal tokens in the same query --
    enriched at the same layer and the same offset, surfacing the same common
    meta-token, which is the *normal* case for a real model's top-k readout --
    collide on ``event_id``, and storage silently drops one of them while the
    other keeps a link pointing at the dropped id.
    """
    meta = {"readout_method": readout_method}
    if source_event_id:
        meta["source_event_id"] = source_event_id
    if metadata:
        meta.update(metadata)

    tags = ["latent_pts", "metatoken", readout_method]

    return CausalReasoningEvent(
        event_id=make_event_id(
            EVENT_LATENT, model_id, query, context, source_event_id,
            layer, position, metatoken,
        ),
        event_type=EVENT_LATENT,
        granularity=GRANULARITY_LATENT,
        visibility=VISIBILITY_LATENT,
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_item_id=dataset_item_id,
        task_type=task_type,
        query=query,
        context=context,
        label=metatoken,
        position=position,
        token_id=token_id,
        layer=layer,
        layer_name=layer_name,
        score=score,
        prob_before=None,
        prob_after=None,
        prob_delta=None,
        is_positive=None,
        search_method=search_method,
        readout_method=readout_method,
        category=category,
        tags=tags,
        metadata=meta,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Migration from v1 records
# ---------------------------------------------------------------------------

def is_v1_pivotal_token(record: Dict[str, Any]) -> bool:
    return "pivot_token" in record


def is_v1_thought_anchor(record: Dict[str, Any]) -> bool:
    return "sentence" in record and "sentence_id" in record


def is_v2_event(record: Dict[str, Any]) -> bool:
    return "event_type" in record and "granularity" in record


def from_v1_pivotal_token(record: Dict[str, Any]) -> CausalReasoningEvent:
    """Convert a v1 pivotal-token record into a unified event.

    v1 stored ``prob_before``/``prob_after``/``prob_delta`` explicitly. We trust
    the stored ``prob_delta`` when present rather than recomputing, because some
    published datasets were written before rounding changes.
    """
    prob_before = record.get("prob_before")
    prob_after = record.get("prob_after")
    prob_delta = record.get("prob_delta")
    if prob_delta is None and prob_before is not None and prob_after is not None:
        prob_delta = prob_after - prob_before

    # Everything v1 carried that has no home in a top-level v2 field is kept.
    reserved = {
        "query", "pivot_context", "pivot_token", "pivot_token_id",
        "prob_before", "prob_after", "prob_delta", "is_positive",
        "model_id", "task_type", "dataset_id", "dataset_item_id", "timestamp",
    }
    metadata = {k: v for k, v in record.items() if k not in reserved}
    metadata.update({
        "pivot_context": record.get("pivot_context", ""),
        "pivot_token": record.get("pivot_token", ""),
        "migrated_from": "v1_pivotal_token",
    })

    model_id = record.get("model_id", "unknown")
    query = record.get("query", "")
    context = record.get("pivot_context", "")
    token = record.get("pivot_token", "")

    return CausalReasoningEvent(
        event_id=make_event_id(EVENT_TOKEN, model_id, query, context, token),
        event_type=EVENT_TOKEN,
        granularity=GRANULARITY_TOKEN,
        visibility=VISIBILITY_EMITTED,
        model_id=model_id,
        dataset_id=record.get("dataset_id"),
        dataset_item_id=record.get("dataset_item_id"),
        task_type=record.get("task_type", "generic"),
        query=query,
        context=context,
        label=token,
        token_id=record.get("pivot_token_id"),
        prob_before=prob_before,
        prob_after=prob_after,
        prob_delta=prob_delta,
        score=abs(prob_delta) if prob_delta is not None else 0.0,
        is_positive=record.get("is_positive", _resolve_positivity(prob_delta)),
        search_method="token_pts",
        intervention_type="append_token",
        # v1 never classified pivotal tokens. Leaving them uncategorized would
        # make every cross-scale comparison vacuous: a latent event's category
        # can never match `None`, so category-match rates against migrated
        # tokens would come back at exactly zero -- a null produced by the
        # migration, not by the data.
        category=_classify(token, "token"),
        tags=["token_pts", "migrated"],
        metadata=metadata,
        timestamp=record.get("timestamp", _now()),
    )


def from_v1_thought_anchor(record: Dict[str, Any]) -> CausalReasoningEvent:
    """Convert a v1 thought-anchor record into a unified event.

    v1 named its probabilities ``prob_with_sentence`` / ``prob_without_sentence``;
    those map onto ``prob_after`` / ``prob_before`` respectively.
    """
    prob_after = record.get("prob_with_sentence")
    prob_before = record.get("prob_without_sentence")
    prob_delta = record.get("prob_delta")
    if prob_delta is None and prob_before is not None and prob_after is not None:
        prob_delta = prob_after - prob_before

    reserved = {
        "query", "sentence", "sentence_id", "prefix_context",
        "prob_with_sentence", "prob_without_sentence", "prob_delta",
        "is_positive", "importance_score", "sentence_category",
        "model_id", "task_type", "dataset_id", "dataset_item_id", "timestamp",
    }
    metadata = {k: v for k, v in record.items() if k not in reserved}
    metadata.update({
        "sentence": record.get("sentence", ""),
        "sentence_id": record.get("sentence_id"),
        "prefix_context": record.get("prefix_context", ""),
        "migrated_from": "v1_thought_anchor",
    })

    model_id = record.get("model_id", "unknown")
    query = record.get("query", "")
    sentence = record.get("sentence", "")
    sentence_id = record.get("sentence_id")

    return CausalReasoningEvent(
        event_id=make_event_id(EVENT_SENTENCE, model_id, query, sentence_id, sentence),
        event_type=EVENT_SENTENCE,
        granularity=GRANULARITY_SENTENCE,
        visibility=VISIBILITY_EMITTED,
        model_id=model_id,
        dataset_id=record.get("dataset_id"),
        dataset_item_id=record.get("dataset_item_id"),
        task_type=record.get("task_type", "generic"),
        query=query,
        context=record.get("prefix_context", ""),
        label=sentence,
        position=sentence_id,
        prob_before=prob_before,
        prob_after=prob_after,
        prob_delta=prob_delta,
        score=abs(prob_delta) if prob_delta is not None else 0.0,
        is_positive=record.get("is_positive", _resolve_positivity(prob_delta)),
        confidence=record.get("verification_score"),
        search_method="sentence_pts",
        intervention_type="replace_sentence",
        # v1's own 8-category vocabulary ("self_checking", "active_computation")
        # is remapped onto the unified taxonomy shared by all three scales.
        # Without this, a v1 sentence category could never equal a latent event's
        # category, and every latent -> token -> sentence chain would be invisible
        # by construction. Falls back to classifying the sentence text when v1
        # recorded no category at all.
        category=(
            map_v1_category(record.get("sentence_category"))
            or _classify(sentence, "sentence")
        ),
        tags=["sentence_pts", "thought_anchor", "migrated"],
        metadata=metadata,
        timestamp=record.get("timestamp", _now()),
    )


def from_any_record(record: Any) -> CausalReasoningEvent:
    """Read a record of any known PTS vintage into a unified event.

    Accepts v2 events, v1 pivotal tokens, v1 thought anchors, and objects that
    expose ``to_dict()`` (the legacy ``PivotalToken`` / ``ThoughtAnchor``
    dataclasses).
    """
    if isinstance(record, CausalReasoningEvent):
        return record

    if hasattr(record, "to_dict"):
        record = record.to_dict()

    if not isinstance(record, dict):
        raise TypeError(f"Cannot read event from {type(record).__name__}")

    if is_v2_event(record):
        return CausalReasoningEvent.from_dict(record)
    if is_v1_pivotal_token(record):
        return from_v1_pivotal_token(record)
    if is_v1_thought_anchor(record):
        return from_v1_thought_anchor(record)

    raise ValueError(
        "Unrecognized PTS record: expected a v2 event (event_type + granularity), "
        "a v1 pivotal token (pivot_token), or a v1 thought anchor "
        f"(sentence + sentence_id). Got keys: {sorted(record)[:10]}"
    )


# ---------------------------------------------------------------------------
# Derived views: recover the v1 shapes from a unified event
# ---------------------------------------------------------------------------

def to_v1_pivotal_token(event: CausalReasoningEvent) -> Dict[str, Any]:
    """Render a token event back into the v1 pivotal-token shape."""
    if event.event_type != EVENT_TOKEN:
        raise ValueError(f"Not a token event: {event.event_type}")
    return {
        "model_id": event.model_id,
        "query": event.query,
        "pivot_context": event.context,
        "pivot_token": event.label,
        "pivot_token_id": event.token_id,
        "prob_before": event.prob_before,
        "prob_after": event.prob_after,
        "prob_delta": event.prob_delta,
        "is_positive": event.is_positive,
        "task_type": event.task_type,
        "dataset_id": event.dataset_id,
        "dataset_item_id": event.dataset_item_id,
        "timestamp": event.timestamp,
    }


def to_v1_thought_anchor(event: CausalReasoningEvent) -> Dict[str, Any]:
    """Render a sentence event back into the v1 thought-anchor shape."""
    if event.event_type != EVENT_SENTENCE:
        raise ValueError(f"Not a sentence event: {event.event_type}")
    meta = event.metadata or {}
    return {
        "model_id": event.model_id,
        "query": event.query,
        "sentence": event.label,
        "sentence_id": event.position,
        "prefix_context": event.context,
        "prob_with_sentence": event.prob_after,
        "prob_without_sentence": event.prob_before,
        "prob_delta": event.prob_delta,
        "is_positive": event.is_positive,
        "importance_score": event.score,
        "sentence_category": event.category,
        "task_type": event.task_type,
        "suffix_context": meta.get("suffix_context", ""),
        "full_reasoning_trace": meta.get("full_reasoning_trace", ""),
        "alternatives_tested": meta.get("alternatives_tested", []),
        "causal_dependencies": meta.get("causal_dependencies", []),
        "causal_dependents": meta.get("causal_dependents", []),
        "failure_mode": meta.get("failure_mode"),
        "error_type": meta.get("error_type"),
        "dataset_id": event.dataset_id,
        "dataset_item_id": event.dataset_item_id,
        "timestamp": event.timestamp,
    }


# Thin named constructors kept for readability at call sites. Subclassing a
# dataclass with defaults is awkward, so these are aliases, not subclasses.
PivotalTokenEvent = make_token_event
ThoughtAnchorEvent = make_sentence_event
MetaTokenEvent = make_latent_event
