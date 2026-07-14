"""
Linking latent, token, and sentence events into causal chains.

The central empirical claim PTS v2 exists to test is::

    Many emitted pivotal tokens and thought-anchor sentences are preceded by
    latent verbalizable meta-tokens in the model's workspace.

Testing it requires deciding which latent event goes with which emitted event.
That is what this module does. Links are *hypotheses scored by evidence*, not
ground truth: a high link score means several weak signals agree, not that a
causal path was verified. Use ``pts link --shuffle-control`` to compare the
observed link-score distribution against a shuffled baseline before believing
any of it.
"""

import logging
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

from .events import (
    CausalReasoningEvent,
    EVENT_LATENT,
    EVENT_SENTENCE,
    EVENT_TOKEN,
)

logger = logging.getLogger(__name__)

LINK_LATENT_PRECEDES_TOKEN = "latent_precedes_token"
LINK_LATENT_PRECEDES_SENTENCE = "latent_precedes_sentence"
LINK_TOKEN_EXPANDS_INTO_SENTENCE = "token_expands_into_sentence"
LINK_SAME_QUERY = "same_query"
LINK_SAME_CONTEXT = "same_context"
LINK_SEMANTIC_MATCH = "semantic_match"
LINK_CATEGORY_MATCH = "category_match"
LINK_CAUSAL_PARENT = "causal_parent"

# Weights for the composite link score. These are a starting point, not a
# fitted model -- they encode "same query matters most, category agreement and
# context overlap matter next". Tune with --link-weights if you have labels.
DEFAULT_WEIGHTS = {
    "same_query": 0.30,
    "context_overlap": 0.25,
    "category_match": 0.20,
    "temporal_proximity": 0.15,
    "semantic_similarity": 0.10,
}


@dataclass
class EventLink:
    source_event_id: str
    target_event_id: str
    link_type: str
    score: float
    evidence: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokenize(text: str) -> set:
    return {w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split() if len(w) > 2}


def context_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two contexts' word sets."""
    ta, tb = _tokenize(a or ""), _tokenize(b or "")
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def semantic_similarity(a: str, b: str) -> float:
    """Cheap lexical proxy for semantic similarity of two event labels.

    A real embedding model would be better; this keeps the base package free of
    sentence-transformers. ``EventLinker(embed_fn=...)`` swaps in a real one.
    """
    ta, tb = _tokenize(a or ""), _tokenize(b or "")
    if not ta or not tb:
        # Fall back to character overlap for single sub-word meta-tokens, which
        # tokenize to nothing under the len>2 word filter.
        sa, sb = set((a or "").lower()), set((b or "").lower())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)
    return len(ta & tb) / len(ta | tb)


def temporal_proximity(
    latent_pos: Optional[int],
    emitted_pos: Optional[int],
    window: int = 8,
) -> float:
    """1.0 when the latent event sits immediately before the emitted one, decaying to 0.

    Latent events *after* the emitted event score 0: the claim under test is
    that workspace activity *precedes* emission, so a link running the wrong way
    in time is not evidence for it.
    """
    if latent_pos is None or emitted_pos is None:
        return 0.0
    lead = emitted_pos - latent_pos
    if lead < 0:
        return 0.0
    if lead > window:
        return 0.0
    return 1.0 - (lead / (window + 1))


class EventLinker:
    """Score and attach links between events across the three PTS scales."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        window: int = 8,
        embed_fn: Optional[Any] = None,
    ):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.threshold = threshold
        self.window = window
        self.embed_fn = embed_fn

    def score_pair(
        self,
        latent: CausalReasoningEvent,
        emitted: CausalReasoningEvent,
    ) -> EventLink:
        same_query = 1.0 if latent.query and latent.query == emitted.query else 0.0
        overlap = context_overlap(latent.context, emitted.context)
        category = (
            1.0
            if latent.category and emitted.category and latent.category == emitted.category
            else 0.0
        )

        # Prefer the explicit offset the enricher recorded; fall back to raw
        # positions, which are only comparable when both are token indices.
        offset = (latent.metadata or {}).get("position_offset_from_linked_event")
        if offset is not None:
            lead = -offset if offset <= 0 else None
            proximity = (
                1.0 - (lead / (self.window + 1))
                if lead is not None and 0 <= lead <= self.window
                else 0.0
            )
        else:
            proximity = temporal_proximity(latent.position, emitted.position, self.window)

        if self.embed_fn is not None:
            semantic = float(self.embed_fn(latent.label, emitted.label))
        else:
            semantic = semantic_similarity(latent.label, emitted.label)

        evidence = {
            "same_query": same_query,
            "context_overlap": overlap,
            "category_match": category,
            "temporal_proximity": proximity,
            "semantic_similarity": semantic,
        }
        score = sum(self.weights[k] * v for k, v in evidence.items())

        link_type = (
            LINK_LATENT_PRECEDES_TOKEN
            if emitted.event_type == EVENT_TOKEN
            else LINK_LATENT_PRECEDES_SENTENCE
        )

        return EventLink(
            source_event_id=latent.event_id,
            target_event_id=emitted.event_id,
            link_type=link_type,
            score=round(score, 4),
            evidence={k: round(v, 4) for k, v in evidence.items()},
        )

    def link(
        self,
        events: Sequence[CausalReasoningEvent],
        attach: bool = True,
    ) -> List[EventLink]:
        """Link every latent event to the emitted events it plausibly precedes.

        Also links token events to the sentence events that contain them. When
        ``attach`` is set, the links are written into the events' own
        ``linked_event_ids`` / ``precedes_event_ids`` / ``follows_event_ids``.
        """
        latent = [e for e in events if e.event_type == EVENT_LATENT]
        tokens = [e for e in events if e.event_type == EVENT_TOKEN]
        sentences = [e for e in events if e.event_type == EVENT_SENTENCE]

        by_query: Dict[str, List[CausalReasoningEvent]] = {}
        for e in tokens + sentences:
            by_query.setdefault(e.query, []).append(e)

        links: List[EventLink] = []

        for lat in latent:
            # A latent event can only precede an emitted event from the same
            # query -- cross-query links are meaningless and would dominate the
            # score distribution by sheer count.
            for emitted in by_query.get(lat.query, []):
                link = self.score_pair(lat, emitted)
                if link.score < self.threshold:
                    continue
                links.append(link)
                if attach:
                    lat.link_to(emitted.event_id, precedes=True)
                    emitted.link_to(lat.event_id, follows=True)

        links.extend(self._link_tokens_to_sentences(tokens, sentences, attach=attach))

        logger.info(f"Linked {len(links)} event pairs from {len(events)} events")
        return links

    def _link_tokens_to_sentences(
        self,
        tokens: Sequence[CausalReasoningEvent],
        sentences: Sequence[CausalReasoningEvent],
        attach: bool = True,
    ) -> List[EventLink]:
        """A token event expands into the sentence event that contains its token."""
        links: List[EventLink] = []
        by_query: Dict[str, List[CausalReasoningEvent]] = {}
        for s in sentences:
            by_query.setdefault(s.query, []).append(s)

        for tok in tokens:
            label = (tok.label or "").strip()
            if not label:
                continue
            for sent in by_query.get(tok.query, []):
                contains = label and label in (sent.label or "")
                overlap = context_overlap(tok.context, sent.context)
                category = (
                    1.0 if tok.category and tok.category == sent.category else 0.0
                )
                evidence = {
                    "same_query": 1.0,
                    "context_overlap": overlap,
                    "category_match": category,
                    "temporal_proximity": 0.0,
                    "semantic_similarity": 1.0 if contains else 0.0,
                }
                score = sum(self.weights[k] * v for k, v in evidence.items())
                if score < self.threshold:
                    continue
                links.append(
                    EventLink(
                        source_event_id=tok.event_id,
                        target_event_id=sent.event_id,
                        link_type=LINK_TOKEN_EXPANDS_INTO_SENTENCE,
                        score=round(score, 4),
                        evidence={k: round(v, 4) for k, v in evidence.items()},
                    )
                )
                if attach:
                    tok.link_to(sent.event_id, precedes=True)
                    sent.link_to(tok.event_id, follows=True)
                    sent.parent_event_id = sent.parent_event_id or tok.event_id
        return links

    def shuffle_control(
        self,
        events: Sequence[CausalReasoningEvent],
        seed: int = 42,
    ) -> Dict[str, float]:
        """Score links against shuffled query assignments.

        If the real link scores are not meaningfully higher than this baseline,
        the observed latent-precedes-token structure is not evidence of
        anything. Reported by ``pts link --shuffle-control``.
        """
        latent = [e for e in events if e.event_type == EVENT_LATENT]
        emitted = [e for e in events if e.event_type in (EVENT_TOKEN, EVENT_SENTENCE)]
        if not latent or not emitted:
            return {"observed_mean": 0.0, "shuffled_mean": 0.0, "n": 0}

        observed = [
            self.score_pair(lat, em).score
            for lat in latent
            for em in emitted
            if lat.query == em.query
        ]

        rng = random.Random(seed)
        shuffled = []
        for lat in latent:
            # Pair each latent event with an emitted event from a *different*
            # query, holding everything else fixed.
            others = [e for e in emitted if e.query != lat.query]
            if not others:
                continue
            for em in rng.sample(others, min(len(others), 5)):
                shuffled.append(self.score_pair(lat, em).score)

        return {
            "observed_mean": sum(observed) / len(observed) if observed else 0.0,
            "shuffled_mean": sum(shuffled) / len(shuffled) if shuffled else 0.0,
            "observed_n": len(observed),
            "shuffled_n": len(shuffled),
        }


def link_events(
    events: Sequence[CausalReasoningEvent],
    threshold: float = 0.5,
    window: int = 8,
    weights: Optional[Dict[str, float]] = None,
) -> List[EventLink]:
    return EventLinker(weights=weights, threshold=threshold, window=window).link(events)
