"""
Turning J-lens readouts into latent meta-token events, and enriching existing
PTS datasets with them.

The enrichment path is the point of PTS: we already have curated token-level
and sentence-level datasets, and re-running the full search to get latent events
would be wasteful. Instead, replay each existing event's context through the
model, read the workspace just before the event fired, and attach whatever the
lens surfaces.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..classification import classify_event_label
from ..events import (
    CausalReasoningEvent,
    EVENT_LATENT,
    EVENT_SENTENCE,
    EVENT_TOKEN,
    make_latent_event,
)
from .activations import ResidualCapture, get_layer_modules, resolve_workspace_layers
from .jlens import Readout, ReadoutResult

logger = logging.getLogger(__name__)


class MetaTokenExtractor:
    """Read latent meta-tokens out of the workspace around an event."""

    def __init__(
        self,
        readout: Readout,
        layers: Sequence[int],
        top_k: int = 25,
        min_score: float = 0.01,
        keep_per_position: int = 3,
        watch_lexicon: Optional[Sequence[str]] = None,
        require_category: bool = False,
    ):
        self.readout = readout
        self.layers = list(layers)
        self.top_k = top_k
        self.min_score = min_score
        self.keep_per_position = keep_per_position
        self.watch_lexicon = {w.lower() for w in watch_lexicon} if watch_lexicon else None
        self.require_category = require_category

        # Tracked so a run that filters away every readout can say so, rather
        # than quietly writing an empty dataset that looks like "the model has
        # no workspace activity".
        self.seen = 0
        self.kept = 0
        self.max_seen_score = 0.0

    def _keep(self, result: ReadoutResult) -> bool:
        """Decide whether a readout token is worth emitting as an event.

        Top-k readouts are mostly filler (punctuation, articles, the literal next
        token). A meta-token is worth keeping if it clears the score floor and
        either lands in a category we have a name for, or sits in an explicit
        watch lexicon.
        """
        self.seen += 1
        self.max_seen_score = max(self.max_seen_score, result.score)

        if result.score < self.min_score:
            return False

        token = result.token.strip()
        if not token or len(token) < 2:
            return False

        if self.watch_lexicon is not None:
            if token.lower().strip("Ġ▁ ") in self.watch_lexicon:
                return True

        category = classify_event_label(result.token, "latent")
        if self.require_category and category is None:
            return False

        return True

    def note_kept(self, n: int) -> None:
        """Record how many readouts actually became events.

        Counted *after* the keep_per_position slice, not inside ``_keep``.
        Counting inside ``_keep`` overcounts, which means an over-aggressive
        ``keep_per_position`` never trips the "thresholds ate everything"
        warning that exists to catch exactly that.
        """
        self.kept += n

    def report_filtering(self) -> None:
        """Say plainly when the thresholds ate everything."""
        if self.seen and not self.kept:
            logger.warning(
                f"All {self.seen} readouts were filtered out. The highest score seen "
                f"was {self.max_seen_score:.4g}, below --min-score={self.min_score:g}. "
                "This is an empty result from thresholding, not evidence that the "
                "model has no workspace activity. Lower --min-score and re-run."
            )

    def extract_for_context(
        self,
        context: str,
        window_before: int = 8,
        max_length: int = 2048,
    ) -> Dict[int, List[ReadoutResult]]:
        """Read the workspace over the last ``window_before`` positions of ``context``."""
        positions = "last" if window_before <= 1 else f"-{window_before}:"
        return self.readout.read_context(
            context, self.layers, positions=positions, top_k=self.top_k
        )

    def events_for(
        self,
        source: CausalReasoningEvent,
        window_before: int = 8,
        latent_model_id: Optional[str] = None,
    ) -> List[CausalReasoningEvent]:
        """Build latent events for the workspace activity preceding ``source``.

        The returned events are linked to ``source`` in both directions: the
        latent event *precedes* the emitted one, and the emitted one *follows*
        the latent one.
        """
        if not source.context:
            return []

        try:
            by_layer = self.extract_for_context(source.context, window_before=window_before)
        except Exception as e:
            logger.warning(f"Readout failed for event {source.event_id}: {e}")
            return []

        model_id = latent_model_id or source.model_id
        events: List[CausalReasoningEvent] = []

        for layer, results in by_layer.items():
            # Group by position so we keep the best few per position rather than
            # the best few overall, which would otherwise all come from one spot.
            by_position: Dict[Optional[int], List[ReadoutResult]] = {}
            for r in results:
                by_position.setdefault(r.position, []).append(r)

            for position, at_pos in by_position.items():
                kept = [r for r in at_pos if self._keep(r)][: self.keep_per_position]
                self.note_kept(len(kept))

                for r in kept:
                    # position is a negative offset from the end of the context,
                    # i.e. how many tokens before the emitted event this
                    # workspace state was read. -1 == immediately before.
                    offset = position if position is not None else -1

                    latent = make_latent_event(
                        query=source.query,
                        context=source.context,
                        metatoken=r.token,
                        token_id=r.token_id,
                        score=r.score,
                        layer=layer,
                        model_id=model_id,
                        position=offset,
                        layer_name=f"model.layers.{layer}",
                        task_type=source.task_type,
                        dataset_id=source.dataset_id,
                        dataset_item_id=source.dataset_item_id,
                        category=classify_event_label(r.token, "latent"),
                        readout_method=r.readout_method,
                        confidence=r.score,
                        # Part of the event's identity: two different source
                        # events in the same query, read at the same layer and
                        # offset, routinely surface the same common meta-token.
                        # Without this they collide and one is silently dropped.
                        source_event_id=source.event_id,
                        metadata={
                            "j_score": r.score,
                            "rank": r.rank,
                            "readout_top_k": self.top_k,
                            "workspace_layers": self.layers,
                            "position_offset_from_linked_event": offset,
                            "source_event_type": source.event_type,
                            "source_model_id": source.model_id,
                            "latent_model_id": model_id,
                        },
                    )

                    latent.link_to(source.event_id, precedes=True)
                    source.link_to(latent.event_id, follows=True)
                    events.append(latent)

        return events


def enrich_events_with_latent(
    events: Sequence[CausalReasoningEvent],
    model: Any,
    tokenizer: Any,
    readout: Readout,
    layers: Optional[Sequence[int]] = None,
    layer_fraction: Tuple[float, float] = (0.38, 0.92),
    window_before: int = 8,
    top_k: int = 25,
    min_score: float = 0.01,
    keep_per_position: int = 3,
    allow_model_mismatch: bool = False,
    latent_model_id: Optional[str] = None,
    show_progress: bool = True,
) -> List[CausalReasoningEvent]:
    """Add latent meta-token events to an existing set of emitted events.

    Returns the full event list: the original events (now carrying links) plus
    the new latent events.

    PTS records are model-specific -- a pivotal token found in one model's
    generation says nothing about another model's workspace. Enriching with a
    different model than the one that produced the events is refused unless
    ``allow_model_mismatch`` is set, in which case both model ids are recorded
    on every latent event.
    """
    from tqdm import tqdm

    emitted = [e for e in events if e.event_type in (EVENT_TOKEN, EVENT_SENTENCE)]
    if not emitted:
        logger.warning("No token or sentence events to enrich")
        return list(events)

    enrich_model_id = latent_model_id or getattr(model.config, "_name_or_path", "unknown")
    source_models = {e.model_id for e in emitted if e.model_id}
    mismatched = {m for m in source_models if m != enrich_model_id}

    if mismatched:
        message = (
            f"These events were produced by {sorted(mismatched)} but you are reading "
            f"the workspace of {enrich_model_id}. Pivotal tokens are model-specific: "
            "the activations of a different model at the same context are not evidence "
            "about the event that model never produced."
        )
        if not allow_model_mismatch:
            raise ValueError(
                message + " Pass --allow-model-mismatch to proceed anyway (results are "
                "exploratory and both model ids will be recorded)."
            )
        logger.warning(message + " Proceeding because --allow-model-mismatch was set.")

    num_layers = len(get_layer_modules(model))
    workspace_layers = resolve_workspace_layers(
        num_layers, layers=layers, fraction=layer_fraction
    )

    extractor = MetaTokenExtractor(
        readout=readout,
        layers=workspace_layers,
        top_k=top_k,
        min_score=min_score,
        keep_per_position=keep_per_position,
    )

    latent_events: List[CausalReasoningEvent] = []
    for event in tqdm(emitted, desc="Enriching with latent events", disable=not show_progress):
        latent_events.extend(
            extractor.events_for(
                event, window_before=window_before, latent_model_id=enrich_model_id
            )
        )

    logger.info(
        f"Extracted {len(latent_events)} latent meta-token events from "
        f"{len(emitted)} emitted events across layers {workspace_layers}"
    )
    extractor.report_filtering()

    return list(events) + latent_events
