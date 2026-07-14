"""
Multiscale PTS -- all three scales over one query, linked into causal chains.

This is what ``pts run --granularity all`` drives. It exists to produce the one
artifact the whole framework is aimed at: for a single query, the latent
workspace events, the emitted pivotal tokens, and the sentence-level anchors,
with links between them, so the chain

    latent: verification  ->  token: " Wait"  ->  sentence: "Let me check..."

can be read off a single dataset.

One model is loaded and shared across all three searchers. Loading it three
times would triple the memory for no benefit.
"""

import logging
from typing import Any, List, Optional, Sequence, Tuple

from ..events import EVENT_LATENT, CausalReasoningEvent
from ..event_storage import EventStorage
from ..linking import EventLinker
from .base import BasePTSSearcher
from .latent import LatentPTSSearcher
from .sentence import SentencePTSSearcher
from .token import TokenPTSSearcher

logger = logging.getLogger(__name__)


class MultiScaleSearcher:
    """Run token, sentence, and latent PTS over the same query and link the results."""

    def __init__(
        self,
        model_name: str,
        oracle: Optional[Any] = None,
        granularities: Sequence[str] = ("token", "sentence", "latent"),
        event_storage: Optional[EventStorage] = None,
        readout_method: str = "logit_lens",
        jlens_path: Optional[str] = None,
        workspace_layers: Optional[Sequence[int]] = None,
        layer_fraction: Tuple[float, float] = (0.38, 0.92),
        window_before: int = 8,
        link_threshold: float = 0.5,
        readout_top_k: int = 25,
        min_score: float = 0.01,
        keep_per_position: int = 3,
        enable_verification: bool = False,
        skip_embeddings: bool = False,
        **searcher_kwargs,
    ):
        self.granularities = list(granularities)
        self.event_storage = event_storage or EventStorage()
        self.window_before = window_before
        self.linker = EventLinker(threshold=link_threshold, window=window_before)

        # Load once, share everywhere.
        base = BasePTSSearcher(model_name=model_name, oracle=oracle, **searcher_kwargs)
        shared = {
            "model": base.model,
            "tokenizer": base.tokenizer,
            "device": base.device,
        }
        common = dict(searcher_kwargs)
        for key in ("model", "tokenizer", "device"):
            common.pop(key, None)

        self.token_searcher: Optional[TokenPTSSearcher] = None
        self.sentence_searcher: Optional[SentencePTSSearcher] = None
        self.latent_searcher: Optional[LatentPTSSearcher] = None

        if "token" in self.granularities:
            self.token_searcher = TokenPTSSearcher(
                model_name=model_name, oracle=oracle, **shared, **common
            )
        if "sentence" in self.granularities:
            self.sentence_searcher = SentencePTSSearcher(
                model_name=model_name,
                oracle=oracle,
                enable_verification=enable_verification,
                skip_embeddings=skip_embeddings,
                **shared,
                **common,
            )
        if "latent" in self.granularities:
            self.latent_searcher = LatentPTSSearcher(
                model_name=model_name,
                readout_method=readout_method,
                jlens_path=jlens_path,
                workspace_layers=workspace_layers,
                layer_fraction=layer_fraction,
                top_k=readout_top_k,
                min_score=min_score,
                keep_per_position=keep_per_position,
                **shared,
                **{k: v for k, v in common.items() if k != "oracle"},
            )

        self.model = base.model
        self.tokenizer = base.tokenizer

    def search(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        task_type: str = "generic",
        dataset_id: Optional[str] = None,
        item_id: Optional[str] = None,
        max_generations: int = 10,
        min_prob: float = 0.2,
        max_prob: float = 0.8,
        category: Optional[str] = None,
    ) -> List[CausalReasoningEvent]:
        """Find events at every requested scale for one query, then link them."""
        events: List[CausalReasoningEvent] = []

        if self.token_searcher is not None:
            token_events = list(
                self.token_searcher.search(
                    query=query,
                    system_prompt=system_prompt,
                    task_type=task_type,
                    dataset_id=dataset_id,
                    item_id=item_id,
                    max_generations=max_generations,
                    min_prob=min_prob,
                    max_prob=max_prob,
                    category=category,
                )
            )
            logger.info(f"Token PTS found {len(token_events)} events")
            events.extend(token_events)

        if self.sentence_searcher is not None:
            prompt = self.sentence_searcher.format_prompt(
                query, system_prompt=system_prompt, category=category
            )
            completions = self.sentence_searcher.generate_completions(prompt, num_samples=1)
            trace = completions[0] if completions else ""

            if trace.strip():
                sentence_events = list(
                    self.sentence_searcher.search(
                        query=query,
                        reasoning_trace=trace,
                        system_prompt=system_prompt,
                        task_type=task_type,
                        dataset_id=dataset_id,
                        item_id=item_id,
                        min_prob=min_prob,
                        max_prob=max_prob,
                        category=category,
                    )
                )
                logger.info(f"Sentence PTS found {len(sentence_events)} events")
                events.extend(sentence_events)

        if self.latent_searcher is not None:
            if events:
                # Read the workspace at the contexts of the emitted events we just
                # found. Anchoring on those contexts is what makes the latent
                # events alignable with them at all.
                enriched = self.latent_searcher.enrich(
                    events, window_before=self.window_before, allow_model_mismatch=True
                )
                latent_events = [e for e in enriched if e.event_type == EVENT_LATENT]
                events = enriched
            else:
                # No emitted events to anchor on -- either latent is the only
                # requested scale, or the other scales found nothing. Walk the
                # trace and read the workspace as we go instead of returning
                # nothing.
                latent_events = list(
                    self.latent_searcher.probe(
                        query=query,
                        system_prompt=system_prompt,
                        task_type=task_type,
                        dataset_id=dataset_id,
                        item_id=item_id,
                        category=category,
                    )
                )
                events = latent_events

            logger.info(f"Latent PTS found {len(latent_events)} events")

        self.linker.link(events)
        self.event_storage.add_events(events)
        return events
