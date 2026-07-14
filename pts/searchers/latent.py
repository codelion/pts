"""
Latent PTS -- searching the hidden workspace directly.

Two modes:

*Enrichment* replays the contexts of existing token/sentence events and reads
the workspace just before each one fired. This is the cheap path and the one
that reuses curated PTS datasets.

*Probe* runs on raw queries: generate a trace, then read the workspace at every
step, emitting latent events wherever the lens surfaces something categorizable.

Both are **observational**: they report what the lens sees, not what happens if
you change it. No ``prob_delta`` is measured, so latent events carry a readout
score rather than a causal effect, and the two are not comparable. Establishing
that a meta-token *causes* the emitted event needs intervention (steer/ablate
and re-estimate), which is future work -- see ``docs/latent_pts.md``.
"""

import logging
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import torch

from ..events import CausalReasoningEvent
from ..event_storage import EventStorage
from ..latent.activations import get_layer_modules, resolve_workspace_layers
from ..latent.jlens import Readout, load_readout
from ..latent.metatokens import MetaTokenExtractor, enrich_events_with_latent
from .base import BasePTSSearcher

logger = logging.getLogger(__name__)


class LatentPTSSearcher(BasePTSSearcher):
    """Search the model's workspace for latent meta-token events."""

    def __init__(
        self,
        *args,
        readout_method: str = "logit_lens",
        jlens_path: Optional[str] = None,
        workspace_layers: Optional[Sequence[int]] = None,
        layer_fraction: Tuple[float, float] = (0.38, 0.92),
        top_k: int = 25,
        min_score: float = 0.01,
        keep_per_position: int = 3,
        event_storage: Optional[EventStorage] = None,
        readout: Optional[Readout] = None,
        **kwargs,
    ):
        # Latent PTS is observational, so it needs no oracle -- there is no
        # success probability to estimate.
        kwargs.setdefault("oracle", None)
        super().__init__(*args, **kwargs)

        self.event_storage = event_storage or EventStorage()
        self.readout_method = readout_method
        self.readout = readout or load_readout(
            self.model, self.tokenizer, method=readout_method, jlens_path=jlens_path
        )

        self.workspace_layers = resolve_workspace_layers(
            len(get_layer_modules(self.model)),
            layers=workspace_layers,
            fraction=layer_fraction,
        )

        self.extractor = MetaTokenExtractor(
            readout=self.readout,
            layers=self.workspace_layers,
            top_k=top_k,
            min_score=min_score,
            keep_per_position=keep_per_position,
        )

    def enrich(
        self,
        events: Sequence[CausalReasoningEvent],
        window_before: int = 8,
        allow_model_mismatch: bool = False,
    ) -> List[CausalReasoningEvent]:
        """Attach latent events to existing token/sentence events."""
        return enrich_events_with_latent(
            events,
            self.model,
            self.tokenizer,
            self.readout,
            layers=self.workspace_layers,
            window_before=window_before,
            top_k=self.extractor.top_k,
            min_score=self.extractor.min_score,
            keep_per_position=self.extractor.keep_per_position,
            allow_model_mismatch=allow_model_mismatch,
            latent_model_id=self.model_name,
        )

    def probe(
        self,
        query: str,
        reasoning_trace: Optional[str] = None,
        system_prompt: Optional[str] = None,
        task_type: str = "generic",
        dataset_id: Optional[str] = None,
        item_id: Optional[str] = None,
        category: Optional[str] = None,
        stride: int = 4,
    ) -> Generator[CausalReasoningEvent, None, None]:
        """Read the workspace along a reasoning trace, emitting latent events.

        Generates a trace if none is supplied, then walks it at ``stride``-token
        intervals reading the workspace at each step.
        """
        prompt = self.format_prompt(query, system_prompt=system_prompt, category=category)

        if reasoning_trace is None:
            completions = self.generate_completions(prompt, num_samples=1)
            reasoning_trace = completions[0] if completions else ""

        if not reasoning_trace.strip():
            self.logger.warning("Empty reasoning trace; nothing to probe")
            return

        from ..classification import classify_event_label
        from ..events import make_latent_event

        prompt_len = len(self.tokenizer(prompt).input_ids)
        trace_ids = self.tokenizer(reasoning_trace, add_special_tokens=False).input_ids

        for end in range(stride, len(trace_ids) + 1, stride):
            context = prompt + self.tokenizer.decode(trace_ids[:end])

            try:
                by_layer = self.extractor.extract_for_context(context, window_before=1)
            except Exception as e:
                self.logger.debug(f"Readout failed at position {end}: {e}")
                continue

            for layer, results in by_layer.items():
                # Honour keep_per_position here as well as in enrichment. Without
                # the cap, a 400-token trace at stride 4 over 4 layers with
                # top_k=25 emits 10,000 latent events for a single query -- the
                # flag promises at most 3.
                kept = [r for r in results if self.extractor._keep(r)]
                kept = kept[: self.extractor.keep_per_position]
                self.extractor.note_kept(len(kept))

                for r in kept:
                    event = make_latent_event(
                        query=query,
                        context=context,
                        metatoken=r.token,
                        token_id=r.token_id,
                        score=r.score,
                        layer=layer,
                        # Absolute token index including the prompt, so a probe
                        # event and a token event are in the same index frame and
                        # the linker can actually compare them. (Token PTS records
                        # prompt-inclusive indices too.)
                        position=prompt_len + end,
                        model_id=self.model_name,
                        layer_name=f"model.layers.{layer}",
                        task_type=task_type,
                        dataset_id=dataset_id,
                        dataset_item_id=item_id,
                        category=classify_event_label(r.token, "latent"),
                        readout_method=r.readout_method,
                        confidence=r.score,
                        search_method="latent_pts",
                        metadata={
                            "j_score": r.score,
                            "rank": r.rank,
                            "workspace_layers": self.workspace_layers,
                            "trace_position": end,
                            "prompt_len": prompt_len,
                            "probe_stride": stride,
                        },
                    )
                    self.event_storage.add_event(event)
                    yield event

        self.extractor.report_filtering()

        if self.event_storage.filepath:
            self.event_storage.save()
