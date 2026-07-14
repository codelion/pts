"""
Token PTS -- emitted pivotal tokens.

A token is pivotal if appending it to a prefix shifts the estimated probability
of eventually succeeding::

    P(success | prefix + token) - P(success | prefix)

Found by recursively bisecting a generated sequence until each segment either
holds a single token or moves the probability by less than the threshold. This
is the original Pivotal Token Search from the Phi-4 technical report.
"""

import logging
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch

from ..classification import classify_event_label
from ..events import CausalReasoningEvent, make_token_event
from ..event_storage import EventStorage
from .base import BasePTSSearcher

logger = logging.getLogger(__name__)


class TokenPTSSearcher(BasePTSSearcher):
    """Search for emitted pivotal tokens in model generations."""

    def __init__(self, *args, event_storage: Optional[EventStorage] = None, **kwargs):
        # v1 called this `token_storage`; accept either.
        token_storage = kwargs.pop("token_storage", None)
        super().__init__(*args, **kwargs)
        self.event_storage = event_storage or token_storage or EventStorage()

    # Kept so v1 code reaching for `.token_storage` still works.
    @property
    def token_storage(self) -> EventStorage:
        return self.event_storage

    def subdivide_sequence(
        self,
        query: str,
        sequence: List[int],
        prefix: Optional[List[int]] = None,
        system_prompt: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[List[int]]:
        """Bisect until each segment is one token or moves probability below threshold."""
        prefix = prefix or []

        if len(sequence) <= 1:
            return [sequence]

        prefix_str = self.tokenizer.decode(prefix, skip_special_tokens=False)
        full_str = self.tokenizer.decode(prefix + sequence, skip_special_tokens=False)

        prob_before = self.estimate_success_probability(
            query, prefix_str, system_prompt=system_prompt, category=category
        )
        prob_after = self.estimate_success_probability(
            query, full_str, system_prompt=system_prompt, category=category
        )

        if abs(prob_after - prob_before) < self.prob_threshold:
            return [sequence]

        mid = len(sequence) // 2
        left, right = sequence[:mid], sequence[mid:]

        left_segments = self.subdivide_sequence(
            query, left, prefix, system_prompt, category
        )
        right_segments = self.subdivide_sequence(
            query, right, prefix + left, system_prompt, category
        )
        return left_segments + right_segments

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
    ) -> Generator[CausalReasoningEvent, None, None]:
        """Yield pivotal-token events for one query.

        Only queries whose baseline success probability falls inside
        ``[min_prob, max_prob]`` are searched: a query the model always solves
        (or never solves) has no pivotal tokens to find, because no single token
        can move a probability that is already saturated.
        """
        init_prob = self.estimate_success_probability(
            query, system_prompt=system_prompt, category=category
        )

        if not (min_prob <= init_prob <= max_prob):
            self.logger.info(
                f"Baseline success probability {init_prob:.2f} outside "
                f"[{min_prob:.2f}, {max_prob:.2f}], skipping query"
            )
            return

        self.logger.info(f"Baseline success probability: {init_prob:.4f}")

        prompt = self.format_prompt(query, system_prompt=system_prompt, category=category)
        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True)
        prompt_ids = tokenized.input_ids[0].to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        prompt_len = len(prompt_ids)

        for i in range(max_generations):
            self.logger.info(f"Generating sequence {i + 1}/{max_generations}")

            with torch.no_grad():
                output = self.model.generate(
                    prompt_ids.unsqueeze(0),
                    attention_mask=attention_mask,
                    do_sample=True,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=self.min_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )

            sequence = output.sequences[0][prompt_len:].tolist()
            if len(sequence) < 5:
                self.logger.info("Generated sequence too short, skipping")
                continue

            segments = self.subdivide_sequence(
                query,
                sequence,
                prefix=prompt_ids.tolist(),
                system_prompt=system_prompt,
                category=category,
            )

            current_prefix = prompt_ids.tolist()
            for segment in segments:
                if not segment:
                    continue

                if len(segment) == 1:
                    token_id = segment[0]
                    token_str = self.tokenizer.decode([token_id])

                    prefix_str = self.tokenizer.decode(
                        current_prefix, skip_special_tokens=False
                    )
                    with_token_str = self.tokenizer.decode(
                        current_prefix + segment, skip_special_tokens=False
                    )

                    prob_before = self.estimate_success_probability(
                        query, prefix_str, system_prompt=system_prompt, category=category
                    )
                    prob_after = self.estimate_success_probability(
                        query, with_token_str, system_prompt=system_prompt, category=category
                    )
                    prob_delta = prob_after - prob_before

                    if abs(prob_delta) >= self.prob_threshold:
                        event = make_token_event(
                            query=query,
                            context=prefix_str,
                            token=token_str,
                            token_id=token_id,
                            prob_before=prob_before,
                            prob_after=prob_after,
                            model_id=self.model_name,
                            task_type=task_type,
                            dataset_id=dataset_id,
                            dataset_item_id=item_id,
                            # Absolute index in the full sequence, so latent
                            # events (indexed the same way) can be aligned.
                            position=len(current_prefix),
                            category=classify_event_label(token_str, "token"),
                            metadata={
                                "prob_threshold": self.prob_threshold,
                                "baseline_prob": init_prob,
                                "generation_index": i,
                            },
                        )

                        # The searcher owns the write. v1 also had the CLI write
                        # every yielded token, duplicating each record; storage
                        # now de-duplicates by event_id so that is harmless.
                        self.event_storage.add_event(event)
                        yield event

                current_prefix = current_prefix + segment

        if self.event_storage.filepath:
            self.event_storage.save()

    # -- rejected tokens (for DPO) -----------------------------------------

    def find_rejected_token(
        self,
        event: CausalReasoningEvent,
        num_candidates: int = 10,
        category: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[Tuple[str, int, float]]:
        """Find a token that, in place of a positive pivotal token, hurts success.

        Requires a *real* oracle. With ``DummyOracle`` (which returns True for
        everything) every candidate scores exactly 1.0, the acceptance test can
        never fire, and this silently returns None for every token -- which is
        how v1 managed to drop every positive token from every DPO export.
        """
        if not event.is_positive:
            return None

        from ..oracle import DummyOracle

        if isinstance(self.oracle, DummyOracle):
            raise ValueError(
                "find_rejected_token needs a real oracle: DummyOracle reports every "
                "completion as a success, so every candidate scores P(success)=1.0 and "
                "no rejected token can ever be found. Pass the original dataset so an "
                "oracle can be reconstructed (see `pts export --dataset ...`)."
            )

        context_ids = self.tokenizer.encode(event.context, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(context_ids).logits[0, -1, :]

        probs = torch.softmax(logits, dim=0)
        top = torch.topk(probs, k=num_candidates)

        for i in range(num_candidates):
            token_id = top.indices[i].item()
            if token_id == event.token_id:
                continue
            token_str = self.tokenizer.decode([token_id])

            prob_after = self.estimate_success_probability(
                event.query,
                event.context + token_str,
                system_prompt=system_prompt,
                category=category,
            )

            if event.prob_before - prob_after >= self.prob_threshold:
                return (token_str, token_id, prob_after)

        return None


# ---------------------------------------------------------------------------
# v1 compatibility
# ---------------------------------------------------------------------------

class PivotalTokenSearcher(TokenPTSSearcher):
    """v1 name and method surface, kept working on top of the v2 searcher."""

    def search_pivotal_tokens(self, *args, **kwargs):
        return self.search(*args, **kwargs)

    def find_rejected_tokens(self, pivotal_token, num_candidates: int = 10, category=None):
        from ..events import from_any_record

        return self.find_rejected_token(
            from_any_record(pivotal_token), num_candidates=num_candidates, category=category
        )
