"""
Sentence PTS -- pivotal reasoning sentences (thought anchors).

A sentence is pivotal if including it shifts the estimated probability of
success relative to removing it or substituting an alternative::

    P(success | prefix + sentence) - P(success | prefix + alternative)

This is the thought-anchor method, expressed as one scale of the unified PTS
framework.
"""

import logging
import re
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch

from ..classification import COMPUTATION, VERIFICATION, classify_event_label
from ..events import CausalReasoningEvent, make_sentence_event
from ..event_storage import EventStorage
from .base import BasePTSSearcher

logger = logging.getLogger(__name__)


class SentenceSegmenter:
    """Split a reasoning trace into sentence-level reasoning steps."""

    def segment(self, text: str) -> List[str]:
        text = self.extract_final_response(text)

        sentences = re.split(
            r"[.!?]+|\n+|(?:Therefore)|(?:Thus)|(?:So,)|(?:Hence)", text
        )

        # Equations on their own line start a new reasoning step.
        expanded: List[str] = []
        for sentence in sentences:
            expanded.extend(re.split(r"\n(?=[=])|(?<=[0-9])\s*\n(?=[0-9])", sentence))

        cleaned: List[str] = []
        for sentence in expanded:
            sentence = sentence.strip()
            if not sentence:
                continue
            if re.match(r"^[=\-#*]+$", sentence):
                continue

            # Math steps are often terse ("= 42"), so they get a lower length
            # bar than prose, which needs to look like an actual clause.
            if re.search(r"\d|[+\-*/=]|\\boxed|answer|result", sentence, re.IGNORECASE):
                if len(sentence) >= 5:
                    cleaned.append(sentence)
            elif len(sentence) >= 15 and len(sentence.split()) >= 3:
                cleaned.append(sentence)

        return cleaned

    @staticmethod
    def extract_final_response(response: str) -> str:
        match = re.search(r"</think>(.*)", response, re.DOTALL)
        return match.group(1).strip() if match else response

    # v1 name
    def segment_reasoning_trace(self, text: str) -> List[str]:
        return self.segment(text)


class SentencePTSSearcher(BasePTSSearcher):
    """Search for pivotal sentences (thought anchors) in a reasoning trace."""

    def __init__(
        self,
        *args,
        similarity_threshold: float = 0.8,
        enable_verification: bool = False,
        skip_embeddings: bool = False,
        event_storage: Optional[EventStorage] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.similarity_threshold = similarity_threshold
        self.segmenter = SentenceSegmenter()
        self.event_storage = event_storage or EventStorage()

        self.skip_embeddings = skip_embeddings or getattr(
            self.oracle, "skip_embeddings", False
        )
        self.embedding_model = None
        if not self.skip_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                self.logger.warning(
                    "sentence-transformers not installed; falling back to lexical "
                    "similarity for alternative-sentence diversity."
                )

        self.verifier = None
        if enable_verification:
            from ..verification import ArithmeticVerifier

            self.verifier = ArithmeticVerifier(self.model, self.tokenizer, self.device)

    # Sentence PTS conditions on prompt + reasoning-so-far, so the prefix is
    # appended to the formatted prompt rather than replacing it (Token PTS
    # passes a fully-decoded prefix that already contains the prompt).
    def build_conditioning_text(
        self,
        query: str,
        prefix: str = "",
        system_prompt: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        prompt = self.format_prompt(query, system_prompt=system_prompt, category=category)
        if prefix:
            return f"{prompt} {prefix}"
        return prompt

    def estimate_success_probability_with_sentences(
        self,
        query: str,
        sentences: List[str],
        num_samples: Optional[int] = None,
        system_prompt: Optional[str] = None,
        category: Optional[str] = None,
    ) -> float:
        prefix = " ".join(sentences) if sentences else ""
        return self.estimate_success_probability(
            query,
            prefix=prefix,
            num_samples=num_samples,
            system_prompt=system_prompt,
            category=category,
        )

    # -- similarity --------------------------------------------------------

    def compute_sentence_similarity(self, a: str, b: str) -> float:
        if self.embedding_model is not None:
            import numpy as np

            emb = self.embedding_model.encode([a, b])
            denom = np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])
            if denom == 0:
                return 0.0
            return float(np.dot(emb[0], emb[1]) / denom)

        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def embed(self, sentence: str) -> List[float]:
        if self.embedding_model is None:
            return []
        return self.embedding_model.encode([sentence])[0].tolist()

    # -- alternatives ------------------------------------------------------

    ALTERNATIVE_PROMPTS = [
        "{context} Let me try a different approach:",
        "{context} Alternatively, I could:",
        "{context} Another way to think about this:",
        "{context} Instead, let me:",
        "{context} Actually, maybe I should:",
    ]

    def generate_alternative_sentence(
        self,
        original_sentence: str,
        prefix_sentences: List[str],
        query: str,
        max_attempts: int = 5,
    ) -> Optional[str]:
        """Generate a semantically *different* sentence to substitute in.

        The counterfactual is only meaningful if the alternative actually says
        something else, so candidates too similar to the original are rejected.
        """
        context = " ".join(prefix_sentences) if prefix_sentences else ""

        for attempt in range(max_attempts):
            template = self.ALTERNATIVE_PROMPTS[attempt % len(self.ALTERNATIVE_PROMPTS)]
            prompt = template.format(context=context)

            try:
                tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = tokenized.input_ids.to(self.device)
                attention_mask = tokenized.attention_mask.to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        max_new_tokens=50,
                        temperature=self.temperature * 1.2,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )

                generated = self.tokenizer.decode(
                    outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True
                )
                candidates = self.segmenter.segment(generated)
                if not candidates:
                    continue

                alternative = candidates[0]
                if self.compute_sentence_similarity(original_sentence, alternative) < self.similarity_threshold:
                    return alternative
            except Exception as e:
                self.logger.debug(f"Error generating alternative sentence: {e}")
                continue

        return None

    # -- analysis ----------------------------------------------------------

    PREMISE = ["because", "since", "given that", "if", "suppose", "assume"]
    CONCLUSION = ["therefore", "thus", "hence", "so", "consequently"]
    ELABORATION = ["specifically", "for example", "in other words", "that is"]
    CONTRADICTION = ["but", "however", "although", "despite", "on the other hand"]

    def analyze_dependencies(
        self, sentence: str, sentence_id: int, all_sentences: List[str]
    ) -> Tuple[List[int], List[int], Optional[str]]:
        deps: List[int] = []
        dependents: List[int] = []
        relationship = None

        lower = sentence.lower()
        if any(i in lower for i in self.CONCLUSION):
            relationship = "conclusion"
        elif any(i in lower for i in self.PREMISE):
            relationship = "premise"
        elif any(i in lower for i in self.ELABORATION):
            relationship = "elaboration"
        elif any(i in lower for i in self.CONTRADICTION):
            relationship = "contradiction"

        back_refs = ["result", "answer", "total", "sum", "product", "this", "that"]
        explicit_refs = ["above", "previously", "earlier", "before"]

        for i in range(min(sentence_id, len(all_sentences))):
            prev = all_sentences[i].lower()
            if any(w in lower for w in back_refs) and re.search(r"\d+|=|\+|\-|\*|/", prev):
                deps.append(i)
            elif any(r in lower for r in explicit_refs):
                deps.append(i)

        if re.search(r"\d+|=|\+|\-|\*|/", lower):
            for i in range(sentence_id + 1, len(all_sentences)):
                nxt = all_sentences[i].lower()
                if any(w in nxt for w in ["result", "answer", "total", "this", "that"]):
                    dependents.append(i)

        return deps, dependents, relationship

    def analyze_failure_mode(
        self, sentence: str, prob_delta: float, alternatives: List[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if prob_delta >= 0:
            return None, None, None

        lower = sentence.lower()

        if re.search(r"\b\d+\s*[+\-*/]\s*\d+\s*=\s*\d+", sentence):
            mode, err = "computational_mistake", "arithmetic_error"
            fix = "Verify mathematical calculations step by step"
        elif any(w in lower for w in ["but", "however", "although", "wrong", "incorrect", "mistake"]):
            mode, err = "logical_error", "contradiction"
            fix = "Resolve logical inconsistency"
        elif len(sentence.strip()) < 10:
            mode, err = "missing_step", "incomplete_reasoning"
            fix = "Provide more detailed explanation"
        elif any(w in lower for w in ["maybe", "perhaps", "possibly", "might", "could be"]):
            mode, err = "logical_error", "excessive_uncertainty"
            fix = "Provide more definitive reasoning"
        elif sentence.count("=") > 1:
            mode, err = "computational_mistake", "calculation_confusion"
            fix = "Separate calculations into distinct steps"
        else:
            mode, err = "logical_error", "reasoning_error"
            fix = "Revise reasoning logic"

        if alternatives:
            best = max(alternatives, key=lambda a: len(a.split()))
            if len(best) > len(sentence):
                fix = f"Consider more detailed approach: '{best[:100]}...'"

        return mode, err, fix

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        reasoning_trace: str,
        system_prompt: Optional[str] = None,
        task_type: str = "generic",
        dataset_id: Optional[str] = None,
        item_id: Optional[str] = None,
        min_prob: float = 0.2,
        max_prob: float = 0.8,
        category: Optional[str] = None,
    ) -> Generator[CausalReasoningEvent, None, None]:
        """Yield pivotal-sentence events for one reasoning trace.

        For each sentence, compare P(success | prefix + sentence) against
        P(success | prefix + alternative), where the alternative is a generated
        sentence that says something different. Falls back to dropping the
        sentence entirely when no sufficiently different alternative is found.
        """
        sentences = self.segmenter.segment(reasoning_trace)
        if not sentences:
            self.logger.info("No sentences segmented from reasoning trace")
            return

        # `sentences` is read for every later iteration (prefixes, suffixes,
        # dependency analysis). v1 blanked already-processed entries in place
        # under memory pressure, which silently invalidated every prob_delta
        # computed afterwards. Nothing here mutates it.
        self.logger.info(f"Segmented reasoning trace into {len(sentences)} sentences")

        baseline = self.estimate_success_probability(
            query, system_prompt=system_prompt, category=category
        )
        if not (min_prob <= baseline <= max_prob):
            self.logger.info(
                f"Baseline success probability {baseline:.2f} outside "
                f"[{min_prob:.2f}, {max_prob:.2f}], skipping query"
            )
            return

        for i, sentence in enumerate(sentences):
            prefix_sentences = sentences[:i]

            prob_with = self.estimate_success_probability_with_sentences(
                query, sentences[: i + 1], system_prompt=system_prompt, category=category
            )

            alternative = self.generate_alternative_sentence(
                sentence, prefix_sentences, query
            )
            alternatives_tested: List[str] = []

            if alternative:
                alternatives_tested.append(alternative)
                prob_without = self.estimate_success_probability_with_sentences(
                    query,
                    prefix_sentences + [alternative],
                    system_prompt=system_prompt,
                    category=category,
                )
                intervention = "replace_sentence"
            else:
                # No sufficiently different alternative: fall back to ablation.
                prob_without = self.estimate_success_probability_with_sentences(
                    query, prefix_sentences, system_prompt=system_prompt, category=category
                )
                intervention = "remove_sentence"

            prob_delta = prob_with - prob_without
            if abs(prob_delta) < self.prob_threshold:
                continue

            deps, dependents, relationship = self.analyze_dependencies(
                sentence, i, sentences
            )
            failure_mode, error_type, correction = self.analyze_failure_mode(
                sentence, prob_delta, alternatives_tested
            )

            unified_category = classify_event_label(sentence, "sentence")

            verification_score = None
            verification_method = None
            arithmetic_errors: List[Dict[str, Any]] = []
            attention_entropy = None
            attention_focus = None
            if self.verifier is not None and unified_category in (VERIFICATION, COMPUTATION):
                try:
                    result = self.verifier.verify_sentence(
                        sentence, prefix_context=" ".join(prefix_sentences)
                    )
                    verification_score = result.get("verification_score")
                    verification_method = result.get("verification_method")
                    arithmetic_errors = result.get("arithmetic_errors", [])
                    attention_entropy = result.get("attention_entropy")
                    attention_focus = result.get("attention_focus_score")
                except Exception as e:
                    self.logger.debug(f"Verification failed for sentence {i}: {e}")

            event = make_sentence_event(
                query=query,
                context=" ".join(prefix_sentences),
                sentence=sentence,
                sentence_id=i,
                prob_before=prob_without,
                prob_after=prob_with,
                model_id=self.model_name,
                task_type=task_type,
                dataset_id=dataset_id,
                dataset_item_id=item_id,
                category=unified_category,
                confidence=verification_score,
                metadata={
                    "suffix_context": " ".join(sentences[i + 1:]),
                    # Stored whole. v1 truncated to 3000 chars with no marker,
                    # so consumers could not tell a complete trace from a
                    # chopped one.
                    "full_reasoning_trace": reasoning_trace,
                    "alternatives_tested": alternatives_tested,
                    "causal_dependencies": deps,
                    "causal_dependents": dependents,
                    "logical_relationship": relationship,
                    "failure_mode": failure_mode,
                    "error_type": error_type,
                    "correction_suggestion": correction,
                    "verification_score": verification_score,
                    "verification_method": verification_method,
                    "arithmetic_errors": arithmetic_errors,
                    "attention_entropy": attention_entropy,
                    "attention_focus_score": attention_focus,
                    "sentence_embedding": self.embed(sentence),
                    "baseline_prob": baseline,
                    "num_sentences": len(sentences),
                },
            )
            event.intervention_type = intervention

            self.event_storage.add_event(event)
            yield event

        if self.event_storage.filepath:
            self.event_storage.save()

    # v1 name
    def search_thought_anchors(self, *args, **kwargs):
        return self.search(*args, **kwargs)
