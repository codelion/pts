"""
P(success | prompt, prefix) for a behavior, by sampling on an mlx-lm model.

The same estimator as token PTS (``estimate_success_probability``): sample
completions to the end and ask an oracle whether each one succeeded. Nothing
is added to the dataset's prompt and nothing is read before the model has
finished, so any model, chat template (thinking included), and dataset whose
success an oracle can judge works the same way.

For multiple-choice behavior evals the oracle is ``ChoiceOracle``: success is
"the model's final answer is the behavior-matching choice". Completions with
no recognisable answer count as failures, as in token PTS; the fraction that
answered at all is reported alongside, because a steering vector that stops
the model answering would otherwise look like it reduced the behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evals import ChoiceItem


def limit_gpu_memory(fraction: float = 0.5) -> int:
    """Cap MLX's memory at ``fraction`` of the device's memory, and return the cap.

    Discovery samples thousands of completions. Left uncapped, a large batch
    can exhaust GPU memory, and on macOS that has panicked the kernel rather
    than failing the job. With a cap MLX frees cached buffers and slows down
    instead.
    """
    import mlx.core as mx

    cap = int(mx.device_info()["memory_size"] * fraction)
    mx.set_memory_limit(cap)
    return cap


@dataclass
class Cost:
    """What a run spent, so efficiency is measured rather than assumed."""

    sampled_tokens: int = 0
    completions: int = 0
    prefilled_once: int = 0     # distinct prompts whose prefill all their samples shared

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty")


@dataclass
class Estimate:
    """Counts behind one P(success) estimate; more samples can be added later."""

    successes: int = 0
    answered_count: int = 0
    n: int = 0
    exact: bool = False     # a finished response: the oracle's verdict, not a sample

    @property
    def p(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    @property
    def answered(self) -> float:
        return self.answered_count / self.n if self.n else float("nan")

    @property
    def se(self) -> float:
        if self.exact:
            return 0.0
        if self.n < 2:
            return float("nan")
        return float(np.sqrt(self.p * (1 - self.p) / (self.n - 1)))

    def __add__(self, other: "Estimate") -> "Estimate":
        return Estimate(self.successes + other.successes, self.answered_count + other.answered_count,
                        self.n + other.n)


class ChoiceScorer:
    """Samples completions on an mlx-lm model and judges them with ``oracle``.

    ``oracle`` is any PTS oracle: ``check_success(query, response) -> bool``,
    called with the item's question as the query and the response text alone.
    If it also has ``answered(query, response) -> bool`` that is used for the
    answered fraction; otherwise every completion counts as answered.
    """

    def __init__(
        self,
        model,
        tokenizer,
        oracle,
        temperature: float = 0.7,
        top_p: float = 0.95,
        chat_template_args: Optional[Dict] = None,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        batch_size: int = 32,
    ):
        self.model = model
        self.tok = tokenizer
        self.oracle = oracle
        # Passed to apply_chat_template, as mlx_lm's --chat-template-args.
        self.chat_template_args = dict(chat_template_args or {})
        self._open_think: Dict[Tuple[int, ...], bool] = {}
        # Sampling, as in a model card's recommended settings (e.g. Qwen3.5
        # thinking: temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5;
        # without the penalty small models can loop in their thinking forever).
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        # Sequences generated at once. One generator runs every prompt, admitting
        # the next as soon as one finishes, so the batch never drains waiting
        # for its longest sequence. A sequence is inserted only when a slot
        # frees, so at most this many prompt-cache copies exist at a time.
        # mlx-lm pads a batch's KV cache to its longest sequence, so memory is
        # about batch_size x the longest context.
        self.batch_size = batch_size
        self.cost = Cost()
        # Prepended to every question; used to compare steering with plain prompting.
        self.extra_instruction: Optional[str] = None

    def prompt_ids(self, item: ChoiceItem) -> List[int]:
        content = item.question.strip()
        if self.extra_instruction:
            content = self.extra_instruction.strip() + "\n\n" + content
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
            **self.chat_template_args,
        )
        # The rendered template already contains its special tokens.
        return list(self.tok.encode(text, add_special_tokens=False))

    share_prefill = True

    def _prefill(self, tokens: Sequence[int]):
        """A KV cache covering ``tokens``."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        for i in range(0, len(tokens), 2048):
            self.model(mx.array(list(tokens[i:i + 2048]))[None], cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    def sample(self, prompts: Sequence[Sequence[int]], max_tokens: int) -> List[List[int]]:
        """One sampled continuation per prompt (stop token excluded), batched.

        A prompt that appears more than once (every sample of an estimate
        continues the same prompt + prefix) is processed once; each sample
        starts from a copy of that cache, as mlx-lm's server reuses prompt
        caches.
        """
        import copy

        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        counts: Dict[Tuple[int, ...], int] = {}
        for x in prompts:
            counts[tuple(x)] = counts.get(tuple(x), 0) + 1
        left = dict(counts)
        shared: Dict[Tuple[int, ...], object] = {}
        prefilled = 0

        gen = BatchGenerator(
            self.model,
            stop_tokens=[[t] for t in self.tok.eos_token_ids],
            sampler=make_sampler(temp=self.temperature, top_p=self.top_p, min_p=self.min_p,
                                 top_k=self.top_k),
            logits_processors=make_logits_processors(
                presence_penalty=self.presence_penalty or None),
            completion_batch_size=self.batch_size,
        )
        results: List[List[int]] = [[] for _ in prompts]
        index: Dict[int, int] = {}
        # Admit prompts shortest first, so sequences sharing the batch have
        # similar lengths and the KV cache is not padded to a much longer one.
        queue = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
        try:
            while queue or index:
                while queue and len(index) < self.batch_size:
                    i = queue.pop()
                    x = list(prompts[i])
                    key = tuple(x)
                    if self.share_prefill and counts[key] > 1 and len(x) > 1:
                        if key not in shared:
                            shared[key] = self._prefill(x[:-1])
                            prefilled += 1
                        cache, tail = copy.deepcopy(shared[key]), x[-1:]
                        left[key] -= 1
                        if left[key] == 0:
                            del shared[key]
                    else:
                        cache, tail = None, x
                    (uid,) = gen.insert([tail], [max_tokens], caches=[cache],
                                        all_tokens=[x[:len(x) - len(tail)]])
                    index[uid] = i
                for r in gen.next_generated():
                    if r.finish_reason != "stop":
                        results[index[r.uid]].append(r.token)
                    if r.finish_reason is not None:
                        del index[r.uid]
        finally:
            gen.close()
            shared.clear()
            mx.clear_cache()
        self.cost.sampled_tokens += sum(len(v) for v in results)
        self.cost.completions += len(prompts)
        self.cost.prefilled_once += prefilled
        return results

    def _opens_thinking(self, item: ChoiceItem) -> bool:
        """Whether the template already opened a thinking block in the prompt."""
        key = (item.question, self.extra_instruction)
        if key not in self._open_think:
            text = self.tok.decode(self.prompt_ids(item))
            self._open_think[key] = text.rfind("<think>") > text.rfind("</think>")
        return self._open_think[key]

    def judge(self, item: ChoiceItem, response_ids: Sequence[int]) -> Tuple[bool, bool]:
        """(success, answered) for one finished response.

        If the chat template opened a thinking block, the response is judged
        with that tag in front of it, so the oracle sees where thinking ends.
        """
        text = self.tok.decode(list(response_ids))
        if self._opens_thinking(item):
            text = "<think>" + text
        success = bool(self.oracle.check_success(item.question, text))
        answered = getattr(self.oracle, "answered", None)
        return success, (bool(answered(item.question, text)) if answered else True)

    def estimate_many(
        self,
        requests: Sequence[Tuple[ChoiceItem, Sequence[int], Sequence[int]]],
        num_samples,
        max_tokens: int,
    ) -> List[Estimate]:
        """``estimate`` for several (item, prompt, prefix) at once, in one batch.

        ``num_samples`` is one count for every request or a count per request.
        """
        counts = [num_samples] * len(requests) if isinstance(num_samples, int) else list(num_samples)
        prompts = [list(p) + list(x) for (_, p, x), k in zip(requests, counts) for _ in range(k)]
        conts = iter(self.sample(prompts, max_tokens))
        out = []
        for (item, _, prefix), k in zip(requests, counts):
            judged = [self.judge(item, list(prefix) + next(conts)) for _ in range(k)]
            out.append(Estimate(sum(j[0] for j in judged), sum(j[1] for j in judged), len(judged)))
        return out

    def estimate(
        self,
        item: ChoiceItem,
        prompt_ids: Sequence[int],
        prefix_ids: Sequence[int],
        num_samples: int,
        max_tokens: int,
    ) -> Estimate:
        """P(success | prompt + prefix): sample completions, judge each whole response."""
        return self.estimate_many([(item, prompt_ids, prefix_ids)], num_samples, max_tokens)[0]
