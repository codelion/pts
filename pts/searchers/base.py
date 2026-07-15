"""
Shared machinery for every PTS searcher.

Model loading, device selection, prompt formatting, batched generation, and
success-probability estimation live here exactly once. In v1 this code was
copy-pasted between ``PivotalTokenSearcher`` and ``ThoughtAnchorSearcher``, and
the two copies had drifted apart (only one handled MPS dtype; only one bounded
its cache).

The probability cache is the subtle part -- see ``estimate_success_probability``.
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..oracle import Oracle

logger = logging.getLogger(__name__)


def select_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BasePTSSearcher:
    """Base for all PTS searchers: owns the model, the oracle, and the prob cache."""

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        oracle: Optional[Oracle] = None,
        device: Optional[str] = None,
        prob_threshold: float = 0.2,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_new_tokens: int = 512,
        num_samples: int = 20,
        batch_size: int = 5,
        trust_remote_code: bool = True,
        max_cache_size: int = 4096,
        log_level: int = logging.INFO,
        debug_mode: bool = False,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(self.__class__.__module__)

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.device = select_device(device)
        self.prob_threshold = prob_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.oracle = oracle
        self.debug_mode = debug_mode

        # Bounded LRU. the legacy code responded to memory pressure by clearing the whole
        # cache, which freed almost nothing (these are floats keyed by strings)
        # while forcing every subsequent estimate to re-run num_samples
        # generations. A bounded cache makes that unnecessary.
        self.prob_cache: "OrderedDict[tuple, float]" = OrderedDict()
        self.max_cache_size = max_cache_size

        if model is not None and tokenizer is not None:
            # Sharing one loaded model across searchers is what makes
            # `--granularity all` affordable.
            self.model = model
            self.tokenizer = tokenizer
            self.logger.info(f"Reusing already-loaded model {model_name}")
        else:
            self.model, self.tokenizer = self._load_model()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self):
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device,
        }

        try:
            import flash_attn  # noqa: F401
            has_flash = True
        except ImportError:
            has_flash = False

        if has_flash and self.device == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["attn_implementation"] = "flash_attention_2"
            self.logger.info(
                f"Flash Attention 2 enabled ({model_kwargs['torch_dtype']})"
            )
        elif self.device == "cuda":
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            )
        elif self.device == "mps":
            # MPS has incomplete half-precision kernel coverage; fp32 is the
            # only reliable choice there.
            model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.logger.info(f"Loading model {self.model_name} on {self.device}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, trust_remote_code=self.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        return model, tokenizer

    # -- prompting ---------------------------------------------------------

    def format_prompt(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        """Turn a query into the exact string the model is conditioned on."""
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if category and hasattr(self.oracle, "get_prompt_for_category"):
            return self.oracle.get_prompt_for_category(query, category)

        return query

    def build_conditioning_text(
        self,
        query: str,
        prefix: str = "",
        system_prompt: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        """The text the model actually sees.

        Token PTS passes a fully-decoded prefix that already contains the
        formatted prompt, so the prefix *replaces* it. Sentence PTS overrides
        this to append instead.
        """
        if prefix:
            return prefix
        return self.format_prompt(query, system_prompt=system_prompt, category=category)

    # -- generation --------------------------------------------------------

    def generate_completions(
        self,
        prompt: str,
        num_samples: int,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        completions: List[str] = []
        remaining = num_samples

        with tqdm(
            total=num_samples, desc="Generating samples", disable=num_samples < 10, leave=False
        ) as pbar:
            while remaining > 0:
                current = min(self.batch_size, remaining)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        num_return_sequences=current,
                        max_new_tokens=max_new_tokens or self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        min_p=self.min_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )
                for seq in outputs.sequences:
                    completions.append(
                        self.tokenizer.decode(
                            seq[input_ids.shape[1]:], skip_special_tokens=True
                        )
                    )
                    pbar.update(1)
                remaining -= current

        return completions

    # -- scoring -----------------------------------------------------------

    def _cache_get(self, key: tuple) -> Optional[float]:
        if key in self.prob_cache:
            self.prob_cache.move_to_end(key)
            return self.prob_cache[key]
        return None

    def _cache_put(self, key: tuple, value: float) -> None:
        self.prob_cache[key] = value
        self.prob_cache.move_to_end(key)
        while len(self.prob_cache) > self.max_cache_size:
            self.prob_cache.popitem(last=False)

    def estimate_success_probability(
        self,
        query: str,
        prefix: str = "",
        num_samples: Optional[int] = None,
        system_prompt: Optional[str] = None,
        category: Optional[str] = None,
    ) -> float:
        """P(success | query, prefix), estimated by sampling and asking the oracle."""
        if self.oracle is None:
            raise ValueError("Oracle must be provided to estimate success probability")

        # Two cache-key bugs from v1 are fixed here, both of which returned
        # confidently wrong probabilities:
        #
        #   1. `category` was absent from the key, but it selects a *different
        #      prompt* via oracle.get_prompt_for_category. Two categories with
        #      the same query therefore read each other's cached probability.
        #   2. `num_samples` was keyed before being defaulted, so None and the
        #      value it defaults to were two separate entries for identical work.
        num_samples = num_samples or self.num_samples
        cache_key = (query, prefix, num_samples, system_prompt, category)

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        prompt = self.build_conditioning_text(
            query, prefix=prefix, system_prompt=system_prompt, category=category
        )

        completions = self.generate_completions(prompt, num_samples)

        success_count = 0
        for completion in completions:
            success = self.oracle.check_success(query, prompt + completion)
            if success:
                success_count += 1
            if self.debug_mode:
                print("\n" + "=" * 50)
                print(f"QUERY: {query}")
                if category:
                    print(f"CATEGORY: {category}")
                print("-" * 50)
                print(f"COMPLETION (SUCCESS: {success}):\n{completion}")
                print("=" * 50)

        prob = success_count / num_samples if num_samples else 0.0
        self.logger.debug(f"Success probability: {prob:.4f}")
        self._cache_put(cache_key, prob)
        return prob

    # -- misc --------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def free(self) -> None:
        """Release GPU memory held by this searcher's model."""
        del self.model
        self.prob_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
