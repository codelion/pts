"""Searcher behaviour against a real (tiny) model.

These cover the parts that only break when a model is actually attached: prompt
formatting, the probability cache, and the searchers' event output.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pts.events import EVENT_SENTENCE, EVENT_TOKEN  # noqa: E402
from pts.oracle import DummyOracle, Oracle  # noqa: E402
from pts.searchers.base import BasePTSSearcher, select_device  # noqa: E402
from pts.searchers.sentence import SentencePTSSearcher, SentenceSegmenter  # noqa: E402
from pts.searchers.token import TokenPTSSearcher  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class CountingOracle(Oracle):
    """Succeeds iff the response contains a marker. Counts how often it is asked."""

    def __init__(self, marker="42"):
        self.marker = marker
        self.calls = 0

    def check_success(self, query, response):
        self.calls += 1
        return self.marker in response


class CategoryOracle(Oracle):
    """Formats a different prompt per category -- the thing v1's cache key ignored."""

    def __init__(self):
        self.calls = 0

    def get_prompt_for_category(self, question, category):
        return f"[{category}] {question}"

    def check_success(self, query, response):
        self.calls += 1
        return "[gsm8k]" in response


@pytest.fixture(scope="module")
def base():
    return BasePTSSearcher(
        model_name=TINY,
        oracle=CountingOracle(),
        device="cpu",
        num_samples=2,
        batch_size=2,
        max_new_tokens=4,
    )


# -- prompt formatting -----------------------------------------------------

def test_category_prompt_is_used(base):
    s = BasePTSSearcher(
        model_name=TINY, oracle=CategoryOracle(), device="cpu",
        num_samples=1, batch_size=1, max_new_tokens=2,
        model=base.model, tokenizer=base.tokenizer,
    )
    assert s.format_prompt("What is 2+2?", category="gsm8k") == "[gsm8k] What is 2+2?"
    assert s.format_prompt("What is 2+2?") == "What is 2+2?"


# -- the probability cache -------------------------------------------------

def test_cache_key_includes_category(base):
    """v1 omitted `category` from the cache key.

    Because category selects a *different prompt*, two categories with the same
    query silently returned each other's cached probability. This is the test
    that would have caught it.
    """
    oracle = CategoryOracle()
    s = BasePTSSearcher(
        model_name=TINY, oracle=oracle, device="cpu",
        num_samples=1, batch_size=1, max_new_tokens=2,
        model=base.model, tokenizer=base.tokenizer,
    )

    s.estimate_success_probability("q", category="gsm8k")
    after_first = oracle.calls
    assert after_first > 0

    # Same query, DIFFERENT category -> must not be served from cache.
    s.estimate_success_probability("q", category="boolq")
    assert oracle.calls > after_first, "different category was served a stale cached value"

    # Same query, same category -> must be served from cache.
    before = oracle.calls
    s.estimate_success_probability("q", category="gsm8k")
    assert oracle.calls == before, "identical call was not cached"


def test_cache_defaults_num_samples_before_keying(base):
    """v1 keyed on num_samples *before* defaulting it, so None and its default
    were two entries for identical work."""
    oracle = CountingOracle()
    s = BasePTSSearcher(
        model_name=TINY, oracle=oracle, device="cpu",
        num_samples=2, batch_size=2, max_new_tokens=2,
        model=base.model, tokenizer=base.tokenizer,
    )

    s.estimate_success_probability("q", num_samples=None)
    before = oracle.calls
    s.estimate_success_probability("q", num_samples=2)  # == self.num_samples
    assert oracle.calls == before, "None and its default were cached separately"


def test_cache_is_bounded(base):
    s = BasePTSSearcher(
        model_name=TINY, oracle=CountingOracle(), device="cpu",
        num_samples=1, batch_size=1, max_new_tokens=2, max_cache_size=3,
        model=base.model, tokenizer=base.tokenizer,
    )
    for i in range(10):
        s.estimate_success_probability(f"query {i}")
    # v1 responded to memory pressure by clearing the whole cache, which freed
    # nothing and forced full recomputation. A bound makes that unnecessary.
    assert len(s.prob_cache) == 3


def test_oracle_is_required_for_probability(base):
    s = BasePTSSearcher(
        model_name=TINY, oracle=None, device="cpu",
        model=base.model, tokenizer=base.tokenizer,
    )
    with pytest.raises(ValueError, match="Oracle must be provided"):
        s.estimate_success_probability("q")


# -- token searcher --------------------------------------------------------

def test_token_searcher_emits_unified_events(base):
    s = TokenPTSSearcher(
        model_name=TINY, oracle=CountingOracle(), device="cpu",
        num_samples=2, batch_size=2, max_new_tokens=8, prob_threshold=0.0,
        model=base.model, tokenizer=base.tokenizer,
    )
    events = list(s.search("What is 2+2?", max_generations=1, min_prob=0.0, max_prob=1.0))

    for e in events:
        assert e.event_type == EVENT_TOKEN
        assert e.granularity == "token"
        assert e.visibility == "emitted"
        assert e.prob_delta is not None
        assert e.score == pytest.approx(abs(e.prob_delta))
        assert e.search_method == "token_pts"


def test_searcher_writes_each_event_once(base):
    """v1's searcher AND its CLI both wrote every token, doubling the dataset."""
    s = TokenPTSSearcher(
        model_name=TINY, oracle=CountingOracle(), device="cpu",
        num_samples=2, batch_size=2, max_new_tokens=8, prob_threshold=0.0,
        model=base.model, tokenizer=base.tokenizer,
    )
    events = list(s.search("What is 2+2?", max_generations=1, min_prob=0.0, max_prob=1.0))

    # Simulate the v1 CLI adding every yielded event a second time.
    for e in events:
        s.event_storage.add_event(e)

    ids = [e.event_id for e in s.event_storage]
    assert len(ids) == len(set(ids)), "storage contains duplicates"


def test_rejected_token_search_refuses_a_dummy_oracle(base):
    """The exact v1 failure: DummyOracle makes every candidate score P=1.0, so no
    rejected token can ever be found and every positive token is silently dropped.
    Refusing loudly beats returning None forever."""
    from pts.events import make_token_event

    s = TokenPTSSearcher(
        model_name=TINY, oracle=DummyOracle(), device="cpu",
        model=base.model, tokenizer=base.tokenizer,
    )
    event = make_token_event(
        query="q", context="ctx", token=" Wait", token_id=1,
        prob_before=0.3, prob_after=0.8, model_id=TINY,
    )
    with pytest.raises(ValueError, match="needs a real oracle"):
        s.find_rejected_token(event)


def test_query_outside_the_probability_band_is_skipped(base):
    s = TokenPTSSearcher(
        model_name=TINY, oracle=CountingOracle(marker="\x00impossible\x00"),
        device="cpu", num_samples=2, batch_size=2, max_new_tokens=4,
        model=base.model, tokenizer=base.tokenizer,
    )
    # Baseline will be 0.0, below min_prob -- no single token can move a
    # saturated probability, so there is nothing to find.
    assert list(s.search("q", min_prob=0.2, max_prob=0.8, max_generations=1)) == []


# -- sentence searcher -----------------------------------------------------

def test_segmenter_does_not_mutate_its_input():
    seg = SentenceSegmenter()
    sentences = seg.segment("First I plan. Then I compute 2 + 2 = 4. Therefore the answer is 4.")
    assert len(sentences) >= 2
    assert all(s.strip() for s in sentences), "segmenter emitted a blank sentence"


def test_segmenter_strips_think_tags():
    seg = SentenceSegmenter()
    out = seg.segment("<think>hidden reasoning here</think>The answer is 4.")
    assert not any("hidden" in s for s in out)


def test_sentence_searcher_appends_prefix_to_prompt(base):
    """Sentence PTS conditions on prompt + reasoning-so-far, so the prefix is
    appended. Token PTS passes a decoded prefix that already contains the
    prompt, so there it replaces. Getting this backwards silently changes what
    the model is conditioned on."""
    s = SentencePTSSearcher(
        model_name=TINY, oracle=CountingOracle(), device="cpu",
        num_samples=1, batch_size=1, max_new_tokens=2, skip_embeddings=True,
        model=base.model, tokenizer=base.tokenizer,
    )
    text = s.build_conditioning_text("What is 2+2?", prefix="Let me think.")
    assert text == "What is 2+2? Let me think."

    t = TokenPTSSearcher(
        model_name=TINY, oracle=CountingOracle(), device="cpu",
        model=base.model, tokenizer=base.tokenizer,
    )
    assert t.build_conditioning_text("What is 2+2?", prefix="decoded prefix") == "decoded prefix"


def test_sentence_searcher_emits_unified_events(base):
    s = SentencePTSSearcher(
        model_name=TINY, oracle=CountingOracle(), device="cpu",
        num_samples=2, batch_size=2, max_new_tokens=4, prob_threshold=0.0,
        skip_embeddings=True,
        model=base.model, tokenizer=base.tokenizer,
    )
    events = list(s.search(
        "What is 2+2?",
        reasoning_trace="First I plan the approach. Then I compute 2 + 2 = 4. Therefore the answer is 4.",
        min_prob=0.0, max_prob=1.0,
    ))

    for e in events:
        assert e.event_type == EVENT_SENTENCE
        assert e.granularity == "sentence"
        assert e.search_method == "sentence_pts"
        assert e.prob_delta is not None
        # v1 truncated the stored trace to 3000 chars with no marker.
        assert e.metadata["full_reasoning_trace"].startswith("First I plan")


def test_device_selection():
    assert select_device("cpu") == "cpu"
    assert select_device() in ("cuda", "mps", "cpu")
