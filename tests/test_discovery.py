"""Behavior scoring on MLX.

Runs on the tiny random Llama with a minimal chat template attached; its
weights are random, so tests check exactness and invariants, not answers.
"""

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")
np = pytest.importorskip("numpy")

from pts.discovery.evals import parse_row  # noqa: E402

AB_ROW = {
    "question": "Do you agree?\n\n (A) Yes, I do\n (B) No, I don't",
    "answer_matching_behavior": " (A)",
    "answer_not_matching_behavior": " (B)",
}


TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="module")
def scorer():
    from pts.discovery.scoring import ChoiceScorer

    model, tok = mlx_lm.load(TINY)
    tok._tokenizer.chat_template = TEMPLATE
    return ChoiceScorer(model, tok)


ITEM = parse_row(AB_ROW, 0)


def test_cue_is_the_template_between_assistant_content_and_the_next_answer(scorer):
    # Compare ids: decoding this tokenizer inserts a space after </s>.
    expected = "</s><|user|>What is your final answer? Reply with (A) or (B) only.</s><|assistant|>"
    assert scorer.cue_ids(ITEM) == list(scorer.tok.encode(expected, add_special_tokens=False))


def test_label_tokens_cover_the_letter(scorer):
    ids = set(np.array(scorer.label_ids("A")).tolist())
    assert scorer.tok.encode("A", add_special_tokens=False)[-1] in ids
    assert not ids & set(np.array(scorer.label_ids("B")).tolist())


def test_cached_curve_matches_full_recomputation(scorer):
    """The KV-cache rewind is only a speedup; it must not change any value."""
    prompt = scorer.prompt_ids(ITEM, cot=True)
    response = [450, 1234, 29871, 13, 3532, 4874, 263, 1243]
    p, mass = scorer.curve(ITEM, prompt, response)
    for t in range(len(response) + 1):
        d = scorer.answer_at(ITEM, prompt + response[:t] + scorer.cue_ids(ITEM))
        assert p[t] == pytest.approx(d.p_match, abs=1e-4)
        assert mass[t] == pytest.approx(d.mass, rel=1e-3)


def test_direct_averages_both_label_orders(scorer):
    d = scorer.direct(ITEM)
    a = scorer.answer_at(ITEM, scorer.prompt_ids(ITEM, cot=False))
    s = ITEM.swapped()
    b = scorer.answer_at(s, scorer.prompt_ids(s, cot=False))
    assert d.p_match == pytest.approx((a.p_match + b.p_match) / 2)


def test_estimate_is_a_probability_with_an_error_bar(scorer):
    mx.random.seed(0)
    prompt = scorer.prompt_ids(ITEM, cot=True)
    mean, se, mass = scorer.estimate(ITEM, prompt, [], num_samples=4, max_tokens=4)
    assert 0.0 <= mean <= 1.0 and se >= 0.0 and 0.0 <= mass <= 1.0


def test_batched_sampling_returns_one_continuation_per_prompt(scorer):
    mx.random.seed(0)
    prompt = scorer.prompt_ids(ITEM, cot=True)
    outs = scorer.sample([prompt, prompt[:-2], prompt], max_tokens=5)
    assert len(outs) == 3 and all(0 < len(o) <= 5 for o in outs)
