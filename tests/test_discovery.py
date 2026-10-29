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


def test_returned_caches_cover_exactly_prompt_plus_continuation(scorer):
    """The estimate reads each answer by running only the cue on a sample's
    cache. That must equal a full pass over prompt + continuation + cue, for
    samples that stopped on EOS (whose stop token is trimmed) and that hit the
    length limit."""
    mx.random.seed(3)
    prompt = scorer.prompt_ids(ITEM, cot=True)
    conts, caches = scorer.sample([prompt] * 4, max_tokens=6, return_caches=True)
    cue = scorer.cue_ids(ITEM)
    for cont, cache in zip(conts, caches):
        assert cache[0].offset == len(prompt) + len(cont)
        cached = scorer._dist(scorer.model(mx.array(cue)[None], cache=cache)[0, -1], ITEM)
        full = scorer.answer_at(ITEM, prompt + cont + cue)
        assert cached.p_match == pytest.approx(full.p_match, abs=1e-4)


# -- directions, validation, pipeline ----------------------------------------

def _capture(scorer):
    from pts.steering import mlx_backend

    return lambda ids, ls: mlx_backend.capture(scorer.model, ids, ls)


def _event(scorer, delta, item_id="q0", trace_index=0, position=2, token_id=3532):
    from pts.events import make_token_event

    prompt = scorer.prompt_ids(ITEM, cot=True)
    return make_token_event(
        query=ITEM.question, context=scorer.tok.decode(prompt + TRACE[:position]), token=" x",
        token_id=token_id, prob_before=0.5, prob_after=0.5 + delta, model_id="tiny",
        dataset_item_id=item_id,
        metadata={"trace_index": trace_index, "response_position": position},
    )


TRACE = [450, 1234, 3532, 29871, 13, 4874, 263, 1243, 2304, 1576]


def test_matched_pair_direction_follows_the_events_sign(scorer):
    from pts.discovery.directions import matched_pairs

    up, _, n = matched_pairs(scorer, _capture(scorer), [_event(scorer, +0.3)], [0, 1])
    down, _, _ = matched_pairs(scorer, _capture(scorer), [_event(scorer, -0.3)], [0, 1])
    assert n == 1
    for i in (0, 1):
        assert np.abs(up[i]).max() > 0
        np.testing.assert_allclose(down[i], -up[i], atol=1e-6)


def test_signed_mean_needs_both_signs(scorer):
    from pts.discovery.directions import signed_mean

    with pytest.raises(ValueError, match="raise and events that lower"):
        signed_mean(scorer, _capture(scorer), [_event(scorer, 0.3)], [0])


def test_signed_mean_is_centered_per_question(scorer):
    """One raising event in question A and one lowering event in question B:
    without centering, the vector is just A's prompt minus B's prompt."""
    from pts.discovery.directions import signed_mean

    a, b = _event(scorer, 0.3, item_id="A", position=2), _event(scorer, -0.3, item_id="B", position=5)
    with pytest.raises(ValueError, match="nothing left after centering"):
        signed_mean(scorer, _capture(scorer), [a, b], [0])
    events = [a, _event(scorer, -0.3, item_id="A", position=5), b, _event(scorer, 0.3, item_id="B", position=2)]
    vecs, _, n = signed_mean(scorer, _capture(scorer), events, [0])
    assert n == 4 and np.abs(vecs[0]).max() > 0


def test_caa_uses_both_label_orders(scorer):
    from pts.discovery.directions import caa

    vecs, norms, n = caa(scorer, _capture(scorer), [ITEM], [0, 1])
    assert n == 2 and set(vecs) == {0, 1} and norms[0] > 0


def test_choose_rejects_damaging_settings():
    from pts.discovery.validate import Score, choose

    base = Score(p_match=0.6, mass=0.95)
    rows = [
        {"layer": 1, "alpha": -9, "p_match": 0.50, "mass": 0.40, "kl": 0.1},  # off-format: rejected
        {"layer": 1, "alpha": -5, "p_match": 0.45, "mass": 0.94, "kl": 2.0},  # breaks neutral text
        {"layer": 2, "alpha": -2, "p_match": 0.52, "mass": 0.93, "kl": 0.1},
        {"layer": 2, "alpha": 2, "p_match": 0.70, "mass": 0.95, "kl": 0.1},
    ]
    assert choose(rows, base, -1)["alpha"] == -2
    assert choose(rows, base, +1)["alpha"] == 2
    assert choose(rows[:2], base, -1) is None


def test_pipeline_runs_end_to_end_and_writes_loadable_vectors(scorer, tmp_path, monkeypatch):
    import json as _json

    from pts.discovery import pipeline
    from pts.discovery.search import SearchConfig
    from pts.steering.vector import SteeringVector

    rows = [{**AB_ROW, "question": f"Q{i}? (A) yes (B) no"} for i in range(8)]
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in rows))
    config = pipeline.DiscoveryConfig(
        num_items=8, layers=[0, 1], fractions=[0.1], max_kl=10.0, min_mass_ratio=0.0,
        random_null=2, cot_samples=2,
        search=SearchConfig(traces_per_item=1, max_tokens=4, samples=2,
                            min_direct=0.0, max_direct=1.0),
    )
    report = pipeline.discover(scorer.model, scorer.tok, "tiny", str(path), "toy", str(tmp_path / "out"), config)

    saved = _json.loads((tmp_path / "out" / "report.json").read_text())
    assert saved["split"] == {"discovery": 4, "selection": 2, "test": 2}
    assert "caa" in report["methods"]
    assert len(report["methods"]["caa"]["selection_sweep"]) == 2 * 1 * 2  # layers x fractions x signs
    assert set(saved["prompt_baselines"]) == {"increase", "decrease"}
    assert "test_cot_effect" in saved["prompt_baselines"]["decrease"]
    assert (tmp_path / "out" / "traces.jsonl").exists()
    for label in ("increase", "decrease"):
        r = report["methods"]["caa"][label]
        if r is not None:
            assert len(r["random_null"]["deltas"]) == 2
            assert r["test_effect"]["ci_low"] <= r["test_effect"]["delta"] <= r["test_effect"]["ci_high"]
            vec = SteeringVector.load(str(tmp_path / "out" / f"caa_{label}"))
            assert vec.default_alpha == pytest.approx(r["alpha"])
            assert vec.validation["direction"] == label


class _StepScorer:
    """A stand-in scorer where P(behavior | prefix) steps at known tokens.

    The single trace is tokens 100..131; P is 0.5 until token 105 is emitted,
    0.9 after it, and 0.1 after token 120. Bisection must find exactly those
    two tokens, with no sampling noise to blur them.
    """

    def __init__(self):
        from pts.discovery.scoring import AnswerDist, Cost

        self.cost = Cost()
        self.trace = list(range(100, 132))
        self.calls = 0
        self._AnswerDist = AnswerDist

    def p(self, prefix):
        p = 0.5
        if 105 in prefix:
            p = 0.9
        if 120 in prefix:
            p = 0.1
        return p

    def direct(self, item):
        return self._AnswerDist(0.5, 1.0)

    def prompt_ids(self, item, cot):
        return [1, 2, 3]

    def estimate(self, item, prompt, prefix, n, max_tokens, temperature=0.7):
        self.calls += 1
        return self.p(list(prefix)), 0.0, 1.0

    def sample(self, prompts, max_tokens, temperature=0.7, return_caches=False):
        return [list(self.trace)], ["cache"]

    def cue_ids(self, item):
        return [7]

    def label_ids(self, label):
        return mx.array([120]) if label == "B" else mx.array([999])

    def model(self, cue, cache=None):
        return mx.zeros((1, 1, 4))

    def _dist(self, logits, item):
        return self._AnswerDist(self.p(self.trace), 1.0)

    class tok:
        @staticmethod
        def decode(ids):
            return " ".join(map(str, ids))


def test_bisection_finds_exactly_the_tokens_where_p_steps():
    from pts.discovery.search import SearchConfig, search_item

    fake = _StepScorer()
    cfg = SearchConfig(traces_per_item=1, prob_threshold=0.2, max_estimates_per_trace=50)
    events = list(search_item(fake, ITEM, cfg, "fake", "toy"))
    assert sorted(e.token_id for e in events) == [105, 120]
    by_token = {e.token_id: e for e in events}
    assert by_token[105].prob_delta == pytest.approx(0.4)
    assert by_token[120].prob_delta == pytest.approx(-0.8)
    assert all(e.metadata["estimator"] == "sampled_answer_distribution" for e in events)
    # Token 120 stands in for writing the answer label; it is tagged, not hidden.
    assert by_token[120].metadata["states_answer"] and not by_token[105].metadata["states_answer"]
    # Logarithmic, not linear, in the trace length: far fewer than 32 estimates.
    assert fake.calls < 20


def test_traces_whose_answer_did_not_move_cost_no_sampling():
    from pts.discovery.search import SearchConfig, search_item

    fake = _StepScorer()
    fake.trace = list(range(100, 104))  # never reaches the step tokens: ends at 0.5
    events = list(search_item(fake, ITEM, SearchConfig(traces_per_item=1), "fake", "toy"))
    assert events == [] and fake.calls == 1  # only P(start)


def test_answer_stating_events_are_not_used_for_directions(scorer):
    from pts.discovery.directions import matched_pairs

    e = _event(scorer, 0.3)
    e.metadata["states_answer"] = True
    with pytest.raises(ValueError, match="No token events"):
        matched_pairs(scorer, _capture(scorer), [e], [0])


def test_sign_shuffle_keeps_tokens_and_sign_counts(scorer, monkeypatch):
    from pts.discovery import directions

    seen = []
    monkeypatch.setattr(directions, "_pair_vectors", lambda sc, cap, specs, layers: seen.append(specs) or ({}, {}, len(specs)))
    events = [_event(scorer, d, position=k) for k, d in enumerate([0.3, 0.3, 0.3, -0.3, -0.3], start=1)]
    directions.matched_pairs(scorer, None, events, [0])
    directions.sign_shuffled_pairs(scorer, None, events, [0], seed=1)
    real, shuffled = seen
    assert [s.context_ids for s in real] == [s.context_ids for s in shuffled]
    assert sorted(s.sign for s in real) == sorted(s.sign for s in shuffled)
    assert [s.sign for s in real] != [s.sign for s in shuffled]


def test_random_positions_avoid_pivotal_and_answer_tokens(scorer, monkeypatch):
    from pts.discovery import directions

    seen = []
    monkeypatch.setattr(directions, "_pair_vectors", lambda sc, cap, specs, layers: seen.append(specs) or ({}, {}, len(specs)))
    prompt = scorer.prompt_ids(ITEM, cot=True)
    answer_token = int(np.array(scorer.label_ids("A"))[0])
    trace = TRACE[:6] + [answer_token] + TRACE[6:]
    traces = [{"item_id": "q0", "trace_index": 0, "prompt_ids": prompt, "response_ids": trace,
               "p_start": 0.5, "p_end": 0.1}]
    events = [_event(scorer, 0.3, position=2)]
    for seed in range(20):
        directions.random_position_pairs(scorer, None, events, traces, [0], seed=seed)
    for (spec,) in seen:
        t = len(spec.context_ids) - len(prompt)
        assert t != 2 and trace[t] != answer_token
        assert spec.context_ids == prompt + trace[:t] and spec.token_id == trace[t]
        assert spec.sign == -1.0  # the trace went down, whatever the event's own sign


def test_bootstrap_interval_covers_the_mean_and_shrinks_with_n():
    from pts.discovery.validate import bootstrap_ci

    rng = np.random.default_rng(0)
    small, large = rng.normal(0.1, 0.2, 10), rng.normal(0.1, 0.2, 400)
    lo_s, hi_s = bootstrap_ci(small)
    lo_l, hi_l = bootstrap_ci(large)
    assert lo_s <= small.mean() <= hi_s and lo_l <= large.mean() <= hi_l
    assert hi_l - lo_l < hi_s - lo_s


def test_prompt_baseline_changes_the_prompt_only_while_scoring(scorer):
    from pts.discovery.validate import Evaluator

    ev = Evaluator(scorer)
    plain = scorer.prompt_ids(ITEM, cot=False)
    s = ev.score([ITEM], prompt="Avoid it.")
    assert scorer.extra_instruction is None
    assert scorer.prompt_ids(ITEM, cot=False) == plain
    assert len(s.per_item) == 1


def test_pipeline_reuses_a_saved_search_and_compares_pts_with_every_alternative(scorer, tmp_path):
    import json as _json

    from pts.discovery import pipeline
    from pts.event_storage import EventStorage

    rows = [{**AB_ROW, "question": f"Q{i}? (A) yes (B) no"} for i in range(8)]
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in rows))
    out = tmp_path / "out"
    out.mkdir()
    prompt = scorer.prompt_ids(ITEM, cot=True)
    # Distinct contexts per question: event ids hash the context, as real
    # questions' different text would.
    events = [_event(scorer, d, item_id=q, position=k + off)
              for q, off in (("q0", 0), ("q1", 1)) for k, d in ((2, 0.3), (5, -0.3))]
    store = EventStorage(filepath=str(out / "events.jsonl"))
    store.add_events(events)
    store.save()
    with open(out / "traces.jsonl", "w") as f:
        for q in ("q0", "q1"):
            f.write(_json.dumps({"item_id": q, "trace_index": 0, "prompt_ids": prompt,
                                 "response_ids": TRACE, "p_start": 0.5, "p_end": 0.9}) + "\n")

    config = pipeline.DiscoveryConfig(
        num_items=8, layers=[0, 1], fractions=[0.1, 0.4], max_kl=10.0, min_mass_ratio=0.0,
        random_null=2, cot_samples=0, reuse_search=True,
    )
    report = pipeline.discover(scorer.model, scorer.tok, "tiny", str(path), "toy", str(out), config)
    assert report["events"]["total"] == 4 and report["search_cost"] is None
    assert set(report["methods"]) == {
        "pts_matched_pairs", "pts_signed_mean_centered", "control_sign_shuffle",
        "control_random_positions", "caa",
    }
    keys = report["comparisons"].keys()
    for label in ("increase", "decrease"):
        if report["methods"]["pts_matched_pairs"][label] is not None:
            others = {k.split(" vs ")[1] for k in keys if k.startswith(f"{label}: ")}
            assert {"control_sign_shuffle", "control_random_positions", "caa", "prompt"} <= others
