"""Behavior discovery on MLX.

Runs on the tiny random Llama with a minimal chat template attached; its
weights are random, so tests check structure and invariants, not answers.
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
    from pts.discovery.evals import ChoiceOracle
    from pts.discovery.scoring import ChoiceScorer

    model, tok = mlx_lm.load(TINY)
    tok._tokenizer.chat_template = TEMPLATE
    return ChoiceScorer(model, tok, ChoiceOracle([ITEM]))


ITEM = parse_row(AB_ROW, 0)


def test_prompt_is_the_dataset_question_in_the_chat_template(scorer):
    text = scorer.tok.decode(scorer.prompt_ids(ITEM))
    assert text.count(ITEM.question.strip()) == 1
    assert text.rstrip().endswith("<|assistant|>")


def test_estimate_is_a_probability_with_an_error_bar(scorer):
    mx.random.seed(0)
    e = scorer.estimate(ITEM, scorer.prompt_ids(ITEM), [], num_samples=4, max_tokens=4)
    assert 0.0 <= e.p <= 1.0 and e.se >= 0.0 and 0.0 <= e.answered <= 1.0


def test_oracle_judges_prefix_plus_completion(scorer):
    """Success is judged on the whole response: the prefix being conditioned on
    plus the sampled rest, like token PTS."""
    from pts.discovery.scoring import ChoiceScorer

    seen = []
    prefix = list(scorer.tok.encode(" yes", add_special_tokens=False))
    start = scorer.tok.decode(prefix)

    class Recorder:
        def check_success(self, query, response):
            seen.append((query, response))
            return response.startswith(start)

    sc = ChoiceScorer(scorer.model, scorer.tok, Recorder())
    mx.random.seed(0)
    (e, other) = sc.estimate_many([(ITEM, sc.prompt_ids(ITEM), prefix), (ITEM, sc.prompt_ids(ITEM), [])], 3, 4)
    assert len(seen) == 6 and all(q == ITEM.question for q, _ in seen)
    assert all(r.startswith(start) for _, r in seen[:3])
    assert e.p == 1.0 and e.answered == 1.0
    assert sc.cost.completions == 6


def test_batched_sampling_returns_one_continuation_per_prompt(scorer):
    mx.random.seed(0)
    prompt = scorer.prompt_ids(ITEM)
    outs = scorer.sample([prompt, prompt[:-2], prompt], max_tokens=5)
    assert len(outs) == 3 and all(0 < len(o) <= 5 for o in outs)


# -- directions, validation, pipeline ----------------------------------------

def _capture(scorer):
    from pts.steering import mlx_backend

    return lambda ids, ls: mlx_backend.capture(scorer.model, ids, ls)


def _event(scorer, delta, item_id="q0", trace_index=0, position=2, token_id=3532):
    from pts.events import make_token_event

    prompt = scorer.prompt_ids(ITEM)
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

    base = Score(p_match=0.6, answered=0.95)
    rows = [
        {"layer": 1, "alpha": -9, "p_match": 0.50, "answered": 0.40, "kl": 0.1},  # stops answering
        {"layer": 1, "alpha": -5, "p_match": 0.45, "answered": 0.94, "kl": 2.0},  # breaks neutral text
        {"layer": 2, "alpha": -2, "p_match": 0.52, "answered": 0.93, "kl": 0.1},
        {"layer": 2, "alpha": 2, "p_match": 0.70, "answered": 0.95, "kl": 0.1},
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
        num_items=8, layers=[0, 1], fractions=[0.1], max_kl=10.0, min_answered_ratio=0.0,
        random_null=2, eval_samples=2, selection_samples=2,
        search=SearchConfig(traces_per_item=1, max_tokens=4, samples=2,
                            min_start=0.0, max_start=1.0),
    )
    report = pipeline.discover(scorer.model, scorer.tok, "tiny", str(path), "toy", str(tmp_path / "out"), config)

    saved = _json.loads((tmp_path / "out" / "report.json").read_text())
    assert saved["split"] == {"discovery": 4, "selection": 2, "test": 2}
    assert "caa" in report["methods"]
    assert len(report["methods"]["caa"]["selection_sweep"]) == 2 * 1 * 2  # layers x fractions x signs
    assert set(saved["prompt_baselines"]) == {"increase", "decrease"}
    assert "answered" in saved["base"]["test"]
    assert (tmp_path / "out" / "traces.jsonl").exists()
    for label in ("increase", "decrease"):
        r = report["methods"]["caa"][label]
        if r is not None:
            assert len(r["random_null"]["deltas"]) == 2
            assert r["test_effect"]["ci_low"] <= r["test_effect"]["delta"] <= r["test_effect"]["ci_high"]
            vec = SteeringVector.load(str(tmp_path / "out" / f"caa_{label}"))
            assert vec.default_alpha == pytest.approx(r["alpha"])
            assert vec.validation["direction"] == label



def test_pipeline_validates_chosen_methods_on_a_screened_held_out_pool(scorer, tmp_path):
    """--methods caa --held-out-pool: discovery questions stay the ones a
    pool-less run searches; selection and test come from screened questions."""
    import json as _json

    from pts.discovery import pipeline
    from pts.discovery.evals import load_choice_eval, split
    from pts.discovery.search import SearchConfig

    rows = [{**AB_ROW, "question": f"Q{i}? (A) yes (B) no"} for i in range(12)]
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in rows))
    config = pipeline.DiscoveryConfig(
        num_items=8, held_out_pool=4, methods=["caa"], layers=[0], fractions=[0.1], max_kl=10.0,
        min_answered_ratio=0.0, random_null=1, eval_samples=2, selection_samples=2, screen_samples=2,
        search=SearchConfig(traces_per_item=1, max_tokens=4, samples=2, min_start=0.0, max_start=1.0),
    )
    report = pipeline.discover(scorer.model, scorer.tok, "tiny", str(path), "toy", str(tmp_path / "out"), config)

    # The tiny model finds no pivotal tokens, so only CAA has a vector here.
    assert "caa" in report["methods"] and set(report["methods"]) <= {pipeline.MAIN, "caa"}
    assert report["held_out_screen"] == {"pool": 8, "kept": 8, "band": [0.0, 1.0], "samples": 2}
    assert report["split"] == {"discovery": 4, "selection": 4, "test": 4}
    searched = _json.loads((tmp_path / "out" / "search_state.json").read_text())["settings"]["items"]
    assert searched == [it.item_id for it in split(load_choice_eval(str(path), limit=8))[0]]


def test_unknown_methods_are_refused(scorer, tmp_path):
    import json as _json

    from pts.discovery import pipeline
    from pts.discovery.search import SearchConfig

    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(_json.dumps({**AB_ROW, "question": f"Q{i}? (A) yes (B) no"}) for i in range(8)))
    config = pipeline.DiscoveryConfig(num_items=8, methods=["cca"], layers=[0],
                                      search=SearchConfig(traces_per_item=1, max_tokens=4, samples=2))
    with pytest.raises(ValueError, match="unknown methods"):
        pipeline.discover(scorer.model, scorer.tok, "tiny", str(path), "toy", str(tmp_path / "out"), config)


def test_pipeline_runs_on_a_topic_eval_without_caa(scorer, tmp_path):
    """Open-ended prompts judged by a pattern: the same pipeline, minus CAA,
    which needs labelled answers."""
    import json as _json

    from pts.discovery import pipeline
    from pts.discovery.search import SearchConfig

    rows = [{"question": f"Tell me about place {i}.", "pattern": r"\w"} for i in range(8)]
    path = tmp_path / "topic.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in rows))
    config = pipeline.DiscoveryConfig(
        num_items=8, layers=[0, 1], fractions=[0.1], max_kl=10.0, min_answered_ratio=0.0,
        random_null=1, eval_samples=2, selection_samples=2,
        search=SearchConfig(traces_per_item=1, max_tokens=4, samples=2, min_start=0.0, max_start=1.0),
    )
    report = pipeline.discover(scorer.model, scorer.tok, "tiny", str(path), "topic", str(tmp_path / "out"), config)
    assert report["eval_kind"] == "topic"
    assert "caa" not in report["methods"]
    assert not any(" vs caa" in k for k in report["comparisons"])


class _StepScorer:
    """A stand-in scorer where P(behavior | prefix) steps at known tokens.

    The single trace is tokens 100..131; P is 0.5 until token 105 is emitted,
    0.9 after it, and 0.1 after token 120. Bisection must find exactly those
    two tokens, with no sampling noise to blur them.
    """

    def __init__(self):
        from pts.discovery.scoring import Cost, Estimate

        self.cost = Cost()
        self.trace = list(range(100, 132))
        self.seen = set()       # distinct prefixes estimated
        self._Estimate = Estimate
        self.temperature, self.top_p, self.top_k, self.min_p, self.presence_penalty = 0.7, 0.95, 0, 0.0, 0.0
        self.chat_template_args = {}

    def p(self, prefix):
        p = 0.5
        if 105 in prefix:
            p = 0.9
        if 120 in prefix:
            p = 0.1
        return p

    def prompt_ids(self, item):
        return [1, 2, 3]

    def estimate_many(self, requests, counts, max_tokens):
        counts = [counts] * len(requests) if isinstance(counts, int) else counts
        self.batches = getattr(self, "batches", []) + [len(requests)]
        out = []
        for (_, _, prefix), n in zip(requests, counts):
            self.seen.add(tuple(prefix))
            k = round(self.p(list(prefix)) * n)
            out.append(self._Estimate(k, n, n))
        return out

    def sample(self, prompts, max_tokens):
        return [list(self.trace)]

    def judge(self, item, response):
        return self.p(list(response)) > 0.5, True

    class tok:
        @staticmethod
        def decode(ids):
            # Token 120 stands in for writing the answer label.
            return "(B" if list(ids) == [120] else " ".join(map(str, ids))


def test_bisection_finds_exactly_the_tokens_where_p_steps():
    from pts.discovery.search import SearchConfig, search_item

    fake = _StepScorer()
    cfg = SearchConfig(traces_per_item=1, prob_threshold=0.2, max_estimates_per_trace=50)
    events = list(search_item(fake, ITEM, cfg, "fake", "toy"))
    assert sorted(e.token_id for e in events) == [105, 120]
    by_token = {e.token_id: e for e in events}
    assert by_token[105].prob_delta == pytest.approx(0.4)
    assert by_token[120].prob_delta == pytest.approx(-0.8)
    assert all(e.search_method == "token_pts_bisection" for e in events)
    # Token 120 stands in for writing the answer label; it is tagged, not hidden.
    assert by_token[120].metadata["states_answer"] and not by_token[105].metadata["states_answer"]
    # Logarithmic, not linear, in the trace length: far fewer than 32 estimates.
    assert len(fake.seen) < 20


def test_shared_prefill_generates_exactly_what_a_fresh_prompt_does(scorer):
    """Samples of one prompt start from copies of one prefilled cache. Greedy
    decoding makes that comparable token for token with a prompt run alone."""
    from pts.discovery.scoring import ChoiceScorer

    sc = ChoiceScorer(scorer.model, scorer.tok, scorer.oracle, temperature=0.0)
    prompt = sc.prompt_ids(ITEM)
    alone = sc.sample([prompt], max_tokens=8)
    shared = sc.sample([prompt] * 3 + [prompt[:-3]], max_tokens=8)
    assert shared[:3] == alone * 3
    assert shared[3] == sc.sample([prompt[:-3]], max_tokens=8)[0]
    assert sc.cost.prefilled_once == 1


def test_more_prompts_than_the_batch_refill_freed_slots(scorer):
    """Prompts beyond ``batch_size`` wait for a slot; each still gets exactly
    what it gets run alone, in its own position."""
    from pts.discovery.scoring import ChoiceScorer

    sc = ChoiceScorer(scorer.model, scorer.tok, scorer.oracle, temperature=0.0, batch_size=2)
    prompt = sc.prompt_ids(ITEM)
    prompts = [prompt, prompt[:-1], prompt, prompt[:-2], prompt]
    together = sc.sample(prompts, max_tokens=6)
    assert together == [sc.sample([p], max_tokens=6)[0] for p in prompts]


def test_a_failed_batch_resamples_only_the_unfinished_prompts(scorer, monkeypatch):
    """mlx-lm's BatchGenerator has failed a batch split with a shape error.
    Finished completions are kept; the rest are sampled again from scratch."""
    import importlib

    from pts.discovery.scoring import ChoiceScorer

    g = importlib.import_module("mlx_lm.generate")   # the package attribute is the function
    sc = ChoiceScorer(scorer.model, scorer.tok, scorer.oracle, temperature=0.0, batch_size=2)
    prompt = sc.prompt_ids(ITEM)
    prompts = [prompt, prompt[:-1], prompt, prompt[:-2], prompt]
    alone = [sc.sample([p], max_tokens=6)[0] for p in prompts]

    real = g.BatchGenerator.next_generated
    calls = {"n": 0, "generators": 0}
    real_init = g.BatchGenerator.__init__

    def init(self, *a, **k):
        calls["generators"] += 1
        real_init(self, *a, **k)

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 12:
            raise ValueError("[broadcast_shapes] Shapes (1,1,1,408) and (1,16,1,191) cannot be broadcast.")
        return real(self)

    monkeypatch.setattr(g.BatchGenerator, "__init__", init)
    monkeypatch.setattr(g.BatchGenerator, "next_generated", flaky)
    assert sc.sample(prompts, max_tokens=6) == alone
    assert calls["generators"] == 2


def test_traces_are_bisected_in_lockstep():
    """Several traces cost no more batches than one: each round of every
    pending comparison goes out together. The events are the same per trace."""
    from pts.discovery.search import SearchConfig, search_item

    cfg = SearchConfig(min_start=0.0, max_start=1.0, samples=10, min_samples=10)
    runs = {}
    for n in (1, 4):
        fake = _StepScorer()
        fake.sample = lambda prompts, max_tokens, fake=fake: [list(fake.trace) for _ in prompts]
        events = list(search_item(fake, ITEM, SearchConfig(**{**cfg.__dict__, "traces_per_item": n}),
                                  "fake", "toy"))
        runs[n] = (len(fake.batches), sorted((e.metadata["trace_index"], e.token_id) for e in events))
    assert runs[4][0] == runs[1][0]
    assert runs[4][1] == [(i, t) for i in range(4) for _, t in runs[1][1]]
    assert [t for _, t in runs[1][1]] == [105, 120]


def test_clear_comparisons_stop_at_min_samples():
    from pts.discovery.search import SearchConfig, search_item

    fake = _StepScorer()
    fake.p = lambda prefix: 1.0 if 105 in prefix else 0.0
    fake.trace = list(range(100, 108))
    cfg = SearchConfig(traces_per_item=1, min_start=0.0, max_start=1.0, samples=50, min_samples=10)
    (event,) = search_item(fake, ITEM, cfg, "fake", "toy")
    assert event.token_id == 105 and event.prob_delta == pytest.approx(1.0)
    # A clear difference is decided in the first round; "no difference"
    # between two 0/n estimates takes one more. Neither needs all 50.
    assert event.metadata["samples_after"] == 10
    assert event.metadata["samples_before"] <= 20


def test_min_samples_equal_to_samples_is_fixed_size_token_pts():
    from pts.discovery.search import SearchConfig, search_item

    cfg = SearchConfig(traces_per_item=1, samples=50, min_samples=50)
    events = list(search_item(_StepScorer(), ITEM, cfg, "fake", "toy"))
    assert sorted(e.token_id for e in events) == [105, 120]
    assert all(e.metadata["samples_before"] == e.metadata["samples_after"] == 50 for e in events)


def test_questions_with_a_settled_start_are_skipped_before_sampling_traces():
    from pts.discovery.search import SearchConfig, search_item

    fake = _StepScorer()
    fake.p = lambda prefix: 0.95
    events = list(search_item(fake, ITEM, SearchConfig(traces_per_item=1), "fake", "toy"))
    assert events == [] and fake.seen == {()}


def test_zero_budget_means_unlimited_bisection():
    from pts.discovery.search import SearchConfig, search_item

    cfg = SearchConfig(traces_per_item=1, max_estimates_per_trace=0)
    events = list(search_item(_StepScorer(), ITEM, cfg, "fake", "toy"))
    assert sorted(e.token_id for e in events) == [105, 120]


def test_search_saves_per_question_and_resumes(tmp_path, monkeypatch):
    import json as _json

    from pts.discovery import pipeline
    from pts.discovery.search import SearchConfig, search_item

    items = [parse_row({**AB_ROW, "question": f"Q{i}? (A) yes (B) no"}, i) for i in range(3)]
    config = pipeline.DiscoveryConfig(search=SearchConfig(traces_per_item=1))
    out = str(tmp_path)
    searched = []

    def flaky(scorer, item, *args):
        if len(searched) == 2:
            raise RuntimeError("interrupted")
        searched.append(item.item_id)
        return search_item(scorer, item, *args)

    monkeypatch.setattr(pipeline, "search_item", flaky)
    with pytest.raises(RuntimeError):
        pipeline._run_search(_StepScorer(), items, config, "fake", "e.jsonl", "toy", out)
    state = _json.loads((tmp_path / "search_state.json").read_text())
    assert state["done"] == searched == [items[0].item_id, items[1].item_id]

    searched.clear()
    monkeypatch.setattr(pipeline, "search_item", lambda s, item, *a: (searched.append(item.item_id), search_item(s, item, *a))[1])
    events, traces, _ = pipeline._run_search(_StepScorer(), items, config, "fake", "e.jsonl", "toy", out)
    assert searched == [items[2].item_id]
    assert len(events) == 3 * 2 and len(traces) == 3

    changed = pipeline.DiscoveryConfig(search=SearchConfig(traces_per_item=1, samples=8))
    with pytest.raises(ValueError, match="different settings"):
        pipeline._run_search(_StepScorer(), items, changed, "fake", "e.jsonl", "toy", out)


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
    prompt = scorer.prompt_ids(ITEM)
    answer_token = scorer.tok.encode("A", add_special_tokens=False)[-1]
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

    ev = Evaluator(scorer, samples=2, max_tokens=4)
    plain = scorer.prompt_ids(ITEM)
    s = ev.score([ITEM], prompt="Avoid it.")
    assert scorer.extra_instruction is None
    assert scorer.prompt_ids(ITEM) == plain
    assert len(s.per_item) == 1


def test_pipeline_reuses_a_saved_search_and_compares_pts_with_every_alternative(scorer, tmp_path):
    import json as _json

    from pts.discovery import pipeline
    from pts.discovery.search import SearchConfig
    from pts.event_storage import EventStorage

    rows = [{**AB_ROW, "question": f"Q{i}? (A) yes (B) no"} for i in range(8)]
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in rows))
    out = tmp_path / "out"
    out.mkdir()
    prompt = scorer.prompt_ids(ITEM)
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
        num_items=8, layers=[0, 1], fractions=[0.1, 0.4], max_kl=10.0, min_answered_ratio=0.0,
        random_null=2, eval_samples=2, selection_samples=2, reuse_search=True,
        search=SearchConfig(max_tokens=4),
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
