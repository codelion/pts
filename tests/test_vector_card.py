"""Model cards and publishing for `pts discover` vectors. No network: Hub calls are faked."""

import json
import os

import pytest

from pts.steering.card import MAIN, discovery_card, verdict


def _result(delta, lo, hi, p):
    return {
        "layer": 14, "fraction": 0.5, "alpha": 4.0,
        "selection": {"p_match": 0.7, "answered": 1.0, "kl": 0.1},
        "test": {"p_match": 0.5 + delta, "answered": 0.98, "kl": 0.12},
        "test_effect": {"delta": delta, "ci_low": lo, "ci_high": hi},
        "random_null": {"deltas": [0.0] * 5, "mean": 0.0, "p_value": p},
    }


def _report(decrease=True):
    return {
        "behavior": "sycophancy", "model": "Qwen/Qwen3-0.6B",
        "eval": "hf://Anthropic/model-written-evals/sycophancy/sycophancy_on_nlp_survey.jsonl",
        "split": {"discovery": 30, "selection": 15, "test": 15},
        "events": {"total": 40, "positive": 18, "states_answer": 9, "moved_traces": 120},
        "search_config": {"max_tokens": 1536, "prob_threshold": 0.2},
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
        "chat_template_args": {"enable_thinking": True},
        "eval_samples": 16,
        "base": {"test": {"p_match": 0.5, "answered": 0.99, "kl": 0.0}},
        "methods": {MAIN: {
            "increase": _result(0.2, 0.1, 0.3, 0.0),
            "decrease": _result(-0.15, -0.25, -0.05, 0.0) if decrease else None,
        }},
        "comparisons": {
            f"increase: {MAIN} vs caa": {"advantage": 0.1, "ci_low": 0.02, "ci_high": 0.2},
            f"increase: {MAIN} vs prompt": {"advantage": 0.12, "ci_low": 0.05, "ci_high": 0.2},
            # Another PTS variant, not a control: losing to it does not fail the verdict.
            f"increase: {MAIN} vs pts_signed_mean_centered": {"advantage": -0.02, "ci_low": -0.1, "ci_high": 0.06},
            f"decrease: {MAIN} vs caa": {"advantage": 0.01, "ci_low": -0.08, "ci_high": 0.1},
            f"decrease: {MAIN} vs prompt": {"advantage": 0.1, "ci_low": 0.03, "ci_high": 0.2},
        },
        "prompt_baselines": {"increase": {"prompt": "Agree with the user.",
                                          "test_effect": {"delta": 0.05, "ci_low": -0.02, "ci_high": 0.1}}},
    }


def test_verdict_passes_only_when_every_control_is_beaten():
    r = _report()
    up, down = verdict(r, "increase"), verdict(r, "decrease")
    assert up["passes"] and sorted(up["beats"]) == ["caa", "prompt"]
    # Moves the behavior, but its interval against CAA includes zero.
    assert down["moved"] and not down["passes"] and down["not_beaten"] == ["caa"]


def test_verdict_needs_the_random_null():
    r = _report()
    r["methods"][MAIN]["increase"]["random_null"]["p_value"] = 0.4
    assert not verdict(r, "increase")["passes"]


def test_verdict_needs_the_effect_interval_to_exclude_zero():
    r = _report()
    r["methods"][MAIN]["increase"]["test_effect"] = {"delta": 0.1, "ci_low": -0.01, "ci_high": 0.2}
    v = verdict(r, "increase")
    assert not v["moved"] and not v["passes"]


def test_card_states_the_verdict_and_how_to_reproduce():
    card = discovery_card(_report(), "someone/Qwen3-0.6B-sycophancy")
    assert card.startswith("---\nlibrary_name: pts\nbase_model: Qwen/Qwen3-0.6B\n")
    assert "**Beats every control**" in card
    assert "does not beat every control.** Not beaten: CAA (answer-token contrast)" in card
    assert "PTS signed mean (variant, not a control)" in card
    assert "0 of 5 random vectors of the same norm moved P as far" in card
    assert "--vector hf://someone/Qwen3-0.6B-sycophancy/increase" in card
    # The reproduce command carries the run's own settings.
    assert "--chat-template-args '{\"enable_thinking\": true}'" in card
    assert "--temperature 0.6" in card and "--top-k 20" in card and "--max-tokens 1536" in card
    assert "--presence-penalty" not in card and "--min-p" not in card


def test_card_says_when_a_direction_has_no_vector():
    card = discovery_card(_report(decrease=False), "someone/x")
    assert "so no vector is published for it" in card
    assert "hf://someone/x/decrease" not in card


def _discover_dir(tmp_path, report):
    out = tmp_path / "run"
    for d in ("increase", "decrease"):
        os.makedirs(out / f"{MAIN}_{d}")
        (out / f"{MAIN}_{d}" / "steering_config.json").write_text("{}")
    os.makedirs(out / "caa_increase")
    (out / "report.json").write_text(json.dumps(report))
    return out


def _fake_hub(monkeypatch):
    import huggingface_hub

    uploaded = {}

    def upload_folder(folder_path, repo_id, **kw):
        uploaded["files"] = sorted(
            os.path.relpath(os.path.join(root, f), folder_path)
            for root, _, files in os.walk(folder_path) for f in files
        )
        uploaded["readme"] = open(os.path.join(folder_path, "README.md")).read()
        uploaded["repo_id"] = repo_id

    monkeypatch.setattr(huggingface_hub, "create_repo", lambda *a, **k: None)
    monkeypatch.setattr(huggingface_hub, "upload_folder", upload_folder)
    return uploaded


def test_push_publishes_the_chosen_vectors_report_and_card(tmp_path, monkeypatch):
    from pts.cli import _push

    out = _discover_dir(tmp_path, _report())
    uploaded = _fake_hub(monkeypatch)
    _push(str(out), "someone/q", private=False, model_name=None)
    assert uploaded["files"] == ["README.md", "decrease/steering_config.json",
                                 "increase/steering_config.json", "report.json"]
    assert "Beats every control" in uploaded["readme"]


def test_push_skips_a_direction_that_failed_selection(tmp_path, monkeypatch):
    from pts.cli import _push

    out = _discover_dir(tmp_path, _report(decrease=False))
    uploaded = _fake_hub(monkeypatch)
    _push(str(out), "someone/q", private=False, model_name=None)
    assert "decrease/steering_config.json" not in uploaded["files"]


def test_push_refuses_a_directory_without_a_report(tmp_path):
    from pts.cli import _push

    with pytest.raises(SystemExit, match="no report.json"):
        _push(str(tmp_path), "someone/q", private=False, model_name=None)
