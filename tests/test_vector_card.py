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
    front = card.split("---\n")[1]
    assert card.startswith("---\nlibrary_name: pts\n")
    assert "base_model: Qwen/Qwen3-0.6B\n" in front and "license: apache-2.0\n" in front
    assert "**Beats every control**" in card
    assert "does not beat every control.** Not beaten: CAA (answer-token contrast)" in card
    assert "PTS signed mean (variant, not a control)" in card
    assert "0 of 5 random vectors of the same norm moved P as far" in card
    assert "--vector hf://someone/Qwen3-0.6B-sycophancy/increase" in card
    assert "https://huggingface.co/spaces/codelion/pts-visualizer?vector=someone/Qwen3-0.6B-sycophancy" in card
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


TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


def _python_api_snippet(tmp_path, backend):
    """The card's Python API example, as published, pointed at the tiny model.

    Only what has to change to run offline is replaced: the model id, the
    vector's Hub id (a local vector of the right shape instead), the token
    budget, and a chat template the tiny tokenizer lacks.
    """
    import numpy as np

    from pts.steering.vector import LayerVector, SteeringVector

    card = discovery_card(_report(), "someone/q")
    code = card.split("## Python API", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    head, rest = code.split("# MLX\n", 1)
    mlx_part, torch_part = rest.split("# PyTorch\n", 1)
    path = str(tmp_path / "vec.safetensors")
    SteeringVector([LayerVector(0, "add", np.ones(16) * 3)], 16, 2, base_model=TINY).save(path)
    head = head.replace('"hf://someone/q/increase"', repr(path))
    if backend == "mlx":
        body = mlx_part.replace(f'load("{_report()["model"]}")',
                                f'load("{TINY}")\ntok._tokenizer.chat_template = {TEMPLATE!r}')
    else:
        body = torch_part.replace(f'AutoTokenizer.from_pretrained("{_report()["model"]}")',
                                  f'AutoTokenizer.from_pretrained("{TINY}")\ntok.chat_template = {TEMPLATE!r}')
    body = body.replace(_report()["model"], TINY).replace("1024", "4")
    return head + body


def test_card_pytorch_example_runs_as_published(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    exec(_python_api_snippet(tmp_path, "torch"), {})
    assert capsys.readouterr().out.strip() != ""


def test_card_mlx_example_runs_as_published(tmp_path, capsys):
    pytest.importorskip("mlx_lm")
    exec(_python_api_snippet(tmp_path, "mlx"), {})
    assert capsys.readouterr().out.strip() != ""


def test_every_pts_command_on_the_card_parses():
    import shlex

    from pts.cli import parse_args

    card = discovery_card(_report(), "someone/q", example_prompt='Is "this" fine? I\'m sure.')
    blocks = [b.split("\n", 1)[1] for b in card.split("```bash")[1:]]
    commands = []
    for block in blocks:
        text = block.split("```", 1)[0].replace("\\\n", " ")
        commands += [line for line in text.splitlines() if line.startswith("pts ")]
    assert {c.split()[1] for c in commands} >= {"steer", "serve", "discover", "push"}
    for command in commands:
        argv = shlex.split(command.split("#", 1)[0] if " # " in command else command)[1:]
        argv = [a.replace("<your-model>", "m").replace("<you>/<your-model>-sycophancy", "a/b") for a in argv]
        args, _ = parse_args(argv)
        assert args.command == argv[0]


def test_curl_example_sends_valid_json_even_with_quotes_in_the_prompt():
    import shlex

    prompt = 'Is "this" fine? I\'m sure.'
    card = discovery_card(_report(), "someone/q", example_prompt=prompt)
    curl = next(b for b in card.split("```bash")[1:] if b.lstrip().startswith("curl")).split("```", 1)[0]
    argv = shlex.split(curl)
    body = json.loads(argv[argv.index("-d") + 1])
    assert body["messages"][0]["content"] == prompt
    assert body["steering"] == [{"vector": "sycophancy-increase"}]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
