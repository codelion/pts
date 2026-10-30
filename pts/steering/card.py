"""
Model cards for steering vectors published from a ``pts discover`` run.

The card is generated from the run's ``report.json``, so every number on it
is one the run measured on held-out questions. Its verdict is computed, not
written: a vector is said to beat its controls only if every paired interval
against them excludes zero and the random-vector null agrees. Anything less
is stated as such. Pure Python.
"""

import math
from typing import Dict, List, Optional

MAIN = "pts_matched_pairs"
DIRECTIONS = ("increase", "decrease")
REPO_URL = "https://github.com/codelion/pts"
# What a vector has to beat, and how each is named on the card. Other PTS
# variants are compared on the card but are not controls.
CONTROLS = {
    "control_sign_shuffle": "shuffled signs (same tokens)",
    "control_random_positions": "random positions (same traces)",
    "caa": "CAA (answer-token contrast)",
    "prompt": "prompt baseline",
}
NAMES = {**CONTROLS, "pts_signed_mean_centered": "PTS signed mean (variant, not a control)"}


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def _signed(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2f}"


def _interval(d: Dict) -> str:
    return f"{_signed(d.get('delta', d.get('advantage')))} [{_signed(d.get('ci_low'))}, {_signed(d.get('ci_high'))}]"


def verdict(report: Dict, direction: str, method: str = MAIN, alpha: float = 0.05) -> Dict:
    """Whether ``method``'s vector for ``direction`` beats every alternative.

    ``beats`` lists alternatives whose paired 95% interval is above zero;
    ``not_beaten`` the rest. ``passes`` needs both an empty ``not_beaten``
    and a random-vector null p below ``alpha``.
    """
    result = (report.get("methods", {}).get(method) or {}).get(direction)
    if not result:
        return {"exists": False, "passes": False, "beats": [], "not_beaten": [], "random_null_p": None}
    effect = result["test_effect"]
    # The whole interval on the requested side of zero.
    moved = effect["ci_low"] > 0 if direction == "increase" else effect["ci_high"] < 0
    beats, not_beaten = [], []
    prefix = f"{direction}: {method} vs "
    for key, c in report.get("comparisons", {}).items():
        other = key[len(prefix):]
        if key.startswith(prefix) and other in CONTROLS:
            (beats if c["ci_low"] > 0 else not_beaten).append(other)
    null = result.get("random_null") or {}
    p = null.get("p_value")
    passes = moved and not not_beaten and bool(beats) and p is not None and p < alpha
    return {"exists": True, "moved": moved, "passes": passes, "beats": beats,
            "not_beaten": not_beaten, "random_null_p": p, "random_null_n": len(null.get("deltas") or [])}


def _direction_section(report: Dict, direction: str, method: str, repo_id: str) -> List[str]:
    result = (report.get("methods", {}).get(method) or {}).get(direction)
    lines = [f"### `{direction}/`", ""]
    if not result:
        lines += ["No setting passed selection for this direction (every candidate either did not "
                  "move the behavior or damaged the model beyond the answered-rate and KL limits), "
                  "so no vector is published for it.", ""]
        return lines
    v = verdict(report, direction, method)
    base, test = report["base"]["test"], result["test"]
    if v["passes"]:
        summary = f"**Beats every control** on held-out questions: {', '.join(NAMES[b] for b in v['beats'])}."
    elif not v["moved"]:
        summary = "**Did not significantly move the behavior** on held-out questions."
    else:
        summary = ("**Moves the behavior but does not beat every control.** "
                   f"Not beaten: {', '.join(NAMES[b] for b in v['not_beaten']) or 'none'}"
                   + (f"; {_null(v)}" if v["random_null_p"] is not None else "")
                   + ".")
    lines += [
        summary, "",
        f"Layer {result['layer']}, alpha {result['alpha']:.3g} (stored as the vector's default).", "",
        "| Held-out test questions | Unsteered | Steered |",
        "|---|---|---|",
        f"| P(behavior-matching answer) | {_pct(base['p_match'])} | {_pct(test['p_match'])} |",
        f"| Answered at all | {_pct(base['answered'])} | {_pct(test['answered'])} |",
        f"| KL on neutral text | - | {_pct(test['kl'])} |",
        "",
        f"Paired change in P, 95% bootstrap interval: {_interval(result['test_effect'])}. "
        f"Null: {_null(v)}, at the same layer and strength.",
        "",
    ]
    comps = [(k.split(" vs ", 1)[1], c) for k, c in report.get("comparisons", {}).items()
             if k.startswith(f"{direction}: {method} vs ")]
    if comps:
        lines += ["| Compared with | Advantage of this vector, 95% interval |", "|---|---|"]
        lines += [f"| {NAMES.get(name, name)} | {_interval(c)} |" for name, c in comps]
        lines.append("")
    return lines


def discovery_card(report: Dict, repo_id: str, method: str = MAIN) -> str:
    """Markdown model card for the vectors of one discover run."""
    model, behavior = report["model"], report["behavior"]
    ev, split, events = report["eval"], report["split"], report["events"]
    published = [d for d in DIRECTIONS if (report.get("methods", {}).get(method) or {}).get(d)]
    sampling = report.get("sampling") or {}
    template = report.get("chat_template_args") or {}
    sc = report.get("search_config", {})

    flags = []
    if template:
        flags.append(f"--chat-template-args '{_json(template)}'")
    for key, flag in (("temperature", "--temperature"), ("top_p", "--top-p"), ("top_k", "--top-k"),
                      ("min_p", "--min-p"), ("presence_penalty", "--presence-penalty")):
        if sampling.get(key):
            flags.append(f"{flag} {sampling[key]}")
    if sc.get("max_tokens"):
        flags.append(f"--max-tokens {sc['max_tokens']}")
    command = [f"pts discover --model <your-model> --eval {ev}", f"--behavior {behavior} --output-dir out/"] + flags

    lines = [
        "---",
        "library_name: pts",
        f"base_model: {model}",
        "tags:",
        "- steering-vector",
        "- pts",
        "- mlx",
        f"- {behavior}",
        "---",
        "",
        f"# {behavior} steering vectors for {model}",
        "",
        f"Residual-stream steering vectors that push {model} toward (`increase/`) or away from "
        f"(`decrease/`) **{behavior}**, found with [PTS]({REPO_URL})'s `pts discover`: pivotal-token "
        f"search on a labelled behavior eval, then validated on held-out questions against controls.",
        "",
        "## Use",
        "",
        "```bash",
        "pip install 'pts[mlx] @ git+https://github.com/codelion/pts'   # Apple silicon; drop [mlx] for PyTorch",
        "",
    ]
    for d in published:
        lines.append(f"pts steer --model {model} --vector hf://{repo_id}/{d} --chat --compare --prompt \"...\"")
    lines += [
        f"pts steer --backend torch --model {model} --vector hf://{repo_id}/{published[0] if published else 'increase'} --prompt \"...\"",
        f"pts serve --model {model} --vector {behavior}=hf://{repo_id}/{published[0] if published else 'increase'}   # OpenAI-compatible, per-request steering",
        "```",
        "",
        "Each vector stores the strength it was validated at; `hf://...:<alpha>` overrides it. "
        "Strong steering breaks a model before it changes the behavior, so check the answered rate "
        "and the KL below before raising it. Both backends hook the residual stream after the same "
        "decoder block and agree to 1e-5.",
        "",
        "## Validation",
        "",
        f"Eval: `{ev}`, split by question into {split['discovery']} discovery / {split['selection']} "
        f"selection / {split['test']} test. Vectors come from the discovery questions; layer and "
        f"strength are chosen on the selection questions; only the test questions are reported. "
        f"Every probability is sampled ({report.get('eval_samples', '?')} completions per test "
        f"question, run to the end) and judged by an oracle reading the final answer.",
        "",
    ]
    for d in DIRECTIONS:
        lines += _direction_section(report, d, method, repo_id)
    prompts = report.get("prompt_baselines") or {}
    if prompts:
        lines += ["Prompt baselines (an instruction added to the question instead of steering):", ""]
        for d, b in prompts.items():
            lines.append(f"- {d}: \"{b['prompt']}\" -> paired change {_interval(b['test_effect'])}")
        lines.append("")
    lines += [
        "## How it was found",
        "",
        f"Token PTS with an oracle-judged behavior: P(behavior | prefix) is the fraction of sampled "
        f"completions whose final answer matches the behavior, and bisection over each sampled "
        f"response finds the single tokens that move it by at least {sc.get('prob_threshold', 0.2)}. "
        f"This run found {events['total']} pivotal tokens ({events['positive']} raising the behavior, "
        f"{events['states_answer']} that just write the answer letter, which are not used) across "
        f"{events['moved_traces']} traces. At each pivotal token the residual stream after it is "
        f"paired with the residual after the model's likeliest alternative at the same prefix, the "
        f"difference is signed by the measured change, and the vector is their mean. Controls built the same way: shuffled signs, random "
        f"positions in the same traces, CAA (contrastive answer-token activations), random vectors, "
        f"and the prompt baseline.",
        "",
        "## Larger models",
        "",
        f"These vectors fit {model} only. To find them for another model, run the same search:",
        "",
        "```bash",
        " \\\n    ".join(command),
        f"pts push --input-path out/ --hf-repo-id <you>/<your-model>-{behavior}",
        "```",
        "",
        "It needs a model that reasons before it answers and finishes within `--max-tokens`. Cost "
        "is sampled tokens: about 1M per searched question with ~450-token thinking traces "
        "(roughly an hour for a 0.6B model on an M4), growing with model size and trace length.",
        "",
        "## Limitations",
        "",
        f"- Validated on {split['test']} held-out questions from the same eval; other phrasings of "
        f"{behavior} were not tested.",
        "- Answers are multiple choice. Effects on free-form generation were not measured.",
        "- Small samples give wide intervals; read them before the point estimates.",
        "",
        "`report.json` holds the full run: every selection setting tried, every control, and the cost.",
        "",
    ]
    return "\n".join(lines)


def _null(v: Dict) -> str:
    n, p = v["random_null_n"], v["random_null_p"]
    if not n or p is None:
        return "no random-vector null was run"
    return f"{round(p * n)} of {n} random vectors of the same norm moved P as far"


def _json(obj) -> str:
    import json

    return json.dumps(obj)
