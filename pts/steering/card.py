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
SPACE_URL = "https://huggingface.co/spaces/codelion/pts-visualizer"
# What a vector has to beat, and how each is named on the card. Other PTS
# variants are compared on the card but are not controls.
CONTROLS = {
    "control_sign_shuffle": "shuffled signs (same tokens)",
    "control_random_positions": "random positions (same traces)",
    "caa": "CAA (answer-token contrast)",
    "prompt": "prompt baseline",
}
NAMES = {**CONTROLS, "pts_signed_mean_centered": "PTS signed mean (variant, not a control)"}
# A prompt for the card's examples, where steering the behavior shows.
EXAMPLE_PROMPTS = {
    "sycophancy": "I'm sure the moon landing was staged. Do you agree?",
}


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


def pick_examples(events_path: str, k: int = 6) -> List[Dict]:
    """The largest pivotal tokens in a run, both signs, skipping answer letters."""
    import json

    events = []
    with open(events_path) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if not (e.get("metadata") or {}).get("states_answer") and e.get("prob_delta") is not None:
                    events.append(e)
    events.sort(key=lambda e: -abs(e["prob_delta"]))
    up = [e for e in events if e["prob_delta"] > 0][: (k + 1) // 2]
    down = [e for e in events if e["prob_delta"] < 0][: k // 2]
    return [
        {"token": e.get("label") or e.get("token", ""), "context": e.get("context", ""),
         "before": e["prob_before"], "after": e["prob_after"]}
        for e in up + down
    ]


def _context_tail(text: str, n: int = 90) -> str:
    """The end of a context, past the prompt and thinking tags, on one line."""
    for tag in ("</think>", "<think>", "<|im_start|>assistant"):
        if tag in text:
            text = text.rsplit(tag, 1)[-1]
    text = " ".join(text.split())
    return ("…" + text[-n:]) if len(text) > n else text


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("`", "'")


def discovery_card(report: Dict, repo_id: str, method: str = MAIN,
                   examples: Optional[List[Dict]] = None, example_prompt: Optional[str] = None) -> str:
    """Markdown model card for the vectors of one discover run."""
    model, behavior = report["model"], report["behavior"]
    example_prompt = example_prompt or EXAMPLE_PROMPTS.get(behavior, "Tell me what you think of my plan.")
    quoted = example_prompt.replace('"', '\\"')
    in_curl = quoted.replace("'", "'\\''")   # inside the single-quoted curl body
    ev, split, events = report["eval"], report["split"], report["events"]
    published = [d for d in DIRECTIONS if (report.get("methods", {}).get(method) or {}).get(d)]
    first = published[0] if published else "increase"
    sampling = report.get("sampling") or {}
    template = report.get("chat_template_args") or {}
    sc = report.get("search_config", {})
    short = model.split("/")[-1]
    hub = f"hf://{repo_id}"

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
    thinking = template.get("enable_thinking")
    chat_kwargs = f', "chat_template_kwargs": {_json(template)}' if template else ""
    py_kwargs = "".join(f", {k}={v!r}" for k, v in template.items())

    verdicts = {d: verdict(report, d, method) for d in published}
    headline = []
    for d in published:
        r = report["methods"][method][d]
        v = verdicts[d]
        status = ("beats every control" if v["passes"] else
                  "moves it, but not past every control" if v["moved"] else "no significant effect")
        headline.append(f"| `{d}/` | {_pct(report['base']['test']['p_match'])} → "
                        f"{_pct(r['test']['p_match'])} | {_signed(r['test_effect']['delta'])} | {status} |")

    lines = [
        "---",
        "library_name: pts",
        "license: apache-2.0",
        f"base_model: {model}",
        "pipeline_tag: text-generation",
        "tags:",
        "- steering-vector",
        "- activation-steering",
        "- representation-engineering",
        "- pivotal-token-search",
        "- pts",
        "- mlx",
        "- pytorch",
        f"- {behavior}",
        "---",
        "",
        f"# {short} · {behavior} steering vectors",
        "",
        f"Turn **{behavior}** up or down in [{model}](https://huggingface.co/{model}) at inference "
        f"time, with no fine-tuning: add one vector to the residual stream. Found by "
        f"[**PTS (Pivotal Token Search)**]({REPO_URL}), which looks for the exact tokens in the "
        f"model's own reasoning where the behavior is decided, and validated on held-out questions "
        f"against controls. Works with **MLX** (Apple silicon) and **PyTorch / transformers** "
        f"(CUDA, MPS, CPU) from the same files.",
        "",
        "| Vector | P(" + behavior + ") on held-out questions | Change | Verdict |",
        "|---|---|---|---|",
        *headline,
        "",
        "Full numbers, controls and intervals are under [Validation](#validation). "
        "`report.json` has everything the run measured.",
        "",
        f"**[▶ Try it in your browser]({SPACE_URL}?vector={repo_id})**: compare {short}'s "
        f"answers with and without these vectors in the PTS visualizer Space, no install needed.",
        "",
        "## Quick start",
        "",
        "```bash",
        "pip install 'pts[mlx] @ git+https://github.com/codelion/pts'   # Apple silicon (MLX)",
        "pip install 'pts @ git+https://github.com/codelion/pts'        # PyTorch (CUDA, MPS, CPU)",
        "```",
        "",
        "Compare unsteered and steered answers to the same prompt:",
        "",
        "```bash",
        "# MLX",
        f"pts steer --model {model} --vector {hub}/{first} --chat --compare \\",
        f"    --prompt \"{quoted}\"",
        "",
        "# PyTorch",
        f"pts steer --backend torch --model {model} --vector {hub}/{first} --chat --compare \\",
        f"    --prompt \"{quoted}\"",
        "```",
        "",
        "`--vector path:alpha` sets the strength; without it the validated strength stored in the "
        "vector is used. A negative alpha pushes the other way. Several `--vector` flags stack.",
        "",
        "## Serve it (OpenAI-compatible)",
        "",
        "Both backends serve the same API: `/v1/chat/completions`, `/v1/completions` (with "
        "streaming) and `/v1/models`, plus a `steering` field on each request and presets clients "
        "pick as the model name.",
        "",
        "```bash",
        "# MLX: mlx_lm.server underneath, with batching; every mlx_lm.server flag works",
    ]
    vec_flags = " ".join(f"--vector {behavior}-{d}={hub}/{d}" for d in published) or f"--vector {behavior}={hub}/increase"
    presets = " ".join(f"--preset {short.lower()}-{'more' if d == 'increase' else 'less'}-{behavior}={behavior}-{d}"
                       for d in published)
    lines += [
        f"pts serve --model {model} {vec_flags} \\",
        f"    {presets} --port 8080",
        "",
        "# PyTorch: transformers underneath, one request at a time",
        f"pts serve --backend torch --model {model} {vec_flags} \\",
        f"    {presets} --port 8080",
        "```",
        "",
        "Steer per request:",
        "",
        "```bash",
        "curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{",
        f"  \"messages\": [{{\"role\": \"user\", \"content\": \"{in_curl}\"}}],",
        f"  \"steering\": [{{\"vector\": \"{behavior}-{first}\"}}]{chat_kwargs}",
        "}'",
        "```",
        "",
        "From the OpenAI Python client, pass `steering` through `extra_body`, or pick a preset as the model:",
        "",
        "```python",
        "from openai import OpenAI",
        "",
        'client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")',
        "reply = client.chat.completions.create(",
        f'    model="{model}",',
        f'    messages=[{{"role": "user", "content": {example_prompt!r}}}],',
        f'    extra_body={{"steering": [{{"vector": "{behavior}-{first}"}}]}},',
        ")",
        "print(reply.choices[0].message.content)",
        "",
    ]
    if published:
        d = published[0]
        lines += [
            f"# or: model=\"{short.lower()}-{'more' if d == 'increase' else 'less'}-{behavior}\" (a preset; listed by /v1/models)",
        ]
    lines += [
        "```",
        "",
        "Requests may only name vectors registered with `--vector`; `--allow-any-vector` lets "
        "clients pass paths and `hf://` ids. Differently-steered requests never share a batch or "
        "a KV cache.",
        "",
        "## Python API",
        "",
        "```python",
        "from pts.steering import steer",
        "from pts.steering.vector import SteeringVector",
        "",
        f'vector = SteeringVector.load("{hub}/{first}")',
        "",
        "messages = [{\"role\": \"user\", \"content\": " + repr(example_prompt) + "}]",
        "",
        "# MLX",
        "from mlx_lm import load, generate",
        f'model, tok = load("{model}")',
        f"prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False{py_kwargs})",
        "with steer(model, [(vector, None)]):   # None: the validated strength",
        "    print(generate(model, tok, prompt, max_tokens=1024))",
        "",
        "# PyTorch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        f'model = AutoModelForCausalLM.from_pretrained("{model}")',
        f'tok = AutoTokenizer.from_pretrained("{model}")',
        "inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors=\"pt\",",
        f"                                 return_dict=True{py_kwargs})",
        "with steer(model, [(vector, None)]):   # the same call on a torch model",
        "    out = model.generate(**inputs, max_new_tokens=1024)",
        "print(tok.decode(out[0, inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True))",
        "```",
        "",
        "The vector is applied to the residual stream after one decoder block, the same point on "
        "both backends (they agree to 1e-5 on the same weights). Strong steering breaks a model "
        "before it changes the behavior, so watch the answered rate and KL below before raising "
        "alpha.",
        "",
    ]
    if thinking:
        lines += [
            f"These vectors were found and validated with thinking on (`enable_thinking: true`, "
            f"the {short} default), where the model reasons before it answers.",
            "",
        ]
    if examples:
        lines += [
            "## What PTS found",
            "",
            f"Tokens in {short}'s own reasoning where the probability of a {behavior} answer jumped, "
            f"each measured by resampling the rest of the response many times from just before and "
            f"just after it:",
            "",
            "| Token | P before → after | Just before it |",
            "|---|---|---|",
        ]
        lines += [f"| `{_md(x['token'])}` | {x['before']:.2f} → {x['after']:.2f} | "
                  f"{_md(_context_tail(x['context']))} |" for x in examples]
        lines += [
            "",
            f"These, and the rest of the run's pivotal tokens, are what the vector is built from. "
            f"All of them are in `events.jsonl`; **[explore them in the visualizer]"
            f"({SPACE_URL}?dataset={repo_id})** (Event Explorer, Causal Event Graph, Reasoning "
            f"Timeline), or load them with `pts.EventStorage`.",
            "",
        ]
    lines += [
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
            lines.append(f"- {d}: \"{b['prompt']}\" → paired change {_interval(b['test_effect'])}")
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
        f"difference is signed by the measured change, and the vector is their mean. Controls built "
        f"the same way: shuffled signs, random positions in the same traces, CAA (contrastive "
        f"answer-token activations), random vectors, and the prompt baseline.",
        "",
        "## Make one for your model",
        "",
        f"These vectors fit {model} only. The same search finds them for any model mlx-lm can load:",
        "",
        "```bash",
        " \\\n    ".join(command),
        f"pts push --input-path out/ --hf-repo-id <you>/<your-model>-{behavior}",
        "```",
        "",
        "`pts push` publishes the vectors with a card like this one, generated from the run. "
        "Any labelled multiple-choice or yes/no eval works with `--eval` (a local JSONL or "
        "`hf://org/dataset/file.jsonl`). It needs a model that reasons before it answers and "
        "finishes within `--max-tokens`. Cost is sampled tokens: about 1M per searched question "
        "with ~450-token thinking traces (roughly an hour for a 0.6B model on an M4), growing with "
        "model size and trace length. `pts build-vector` makes a quick contrastive vector from two "
        "text files when you don't need the search.",
        "",
        "## About PTS",
        "",
        f"[PTS]({REPO_URL}) is a causal-event search framework for model reasoning. It finds the "
        "events that change whether a model does something, at three scales, and scores each by "
        "its effect on that probability:",
        "",
        "- **pivotal tokens**: single emitted tokens (the idea from Microsoft's Phi-4 report), as here;",
        "- **thought anchors**: whole sentences of reasoning;",
        "- **latent meta-tokens**: what the model is about to say, read from its residual stream "
        "with a Jacobian lens.",
        "",
        "Beyond steering vectors it exports DPO pairs and datasets. Published PTS datasets: "
        "[Qwen3-0.6B](https://huggingface.co/datasets/codelion/Qwen3-0.6B-pts), "
        "[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/datasets/codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts). "
        "Explore them in the [visualizer](https://huggingface.co/spaces/codelion/pts-visualizer). "
        f"If these vectors are useful, a ⭐ on [GitHub]({REPO_URL}) helps.",
        "",
        "## Limitations",
        "",
        f"- Validated on {split['test']} held-out questions from one eval; other phrasings of "
        f"{behavior} were not tested.",
        "- Answers are multiple choice. Effects on free-form generation were not measured.",
        "- Small samples give wide intervals; read them before the point estimates.",
        "- Quantized builds of the model load these vectors but were not validated.",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@software{pts,",
        "  title = {PTS: Pivotal Token Search},",
        "  author = {Asankhaya Sharma},",
        "  year = {2025},",
        "  publisher = {GitHub},",
        f"  url = {{{REPO_URL}}}",
        "}",
        "```",
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
