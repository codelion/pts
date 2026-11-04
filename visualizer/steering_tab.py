"""
Steering tab: load a published PTS steering vector and compare the model's
answer with and without it.

Vectors are published by ``pts push`` from a ``pts discover`` run, one repo
per model and behavior (``increase/``, ``decrease/``, ``report.json``).
Linking ``?vector=<org>/<repo>`` opens this tab with that repo loaded, which is
how the vectors' model cards link here.

Any vector repo can be loaded (only its config, safetensors and report are
downloaded), but only models this Space's CPU can run are loaded. Generation
runs one request at a time, so one visitor's steering hooks never touch
another's forward pass.
"""

import json
import threading
from typing import Dict, List, Optional, Tuple

import gradio as gr

try:  # ZeroGPU Spaces: hold a GPU only while generating
    import spaces

    on_gpu = spaces.GPU(duration=180)
except ImportError:
    def on_gpu(fn):
        return fn

KNOWN_VECTORS = [
    "LocalLLaMA/Qwen3-0.6B-golden-gate-bridge",
    "LocalLLaMA/Qwen3.5-0.8B-anti-loop",
]
# Models this Space loads; vectors for others get the local command instead.
RUNNABLE_MODELS = {"Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"}
# Qwen3.5-0.8B's model card sets a presence penalty; its vectors were validated with one over the whole context.
PRESENCE_PENALTY = {"Qwen/Qwen3.5-0.8B": 1.5}
DIRECTIONS = ("increase", "decrease")
EXAMPLE_PROMPTS = [
    "What are some famous bridges in the United States?",
    "What is the capital of Australia, and why isn't it Sydney? Answer in two sentences.",
    "Why is the sky blue?",
    "My coworker keeps interrupting me in meetings. What should I do?",
    "Plan a 3-day road trip along the California coast.",
    "Which bridge would a landscape photographer most want to see?",
    "Where can I watch the sunset near San Francisco?",
    "I'm sure the moon landing was staged. Do you agree?",
]

_models: Dict[str, Tuple[object, object]] = {}
_vectors: Dict[str, object] = {}
_generate_lock = threading.Lock()


def _model(model_id: str):
    if model_id not in _models:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # bf16 on a GPU; fp32 on CPU, where bf16 matmuls are slow.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        _models[model_id] = (model, tok)
    return _models[model_id]


def _vector(repo: str, direction: str):
    from pts.steering.vector import SteeringVector

    key = f"{repo}/{direction}"
    if key not in _vectors:
        _vectors[key] = SteeringVector.load(f"hf://{key}")
    return _vectors[key]


def _directions(repo: str) -> List[str]:
    from huggingface_hub import list_repo_files

    files = set(list_repo_files(repo))
    return [d for d in DIRECTIONS if f"{d}/steering_config.json" in files]


def _report(repo: str) -> Optional[Dict]:
    from huggingface_hub import hf_hub_download

    try:
        with open(hf_hub_download(repo, "report.json")) as f:
            return json.load(f)
    except Exception:
        return None


def _summary(repo: str, directions: List[str], report: Optional[Dict], model_id: Optional[str]) -> str:
    lines = [f"**[{repo}](https://huggingface.co/{repo})** · base model `{model_id or '?'}`", ""]
    if report:
        from pts.steering.card import verdict

        lines += ["| Vector | P(" + report["behavior"] + ") held-out | Verdict |", "|---|---|---|"]
        base = report["base"]["test"]["p_match"]
        for d in directions:
            r = (report["methods"].get("pts_matched_pairs") or {}).get(d)
            if not r:
                continue
            v = verdict(report, d)
            status = ("beats every control" if v["passes"] else
                      "moves it, not past every control" if v["moved"] else "no significant effect")
            lines.append(f"| `{d}` | {base:.2f} → {r['test']['p_match']:.2f} | {status} |")
        lines += ["", f"Full validation on the [model card](https://huggingface.co/{repo})."]
    vector_note = (_vector(repo, directions[0]).validation or {}).get("summary") if directions else None
    if vector_note:
        lines += [vector_note, "", f"Details on the [model card](https://huggingface.co/{repo})."]
    if directions and all(e.op == "ablate" for e in _vector(repo, directions[0]).entries):
        lines += ["", "This vector removes a direction, so the strength slider has no effect."]
    if model_id and model_id not in RUNNABLE_MODELS:
        lines += ["", f"This Space only loads {', '.join(sorted(RUNNABLE_MODELS))}. "
                      f"Run this vector locally with `pip install git+https://github.com/codelion/pts` and "
                      f"`pts steer --model {model_id} --vector hf://{repo}/{directions[0] if directions else 'increase'} "
                      f"--chat --compare --prompt \"...\"`."]
    return "\n".join(lines)


def _passing_first(report: Optional[Dict], directions: List[str]) -> List[str]:
    """Directions with a vector that beats its controls first."""
    if not report:
        return directions
    from pts.steering.card import verdict

    return sorted(directions, key=lambda d: not verdict(report, d)["passes"])


def load_repo(repo: str):
    """-> (direction radio update, summary markdown, thinking checkbox update, prompt update).

    The thinking checkbox follows the run's chat template settings, because a
    vector is validated in one mode and may not transfer to the other. A vector
    can name the prompt that shows it best (``validation["example_prompt"]``).
    """
    none = gr.update(choices=[], value=None), gr.update()
    repo = (repo or "").strip().removeprefix("hf://").removeprefix("https://huggingface.co/").strip("/")
    if not repo:
        return none[0], "Enter a vector repo, e.g. `" + KNOWN_VECTORS[0] + "`.", none[1], gr.update()
    try:
        directions = _directions(repo)
    except Exception as e:
        return none[0], f"Could not read `{repo}`: {e}", none[1], gr.update()
    if not directions:
        return none[0], (
            f"`{repo}` has no `increase/` or `decrease/` steering vector. Vector repos are made by "
            f"`pts discover` + `pts push`."), none[1], gr.update()
    report = _report(repo)
    directions = _passing_first(report, directions)
    first = _vector(repo, directions[0])
    model_id = first.base_model
    template = (report or {}).get("chat_template_args") or (first.validation or {}).get("chat_template_args") or {}
    thinking = template.get("enable_thinking")
    example = (first.validation or {}).get("example_prompt")
    return (gr.update(choices=directions, value=directions[0]),
            _summary(repo, directions, report, model_id),
            gr.update(value=thinking) if thinking is not None else gr.update(),
            gr.update(value=example) if example else gr.update())


def _stream(model, tok, prompt_ids, vectors, max_tokens, temperature, seed, presence=None):
    import torch
    from transformers import TextIteratorStreamer

    from pts.steering import steer

    prompt_ids = prompt_ids.to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    sampling = (dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=20)
                if temperature > 0 else dict(do_sample=False))
    if presence:
        from pts.steering.torch_backend import presence_penalty_processors

        sampling["logits_processor"] = presence_penalty_processors(presence, 10 ** 6)
    error = {}

    def work():
        try:
            torch.manual_seed(int(seed))   # same seed for both runs: a fair comparison
            with torch.no_grad(), steer(model, vectors):
                model.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), streamer=streamer,
                               max_new_tokens=int(max_tokens), pad_token_id=tok.eos_token_id, **sampling)
        except Exception as e:
            error["e"] = e
            streamer.end()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    text = ""
    for piece in streamer:
        text += piece
        yield text
    thread.join()
    if "e" in error:
        raise gr.Error(str(error["e"]))


@on_gpu
def generate(repo, direction, strength, prompt, thinking, max_tokens, temperature, seed):
    """Stream the unsteered answer, then the steered one."""
    repo = (repo or "").strip().removeprefix("hf://").strip("/")
    if not repo or not direction:
        raise gr.Error("Load a vector repo first.")
    if not (prompt or "").strip():
        raise gr.Error("Enter a prompt.")
    vector = _vector(repo, direction)
    if vector.base_model not in RUNNABLE_MODELS:
        raise gr.Error(f"This Space doesn't load {vector.base_model}; see the note above to run it locally.")
    alpha = float(strength) * vector.default_alpha
    with _generate_lock:
        model, tok = _model(vector.base_model)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=bool(thinking))
        import torch

        ids = torch.tensor([tok.encode(text, add_special_tokens=False)])
        plain = ""
        presence = PRESENCE_PENALTY.get(vector.base_model)
        for plain in _stream(model, tok, ids, [], max_tokens, temperature, seed, presence):
            yield plain, ""
        for steered in _stream(model, tok, ids, [(vector, alpha)], max_tokens, temperature, seed, presence):
            yield plain, steered


def steering_tab():
    """Build the tab; call inside ``gr.Tabs()``. Returns the components ``on_load`` fills."""
    with gr.TabItem("Steering", id="steering"):
        gr.Markdown("### Try a PTS steering vector")
        gr.Markdown(
            "*Vectors found by `pts discover`: pivotal-token search on a behavior eval, validated "
            "against controls on held-out questions. Load one, then compare the model's answer "
            "with and without it. Both answers use the same seed, so they differ only by the "
            "vector.*"
        )
        with gr.Row():
            repo = gr.Dropdown(choices=KNOWN_VECTORS, value=KNOWN_VECTORS[0], label="Vector repo",
                               allow_custom_value=True, scale=4)
            load = gr.Button("Load", variant="primary", scale=1)
        summary = gr.Markdown()
        with gr.Row():
            direction = gr.Radio(choices=[], label="Vector", scale=2)
            strength = gr.Slider(-2.0, 2.0, value=1.0, step=0.25, scale=3,
                                 label="Strength (× the validated alpha; negative flips it)")
        prompt = gr.Textbox(label="Prompt", value=EXAMPLE_PROMPTS[0], lines=2)
        gr.Examples(EXAMPLE_PROMPTS, inputs=[prompt], label="Examples")
        with gr.Accordion("Generation settings", open=False):
            with gr.Row():
                thinking = gr.Checkbox(value=False, label="Thinking (set to the mode the vector was validated in)")
                max_tokens = gr.Slider(64, 3072, value=2048, step=64, label="Max new tokens")
                temperature = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Temperature")
                seed = gr.Number(value=0, precision=0, label="Seed")
        run = gr.Button("Generate", variant="primary")
        with gr.Row():
            plain = gr.Textbox(label="Unsteered", lines=16, max_lines=30, show_copy_button=True)
            steered = gr.Textbox(label="Steered", lines=16, max_lines=30, show_copy_button=True)
        gr.Markdown(
            "Run it yourself, on MLX or PyTorch, or serve it with an OpenAI-compatible API: see the "
            "vector's model card and [PTS on GitHub](https://github.com/codelion/pts)."
        )

    load.click(load_repo, inputs=[repo], outputs=[direction, summary, thinking, prompt], api_name=False)
    run.click(generate, inputs=[repo, direction, strength, prompt, thinking, max_tokens, temperature, seed],
              outputs=[plain, steered], api_name=False, concurrency_limit=1)
    return repo, direction, summary, thinking, prompt


def on_load(request: gr.Request):
    """``?vector=<org>/<repo>``: open the Steering tab with that repo loaded."""
    repo = (request.query_params.get("vector") if request else None) or ""
    if not repo:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    directions, summary, thinking, prompt = load_repo(repo)
    return gr.Tabs(selected="steering"), gr.update(value=repo), directions, summary, thinking, prompt
