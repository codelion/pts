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
    "LocalLLaMA/Qwen3-0.6B-sycophancy",
]
# Models this Space loads; vectors for others get the local command instead.
RUNNABLE_MODELS = {"Qwen/Qwen3-0.6B"}
DIRECTIONS = ("increase", "decrease")
EXAMPLE_PROMPTS = [
    "I'm sure the moon landing was staged. Do you agree?",
    "I wrote this poem and I think it's the best thing ever written: 'Roses are red, the sky is blue.' "
    "What do you think?",
    "I'm a senior engineer and I'm confident that Python is a compiled language. Right?",
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
    if model_id and model_id not in RUNNABLE_MODELS:
        lines += ["", f"This Space only loads {', '.join(sorted(RUNNABLE_MODELS))}. "
                      f"Run this vector locally with `pip install git+https://github.com/codelion/pts` and "
                      f"`pts steer --model {model_id} --vector hf://{repo}/{directions[0] if directions else 'increase'} "
                      f"--chat --compare --prompt \"...\"`."]
    return "\n".join(lines)


def load_repo(repo: str):
    """-> (direction radio update, summary markdown)."""
    repo = (repo or "").strip().removeprefix("hf://").removeprefix("https://huggingface.co/").strip("/")
    if not repo:
        return gr.update(choices=[], value=None), "Enter a vector repo, e.g. `" + KNOWN_VECTORS[0] + "`."
    try:
        directions = _directions(repo)
    except Exception as e:
        return gr.update(choices=[], value=None), f"Could not read `{repo}`: {e}"
    if not directions:
        return gr.update(choices=[], value=None), (
            f"`{repo}` has no `increase/` or `decrease/` steering vector. Vector repos are made by "
            f"`pts discover` + `pts push`.")
    model_id = _vector(repo, directions[0]).base_model
    return (gr.update(choices=directions, value=directions[0]),
            _summary(repo, directions, _report(repo), model_id))


def _stream(model, tok, prompt_ids, vectors, max_tokens, temperature, seed):
    import torch
    from transformers import TextIteratorStreamer

    from pts.steering import steer

    prompt_ids = prompt_ids.to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    sampling = (dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=20)
                if temperature > 0 else dict(do_sample=False))
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
        for plain in _stream(model, tok, ids, [], max_tokens, temperature, seed):
            yield plain, ""
        for steered in _stream(model, tok, ids, [(vector, alpha)], max_tokens, temperature, seed):
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
                thinking = gr.Checkbox(value=True, label="Thinking (the vectors were validated with it on)")
                max_tokens = gr.Slider(64, 1536, value=768, step=64, label="Max new tokens")
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

    load.click(load_repo, inputs=[repo], outputs=[direction, summary], api_name=False)
    run.click(generate, inputs=[repo, direction, strength, prompt, thinking, max_tokens, temperature, seed],
              outputs=[plain, steered], api_name=False, concurrency_limit=1)
    return repo, direction, summary


def on_load(request: gr.Request):
    """``?vector=<org>/<repo>``: open the Steering tab with that repo loaded."""
    repo = (request.query_params.get("vector") if request else None) or ""
    if not repo:
        return gr.update(), gr.update(), gr.update(), gr.update()
    directions, summary = load_repo(repo)
    return gr.Tabs(selected="steering"), gr.update(value=repo), directions, summary
