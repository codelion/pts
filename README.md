<div align="center">

<img src="assets/pts-logo.svg" alt="PTS: Pivotal Token Search" width="560">

**A causal-event search framework for model reasoning.**

Find the reasoning events that change whether a model solves a task, at three scales at once.

<a href="https://github.com/codelion/pts/stargazers"><img src="https://img.shields.io/github/stars/codelion/pts?style=social" alt="GitHub stars"></a>
<a href="https://github.com/codelion/pts/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/pts?color=A78BFA" alt="License"></a>
<img src="https://img.shields.io/badge/python-3.9%2B-38BDF8" alt="Python 3.9+">
<a href="https://huggingface.co/spaces/codelion/pts-visualizer"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20visualizer-live-FBBF24" alt="HF Space"></a>

[Quick start](#quick-start) · [The three scales](#the-three-scales) · [Results](#results) · [Related work](#related-work) · [Visualizer](https://huggingface.co/spaces/codelion/pts-visualizer)

</div>

---

## What PTS is

When a model works through a problem, a few points along the way decide the
answer. Some of those points are hidden in the residual stream as a concept the
model has not said yet. Some are a single emitted token. Some are a whole
sentence of reasoning.

PTS treats all three as the same thing at different scales. It searches for them,
scores each one by how much it changes the model's chance of getting the answer
right, labels them with one shared vocabulary, and links them into a single graph.

```
latent meta-token / workspace event        Latent PTS
        |
emitted pivotal token                       Token PTS
        |
sentence-level thought anchor               Sentence PTS
        |
success / failure probability shift
```

The score is the same at every scale:

```
event_importance = outcome_with_event - outcome_without_or_altered_event
```

## The three scales

Each scale builds on existing work. PTS is the frame that holds the three
together.

| Scale | What it finds | How it's scored | Builds on |
|---|---|---|---|
| **Latent PTS** | Concepts active in the mid-layer workspace, not yet emitted | J-lens readout score | Anthropic's workspace / J-lens [[1]](#references) |
| **Token PTS** | Emitted tokens that flip success probability | `P(success \| prefix+token) - P(success \| prefix)` | Phi-4 Pivotal Token Search [[2]](#references) |
| **Sentence PTS** | Reasoning sentences that flip success probability | `P(success \| prefix+sentence) - P(success \| +alternative)` | Thought Anchors [[3]](#references) |

The question PTS is built to test: do latent meta-tokens in the workspace tend to
show up just before the emitted tokens and sentences that matter?

## Install

```bash
git clone https://github.com/codelion/pts.git && cd pts
pip install -e .
```

`import pts` does not load torch or transformers. The schema, storage,
classification, and linking layers are plain Python. Model code loads only when
you use it.

## Quick start

```bash
# Token PTS: emitted pivotal tokens (the original idea)
pts run --granularity token --model Qwen/Qwen3-0.6B --output-path events.jsonl

# Sentence PTS: thought anchors
pts run --granularity sentence --model Qwen/Qwen3-0.6B --output-path events.jsonl

# Latent PTS: add workspace meta-tokens to a dataset you already have.
pts fit-jlens --model Qwen/Qwen3-0.6B --output-path ./jlens         # calibrate once
pts enrich --input-path events.jsonl --output-path events_latent.jsonl \
           --model Qwen/Qwen3-0.6B --jlens-path ./jlens \
           --readout-method jlens --with-latent --shuffle-control

# All three scales, linked into one graph
pts run --granularity all --model Qwen/Qwen3-0.6B \
        --readout-method jlens --jlens-path ./jlens --output-path events.jsonl
```

No J-lens yet? `--readout-method logit_lens` needs no calibration. It is the same
readout with `J = I`, and it is a weaker signal. See
[docs/latent_pts.md](docs/latent_pts.md).

Explore any result in the [hosted visualizer](https://huggingface.co/spaces/codelion/pts-visualizer),
or run it locally with `cd visualizer && python app.py`.

## Results

We enriched two reasoning models and checked whether the J-lens (what an
activation is pushing the model to say later) beats a logit-lens control (what it
would say now). That comparison is what tells a real workspace apart from plain
next-token structure.

| | Qwen3-0.6B | DeepSeek-R1-1.5B |
|---|---|---|
| Meta-token category matches the event it precedes, vs chance | 2.6x | **3.6x** |
| J-lens lift | 2.76x | **3.64x** |
| logit-lens lift (control) | 2.41x | 3.29x |
| J-lens beats the control? | trends ahead, overlapping (n≈100) | **yes** (n=239) |

The gap is clearer on the bigger model, which is the direction the workspace idea
predicts. These are observational results, not causal ones. The dataset cards
have the full per-model write-ups:
[Qwen](https://huggingface.co/datasets/codelion/Qwen3-0.6B-pts),
[DeepSeek](https://huggingface.co/datasets/codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts).

## Commands

| Command | Does |
|---|---|
| `pts run --granularity token\|sentence\|latent\|all` | Search for pivotal events |
| `pts enrich --with-latent` | Add latent meta-token events to a dataset |
| `pts fit-jlens` | Calibrate a Jacobian lens for a model |
| `pts link` | Link latent, token, and sentence events into chains |
| `pts migrate` | Read older pivotal-token / thought-anchor files into the schema |
| `pts export --format …` | `causal_events`, `metatokens`, `pivotal_tokens`, `thought_anchors`, `dpo`, `steering` |
| `pts push` | Upload to Hugging Face |

DPO pairs and steering vectors are `export` formats. The steering vectors feed
[OptiLLM](https://github.com/codelion/optillm)'s autothink. See
[docs/compatibility.md](docs/compatibility.md).

## Related work

PTS pulls three lines of work into one framework. The framework is the
contribution; each scale rests on prior work.

- **Token PTS** is the Pivotal Token Search idea from the Phi-4 technical report
  [[2]](#references), turned from a standalone token method into one scale of a
  larger object.
- **Sentence PTS** is Thought Anchors [[3]](#references), the reasoning steps that
  matter, recast as the sentence scale of the same event.
- **Latent PTS** reads the hidden workspace with a Jacobian lens, following
  Anthropic's workspace work [[1]](#references). It is an independent
  reimplementation from the paper's published equations (no code was released),
  checked against a brute-force autograd Jacobian but not validated against the
  authors' own results. "Meta-token" is our term, not the paper's.

Put simply: what Phi-4 found in tokens and Thought Anchors found in sentences is
the same thing at different scales, and the workspace work describes where it
lives before it is emitted. PTS is the frame around all three. It is inspired by
and compatible with that work, not the same as it.

## Documentation

- [docs/latent_pts.md](docs/latent_pts.md): the J-lens and how latent search works
- [docs/dataset_schema.md](docs/dataset_schema.md): the event schema
- [docs/compatibility.md](docs/compatibility.md): reading older datasets

## Datasets

- [codelion/Qwen3-0.6B-pts](https://huggingface.co/datasets/codelion/Qwen3-0.6B-pts)
- [codelion/Qwen3-0.6B-pts-thought-anchors](https://huggingface.co/datasets/codelion/Qwen3-0.6B-pts-thought-anchors)
- [codelion/Qwen3-0.6B-pts-steering-vectors](https://huggingface.co/datasets/codelion/Qwen3-0.6B-pts-steering-vectors)
- [codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts](https://huggingface.co/datasets/codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts)

## References

1. Anthropic, *Verbalizable Representations Form a Global Workspace in Language
   Models* (2026). [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html)
2. Microsoft, *Phi-4 Technical Report* (2024), which introduces Pivotal Token
   Search. [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)
3. P. C. Bogdan, U. Macar, N. Nanda, A. Conmy, *Thought Anchors: Which LLM
   Reasoning Steps Matter?* (2025). [arXiv:2506.19143](https://arxiv.org/abs/2506.19143)

## Citation

```bibtex
@software{pts,
  title = {PTS: Pivotal Token Search},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/pts}
}
```
