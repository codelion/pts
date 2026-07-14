# PTS: Pivotal Token/Thought Search

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/codelion/pts-visualizer)

**PTS is a multiscale causal-event search framework for model reasoning.** It
identifies the hidden workspace states, emitted tokens, and sentence-level
reasoning steps that causally shift a model's probability of solving a task.

PTS started as Pivotal Token Search: finding emitted tokens that significantly
change a model's chance of getting the answer right. In v2, PTS generalizes into
a unified mechanistic interpretability framework that searches for pivotal
reasoning events at **three representational scales**.

```
latent meta-token / workspace event        <- Latent PTS
        |
emitted pivotal token                      <- Token PTS
        |
sentence-level thought anchor              <- Sentence PTS
        |
success / failure probability shift
```

All three are the same kind of object, scored by the same principle:

```
event_importance = outcome_with_event - outcome_without_or_altered_event
```

## The three scales

| Scale | What it finds | How it's scored | Prior work |
|---|---|---|---|
| **Latent PTS** | Verbalizable concepts active in the model's mid-layer workspace but not yet said out loud | J-lens readout score | Anthropic's [J-space / J-lens](https://transformer-circuits.pub/2026/workspace/index.html) |
| **Token PTS** | Emitted tokens that flip success probability | `P(success \| prefix + token) - P(success \| prefix)` | [Phi-4 technical report](https://arxiv.org/abs/2412.08905) |
| **Sentence PTS** | Reasoning sentences that flip success probability | `P(success \| prefix + sentence) - P(success \| prefix + alternative)` | Thought Anchors |

The claim the framework exists to test:

> Many emitted pivotal tokens and thought-anchor sentences are preceded by latent
> verbalizable meta-tokens in the model's workspace.

**Latent events are observational hypotheses, not measured causal effects.** See
[Caveats](#caveats) — this matters and it is easy to get wrong.

## Installation

```bash
git clone https://github.com/codelion/pts.git
cd pts
pip install -e .
```

`import pts` does not pull in torch or transformers. The event schema, storage,
classification, and linking layers are pure Python; anything that touches a model
loads lazily.

## Quick start

### Token PTS (the original)

```bash
pts run --granularity token \
        --model Qwen/Qwen3-0.6B \
        --dataset codelion/optillmbench \
        --output-path events.jsonl
```

### Sentence PTS (thought anchors)

```bash
pts run --granularity sentence --model Qwen/Qwen3-0.6B --output-path events.jsonl
```

### Latent PTS — enrich a dataset you already have

This is the highest-value path: it reuses curated PTS datasets instead of
re-running the search from scratch. **v1 files work directly** — they are migrated
on read.

```bash
# 1. Calibrate a Jacobian lens for the model (once).
pts fit-jlens --model Qwen/Qwen3-0.6B --output-path ./jlens/qwen3-0.6b

# 2. Read the workspace around every existing event.
pts enrich --input-path pivotal_tokens.jsonl \
           --output-path events_latent.jsonl \
           --model Qwen/Qwen3-0.6B \
           --jlens-path ./jlens/qwen3-0.6b \
           --readout-method jlens \
           --with-latent --shuffle-control
```

No J-lens yet? `--readout-method logit_lens` needs no calibration and works
immediately — it is the same construction with `J = I`. It is also weaker
evidence; see [docs/latent_pts.md](docs/latent_pts.md).

### All three scales at once

```bash
pts run --granularity all \
        --model Qwen/Qwen3-0.6B \
        --dataset codelion/optillmbench \
        --readout-method jlens --jlens-path ./jlens/qwen3-0.6b \
        --output-path events.jsonl
```

### Visualize

```bash
cd visualizer && python app.py
```

Or use the [hosted visualizer](https://huggingface.co/spaces/codelion/pts-visualizer).

## Commands

| Command | Does |
|---|---|
| `pts run --granularity token\|sentence\|latent\|all` | Search for pivotal events |
| `pts enrich --with-latent` | Add latent meta-token events to an existing dataset |
| `pts fit-jlens` | Calibrate a Jacobian lens for a model |
| `pts link` | Link latent → token → sentence into causal chains |
| `pts migrate` | Convert v1 files to the v2 event schema |
| `pts export --format …` | `causal_events`, `metatokens`, `pivotal_tokens`, `thought_anchors`, `dpo`, `steering` |
| `pts push` | Upload to Hugging Face |

## Downstream uses

```bash
# DPO pairs. --dataset is required so a REAL success oracle can be rebuilt.
pts export --input-path events.jsonl --format dpo --output-path dpo.jsonl \
           --model Qwen/Qwen3-0.6B --dataset codelion/optillmbench --find-rejected-tokens

# Steering vectors (works with OptiLLM's autothink).
pts export --input-path events.jsonl --format steering --output-path steering.jsonl \
           --model Qwen/Qwen3-0.6B
```

## Upgrading from v1

**Your existing datasets keep working.** Every command reads v1 pivotal-token and
thought-anchor JSONL directly.

Several v1 bugs were silently producing **wrong numbers**, and fixing them changes
results. If you have v1 datasets or published results, read
[docs/migration.md](docs/migration.md) — in particular: v1 DPO exports contained
no positive tokens at all (the rejected-token search ran against a dummy oracle
that reported every completion as a success), and thought-anchor probability
deltas were invalid after a memory-cleanup path blanked the sentences it was
still reading.

## Caveats

Latent PTS is the newest and least settled part of this. Do not oversell it.

- **Latent events are hypotheses, not measurements.** A latent event's `score` is
  a readout score, **not** a probability delta. `prob_delta` and `is_positive` are
  `null` on every latent event, deliberately. Do not compare or threshold latent
  and emitted scores together — they are not on the same scale.
- **Latent events are observational.** Nothing was intervened on. Showing a
  meta-token *causes* an emitted event requires steering or ablating it and
  re-measuring success. Not implemented.
- **Readouts are noisy.** Neither the J-lens nor the logit lens is guaranteed
  faithful to what the model represents.
- **This is an independent reimplementation.** No reference code was released with
  the workspace paper. The J-lens here is written from the published equations and
  verified for internal correctness against a brute-force autograd Jacobian, but it
  has **not** been validated against the authors' results.
- **"Meta-token" is our term, not the paper's.**
- **Links are scored guesses.** Always run `--shuffle-control`: if the observed
  link scores are not clearly above the shuffled baseline, the structure is not
  distinguishable from chance.
- **PTS records are model-specific.** A token pivotal for one model says nothing
  about another.

## Relation to prior work

PTS was originally inspired by the Pivotal Token Search idea in the
[Phi-4 technical report](https://arxiv.org/abs/2412.08905). This project extends
it into a multiscale mechanistic interpretability framework. **Token PTS**
corresponds to emitted pivotal tokens. **Sentence PTS** corresponds to
thought-anchor-style reasoning steps. **Latent PTS** uses workspace / J-space-style
readouts, inspired by Anthropic's
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html),
to search for hidden verbalizable meta-tokens that may precede emitted pivotal
tokens.

Inspired by, related to, and compatible with — not the same as, and not validated
against.

## Documentation

- [docs/latent_pts.md](docs/latent_pts.md) — the J-lens, the math, and what would make it convincing
- [docs/dataset_schema_v2.md](docs/dataset_schema_v2.md) — the unified event schema
- [docs/migration.md](docs/migration.md) — upgrading from v1, and the bugs that changed results

## Datasets

Existing v1 datasets on Hugging Face (all still load):

- [codelion/Qwen3-0.6B-pts](https://huggingface.co/datasets/codelion/Qwen3-0.6B-pts)
- [codelion/Qwen3-0.6B-pts-thought-anchors](https://huggingface.co/datasets/codelion/Qwen3-0.6B-pts-thought-anchors)
- [codelion/Qwen3-0.6B-pts-steering-vectors](https://huggingface.co/datasets/codelion/Qwen3-0.6B-pts-steering-vectors)
- [codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts](https://huggingface.co/datasets/codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts)

## License

Apache 2.0
