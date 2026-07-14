# Latent PTS

Latent PTS searches the model's hidden workspace for pivotal reasoning events —
verbalizable concepts that are active in the residual stream but not (yet)
emitted as tokens.

This is the newest and least settled part of PTS. Read the caveats before you
believe anything it tells you.

## The idea

Anthropic's [*Verbalizable Representations Form a Global Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html) describes a
**Jacobian lens** (J-lens): a linear readout that, given a mid-layer activation,
recovers the vocabulary tokens that activation is causally pushing the model to
say *later*. The set of concepts the lens surfaces is the **J-space**, and it
occupies roughly the middle band of the network — about 38% to 92% of depth.

PTS uses that readout as a *search primitive*. If a pivotal token like `" Wait"`
flips a generation from failure to success, the question Latent PTS asks is:

> Was the model already "thinking" `verify` a few tokens earlier, in a form it
> had not yet said out loud?

Each token the lens surfaces above a score floor becomes a `latent_metatoken`
event, linked to the emitted event it precedes.

## The math

For layer `l`, the J-lens is the expected Jacobian of the final residual stream
with respect to the layer-`l` residual stream:

```
J_l = E[ ∂h_final,t' / ∂h_l,t ]
```

averaged over source positions `t`, all later positions `t' ≥ t`, and a set of
calibration prompts. Reading a vocabulary distribution out of an activation is
then:

```
lens(h_l) = softmax( W_U · norm( J_l @ h_l ) )
```

The rows of `W_U @ J_l` are the **J-lens vectors** — one direction per
vocabulary token, each the average causal influence of that direction on
eventually producing that token.

### Why fitting is affordable

A naive Jacobian would need one backward pass per (source position, output
component) pair. Two facts collapse that:

1. **Attention is causal**, so `h_l,t` cannot influence `h_final,t'` for
   `t' < t`. The gradient of the *total* `S_i = Σ_t' (h_final,t')_i` with respect
   to `h_l,t` therefore already equals the sum over exactly the `t' ≥ t` terms —
   the rest are structurally zero.
2. **Autograd returns the gradient with respect to every source position at
   once.**

So one backward pass per output component `i` yields row `i` of the summed
Jacobian for *all* source positions simultaneously. That gives the paper's stated
cost of `O(n × d_model)` backward passes. Dividing row `t` by the number of
`t' ≥ t` converts the sum into the mean the definition asks for.

This is verified against a brute-force `torch.autograd.functional.jacobian` in
`tests/test_jlens.py` — the two agree to ~1e-8, and the causal-mask assumption is
checked directly.

### The logit lens is J = I

Setting `J = I` recovers the **logit lens**: it asks what an activation would
emit *right now* if unembedded directly, rather than what it is pushing the model
to emit *later*. That makes it a principled zero-cost baseline rather than a
hack, and it is exactly what `--readout-method logit_lens` computes. It needs no
calibration, so it is the right way to try Latent PTS before committing to a fit.

It is also **weaker evidence**. A workspace claim is a claim about future
influence; the logit lens does not measure future influence. Events read this way
are tagged `readout_method: "logit_lens"` so they can be filtered out.

## Usage

### 1. Fit a J-lens (once per model)

```bash
pts fit-jlens \
  --model Qwen/Qwen3-0.6B \
  --output-path ./jlens/qwen3-0.6b \
  --num-sequences 25 \
  --seq-len 128
```

The paper averages over ~1000 prompts but shows n=10–25 is nearly as good, which
is why 25 is the default. Calibration text just needs to be generic — the lens is
a property of the model, not the task — so the source dataset works fine and no
labels are needed. Pass `--calibration-file` for your own text, one sequence per
line.

Lower `--basis-chunk` if you hit OOM; it controls how many output basis
directions are differentiated per batched backward call.

### 2. Enrich an existing PTS dataset

This is the path that matters: it reuses curated token/sentence datasets instead
of re-running the full search.

```bash
pts enrich \
  --input-path pivotal_tokens.jsonl \
  --output-path pivotal_tokens_latent.jsonl \
  --model Qwen/Qwen3-0.6B \
  --jlens-path ./jlens/qwen3-0.6b \
  --readout-method jlens \
  --with-latent \
  --window-before 8 \
  --shuffle-control
```

v1 files are migrated on read, so you can point this straight at an old dataset.

### 3. Or search all three scales at once

```bash
pts run --granularity all --model Qwen/Qwen3-0.6B \
        --readout-method jlens --jlens-path ./jlens/qwen3-0.6b \
        --output-path events.jsonl
```

## Caveats

These are not boilerplate. Each one is a way the output can mislead you.

**Latent events are hypotheses, not measurements.** A latent event's `score` is a
readout score — how strongly the lens surfaces that token — **not** a probability
delta. It is not comparable to the `prob_delta` on token and sentence events.
`prob_delta` and `is_positive` are deliberately `null` on every latent event, and
should stay that way until an intervention actually measures one.

**Latent events are observational.** Enrichment and probing report what the lens
*sees*. Neither establishes that a meta-token *caused* the emitted event. Showing
causation requires steering or ablating the direction and re-measuring success —
that is Phase 7 and is not implemented.

**Readouts are noisy.** Neither lens is guaranteed faithful to what the model
actually represents. A high-scoring `verify` meta-token may be an artifact of the
unembedding geometry rather than evidence of a "verification" concept.

**This is an independent reimplementation.** No reference code was released with
the paper. The math here is written from the published equations and validated
for internal correctness (it computes the Jacobian it claims to), but it has
**not** been validated against the authors' results. Numbers from PTS should not
be reported as reproducing theirs.

**"Meta-token" is our term, not the paper's.** The paper does not define it. We
use it to mean "a top-k J-lens readout token". Do not cite it as Anthropic
terminology.

**Links are scored guesses.** `linked_event_ids` come from a weighted heuristic
(query match, context overlap, category agreement, temporal proximity), not from
a verified causal path. Always run `--shuffle-control`: it scores links against
mismatched queries, and if the observed mean is not clearly above the shuffled
mean, the structure you are looking at is not distinguishable from chance.

**PTS records are model-specific.** A pivotal token found in one model's
generation says nothing about another model's workspace. Enriching events with a
different model is refused unless you pass `--allow-model-mismatch`, and even
then both model ids are recorded on every latent event so the mismatch stays
visible.

## What would make this convincing

The framework exists to test one claim:

> Many emitted pivotal tokens and thought-anchor sentences are preceded by latent
> verbalizable meta-tokens in the model's workspace.

Evidence that would support it, in rough order of strength:

1. **Intervention.** Steer or ablate the meta-token direction and show success
   probability moves. Not implemented.
2. **Lead time with a control.** Show that a category-matching meta-token appears
   `k` tokens before the emitted event *more often than chance*, using
   `--shuffle-control` as the null.
3. **Chain consistency.** Show `latent: verification → token: " Wait" → sentence:
   "Let me check..."` holds across many queries, not anecdotally.
4. **J-lens beats logit-lens.** If the effect is equally strong with
   `--readout-method logit_lens`, it is not about future influence and therefore
   not about a workspace.

Until at least (2) is done with a control, treat the output as exploratory.
