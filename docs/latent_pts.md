# Latent PTS

Latent PTS looks inside the model for reasoning events: concepts that are active
in the residual stream but have not been emitted as tokens yet. It is the newest
part of PTS and the least settled, so read the notes at the bottom before you
lean on the output.

## The idea

Anthropic's [*Verbalizable Representations Form a Global Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html) describes a
Jacobian lens (J-lens): a linear readout that takes a mid-layer activation and
recovers the vocabulary tokens that activation is pushing the model to say later.
The set of concepts it surfaces is the J-space, which sits in the middle band of
the network, roughly 38% to 92% of the way through.

PTS uses that readout as a search tool. If a pivotal token like `" Wait"` turns a
failed answer into a correct one, Latent PTS asks: was the model already leaning
toward `verify` a few tokens earlier, before it said anything?

Each token the lens surfaces above a score floor becomes a `latent_metatoken`
event, linked to the emitted event it comes before.

## The math

For layer `l`, the J-lens is the expected Jacobian of the final residual stream
with respect to the layer-`l` residual stream:

```
J_l = E[ d h_final,t' / d h_l,t ]
```

averaged over source positions `t`, all later positions `t' >= t`, and a set of
calibration prompts. To read a vocabulary distribution out of an activation:

```
lens(h_l) = softmax( W_U . norm( J_l @ h_l ) )
```

The rows of `W_U @ J_l` are the J-lens vectors, one direction per vocabulary
token, each the average causal pull of that direction on eventually producing
that token.

### Why fitting is cheap

A naive Jacobian would need one backward pass per (source position, output
component) pair. Two facts cut that down:

1. Attention is causal, so `h_l,t` cannot affect `h_final,t'` when `t' < t`. The
   gradient of the summed final stream `S_i = sum_t' (h_final,t')_i` with respect
   to `h_l,t` is therefore already the sum over exactly the `t' >= t` terms; the
   rest are zero.
2. Autograd returns the gradient with respect to every source position at once.

So one backward pass per output component `i` gives row `i` of the summed
Jacobian for all source positions at the same time. That is the paper's stated
cost of `O(n x d_model)` backward passes. Dividing row `t` by the count of
`t' >= t` turns the sum into the mean the definition asks for.

`tests/test_jlens.py` checks this against a brute-force
`torch.autograd.functional.jacobian`. The two agree to about 1e-8, and a separate
test checks the causal-mask assumption directly.

### The logit lens is J = I

Set `J = I` and you get the logit lens: what an activation would emit right now if
unembedded directly, rather than what it pushes the model to say later. It needs
no calibration, so it is a good way to try Latent PTS before you commit to a fit.
It is also a weaker signal. A workspace claim is a claim about future influence,
and the logit lens does not measure future influence. Events read this way are
tagged `readout_method: "logit_lens"` so you can filter them out.

## Usage

### 1. Fit a J-lens (once per model)

```bash
pts fit-jlens \
  --model Qwen/Qwen3-0.6B \
  --output-path ./jlens/qwen3-0.6b \
  --num-sequences 25 \
  --seq-len 128
```

The paper averages over about 1000 prompts but shows 10 to 25 is nearly as good,
which is why 25 is the default. Calibration text only needs to be generic, since
the lens is a property of the model rather than the task, so the source dataset
works fine with no labels. Pass `--calibration-file` for your own text, one
sequence per line. Lower `--basis-chunk` if you run out of memory.

### 2. Enrich a dataset you already have

This is the main path. It reuses curated token and sentence datasets instead of
searching from scratch, and older files are read directly.

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

### 3. Or search all three scales at once

```bash
pts run --granularity all --model Qwen/Qwen3-0.6B \
        --readout-method jlens --jlens-path ./jlens/qwen3-0.6b \
        --output-path events.jsonl
```

## Notes

- A latent event's `score` is a readout probability, not a Δ-probability. It is
  not comparable to the `prob_delta` on token and sentence events, which is why
  `prob_delta` and `is_positive` are `null` on latent events.
- Latent events are observational. The lens shows what a meta-token leans toward,
  not that it caused anything downstream. A causal claim would need an
  intervention (steer or ablate, then re-measure).
- Run `--shuffle-control`. If the link scores do not beat the shuffled baseline,
  the structure is not above chance.
- Records are model-specific. Enriching with a different model needs
  `--allow-model-mismatch`.
