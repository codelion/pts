# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

PTS is a **multiscale causal-event search framework for model reasoning**. It
finds pivotal reasoning events at three representational scales and scores them
all by their effect on the probability of solving the task:

```
latent meta-token / workspace event   (Latent PTS)    <- J-lens readout
        v
emitted pivotal token                 (Token PTS)     <- Phi-4 PTS
        v
sentence-level thought anchor         (Sentence PTS)  <- Thought Anchors
        v
success / failure probability shift
```

All three are `CausalReasoningEvent` records. The name stays PTS ("Pivotal
Token/Thought Search"); v2 generalizes it rather than replacing it.

## The invariant that matters most

**A latent event's `score` is a J-lens readout probability. It is NOT a
probability delta.**

- Latent events have `prob_delta`, `prob_before`, `prob_after`, and
  `is_positive` set to `None`, deliberately. Do not fill them in.
- Never sort, threshold, compare, or histogram latent `score` together with
  emitted `prob_delta` as though they were the same quantity. The API is built so
  you cannot do it by accident:
  - `EventStorage.filter` has **no `min_score`** — use `min_prob_delta` (emitted)
    or `min_readout_score` (latent). Each ignores the other scale.
  - `most_important()` is emitted-only; `most_surfaced()` is latent-only.
  - `summary()` reports `*_prob_delta` and `*_readout_score` separately and has
    no aggregate mean over `score`.
  - `export_causal_events` takes two thresholds.

  A single 0.5 floor across both scales keeps the filler meta-token `" the"`
  (readout 0.92) and throws away a pivotal token worth +0.45. This has already
  been shipped once by accident; do not reintroduce it.
- Never count latent events as positive or negative — they have no valence.
- `logit_lens` readouts are weaker evidence than `jlens` ones; keep
  `readout_method` visible so they can be filtered apart.
- Latent events are **observational**. Nothing intervened on anything. Do not
  write copy that implies a meta-token *caused* a downstream event.

Breaking any of these turns an interpretability hypothesis into a false claim,
which is the main way this project can mislead people. Tests in
`tests/test_events.py` and `tests/test_exporters.py` guard it.

## Commands

```bash
pip install -e .           # core + model deps
pip install -e '.[all]'    # + sentence-transformers, math-verify, pytest

pytest tests/ -q           # 66 tests, ~15s, uses a tiny random model

pts run --granularity token|sentence|latent|all --model M --output-path events.jsonl
pts fit-jlens --model M --output-path ./jlens/m          # calibrate the Jacobian lens
pts enrich --input-path events.jsonl --with-latent --model M --jlens-path ./jlens/m
pts link --input-path events.jsonl --output-path linked.jsonl --shuffle-control
pts migrate --input-path v1.jsonl --output-path v2.jsonl
pts export --format causal_events|metatokens|pivotal_tokens|thought_anchors|dpo|steering
pts push --input-path X --hf-repo-id user/repo
```

`--generate-thought-anchors` still works as an alias for `--granularity sentence`.

## Architecture

| Module | Role |
|---|---|
| `pts/events.py` | `CausalReasoningEvent` + factories + v1 migration/round-trip. Pure Python. |
| `pts/event_storage.py` | `EventStorage` (JSONL, dedupes by `event_id`). Reads v1 files directly. |
| `pts/classification.py` | One category taxonomy across all three scales. |
| `pts/linking.py` | Latent -> token -> sentence links + `shuffle_control` null. |
| `pts/latent/jlens.py` | The Jacobian lens. `logit_lens` is the same thing with `J = I`. |
| `pts/latent/activations.py` | Residual capture, architecture sniffing, workspace-layer selection. |
| `pts/latent/metatokens.py` | Readouts -> latent events; dataset enrichment. |
| `pts/searchers/base.py` | Model loading, prompt formatting, the probability cache. |
| `pts/searchers/{token,sentence,latent,multiscale}.py` | The four searchers. |
| `pts/oracle.py`, `pts/dataset.py` | Success evaluation and dataset loading (largely unchanged). |
| `pts/exporters.py` | All output formats + dataset cards. |
| `pts/core.py`, `pts/storage.py`, `pts/thought_anchors.py` | v1 compatibility shims. |

`import pts` must **not** require torch or transformers. The schema, storage,
classification, and linking layers are pure Python; model-touching code is
imported lazily via `__getattr__`. Keep it that way.

## The J-lens

`J_l = E[∂h_final,t' / ∂h_l,t]`, averaged over source positions `t`, all later
positions `t' >= t`, and calibration prompts. Readout is
`softmax(W_U · norm(J_l @ h_l))`.

Fitting exploits **causal attention**: `h_l,t` cannot influence `h_final,t'` for
`t' < t`, so the gradient of the summed final stream w.r.t. `h_l` already
restricts to `t' >= t`, and autograd returns it for every source position at
once. One backward pass per output component therefore yields a whole Jacobian
row — the paper's `O(n × d_model)` cost.

**This shortcut is load-bearing.** If it is wrong, every latent event is garbage.
`tests/test_jlens.py::test_fast_jacobian_matches_brute_force` checks it against
`torch.autograd.functional.jacobian`, and `test_causal_mask_holds` checks the
assumption directly. Do not weaken those tests.

No reference code was released with the workspace paper — this is written from
the equations and has **not** been validated against the authors' results. Say so
when writing docs. "Meta-token" is our term, not the paper's.

## Things that will bite you

- **OptiLLM compatibility.** Steering exports carry `reasoning_pattern` from a
  fixed five-value vocabulary (`depth_and_thoroughness`, `numerical_accuracy`,
  `self_correction`, `exploration`, `organization`). OptiLLM's autothink reads
  that field. Do **not** replace it with the PTS category taxonomy — both are
  emitted (`reasoning_pattern` and `category`). Pinned by a test.
- **DPO needs a real oracle.** `--find-rejected-tokens` must never run against
  `DummyOracle` (it calls every completion a success, so no rejected token can
  ever be found). It requires `--dataset` and refuses rather than fabricating
  pairs. This was a v1 bug that silently emptied every DPO dataset of positive
  tokens.
- **The probability cache key must include `category`**, which selects a
  different prompt.
- **Token PTS's prefix *replaces* the prompt** (it is a fully-decoded string that
  already contains it); **Sentence PTS's prefix *appends*.** See
  `build_conditioning_text`.
- Published PTS datasets are **model-specific**. Cross-model enrichment requires
  `--allow-model-mismatch` and is exploratory.
- **A latent event's identity includes its `context` and `source_event_id`.** Two
  pivotal tokens in one query, enriched at the same layer and offset, routinely
  surface the same common meta-token. Drop those from the id and they collide,
  storage silently discards one, and the survivor keeps a link pointing at the
  discarded id. At 20 source events per query this destroyed ~95% of the latent
  events.
- **Auto workspace layers must stay below `num_layers - 1`.** `round(0.92 * n)`
  lands on `n-1` for every `n <= 18`, and `JLens.fit` rejects the final layer — so
  a naive bound makes `fit-jlens` crash on GPT-2, Llama-3.2-1B, and every other
  small model.
- **`position` is not one index frame.** Token events and probe latent events use
  absolute prompt-inclusive token indices; enrichment latent events use a negative
  offset from their source; sentence events use sentence indices. Never compare
  across frames. See `docs/dataset_schema_v2.md`.
- `event.context` already contains its special tokens (it was decoded with
  `skip_special_tokens=False`), so re-encode it with `add_special_tokens=False` or
  you get a double BOS and score candidates on a prefix the model never saw.

## Testing

No GPU needed: tests use `hf-internal-testing/tiny-random-LlamaForCausalLM`. That
model has **random weights**, so its readouts are meaningless by construction —
tests assert on structure and math, never on whether a meta-token looks sensible.
