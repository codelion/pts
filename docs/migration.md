# Migrating to PTS v2

PTS v2 generalizes Pivotal Token Search into a multiscale framework. The package
name, the CLI, and the datasets stay PTS; what changes is that pivotal tokens are
now one of three kinds of *causal reasoning event*.

## TL;DR

**Your existing JSONL datasets keep working.** Every PTS command reads v1
pivotal-token and v1 thought-anchor files directly and migrates them on load. You
do not have to convert anything.

```bash
# Old file, new tooling — just works.
pts export --input-path pivotal_tokens.jsonl --format dpo --output-path dpo.jsonl \
           --model Qwen/Qwen3-0.6B --dataset codelion/optillmbench --find-rejected-tokens
```

To convert explicitly:

```bash
pts migrate --input-path pivotal_tokens.jsonl --output-path events_v2.jsonl
pts migrate --input-path thought_anchors.jsonl --output-path events_v2.jsonl
```

And to get the old shapes back out of a v2 file:

```bash
pts export --format pivotal_tokens   --input-path events_v2.jsonl --output-path v1_tokens.jsonl
pts export --format thought_anchors  --input-path events_v2.jsonl --output-path v1_anchors.jsonl
```

The v1 → v2 → v1 round trip is exact for every field v1 defined; it is covered by
tests.

## Field mapping

### Pivotal tokens

| v1 | v2 |
|---|---|
| `pivot_token` | `label` |
| `pivot_context` | `context` |
| `pivot_token_id` | `token_id` |
| `prob_before` / `prob_after` / `prob_delta` | unchanged |
| `is_positive` | unchanged (now a field, not a method) |
| — | `event_type: "pivotal_token"`, `granularity: "token"`, `score: abs(prob_delta)` |

### Thought anchors

| v1 | v2 |
|---|---|
| `sentence` | `label` |
| `sentence_id` | `position` |
| `prefix_context` | `context` |
| `prob_with_sentence` | `prob_after` |
| `prob_without_sentence` | `prob_before` |
| `importance_score` | `score` |
| `sentence_category` | `category` (remapped, see below) |
| everything else (`suffix_context`, `alternatives_tested`, `causal_dependencies`, …) | `metadata.*` |

### Categories

v1's eight sentence categories now map onto one taxonomy shared by all three
scales, which is what makes cross-scale chains detectable at all:

| v1 | v2 |
|---|---|
| `problem_setup` | `interpretation` |
| `plan_generation` | `planning` |
| `fact_retrieval` | `fact_retrieval` |
| `active_computation` | `computation` |
| `uncertainty_management` | `uncertainty` |
| `result_consolidation` | `consolidation` |
| `self_checking` | `verification` |
| `final_answer_emission` | `final_answer` |

## Python API changes

Imports still resolve:

```python
from pts.core import PivotalToken, PivotalTokenSearcher      # works
from pts.thought_anchors import ThoughtAnchorSearcher        # works
from pts.storage import TokenStorage                         # works
```

But the objects are now unified events:

```python
tok = PivotalToken(query="...", pivot_context="...", pivot_token=" Wait", ...)
tok.label          # " Wait"     (was tok.pivot_token)
tok.context        # "..."       (was tok.pivot_context)
tok.is_positive    # True        (was tok.is_positive(), a method)

from pts.events import to_v1_pivotal_token
to_v1_pivotal_token(tok)   # the old dict shape, if you need it
```

`import pts` no longer pulls in torch or transformers. The event schema, storage,
classification, and linking layers are pure Python; anything that touches a model
loads lazily on first access.

## Bugs fixed in the move — these change results

Several v1 defects were silently producing wrong numbers. If you have datasets or
published results from v1, they may be affected.

**Thought anchors: probability deltas after a memory-cleanup trigger were
invalid.** Under memory pressure v1 blanked already-processed sentences in place
(`sentences[j] = ""`) while later iterations still read that list to build
prefixes. Every `prob_delta` computed after that point was measured against a
blank-padded prefix. These numbers were wrong, not merely noisy. v2 never mutates
the sentence list.

**Thought anchors: whole examples were silently dropped.** The same cleanup did
`del reasoning_trace` on what was actually the *function parameter*, guaranteeing
an `UnboundLocalError` on the next anchor. The CLI caught bare `Exception` and
logged one line, so the example vanished with no indication of why. v2 logs
tracebacks.

**DPO exports contained no positive tokens.** `--find-rejected-tokens` built its
searcher with a `DummyOracle`, which reports *every* completion as a success. So
every candidate token scored `P(success) = 1.0`, the acceptance test
(`prob_before - prob_after >= threshold`) could never fire, and every positive
pivotal token was dropped. v1 DPO datasets therefore contain only negative-delta
tokens, whose partner was chosen by next-token likelihood rather than by any
measured effect.

v2 refuses to do this. `--find-rejected-tokens` now needs a real oracle:

```bash
pts export --format dpo --find-rejected-tokens \
           --model Qwen/Qwen3-0.6B \
           --dataset codelion/optillmbench   # <-- rebuilds a real oracle
```

Without it, discovery is skipped with a warning rather than emitting pairs that
look measured but are not. Every pair carries `metadata.counterpart_verified`.

**Every dataset built through the CLI was doubled.** The searcher wrote each token
to storage and the CLI wrote it again. Storage is now idempotent by `event_id`.

**Cached probabilities leaked across categories.** The cache key omitted
`category`, which selects a *different prompt* — so two categories with the same
query returned each other's cached probability. The key now includes it.

**`PivotalToken.rejected_token` did not exist.** `generate_dpo_pairs` read it off
a dataclass that never declared the field, raising `AttributeError` on every
positive token. It lives in `metadata` now.

**Steering vectors read the wrong position under left-padding.** The extractor
indexed the last non-pad token assuming right-padding. It now checks
`tokenizer.padding_side`.

## What is *not* in v2

- **Latent interventions** (steering/ablating a meta-token and re-measuring
  success). Latent events are observational only. See `docs/latent_pts.md`.
- **Published v2 datasets.** The tooling is here; nothing has been pushed.
