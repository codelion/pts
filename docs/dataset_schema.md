# PTS dataset schema

Every PTS record is one `CausalReasoningEvent`, written as a line of JSONL.
The same schema carries all three scales, which is what lets a single file hold a
latent → token → sentence chain.

`schema_version` is `"1.0"`.

## Fields

### Identity

| Field | Type | Notes |
|---|---|---|
| `event_id` | string | Deterministic hash of the identifying parts. Re-migrating the same legacy record yields the same id, so links survive re-runs. |
| `event_type` | string | `latent_metatoken` \| `pivotal_token` \| `thought_anchor` |
| `granularity` | string | `latent` \| `token` \| `sentence` |
| `visibility` | string | `latent` \| `emitted` |

### Provenance

| Field | Type | Notes |
|---|---|---|
| `model_id` | string | **Records are specific to this model.** They do not transfer. |
| `dataset_id` | string? | Source dataset |
| `dataset_item_id` | string? | Item within it |
| `task_type` | string | `math`, `code`, `qa`, `generic`, … |
| `query` | string | The original problem |
| `timestamp` | string | ISO 8601 |

### Location

| Field | Type | Notes |
|---|---|---|
| `context` | string | The prefix before the event |
| `label` | string | The event itself: the token, the sentence, or the meta-token |
| `position` | int? | **Frame depends on the event (see below)** |
| `token_id` | int? | Vocabulary id, where one applies |
| `layer` | int? | Latent events only |
| `layer_name` | string? | Latent events only |

#### `position` is not one index frame

| Event | `position` means |
|---|---|
| `pivotal_token` | Absolute token index in the full sequence, **prompt included** |
| `latent_metatoken` from `pts run --granularity latent` (probe) | Absolute token index, prompt included. Same frame as token events. |
| `latent_metatoken` from `pts enrich` | A **negative offset from its source event**, e.g. `-3` = three tokens before it. Meaningful only against the event named in `metadata.source_event_id` |
| `thought_anchor` | **Sentence** index. Not comparable to either of the above. |

Comparing a sentence index against a token index is meaningless, and applying an
enrichment offset to any event other than its own source is meaningless. The
linker enforces both; if you compute temporal relationships yourself, you must
too.

### Scoring (read this carefully)

| Field | Type | Notes |
|---|---|---|
| `score` | float | The common currency. **Its meaning depends on the scale.** |
| `prob_before` | float? | P(success) without the event |
| `prob_after` | float? | P(success) with it |
| `prob_delta` | float? | `prob_after - prob_before` |
| `is_positive` | bool? | Whether the event helps |
| `confidence` | float? | Method-specific |

**`score` is not one quantity.**

- For `pivotal_token` and `thought_anchor` events, `score = abs(prob_delta)`. It
  is a measured causal effect on success probability.
- For `latent_metatoken` events, `score` is a **readout score**: the lens's
  probability for that vocabulary token. It is **not** a probability delta, it is
  **not** on the same scale, and it does **not** mean the concept caused
  anything.

`prob_before`, `prob_after`, `prob_delta`, and `is_positive` are **`null` on
every latent event**, because nothing measured them. That is deliberate. Do not
fill them in, and do not sort or threshold latent and emitted events together on
`score` as though the numbers were comparable. They are not.

The API enforces this rather than trusting you to remember it:

| Instead of | Use |
|---|---|
| one `min_score` across both scales | `filter(min_prob_delta=…)` for emitted, `filter(min_readout_score=…)` for latent |
| `most_important()` over everything | `most_important()` (emitted only) and `most_surfaced()` (latent only) |
| `summary()['average_score']` over everything | `average_abs_prob_delta` / `max_abs_prob_delta` (emitted) and `average_readout_score` / `max_readout_score` (latent) |

A single 0.5 floor across both scales keeps the banal meta-token `" the"`
(readout probability 0.92) and discards a pivotal token worth +0.45. That is not
a hypothetical. It is what the first implementation did.

### Method

| Field | Type | Notes |
|---|---|---|
| `search_method` | string | `token_pts` \| `sentence_pts` \| `latent_pts` \| `latent_pts_enrichment` |
| `intervention_type` | string? | `append_token` \| `replace_sentence` \| `remove_sentence` |
| `readout_method` | string? | Latent only: `jlens` \| `logit_lens`. **`logit_lens` is weaker evidence**, so filter on this. |

### Classification and links

| Field | Type | Notes |
|---|---|---|
| `category` | string? | Unified taxonomy, shared across all three scales |
| `tags` | string[] | |
| `parent_event_id` | string? | |
| `linked_event_ids` | string[] | |
| `precedes_event_ids` | string[] | |
| `follows_event_ids` | string[] | |
| `metadata` | object | Scale-specific extras; also where unknown fields from a future schema version are parked rather than dropped |

Links are **scored heuristics**, not verified causal paths. See
`docs/latent_pts.md`.

## Categories

One taxonomy for all scales:

`interpretation`, `planning`, `decomposition`, `fact_retrieval`, `computation`,
`verification`, `self_correction`, `backtracking`, `uncertainty`, `exploration`,
`consolidation`, `final_answer`, `tool_use`, `safety`, `deception_or_shortcut`,
`unknown`

Override the lexicon with `PTS_CATEGORY_LEXICON=/path/to/lexicon.json`
(`{category: [regex, ...]}`).

## Examples

### Token event

```json
{
  "schema_version": "2.0",
  "event_id": "token_evt_d1ac89c8fe71",
  "event_type": "pivotal_token",
  "granularity": "token",
  "visibility": "emitted",
  "model_id": "Qwen/Qwen3-0.6B",
  "query": "What is 17 * 24?",
  "context": "Let me compute 17 * 24.",
  "label": " Wait",
  "position": 73,
  "token_id": 13824,
  "prob_before": 0.35,
  "prob_after": 0.67,
  "prob_delta": 0.32,
  "score": 0.32,
  "is_positive": true,
  "search_method": "token_pts",
  "intervention_type": "append_token",
  "category": "self_correction",
  "linked_event_ids": ["latent_evt_81495cd5629b"],
  "follows_event_ids": ["latent_evt_81495cd5629b"],
  "metadata": {"pivot_token": " Wait", "prob_threshold": 0.2}
}
```

### Latent event

Note what is `null`.

```json
{
  "schema_version": "2.0",
  "event_id": "latent_evt_81495cd5629b",
  "event_type": "latent_metatoken",
  "granularity": "latent",
  "visibility": "latent",
  "model_id": "Qwen/Qwen3-0.6B",
  "query": "What is 17 * 24?",
  "context": "Let me compute 17 * 24.",
  "label": " verify",
  "position": -2,
  "layer": 18,
  "layer_name": "model.layers.18",
  "prob_before": null,
  "prob_after": null,
  "prob_delta": null,
  "is_positive": null,
  "score": 0.83,
  "search_method": "latent_pts_enrichment",
  "readout_method": "jlens",
  "category": "verification",
  "precedes_event_ids": ["token_evt_d1ac89c8fe71"],
  "metadata": {
    "j_score": 0.83,
    "rank": 2,
    "position_offset_from_linked_event": -2,
    "workspace_layers": [12, 16, 18, 20],
    "source_event_id": "token_evt_d1ac89c8fe71"
  }
}
```

### Sentence event

```json
{
  "schema_version": "2.0",
  "event_id": "sentence_evt_85fd5ca280a3",
  "event_type": "thought_anchor",
  "granularity": "sentence",
  "visibility": "emitted",
  "model_id": "Qwen/Qwen3-0.6B",
  "query": "What is 17 * 24?",
  "context": "Let me compute 17 * 24.",
  "label": "Wait, I should verify the arithmetic before finalizing.",
  "position": 5,
  "prob_before": 0.38,
  "prob_after": 0.72,
  "prob_delta": 0.34,
  "score": 0.34,
  "is_positive": true,
  "search_method": "sentence_pts",
  "intervention_type": "replace_sentence",
  "category": "verification",
  "parent_event_id": "token_evt_d1ac89c8fe71",
  "metadata": {
    "sentence_id": 5,
    "alternatives_tested": ["Alternatively, I can guess the answer."],
    "causal_dependencies": [3, 4]
  }
}
```

## Reading a dataset

```python
from pts import EventStorage

# Reads legacy pivotal-token and legacy thought-anchor files too, migrating on load.
storage = EventStorage(filepath="events.jsonl")

print(storage.summary())

latent = storage.by_event_type("latent_metatoken")

# Each scale on its own threshold.
strong = storage.filter(granularity="token", min_prob_delta=0.3, is_positive=True)
surfaced = storage.filter(granularity="latent", min_readout_score=0.1)

for tok in strong:
    preceded_by = [storage.get(i) for i in tok.follows_event_ids]
    print(tok.label, "<-", [e.label for e in preceded_by if e])
```
