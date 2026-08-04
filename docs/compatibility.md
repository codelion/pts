# Reading older datasets

PTS reads the pivotal-token and thought-anchor JSONL files that earlier versions
produced. You do not have to convert anything; every command reads them and maps
them to the event schema on the way in.

```bash
# An older pivotal-token file, read directly.
pts export --input-path pivotal_tokens.jsonl --format dpo --output-path dpo.jsonl \
           --model Qwen/Qwen3-0.6B --dataset codelion/optillmbench --find-rejected-tokens
```

To convert a file explicitly:

```bash
pts migrate --input-path pivotal_tokens.jsonl  --output-path events.jsonl
pts migrate --input-path thought_anchors.jsonl --output-path events.jsonl
```

To go the other way and get the older shapes back out:

```bash
pts export --format pivotal_tokens  --input-path events.jsonl --output-path tokens.jsonl
pts export --format thought_anchors --input-path events.jsonl --output-path anchors.jsonl
```

The round trip keeps every field the older formats defined. Tests cover it.

## How older fields map to the schema

### Pivotal tokens

| older field | event field |
|---|---|
| `pivot_token` | `label` |
| `pivot_context` | `context` |
| `pivot_token_id` | `token_id` |
| `prob_before` / `prob_after` / `prob_delta` | unchanged |
| `is_positive` | unchanged (now a field, not a method) |
| | plus `event_type: "pivotal_token"`, `granularity: "token"`, `score: abs(prob_delta)` |

### Thought anchors

| older field | event field |
|---|---|
| `sentence` | `label` |
| `sentence_id` | `position` |
| `prefix_context` | `context` |
| `prob_with_sentence` | `prob_after` |
| `prob_without_sentence` | `prob_before` |
| `importance_score` | `score` |
| `sentence_category` | `category` (remapped, below) |
| everything else (`suffix_context`, `alternatives_tested`, `causal_dependencies`, …) | `metadata` |

### Categories

The older eight-category sentence vocabulary maps to the one taxonomy shared by
all three scales, which is what lets cross-scale chains line up:

| older category | category |
|---|---|
| `problem_setup` | `interpretation` |
| `plan_generation` | `planning` |
| `fact_retrieval` | `fact_retrieval` |
| `active_computation` | `computation` |
| `uncertainty_management` | `uncertainty` |
| `result_consolidation` | `consolidation` |
| `self_checking` | `verification` |
| `final_answer_emission` | `final_answer` |

## Python API

Older imports still work:

```python
from pts.core import PivotalToken, PivotalTokenSearcher
from pts.thought_anchors import ThoughtAnchorSearcher
from pts.storage import TokenStorage
```

The objects they return are now events:

```python
tok = PivotalToken(query="...", pivot_context="...", pivot_token=" Wait", ...)
tok.label          # " Wait"     (was tok.pivot_token)
tok.context        # "..."       (was tok.pivot_context)
tok.is_positive    # True        (was tok.is_positive(), a method)

from pts.events import to_legacy_pivotal_token
to_legacy_pivotal_token(tok)   # the older dict shape, if you need it
```

`import pts` does not load torch or transformers. The schema, storage,
classification, and linking layers are plain Python; model code loads only when
you use it.
