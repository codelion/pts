"""
Rendering unified events into downstream formats.

``causal_events``    the full v2 event stream (the canonical format)
``metatokens``       latent events only
``pivotal_tokens``   the v1 token shape, for consumers of the old datasets
``thought_anchors``  the v1 sentence shape
``dpo``              preference pairs from token events
``steering``         activation vectors clustered by reasoning pattern

Two things here are load-bearing for downstream consumers and were kept
deliberately:

* The steering export's ``reasoning_pattern`` values come from a fixed
  five-bucket vocabulary (``depth_and_thoroughness``, ``numerical_accuracy``,
  ``self_correction``, ``exploration``, ``organization``). OptiLLM's autothink
  reads that field, so it is *not* replaced by the unified category taxonomy.
  Both are emitted: ``reasoning_pattern`` for OptiLLM, ``category`` for PTS.
* The dataset cards keep the OptiLLM integration notes and the DPO Colab link.

The DPO path is the one that changed most. See ``export_dpo``.
"""

import json
import logging
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .event_storage import EventStorage
from .events import (
    CausalReasoningEvent,
    EVENT_LATENT,
    EVENT_SENTENCE,
    EVENT_TOKEN,
    to_v1_pivotal_token,
    to_v1_thought_anchor,
)

logger = logging.getLogger(__name__)


# The vocabulary OptiLLM's autothink expects on steering-vector records. Do not
# rename these without coordinating with optillm.
OPTILLM_REASONING_PATTERNS: Dict[str, List[str]] = {
    "depth_and_thoroughness": [
        "therefore", "alternatively", "however", "wait", "let's", "so", "think",
        "analyze", "additional", "furthermore", "moreover", "deeper", "detailed",
        "comprehensive", "examine", "investigate", "explore", "consider",
        "important", "significant", "critical", "careful", "precise", "nuanced",
        "full", "complete", "exhaustive", "rigorous",
    ],
    "numerical_accuracy": [
        "calculate", "compute", "equation", "correct", "check", "verify", "math",
        "number", "calculation", "result", "answer", "precision", "formula",
        "computation", "sum", "total", "value", "exact", "accurate", "integer",
        "decimal", "fraction", "multiply", "divide",
    ],
    "self_correction": [
        "mistake", "incorrect", "wrong", "error", "let me reconsider", "actually",
        "revise", "correction", "revising", "mistaken", "fix", "adjust", "rectify",
        "amend", "correct", "misunderstood", "misinterpreted", "miscalculated",
    ],
    "exploration": [
        "alternative", "approach", "method", "strategy", "consider", "explore",
        "possibility", "different", "solution", "examine", "investigate", "option",
        "alternatives", "pathway", "direction", "route", "perspective", "viewpoint",
    ],
    "organization": [
        "first", "second", "next", "finally", "step", "organize", "list", "sequence",
        "order", "structure", "outline", "categorize", "classify", "group", "arrange",
        "prioritize", "rank", "sort", "divide", "section", "segment",
    ],
}


def detect_file_type(path: str) -> str:
    """Guess what a JSONL file holds, from its first record."""
    try:
        with open(path) as f:
            first = f.readline().strip()
        if not first:
            return "causal_events"
        record = json.loads(first)
    except (OSError, json.JSONDecodeError):
        return "causal_events"

    if "chosen" in record and "rejected" in record:
        return "dpo"
    if "steering_vector" in record:
        return "steering"
    if record.get("event_type") == EVENT_LATENT:
        return "metatokens"
    if "event_type" in record and "granularity" in record:
        return "causal_events"
    if "sentence" in record and "sentence_id" in record:
        return "thought_anchors"
    if "pivot_token" in record:
        return "pivotal_tokens"
    return "causal_events"


class EventExporter:
    """Export unified events into downstream formats."""

    def __init__(self, storage: Optional[EventStorage] = None):
        self.storage = storage if storage is not None else EventStorage()

    def _write(self, path: str, records: Sequence[Dict[str, Any]]) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        logger.info(f"Wrote {len(records)} records to {path}")

    # -- v2-native ---------------------------------------------------------

    def export_causal_events(
        self,
        output_path: str,
        min_prob_delta: float = 0.0,
        min_score: float = 0.0,
    ) -> None:
        """Export the full event stream.

        Emitted and latent events are filtered by *different* thresholds on
        purpose. A token event's score is a probability delta; a latent event's
        score is a readout probability over the vocabulary. They live on
        different scales, so applying one threshold to both silently deletes
        every latent event (readout scores are routinely well below a 0.1
        prob-delta floor) while looking like a principled filter.
        """
        events = []
        for e in self.storage:
            if e.event_type == EVENT_LATENT:
                if e.score >= min_score:
                    events.append(e)
            elif e.prob_delta is None or abs(e.prob_delta) >= min_prob_delta:
                events.append(e)
        self._write(output_path, [e.to_dict() for e in events])

    def export_metatokens(self, output_path: str, min_score: float = 0.0) -> None:
        events = [
            e for e in self.storage
            if e.event_type == EVENT_LATENT and e.score >= min_score
        ]
        if not events:
            logger.warning(
                "No latent events in this dataset. Add them with `pts enrich --with-latent`."
            )
        self._write(output_path, [e.to_dict() for e in events])

    # -- v1-shaped views ---------------------------------------------------

    def export_pivotal_tokens(self, output_path: str, min_prob_delta: float = 0.0) -> None:
        events = [
            e for e in self.storage
            if e.event_type == EVENT_TOKEN
            and e.prob_delta is not None
            and abs(e.prob_delta) >= min_prob_delta
        ]
        self._write(output_path, [to_v1_pivotal_token(e) for e in events])

    def export_thought_anchors(self, output_path: str, min_prob_delta: float = 0.0) -> None:
        events = [
            e for e in self.storage
            if e.event_type == EVENT_SENTENCE
            and e.prob_delta is not None
            and abs(e.prob_delta) >= min_prob_delta
        ]
        self._write(output_path, [to_v1_thought_anchor(e) for e in events])

    # -- DPO ---------------------------------------------------------------

    def export_dpo(
        self,
        output_path: str,
        model_name: Optional[str] = None,
        dataset: Optional[str] = None,
        split: str = "train",
        find_rejected_tokens: bool = False,
        num_candidates: int = 10,
        min_prob_delta: float = 0.1,
        max_pairs: Optional[int] = None,
        balance: bool = False,
        seed: int = 42,
        device: Optional[str] = None,
        oracle: Optional[Any] = None,
    ) -> None:
        """Build DPO preference pairs from token events.

        A positive pivotal token is the *chosen* continuation; the *rejected*
        one must be found by testing candidate tokens against a real success
        oracle.

        v1 built that oracle as a ``DummyOracle``, which reports every completion
        as a success. Every candidate therefore scored ``P(success) = 1.0``, the
        "does this token hurt?" test (``prob_before - prob_after >= threshold``)
        could never fire, and every positive token was silently dropped. v1's DPO
        exports consequently contained only negative-delta tokens, whose partner
        was picked by next-token likelihood rather than by any measured effect.

        So rejected-token discovery now requires a real oracle. Supply one of:

          * ``--dataset`` (the source dataset, from which an oracle is rebuilt),
          * an ``oracle=`` object, or
          * events that already carry ``rejected_token`` in their metadata.

        Without one, discovery is skipped with a warning rather than emitting
        pairs that look measured but are not.
        """
        rng = random.Random(seed)

        token_events = [
            e for e in self.storage
            if e.event_type == EVENT_TOKEN
            and e.prob_delta is not None
            and abs(e.prob_delta) >= min_prob_delta
        ]
        if not token_events:
            logger.error(
                f"No token events with |prob_delta| >= {min_prob_delta}. "
                "DPO export needs token-level events."
            )
            return

        searcher = None
        if find_rejected_tokens:
            oracle = oracle or self._rebuild_oracle(dataset, split)

            if oracle is None:
                logger.warning(
                    "--find-rejected-tokens was requested but no usable oracle is "
                    "available. Discovering a rejected token means measuring P(success) "
                    "for candidate continuations, which needs ground-truth answers. "
                    "Pass --dataset <source dataset> so an oracle can be rebuilt. "
                    "Skipping discovery: only events that already carry a rejected_token "
                    "will produce pairs."
                )
                find_rejected_tokens = False
            elif not model_name:
                logger.warning(
                    "--find-rejected-tokens needs --model to score candidate tokens. "
                    "Skipping discovery."
                )
                find_rejected_tokens = False
            else:
                from .searchers.token import TokenPTSSearcher

                logger.info(f"Loading {model_name} to search for rejected tokens")
                searcher = TokenPTSSearcher(
                    model_name=model_name,
                    oracle=oracle,
                    device=device,
                    prob_threshold=min_prob_delta,
                )

        pairs: List[Dict[str, Any]] = []
        skipped = 0

        for event in token_events:
            meta = event.metadata or {}

            if event.is_positive:
                rejected_token = meta.get("rejected_token")
                rejected_prob = meta.get("rejected_prob")

                if rejected_token is None and searcher is not None:
                    found = searcher.find_rejected_token(
                        event, num_candidates=num_candidates, category=event.category
                    )
                    if found:
                        rejected_token, _, rejected_prob = found

                if rejected_token is None:
                    skipped += 1
                    continue

                pairs.append(self._pair(
                    event,
                    chosen=event.label,
                    rejected=rejected_token,
                    prob_chosen=event.prob_after,
                    prob_rejected=rejected_prob,
                ))
            else:
                # A negative pivotal token is itself the rejected continuation.
                # Its partner is only kept if it was *measured* to be better --
                # otherwise it is a likelihood guess dressed up as a preference.
                chosen_token = meta.get("chosen_token")
                chosen_prob = meta.get("chosen_prob")

                if chosen_token is None and searcher is not None:
                    found = self._find_better_token(searcher, event, num_candidates)
                    if found:
                        chosen_token, chosen_prob = found

                if chosen_token is None:
                    skipped += 1
                    continue

                pairs.append(self._pair(
                    event,
                    chosen=chosen_token,
                    rejected=event.label,
                    prob_chosen=chosen_prob,
                    prob_rejected=event.prob_after,
                ))

        if balance:
            pos = [p for p in pairs if (p["metadata"]["prob_delta"] or 0) > 0]
            neg = [p for p in pairs if (p["metadata"]["prob_delta"] or 0) <= 0]
            n = min(len(pos), len(neg))
            pairs = rng.sample(pos, n) + rng.sample(neg, n)
            rng.shuffle(pairs)

        if max_pairs:
            pairs = pairs[:max_pairs]

        if skipped:
            logger.warning(
                f"Skipped {skipped} token events with no verified counterpart. Run with "
                "--find-rejected-tokens --model <model> --dataset <source dataset> to "
                "search for them."
            )

        if not pairs:
            logger.error(
                "No DPO pairs produced: no token event had a verified counterpart. "
                "Re-run with --find-rejected-tokens --model <model> --dataset <dataset>."
            )
            return

        self._write(output_path, pairs)

    @staticmethod
    def _pair(
        event: CausalReasoningEvent,
        chosen: str,
        rejected: str,
        prob_chosen: Optional[float],
        prob_rejected: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "prompt": event.context,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "original_query": event.query,
                "prob_chosen": prob_chosen,
                "prob_rejected": prob_rejected,
                "prob_delta": event.prob_delta,
                "task_type": event.task_type,
                "event_id": event.event_id,
                "category": event.category,
                # False would mean the partner token came from stored metadata
                # rather than a measured probability. It is never silently true.
                "counterpart_verified": prob_chosen is not None and prob_rejected is not None,
            },
        }

    def _rebuild_oracle(self, dataset: Optional[str], split: str) -> Optional[Any]:
        """Rebuild a real oracle from the source dataset."""
        if not dataset:
            ids = {e.dataset_id for e in self.storage if e.dataset_id}
            if len(ids) == 1:
                dataset = ids.pop()
                logger.info(f"Rebuilding oracle from the events' own dataset_id: {dataset}")
            else:
                return None

        try:
            from .dataset import create_oracle_from_dataset, load_dataset
            from .oracle import DummyOracle

            examples = load_dataset(dataset_name=dataset, split=split)
            if not examples:
                logger.warning(f"Could not load {dataset} to rebuild an oracle")
                return None

            oracle = create_oracle_from_dataset(examples)
            if isinstance(oracle, DummyOracle):
                logger.warning(
                    f"Rebuilding an oracle from {dataset} produced a DummyOracle, which "
                    "reports every completion as a success. It cannot verify rejected "
                    "tokens, so it is refused here."
                )
                return None
            return oracle
        except Exception as e:
            logger.warning(f"Could not rebuild an oracle from {dataset}: {e}")
            return None

    @staticmethod
    def _find_better_token(searcher, event: CausalReasoningEvent, num_candidates: int = 10):
        """For a negative token, find a candidate that measurably beats it."""
        import torch

        context_ids = searcher.tokenizer.encode(event.context, return_tensors="pt").to(
            searcher.device
        )
        with torch.no_grad():
            logits = searcher.model(context_ids).logits[0, -1, :]

        top = torch.topk(torch.softmax(logits, dim=0), k=num_candidates)
        for i in range(num_candidates):
            token_id = top.indices[i].item()
            if token_id == event.token_id:
                continue
            token_str = searcher.tokenizer.decode([token_id])
            prob = searcher.estimate_success_probability(
                event.query, event.context + token_str, category=event.category
            )
            if prob - event.prob_before >= searcher.prob_threshold:
                return token_str, prob
        return None

    # -- steering ----------------------------------------------------------

    def export_steering_vectors(
        self,
        output_path: str,
        model_name: str,
        layer_nums: Sequence[int] = (19, 23, 27),
        select_layer: Optional[int] = None,
        num_clusters: int = 10,
        pca_components: int = 50,
        batch_size: int = 4,
        min_prob_delta: float = 0.2,
        device: Optional[str] = None,
    ) -> None:
        """Extract each event's context activation and cluster into steering vectors."""
        import numpy as np
        import torch
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .latent.activations import ResidualCapture, get_layer_modules
        from .searchers.base import select_device

        events = [
            e for e in self.storage
            if e.prob_delta is not None
            and abs(e.prob_delta) >= min_prob_delta
            and e.context
        ]
        if not events:
            logger.error(f"No events with |prob_delta| >= {min_prob_delta}")
            return

        device = select_device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device in ("cpu", "mps") else torch.float16,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        layer_modules = get_layer_modules(model)
        num_layers = len(layer_modules)
        layer = select_layer if select_layer is not None else layer_nums[0]
        if not (0 <= layer < num_layers):
            raise ValueError(
                f"Steering layer {layer} is out of range for a {num_layers}-layer model. "
                f"Pass --select-layer with a value in [0, {num_layers - 1}]."
            )
        # v1 hooked every layer in --layer-nums, then used only the first and threw
        # the rest away. Only the layer actually used is captured.
        logger.info(f"Extracting activations at layer {layer} of {num_layers}")

        capture = ResidualCapture(model, layer_modules)
        vectors: List[Any] = []

        with capture.capture([layer]) as acts:
            for start in range(0, len(events), batch_size):
                batch = events[start : start + batch_size]
                encoded = tokenizer(
                    [e.context for e in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                input_ids = encoded.input_ids.to(device)
                attention_mask = encoded.attention_mask.to(device)

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                hidden = acts[layer]  # [B, T, d]
                for i in range(len(batch)):
                    # The last *real* token is not the last column when the
                    # tokenizer pads on the left, which v1 assumed it never did.
                    if tokenizer.padding_side == "left":
                        idx = hidden.shape[1] - 1
                    else:
                        idx = int(attention_mask[i].sum().item()) - 1
                    vectors.append(hidden[i, idx, :].float().cpu().numpy())

        matrix = np.vstack(vectors)
        logger.info(f"Extracted {matrix.shape[0]} activation vectors of dim {matrix.shape[1]}")

        reduced = matrix
        if matrix.shape[0] > pca_components and matrix.shape[1] > pca_components:
            reduced = PCA(n_components=pca_components).fit_transform(matrix)

        k = min(num_clusters, matrix.shape[0])
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(reduced)

        # Cluster means live in the original activation space, because that is the
        # space a steering vector has to be added into.
        cluster_means = {c: matrix[labels == c].mean(axis=0) for c in range(k)}
        cluster_events = {
            c: [events[i] for i in range(len(events)) if labels[i] == c] for c in range(k)
        }
        patterns = {
            c: self._optillm_pattern(cluster_events[c], c) for c in range(k)
        }

        records = []
        for i, event in enumerate(events):
            c = int(labels[i])
            base = (
                to_v1_pivotal_token(event)
                if event.event_type == EVENT_TOKEN
                else {
                    "query": event.query,
                    "pivot_context": event.context,
                    "pivot_token": event.label,
                    "pivot_token_id": event.token_id,
                    "prob_before": event.prob_before,
                    "prob_after": event.prob_after,
                    "prob_delta": event.prob_delta,
                    "is_positive": event.is_positive,
                    "model_id": event.model_id,
                    "task_type": event.task_type,
                }
            )
            records.append(
                {
                    **base,
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "steering_vector": matrix[i].tolist(),
                    "cluster_id": c,
                    "cluster_vector": cluster_means[c].tolist(),
                    # OptiLLM autothink reads `reasoning_pattern`; PTS reads `category`.
                    "reasoning_pattern": patterns[c],
                    "category": event.category,
                    "steering_layer": layer,
                }
            )

        self._write(output_path, records)

    @staticmethod
    def _optillm_pattern(events: Sequence[CausalReasoningEvent], cluster_id: int) -> str:
        """Label a cluster with one of OptiLLM's five reasoning patterns."""
        text = " ".join((e.label or "") for e in events).lower()
        scores = {
            pattern: sum(text.count(kw) for kw in keywords)
            for pattern, keywords in OPTILLM_REASONING_PATTERNS.items()
        }
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        names = list(OPTILLM_REASONING_PATTERNS)
        return names[cluster_id % len(names)]


# ---------------------------------------------------------------------------
# Dataset cards
# ---------------------------------------------------------------------------

def generate_dataset_card(
    path: str,
    file_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Write a dataset card describing what is actually in the file.

    Counts are read from the file rather than asserted. When latent events are
    present the caveats are mandatory: a reader must not come away thinking a
    J-lens readout is a measured causal effect.
    """
    file_type = file_type or detect_file_type(path)

    counts: Counter = Counter()
    readouts: Counter = Counter()
    total = 0
    models = set()

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if "event_type" in record:
                    counts[record["event_type"]] += 1
                if record.get("readout_method"):
                    readouts[record["readout_method"]] += 1
                if record.get("model_id"):
                    models.add(record["model_id"])
    except OSError:
        pass

    model = model_name or (sorted(models)[0] if models else "MODEL_NAME")

    lines = [
        "---",
        "license: apache-2.0",
        "language:",
        "- en",
        "tags:",
        "- pts",
        "- pivotal-token-search",
        "- mechanistic-interpretability",
        "- reasoning",
        "task_categories:",
        "- other",
        "---",
        "",
        f"# PTS dataset ({file_type})",
        "",
        "Generated with [PTS](https://github.com/codelion/pts) v2 (schema version 2.0).",
        "",
        f"- **Model:** `{model}`",
        f"- **Records:** {total}",
    ]

    if counts:
        lines.append("- **Events by type:**")
        for name, n in counts.most_common():
            lines.append(f"  - `{name}`: {n}")
    if readouts:
        lines.append(
            "- **Latent readout:** "
            + ", ".join(f"`{m}` ({n})" for m, n in readouts.most_common())
        )

    lines += [
        "",
        "## What PTS events are",
        "",
        "PTS searches for pivotal reasoning events at three scales, each scored by its "
        "effect on the probability of solving the task:",
        "",
        "| Scale | `event_type` | What it is |",
        "|---|---|---|",
        "| Latent | `latent_metatoken` | A verbalizable concept read out of the model's mid-layer workspace |",
        "| Token | `pivotal_token` | An emitted token that shifts success probability |",
        "| Sentence | `thought_anchor` | A reasoning sentence that shifts success probability |",
        "",
    ]

    if counts.get(EVENT_LATENT):
        lines += [
            "## Caveats on the latent events",
            "",
            "Read these before using any `latent_metatoken` record.",
            "",
            "- **They are hypotheses, not measurements.** A latent event's `score` is a "
            "readout score -- how strongly the lens surfaces that token -- **not** a "
            "probability delta. It is not comparable to the `prob_delta` on token and "
            "sentence events, and it does not establish that the concept caused anything. "
            "`prob_delta` and `is_positive` are deliberately `null` on these records.",
            "- **They are observational.** No intervention was run. Showing that a "
            "meta-token *causes* a downstream event requires steering or ablating it and "
            "re-measuring success. These records do not do that.",
            "- **Readouts are noisy.** Neither the J-lens nor the logit lens is guaranteed "
            "to be faithful to what the model actually represents.",
            "- **`logit_lens` is weaker evidence than `jlens`.** The logit lens reads what "
            "an activation would emit *now*; the J-lens reads what it pushes the model to "
            "emit *later*. Only the latter speaks to a workspace. Check the "
            "`readout_method` field.",
            "- **Links are scored guesses.** `linked_event_ids` come from a weighted "
            "heuristic (query match, context overlap, category agreement, temporal "
            "proximity), not from a verified causal path.",
            "",
        ]

    if file_type == "dpo":
        lines += [
            "## Usage",
            "",
            "Fine-tune with Direct Preference Optimization. A Colab notebook is available:",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            "(https://colab.research.google.com/drive/1FggA9EQ1eFBjE0Qbsl0-EFzyWIxpdhlH?usp=sharing)",
            "",
            "```python",
            "from datasets import load_dataset",
            "from trl import DPOTrainer",
            "",
            'dataset = load_dataset("USERNAME/REPO_NAME")',
            "trainer = DPOTrainer(model=model, args=training_args, beta=0.1,",
            "                     train_dataset=dataset, tokenizer=tokenizer)",
            "trainer.train()",
            "```",
            "",
            "Each pair carries `metadata.counterpart_verified`. When it is `false`, the "
            "partner token was taken from stored metadata rather than measured against a "
            "success oracle -- filter those out if you need only verified preferences.",
            "",
        ]

    if file_type == "steering":
        lines += [
            "## Usage",
            "",
            "These vectors can be added to a residual stream at inference to steer the "
            "model toward a reasoning pattern. Each record carries `steering_vector` (that "
            "event's own activation), `cluster_vector` (its cluster mean), `steering_layer`, "
            "and `reasoning_pattern`.",
            "",
            "### OptiLLM integration",
            "",
            "These work with [OptiLLM](https://github.com/codelion/optillm)'s `autothink` "
            "approach, which reads the `reasoning_pattern` field. That field uses OptiLLM's "
            "five-value vocabulary (`depth_and_thoroughness`, `numerical_accuracy`, "
            "`self_correction`, `exploration`, `organization`). The PTS category taxonomy is "
            "carried separately in `category`.",
            "",
        ]

    if file_type == "thought_anchors":
        lines += [
            "## Usage",
            "",
            "Thought anchors are Sentence PTS events rendered in the v1 shape. They can be "
            "used to focus attention on critical reasoning steps, validate reasoning by "
            "checking for anchor patterns, or guide search toward high-value steps. They "
            "also work with [OptiLLM](https://github.com/codelion/optillm)'s autothink.",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            'anchors = load_dataset("USERNAME/REPO_NAME")["train"]',
            'positive = anchors.filter(lambda x: x["is_positive"] and x["importance_score"] > 0.3)',
            "```",
            "",
        ]

    lines += [
        "## Relation to prior work",
        "",
        "PTS began as Pivotal Token Search, the idea described in the Phi-4 technical "
        "report; Token PTS corresponds to that. Sentence PTS corresponds to "
        "thought-anchor-style reasoning steps. Latent PTS is inspired by the workspace / "
        "J-space readouts in Anthropic's *Verbalizable Representations Form a Global "
        "Workspace in Language Models*, and is compatible with that framing -- but it is an "
        "independent reimplementation from the paper's published equations. No reference "
        "code was released, and these readouts have not been validated against the authors' "
        "results. The term \"meta-token\" is ours, not the paper's.",
        "",
        "## Model specificity",
        "",
        f"Every record here is specific to `{model}`. Pivotal tokens, thought anchors, and "
        "workspace readouts do not transfer across models: a token that is pivotal for one "
        "model tells you nothing about another.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v1 compatibility
# ---------------------------------------------------------------------------

def generate_readme_content(
    file_type: str, model_name: Optional[str] = None, dataset_info: Optional[str] = None
) -> str:
    """v1 name. Kept so `from pts.exporters import generate_readme_content` resolves."""
    content = "\n".join(
        [
            "---",
            "license: apache-2.0",
            "tags:",
            "- pts",
            "---",
            "",
            f"# PTS dataset ({file_type})",
            "",
            f"Model: `{model_name or 'Unknown'}`",
            "",
            "Generated with [PTS](https://github.com/codelion/pts).",
        ]
    )
    if dataset_info:
        content += f"\n\n## Additional Information\n\n{dataset_info}\n"
    return content


class TokenExporter(EventExporter):
    """v1 name for ``EventExporter``."""

    def __init__(self, token_storage: Optional[EventStorage] = None, searcher=None):
        super().__init__(token_storage)
        self.searcher = searcher

    def export_dpo_dataset(self, output_path: str, **kwargs) -> None:
        for dropped in (
            "filter_criteria", "save_tokens", "tokens_output_path",
            "hf_push", "hf_repo_id", "private",
        ):
            kwargs.pop(dropped, None)
        if "balance_positive_negative" in kwargs:
            kwargs["balance"] = kwargs.pop("balance_positive_negative")
        return self.export_dpo(output_path, **kwargs)
