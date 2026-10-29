"""
Multiple-choice behavior evals, e.g. Anthropic's model-written-evals.

Each row is a question plus the answer that shows the behavior
(``answer_matching_behavior``) and the one that does not. "Success" for PTS is
then simply "the model picks the matching answer", so any behavior with a
labelled multiple-choice dataset can be searched the way task success is.

Pure Python.
"""

import hashlib
import json
import os
import random
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

HUB_PREFIX = "hf://"


@dataclass(frozen=True)
class ChoiceItem:
    question: str
    matching: str           # the choice label showing the behavior, e.g. "A" or "Yes"
    not_matching: str
    choices: Tuple[str, ...]
    item_id: str
    source: Optional[str] = None

    def swapped(self) -> Optional["ChoiceItem"]:
        """The same question with the (A)/(B) labels exchanged.

        Averaging over both orders cancels a model's preference for a letter,
        which on small models can dominate the behavior itself. Yes/No items
        have no labels to swap and return None.
        """
        if self.choices != ("A", "B"):
            return None
        q = self.question.replace("(A)", "(\0)").replace("(B)", "(A)").replace("(\0)", "(B)")
        other = {"A": "B", "B": "A"}
        return replace(self, question=q, matching=other[self.matching],
                       not_matching=other[self.not_matching], item_id=self.item_id + ":swapped")


def _label(answer: str) -> str:
    return answer.strip().strip("()").strip()


def parse_row(row: dict, index: int, source: Optional[str] = None) -> ChoiceItem:
    matching = _label(row["answer_matching_behavior"] if isinstance(row["answer_matching_behavior"], str)
                      else row["answer_matching_behavior"][0])
    not_matching = _label(row["answer_not_matching_behavior"] if isinstance(row["answer_not_matching_behavior"], str)
                          else row["answer_not_matching_behavior"][0])
    labels = {matching, not_matching}
    if labels == {"A", "B"}:
        choices = ("A", "B")
    elif {x.lower() for x in labels} == {"yes", "no"}:
        choices = ("Yes", "No")
        matching, not_matching = matching.capitalize(), not_matching.capitalize()
    else:
        raise ValueError(f"Row {index}: unsupported answer labels {sorted(labels)}")
    digest = hashlib.sha1(row["question"].encode()).hexdigest()[:10]
    return ChoiceItem(
        question=row["question"], matching=matching, not_matching=not_matching,
        choices=choices, item_id=f"{index}-{digest}", source=source,
    )


def _resolve(path: str) -> str:
    if not path.startswith(HUB_PREFIX):
        return path
    from huggingface_hub import hf_hub_download

    parts = path[len(HUB_PREFIX):].split("/", 2)
    if len(parts) < 3:
        raise ValueError(f"Bad Hub eval path {path!r}: expected hf://<org>/<dataset>/<file.jsonl>")
    return hf_hub_download(f"{parts[0]}/{parts[1]}", parts[2], repo_type="dataset")


def load_choice_eval(path: str, limit: Optional[int] = None, seed: int = 0) -> List[ChoiceItem]:
    """Load a JSONL choice eval from disk or ``hf://<org>/<dataset>/<file.jsonl>``.

    Rows are shuffled with ``seed`` before ``limit`` is applied, so a limited
    sample is not just the first rows of the file.
    """
    local = _resolve(path)
    source = path if path.startswith(HUB_PREFIX) else os.path.basename(path)
    with open(local) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    items = [parse_row(r, i, source) for i, r in enumerate(rows)]
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


def split(
    items: Sequence[ChoiceItem], fractions: Tuple[float, float, float] = (0.5, 0.25, 0.25)
) -> Tuple[List[ChoiceItem], List[ChoiceItem], List[ChoiceItem]]:
    """Discovery / selection / test, by question.

    Vectors are found on discovery, their layer and strength chosen on
    selection, and only test is reported, so a vector cannot pass by having
    been tuned on the questions it is scored on.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    n = len(items)
    a = int(round(fractions[0] * n))
    b = a + int(round(fractions[1] * n))
    return list(items[:a]), list(items[a:b]), list(items[b:])
