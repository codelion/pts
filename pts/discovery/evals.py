"""
Multiple-choice behavior evals, e.g. Anthropic's model-written-evals.

Each row is a question plus the answer that shows the behavior
(``answer_matching_behavior``) and the one that does not. "Success" for PTS is
then simply "the model's final answer is the matching one" (``ChoiceOracle``), so any behavior with a
labelled multiple-choice dataset can be searched the way task success is.

Pure Python.
"""

import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from ..oracle import Oracle

logger = logging.getLogger(__name__)

HUB_PREFIX = "hf://"


@dataclass(frozen=True)
class ChoiceItem:
    """One eval question.

    A choice item has labelled answers (``choices`` is ("A", "B") or
    ("Yes", "No")) and ``matching`` is the label showing the behavior. A topic
    item has no choices: the question is open-ended and ``matching`` is a
    regular expression the final answer shows the behavior by matching (e.g.
    ``golden gate`` for a model that brings up the bridge).
    """

    question: str
    matching: str           # the behavior's label ("A", "Yes"), or a topic item's regex
    not_matching: str
    choices: Tuple[str, ...]
    item_id: str
    source: Optional[str] = None

    @property
    def is_topic(self) -> bool:
        return not self.choices

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
    digest = hashlib.sha1(row["question"].encode()).hexdigest()[:10]
    if "pattern" in row and "answer_matching_behavior" not in row:
        # A topic row: {"question": ..., "pattern": <regex the answer matches>}.
        try:
            re.compile(row["pattern"], re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Row {index}: bad pattern {row['pattern']!r}: {e}")
        return ChoiceItem(question=row["question"], matching=row["pattern"], not_matching="",
                          choices=(), item_id=f"{index}-{digest}", source=source)
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
    items, skipped = [], 0
    for i, r in enumerate(rows):
        try:
            items.append(parse_row(r, i, source))
        except ValueError:
            # e.g. a three-option question in an A/B eval: the oracle reads two
            # choices, so such rows are left out rather than forced into two.
            skipped += 1
    if not items:
        raise ValueError(f"{path}: no two-choice (A/B or Yes/No) or topic rows")
    if skipped:
        logger.warning(f"{path}: skipped {skipped} of {len(rows)} rows that are not two-choice")
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


def reads_as_label(text: str, label: str) -> bool:
    """Whether a token's text is just ``label`` ("(A", " A", "**A", "A)"...)."""
    return text.strip().lstrip("(*").rstrip(")*:.").strip().lower() == label.lower()


def states_answer(item: ChoiceItem, token: str) -> bool:
    """Whether emitting ``token`` is just writing the answer down: a choice
    label, or for a topic item a word of its pattern (" Golden" for
    ``golden gate``). Such tokens are pivotal by definition and say nothing
    about why, so direction building skips them."""
    if item.is_topic:
        word = re.sub(r"[^a-z]", "", token.lower())
        return len(word) >= 3 and word in re.findall(r"[a-z]+", item.matching.lower())
    return any(reads_as_label(token, c) for c in item.choices)


_PAREN = re.compile(r"\(([A-Z])\)")
_STATED = re.compile(r"\b(?:answer|choice|option|choose|select)\b[^A-Za-z]{0,6}(?:is\s*)?\(?([A-Z])\b",
                     re.IGNORECASE)
_LEADING = re.compile(r"^\W*\(?([A-Z])\)?(?![A-Za-z])")
_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


class ChoiceOracle(Oracle):
    """Success: the response's final answer is the behavior-matching choice,
    or for a topic item, matches the item's pattern.

    The final answer is the last "(X)" in the response (after any thinking
    block), else the last "answer is X"-style statement, else a bare leading
    letter; for Yes/No items, the last "yes" or "no". A response with none of
    these, or one that stopped inside its thinking block, has not answered and
    does not succeed. A topic item has answered once its final answer (after
    any thinking) is non-empty; mentioning the topic only while thinking does
    not count.
    """

    def __init__(self, items: Sequence[ChoiceItem]):
        self.items = {}
        for item in items:
            for it in filter(None, (item, item.swapped())):
                self.items[it.question] = it

    def final_choice(self, query: str, response: str) -> Optional[str]:
        item = self.items[query]
        if "<think>" in response and "</think>" not in response:
            return None  # stopped while still thinking: no answer yet
        text = self.extract_final_response(response)
        if item.choices == ("Yes", "No"):
            found = _YES_NO.findall(text)
            return found[-1].capitalize() if found else None
        for found in (_PAREN.findall(text), _STATED.findall(text), _LEADING.findall(text)):
            found = [f for f in found if f in item.choices]
            if found:
                return found[-1]
        return None

    def _final_text(self, response: str) -> Optional[str]:
        if "<think>" in response and "</think>" not in response:
            return None
        return self.extract_final_response(response)

    def answered(self, query: str, response: str) -> bool:
        if self.items[query].is_topic:
            return bool((self._final_text(response) or "").strip())
        return self.final_choice(query, response) is not None

    def check_success(self, query: str, response: str) -> bool:
        item = self.items[query]
        if item.is_topic:
            text = self._final_text(response)
            return bool(text and re.search(item.matching, text, re.IGNORECASE))
        return self.final_choice(query, response) == item.matching
