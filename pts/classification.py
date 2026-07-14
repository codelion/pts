"""
Category taxonomy shared by latent, token, and sentence events.

One taxonomy across all three scales is what makes the multiscale claim
testable: if a latent ``verification`` meta-token precedes a token
``verification`` event which expands into a sentence ``verification`` anchor,
that chain is only visible when all three are labelled from the same vocabulary.

The rules here are keyword/regex based and deliberately shallow. They are a
first pass, not a classifier -- see ``docs/latent_pts.md`` for the caveats.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

INTERPRETATION = "interpretation"
PLANNING = "planning"
DECOMPOSITION = "decomposition"
FACT_RETRIEVAL = "fact_retrieval"
COMPUTATION = "computation"
VERIFICATION = "verification"
SELF_CORRECTION = "self_correction"
BACKTRACKING = "backtracking"
UNCERTAINTY = "uncertainty"
EXPLORATION = "exploration"
CONSOLIDATION = "consolidation"
FINAL_ANSWER = "final_answer"
TOOL_USE = "tool_use"
SAFETY = "safety"
DECEPTION_OR_SHORTCUT = "deception_or_shortcut"
UNKNOWN = "unknown"

CATEGORIES: List[str] = [
    INTERPRETATION, PLANNING, DECOMPOSITION, FACT_RETRIEVAL, COMPUTATION,
    VERIFICATION, SELF_CORRECTION, BACKTRACKING, UNCERTAINTY, EXPLORATION,
    CONSOLIDATION, FINAL_ANSWER, TOOL_USE, SAFETY, DECEPTION_OR_SHORTCUT,
    UNKNOWN,
]

# Order matters: the first matching category wins. The specific, high-signal
# categories (self-correction, verification) are checked before the generic
# ones (computation, consolidation), because a sentence like "Wait, let me
# check 2 + 2 = 4" is about self-correction, not arithmetic. The v1 classifier
# in thought_anchors.py got this backwards -- any sentence containing "=" was
# labelled computation regardless of its verification wording.
CATEGORY_PATTERNS: Dict[str, List[str]] = {
    SELF_CORRECTION: [
        r"\bwait\b", r"\bactually\b", r"\bmistake\b", r"\bwrong\b", r"\berror\b",
        r"\breconsider\b", r"\bi was\b", r"\bcorrection\b", r"\boops\b",
    ],
    BACKTRACKING: [
        r"\bbacktrack\b", r"\bgo back\b", r"\bstart over\b", r"\bscratch that\b",
        r"\bnever mind\b", r"\bundo\b", r"\bretrace\b",
    ],
    VERIFICATION: [
        r"\bcheck\b", r"\bverify\b", r"\bconfirm\b", r"\bvalidate\b",
        r"\bdouble[- ]check\b", r"\bmake sure\b", r"\bsanity\b", r"\btest\b",
    ],
    DECEPTION_OR_SHORTCUT: [
        r"\bfake\b", r"\btrick\b", r"\bcheat\b", r"\bshortcut\b", r"\bhidden\b",
        r"\bsecret\b", r"\breward\b", r"\bgame the\b",
    ],
    SAFETY: [
        r"\bharmful\b", r"\bunsafe\b", r"\brefuse\b", r"\bcannot help\b",
        r"\bpolicy\b", r"\binappropriate\b",
    ],
    TOOL_USE: [
        r"\btool\b", r"\bsearch\b", r"\bexecute\b", r"\bpython\b", r"\bcall the\b",
        r"\bapi\b", r"\bquery the\b",
    ],
    UNCERTAINTY: [
        r"\bnot sure\b", r"\bunsure\b", r"\bunclear\b", r"\bhmm\b", r"\bperhaps\b",
        r"\bmight be\b", r"\bpossibly\b", r"\bi think\b", r"\bprobably\b",
    ],
    EXPLORATION: [
        r"\balternative\b", r"\banother\b", r"\bdifferent\b", r"\bmaybe\b",
        r"\bwhat if\b", r"\bpossibility\b", r"\btry\b", r"\bconsider\b",
    ],
    INTERPRETATION: [
        r"\bmeaning\b", r"\binterpret\b", r"\bambiguous\b", r"\bwhat does\b",
        r"\bpun\b", r"\briddle\b", r"\bunderstand\b", r"\basking\b", r"\bthe question\b",
    ],
    PLANNING: [
        r"\bplan\b", r"\bapproach\b", r"\bstrategy\b", r"\bfirst\b", r"\bnext\b",
        r"\bstep\b", r"\blet me\b", r"\bi will\b", r"\bwe need to\b",
    ],
    DECOMPOSITION: [
        r"\bbreak (?:it |this )?down\b", r"\bsub-?problem\b", r"\bpart (?:one|two|1|2)\b",
        r"\bsplit\b", r"\bseparately\b", r"\bcase \d\b",
    ],
    FACT_RETRIEVAL: [
        r"\brecall\b", r"\bknown\b", r"\bformula\b", r"\bdefinition\b",
        r"\bwe know\b", r"\btheorem\b", r"\bby definition\b",
    ],
    COMPUTATION: [
        r"\bcalculate\b", r"\bcompute\b", r"\bequation\b", r"\bsum\b",
        r"\bmultiply\b", r"\bdivide\b", r"\bsubtract\b", r"\badd\b",
        r"\d+\s*[+\-*/x]\s*\d+", r"=",
    ],
    CONSOLIDATION: [
        r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bso the\b",
        r"\bin conclusion\b", r"\bit follows\b", r"\bwhich means\b",
    ],
    FINAL_ANSWER: [
        r"\bfinal answer\b", r"\bthe answer is\b", r"\banswer:\b", r"####",
        r"\\boxed", r"\bin summary\b",
    ],
}

# The v1 thought-anchor taxonomy, mapped onto the unified one so that migrated
# sentence events land in the same category space as everything else.
V1_SENTENCE_CATEGORY_MAP: Dict[str, str] = {
    "problem_setup": INTERPRETATION,
    "plan_generation": PLANNING,
    "fact_retrieval": FACT_RETRIEVAL,
    "active_computation": COMPUTATION,
    "uncertainty_management": UNCERTAINTY,
    "result_consolidation": CONSOLIDATION,
    "self_checking": VERIFICATION,
    "final_answer_emission": FINAL_ANSWER,
}


class EventClassifier:
    """Assign a unified category to an event label (token, sentence, or meta-token)."""

    def __init__(self, patterns: Optional[Dict[str, List[str]]] = None):
        raw = patterns if patterns is not None else CATEGORY_PATTERNS
        self.patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in pats]
            for category, pats in raw.items()
        }

    @classmethod
    def from_file(cls, path: str) -> "EventClassifier":
        """Load a category lexicon from JSON: ``{category: [pattern, ...]}``."""
        with open(path) as f:
            return cls(patterns=json.load(f))

    def classify(self, text: str) -> Optional[str]:
        """Return the first matching category, or ``None`` if nothing matches.

        Callers that need a value rather than ``None`` should substitute
        ``UNKNOWN`` themselves -- the distinction between "no rule fired" and
        "explicitly unknown" is worth keeping.
        """
        if not text or not text.strip():
            return None

        for category, compiled in self.patterns.items():
            for pattern in compiled:
                if pattern.search(text):
                    return category
        return None

    def classify_metatoken(self, token: str) -> Optional[str]:
        """Classify a single latent meta-token.

        Meta-tokens arrive as raw vocabulary pieces, often with a leading space
        or subword marker, so they are normalized before matching. Word-boundary
        patterns are relaxed to substring matching here because a single subword
        like "verif" would otherwise never match ``\\bverify\\b``.
        """
        if not token:
            return None

        cleaned = token.strip().strip("ĠĊ▁").lower()
        if not cleaned:
            return None

        direct = self.classify(cleaned)
        if direct:
            return direct

        # Subword fallback: match against the literal keywords behind the
        # patterns rather than the patterns themselves.
        for category, compiled in self.patterns.items():
            for pattern in compiled:
                literal = _pattern_literal(pattern.pattern)
                if literal and len(cleaned) >= 4 and literal.startswith(cleaned):
                    return category
        return None


_WORD_BOUNDARY = re.compile(r"^\\b(.+?)\\b$")


def _pattern_literal(pattern: str) -> Optional[str]:
    """Recover the plain keyword from a simple ``\\bword\\b`` pattern."""
    m = _WORD_BOUNDARY.match(pattern)
    if not m:
        return None
    literal = m.group(1)
    if re.search(r"[\\\[\](){}|+*?^$]", literal):
        return None
    return literal.lower()


_default_classifier: Optional[EventClassifier] = None


def get_classifier() -> EventClassifier:
    global _default_classifier
    if _default_classifier is None:
        path = os.environ.get("PTS_CATEGORY_LEXICON")
        if path and os.path.exists(path):
            logger.info(f"Loading category lexicon from {path}")
            _default_classifier = EventClassifier.from_file(path)
        else:
            _default_classifier = EventClassifier()
    return _default_classifier


def classify_event_label(text: str, granularity: str = "token") -> Optional[str]:
    clf = get_classifier()
    if granularity == "latent":
        return clf.classify_metatoken(text)
    return clf.classify(text)


def map_v1_category(v1_category: Optional[str]) -> Optional[str]:
    if not v1_category:
        return None
    return V1_SENTENCE_CATEGORY_MAP.get(v1_category, v1_category)
