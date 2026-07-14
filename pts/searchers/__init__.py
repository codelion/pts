"""
PTS searchers -- one framework, three representational scales.

Each searcher finds pivotal reasoning events at a different granularity and
emits them as ``CausalReasoningEvent`` records:

    TokenPTSSearcher     emitted pivotal tokens        (Phi-4-style PTS)
    SentencePTSSearcher  sentence-level thought anchors
    LatentPTSSearcher    latent workspace meta-tokens  (J-space readouts)
    MultiScaleSearcher   all three, linked together

They share ``BasePTSSearcher``, which owns model loading, prompt formatting,
generation, and success-probability estimation.
"""

from .base import BasePTSSearcher
from .token import TokenPTSSearcher, PivotalTokenSearcher
from .sentence import SentencePTSSearcher
from .latent import LatentPTSSearcher
from .multiscale import MultiScaleSearcher

__all__ = [
    "BasePTSSearcher",
    "TokenPTSSearcher",
    "PivotalTokenSearcher",
    "SentencePTSSearcher",
    "LatentPTSSearcher",
    "MultiScaleSearcher",
]
