"""
PTS -- a multiscale causal-event search framework for model reasoning.

PTS identifies the hidden workspace states, emitted tokens, and sentence-level
reasoning steps that causally shift downstream success probability::

    latent meta-token / workspace event   (Latent PTS)
            v
    emitted pivotal token                 (Token PTS)
            v
    sentence-level thought anchor         (Sentence PTS)
            v
    success/failure probability shift

Token PTS is the original Pivotal Token Search idea from the Phi-4 technical
report. Sentence PTS corresponds to thought-anchor-style reasoning steps.
Latent PTS uses workspace/J-space-style readouts to search for hidden
verbalizable meta-tokens that may precede emitted events.

The event schema, storage, classification, and linking layers are pure Python
and import without torch or transformers. Anything that touches a model
(searchers, latent readouts, exporters) is imported lazily on first access, so
``import pts`` stays cheap and works in environments without a GPU stack.
"""

from importlib import import_module
from typing import Any

__version__ = "2.0.0"

# Light: no torch, no transformers.
from .events import (  # noqa: F401
    CausalReasoningEvent,
    SCHEMA_VERSION,
    EVENT_LATENT,
    EVENT_TOKEN,
    EVENT_SENTENCE,
    make_token_event,
    make_sentence_event,
    make_latent_event,
    from_any_record,
    from_v1_pivotal_token,
    from_v1_thought_anchor,
    to_v1_pivotal_token,
    to_v1_thought_anchor,
)
from .event_storage import EventStorage, TokenStorage  # noqa: F401
from .classification import (  # noqa: F401
    CATEGORIES,
    EventClassifier,
    classify_event_label,
    get_classifier,
)
from .linking import EventLinker, link_events  # noqa: F401

# Heavy: resolved on first attribute access.
_LAZY = {
    "PivotalToken": ("pts.core", "PivotalToken"),
    "PivotalTokenSearcher": ("pts.core", "PivotalTokenSearcher"),
    "TokenPTSSearcher": ("pts.searchers.token", "TokenPTSSearcher"),
    "SentencePTSSearcher": ("pts.searchers.sentence", "SentencePTSSearcher"),
    "LatentPTSSearcher": ("pts.searchers.latent", "LatentPTSSearcher"),
    "BasePTSSearcher": ("pts.searchers.base", "BasePTSSearcher"),
    "ThoughtAnchor": ("pts.thought_anchors", "ThoughtAnchor"),
    "ThoughtAnchorSearcher": ("pts.thought_anchors", "ThoughtAnchorSearcher"),
    "ThoughtAnchorStorage": ("pts.thought_anchors", "ThoughtAnchorStorage"),
    "Oracle": ("pts.oracle", "Oracle"),
    "MathOracle": ("pts.oracle", "MathOracle"),
    "CodeOracle": ("pts.oracle", "CodeOracle"),
    "QAOracle": ("pts.oracle", "QAOracle"),
    "DummyOracle": ("pts.oracle", "DummyOracle"),
    "OptiBenchOracle": ("pts.oracle", "OptiBenchOracle"),
    "load_dataset": ("pts.dataset", "load_dataset"),
    "create_oracle_from_dataset": ("pts.dataset", "create_oracle_from_dataset"),
    "TokenExporter": ("pts.exporters", "TokenExporter"),
    "EventExporter": ("pts.exporters", "EventExporter"),
    "JLens": ("pts.latent.jlens", "JLens"),
    "Readout": ("pts.latent.jlens", "Readout"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        try:
            module = import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"pts.{name} requires optional dependencies that are not installed "
                f"({e}). Install the full stack with: pip install 'pts[all]'"
            ) from e
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pts' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
