"""
Latent PTS -- searching the model's hidden workspace for pivotal events.

Anthropic's "Verbalizable Representations Form a Global Workspace in Language
Models" describes a Jacobian lens (J-lens) that reads, from a mid-layer
residual stream, the vocabulary tokens that activation is causally pushing the
model to say later. The set of concepts it surfaces is the J-space.

PTS uses that readout as a *search* primitive: the tokens a J-lens surfaces
just before an emitted pivotal token are candidate latent meta-tokens -- hidden
verbalizable concepts that may precede, and perhaps cause, the emitted event.

Important framing, stated once here and repeated in the docs: the paper does
not define "meta-token". That is our term for a top-k J-lens readout. These
readouts are noisy interpretability hypotheses, not ground truth, and until
they are validated by intervention they remain observational.

This subpackage is optional. It needs torch and transformers; ``import pts``
does not.
"""

from .activations import (
    ResidualCapture,
    collect_residual_activations,
    get_layer_modules,
    resolve_workspace_layers,
)
from .jlens import JLens, LogitLens, Readout, ReadoutResult, load_readout
from .metatokens import MetaTokenExtractor, enrich_events_with_latent

__all__ = [
    "ResidualCapture",
    "collect_residual_activations",
    "get_layer_modules",
    "resolve_workspace_layers",
    "JLens",
    "LogitLens",
    "Readout",
    "ReadoutResult",
    "load_readout",
    "MetaTokenExtractor",
    "enrich_events_with_latent",
]
