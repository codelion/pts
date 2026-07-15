"""
Residual-stream capture for decoder-only transformers.

The J-lens needs two things from a forward pass: the residual stream at some
mid-network layer, and the residual stream at the final layer. Getting them
means hooking the right modules, and HuggingFace has no single convention for
where the decoder blocks live -- hence the sniffing below.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


def get_layer_modules(model: Any) -> Sequence[Any]:
    """Return the list of decoder blocks, whatever the architecture calls it."""
    candidates = [
        ("model.layers", lambda m: m.model.layers),                 # Llama, Qwen, Mistral, Gemma
        ("transformer.h", lambda m: m.transformer.h),               # GPT-2, GPT-J, Falcon
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),           # GPT-NeoX, Pythia
        ("model.decoder.layers", lambda m: m.model.decoder.layers), # OPT
        ("layers", lambda m: m.layers),                             # already-unwrapped
    ]

    for name, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            logger.debug(f"Found {len(layers)} decoder layers at {name}")
            return layers

    raise AttributeError(
        f"Could not locate decoder layers on {type(model).__name__}. Tried: "
        f"{', '.join(n for n, _ in candidates)}. If this is a custom architecture, "
        "pass the layer list explicitly to ResidualCapture(layers=...)."
    )


def get_final_norm(model: Any) -> Optional[Any]:
    """The norm applied to the final residual stream before unembedding."""
    for getter in (
        lambda m: m.model.norm,           # Llama, Qwen, Mistral
        lambda m: m.transformer.ln_f,     # GPT-2, Falcon
        lambda m: m.gpt_neox.final_layer_norm,
        lambda m: m.model.decoder.final_layer_norm,
        lambda m: m.model.final_layernorm,
    ):
        try:
            norm = getter(model)
        except AttributeError:
            continue
        if norm is not None:
            return norm
    logger.warning("Could not locate the final norm; readouts will skip normalization.")
    return None


def get_unembedding(model: Any) -> torch.Tensor:
    """The unembedding matrix W_U, shape [vocab, d_model]."""
    head = getattr(model, "lm_head", None)
    if head is not None and hasattr(head, "weight"):
        return head.weight
    head = getattr(model, "embed_out", None)  # GPT-NeoX
    if head is not None and hasattr(head, "weight"):
        return head.weight
    try:
        return model.get_output_embeddings().weight
    except (AttributeError, TypeError):
        pass
    raise AttributeError(
        f"Could not locate the unembedding matrix on {type(model).__name__}"
    )


def resolve_workspace_layers(
    num_layers: int,
    layers: Optional[Sequence[int]] = None,
    fraction: Tuple[float, float] = (0.38, 0.92),
    max_layers: int = 4,
) -> List[int]:
    """Pick the layers to read the workspace from.

    The paper locates the workspace between roughly 38% and 92% of depth --
    after the early "parsing" layers and before the final "output" layers -- so
    that band is the default rather than any hardcoded layer numbers, which
    would not transfer across model sizes.
    """
    if layers:
        bad = [l for l in layers if not (0 <= l < num_layers)]
        if bad:
            raise ValueError(
                f"Layers {bad} are out of range for a {num_layers}-layer model"
            )
        return sorted(layers)

    lo_frac, hi_frac = fraction
    lo = max(0, int(round(lo_frac * num_layers)))
    # A workspace layer must sit strictly below the final layer: the Jacobian
    # from the final layer to itself is the identity and carries no information,
    # and JLens.fit rejects it. For any model with <= 18 layers, round(0.92 * n)
    # lands on n-1, so the naive bound made fit-jlens unusable on GPT-2,
    # Llama-3.2-1B, and every other small model.
    hi = min(num_layers - 2, int(round(hi_frac * num_layers)))
    if hi <= lo:
        return [max(0, min(num_layers - 2, num_layers // 2))]

    n = min(max_layers, hi - lo + 1)
    if n == 1:
        return [(lo + hi) // 2]

    step = (hi - lo) / (n - 1)
    selected = sorted({int(round(lo + i * step)) for i in range(n)})
    logger.info(
        f"Auto-selected workspace layers {selected} "
        f"({lo_frac:.0%}-{hi_frac:.0%} of {num_layers} layers)"
    )
    return selected


class ResidualCapture:
    """Capture the residual stream at chosen layers during a forward pass.

    Hooks are registered on entry and always removed on exit, including when the
    forward pass raises -- a leaked hook would silently corrupt every later
    forward pass on the same model.
    """

    def __init__(self, model: Any, layers: Optional[Sequence[Any]] = None):
        self.model = model
        self.layer_modules = layers if layers is not None else get_layer_modules(model)
        self.num_layers = len(self.layer_modules)
        self.activations: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []

    def _make_hook(self, layer_idx: int, keep_graph: bool):
        def hook(module, inputs, output):
            # Decoder blocks return either a tensor or a tuple whose first
            # element is the hidden state.
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden if keep_graph else hidden.detach()
        return hook

    @contextmanager
    def capture(self, layer_indices: Sequence[int], keep_graph: bool = False):
        """Capture ``layer_indices`` for the duration of the block.

        ``keep_graph`` keeps the activations attached to the autograd graph,
        which J-lens fitting needs in order to differentiate through them.
        """
        self.activations = {}
        try:
            for idx in layer_indices:
                handle = self.layer_modules[idx].register_forward_hook(
                    self._make_hook(idx, keep_graph)
                )
                self._handles.append(handle)
            yield self.activations
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles = []


def collect_residual_activations(
    model: Any,
    tokenizer: Any,
    text: str,
    layers: Sequence[int],
    positions: str = "last",
    device: Optional[str] = None,
    max_length: int = 2048,
) -> Dict[int, torch.Tensor]:
    """Run ``text`` through the model and return the residual stream at ``layers``.

    ``positions`` is either ``"last"`` (the final token only, shape [d_model]),
    ``"all"`` (every position, shape [seq, d_model]), or a window like ``"-8:"``
    meaning the last 8 positions.
    """
    device = device or next(model.parameters()).device
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    capture = ResidualCapture(model)
    with capture.capture(layers) as acts:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        out: Dict[int, torch.Tensor] = {}
        for layer in layers:
            if layer not in acts:
                continue
            hidden = acts[layer][0]  # drop the batch dim -> [seq, d_model]
            if positions == "last":
                out[layer] = hidden[-1]
            elif positions == "all":
                out[layer] = hidden
            elif positions.startswith("-") and positions.endswith(":"):
                n = int(positions[1:-1])
                out[layer] = hidden[-n:]
            else:
                raise ValueError(
                    f"positions must be 'last', 'all', or '-N:'; got {positions!r}"
                )
        return out
