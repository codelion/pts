"""
The Jacobian lens (J-lens) and its cheaper relatives.

Following Anthropic's "Verbalizable Representations Form a Global Workspace in
Language Models", the J-lens for layer l is the expected Jacobian of the final
residual stream with respect to the layer-l residual stream::

    J_l = E[ d h_final,t' / d h_l,t ]

averaged over source positions t, all later positions t' >= t, and a set of
calibration prompts. Reading a vocabulary distribution out of an activation is
then::

    lens(h_l) = softmax( W_U . norm( J_l @ h_l ) )

The rows of ``W_U @ J_l`` are the J-lens vectors: one direction per vocabulary
token, each the average causal influence of that direction on eventually
producing that token.

No reference implementation was released with the paper, so this is written
from the equations. It has not been validated against the authors' results --
treat the readouts as hypotheses. See ``docs/latent_pts.md``.

Note that the **logit lens is exactly this construction with J = I**: it asks
what the activation would say if emitted right now, rather than what it is
pushing the model to say later. That makes it a principled zero-cost baseline
rather than a hack, and it is what ``--readout-method logit_lens`` uses.

Computing J efficiently
-----------------------
A naive Jacobian would need one backward pass per (source position, output
component) pair. Two observations collapse that:

1. Attention is causal, so ``h_l,t`` cannot influence ``h_final,t'`` for
   ``t' < t``. The gradient of the *total* ``S_i = sum_t' (h_final,t')_i`` with
   respect to ``h_l,t`` therefore already equals the sum over exactly the
   ``t' >= t`` terms we want -- the rest are structurally zero.
2. Autograd returns the gradient with respect to every source position at once.

So one backward pass per output component ``i`` yields row ``i`` of the summed
Jacobian for *all* source positions simultaneously, giving the paper's stated
``O(n x d_model)`` backward passes.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .activations import (
    ResidualCapture,
    get_final_norm,
    get_layer_modules,
    get_unembedding,
    resolve_workspace_layers,
)

logger = logging.getLogger(__name__)

READOUT_JLENS = "jlens"
READOUT_LOGIT_LENS = "logit_lens"
READOUT_METHODS = (READOUT_JLENS, READOUT_LOGIT_LENS)


@dataclass
class ReadoutResult:
    """One vocabulary token surfaced from an activation."""

    token: str
    token_id: int
    score: float
    rank: int
    layer: int
    position: Optional[int] = None
    readout_method: str = READOUT_JLENS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "token_id": self.token_id,
            "score": self.score,
            "rank": self.rank,
            "layer": self.layer,
            "position": self.position,
            "readout_method": self.readout_method,
        }


class Readout:
    """Base: turn a residual-stream activation into ranked vocabulary tokens."""

    method: str = "base"

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.final_norm = get_final_norm(model)
        self._W_U: Optional[torch.Tensor] = None

    @property
    def W_U(self) -> torch.Tensor:
        if self._W_U is None:
            self._W_U = get_unembedding(self.model)
        return self._W_U

    def transform(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a layer-l activation into the final-residual-stream basis."""
        raise NotImplementedError

    def read(
        self,
        activations: torch.Tensor,
        layer: int,
        top_k: int = 25,
        position: Optional[int] = None,
    ) -> List[ReadoutResult]:
        """Read the top-k vocabulary tokens out of one activation vector."""
        if activations.dim() != 1:
            raise ValueError(
                f"read() expects a single [d_model] activation, got {tuple(activations.shape)}"
            )

        with torch.no_grad():
            h = activations.to(self.W_U.dtype).to(self.W_U.device)
            projected = self.transform(h, layer)

            if self.final_norm is not None:
                projected = self.final_norm(projected)

            logits = self.W_U @ projected
            probs = torch.softmax(logits.float(), dim=-1)

            k = min(top_k, probs.shape[-1])
            top = torch.topk(probs, k=k)

        results = []
        for rank, (score, token_id) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
            results.append(
                ReadoutResult(
                    token=self.tokenizer.decode([token_id]),
                    token_id=token_id,
                    score=float(score),
                    rank=rank,
                    layer=layer,
                    position=position,
                    readout_method=self.method,
                )
            )
        return results

    def read_context(
        self,
        text: str,
        layers: Sequence[int],
        positions: str = "last",
        top_k: int = 25,
    ) -> Dict[int, List[ReadoutResult]]:
        """Read meta-tokens out of the workspace while the model processes ``text``."""
        from .activations import collect_residual_activations

        acts = collect_residual_activations(
            self.model, self.tokenizer, text, layers, positions=positions
        )

        out: Dict[int, List[ReadoutResult]] = {}
        for layer, hidden in acts.items():
            if hidden.dim() == 1:
                out[layer] = self.read(hidden, layer, top_k=top_k)
            else:
                # A window of positions: index them relative to the end, so
                # position -1 is the last token, -2 the one before it, etc.
                seq_len = hidden.shape[0]
                results: List[ReadoutResult] = []
                for i in range(seq_len):
                    offset = i - seq_len  # negative
                    results.extend(
                        self.read(hidden[i], layer, top_k=top_k, position=offset)
                    )
                out[layer] = results
        return out


class LogitLens(Readout):
    """The zero-cost baseline: J = I.

    Asks what the activation would emit *right now* if unembedded directly.
    That is a different question from what it is pushing the model to say
    *later*, which is what the J-lens measures -- so logit-lens readouts are
    weaker evidence for a workspace, and events produced this way are tagged
    ``readout_method="logit_lens"`` so they can be told apart downstream.
    """

    method = READOUT_LOGIT_LENS

    def transform(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        return h


class JLens(Readout):
    """The fitted Jacobian lens."""

    method = READOUT_JLENS

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        matrices: Optional[Dict[int, torch.Tensor]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model, tokenizer)
        self.matrices: Dict[int, torch.Tensor] = matrices or {}
        self.config: Dict[str, Any] = config or {}

    @property
    def layers(self) -> List[int]:
        return sorted(self.matrices)

    def transform(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        if layer not in self.matrices:
            raise KeyError(
                f"No J-lens matrix fitted for layer {layer}. "
                f"Fitted layers: {self.layers or 'none'}. "
                f"Run `pts fit-jlens --layers {layer}` to add it."
            )
        J = self.matrices[layer].to(h.dtype).to(h.device)
        return J @ h

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        calibration_texts: Sequence[str],
        layers: Sequence[int],
        seq_len: int = 128,
        basis_chunk: int = 64,
        device: Optional[str] = None,
        show_progress: bool = True,
    ) -> "JLens":
        """Estimate J_l for each layer by averaging Jacobians over calibration text.

        ``basis_chunk`` trades memory for speed: it is how many output basis
        directions are differentiated in one batched backward call. Lower it if
        you hit OOM.
        """
        from tqdm import tqdm

        device = device or str(next(self.model.parameters()).device)
        layer_modules = get_layer_modules(self.model)
        num_layers = len(layer_modules)
        final_layer = num_layers - 1

        for l in layers:
            if not (0 <= l < final_layer):
                raise ValueError(
                    f"Layer {l} cannot be a workspace layer: it must be below the "
                    f"final layer ({final_layer}) for a Jacobian to it to be nonzero."
                )

        d_model = self.model.config.hidden_size
        # Accumulate on CPU in float64: the running sum is small, it never needs
        # to be on the accelerator, float64 keeps it from drifting across many
        # prompts, and MPS has no float64 at all.
        accum = {l: torch.zeros(d_model, d_model, dtype=torch.float64) for l in layers}
        counts = {l: 0 for l in layers}

        capture = ResidualCapture(self.model, layer_modules)
        was_training = self.model.training
        self.model.eval()

        texts = list(calibration_texts)
        iterator = tqdm(texts, desc="Fitting J-lens", disable=not show_progress)

        for text in iterator:
            encoded = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=seq_len
            )
            input_ids = encoded.input_ids.to(device)
            if input_ids.shape[1] < 2:
                continue
            attention_mask = encoded.attention_mask.to(device)
            T = input_ids.shape[1]

            # Gradients flow to activations, not parameters.
            with capture.capture(list(layers) + [final_layer], keep_graph=True) as acts:
                with torch.enable_grad():
                    self.model(input_ids=input_ids, attention_mask=attention_mask)

                    h_final = acts[final_layer][0]  # [T, d_model]

                    for l in layers:
                        h_l = acts[l]  # [1, T, d_model], still in the graph
                        J_l = self._jacobian_for_layer(
                            h_final, h_l, T, d_model, basis_chunk, device
                        )
                        accum[l] += J_l.detach().cpu().to(torch.float64)
                        counts[l] += 1

            del acts

        if was_training:
            self.model.train()

        for l in layers:
            if counts[l] == 0:
                logger.warning(f"No usable calibration text for layer {l}; skipping")
                continue
            self.matrices[l] = (accum[l] / counts[l]).to(torch.float32)

        self.config.update(
            {
                "model_id": getattr(self.model.config, "_name_or_path", "unknown"),
                "d_model": d_model,
                "num_layers": num_layers,
                "layers": sorted(self.matrices),
                "num_sequences": len(texts),
                "seq_len": seq_len,
                "method": "averaged_jacobian",
            }
        )
        logger.info(f"Fitted J-lens for layers {sorted(self.matrices)} on {len(texts)} sequences")
        return self

    def _jacobian_for_layer(
        self,
        h_final: torch.Tensor,
        h_l: torch.Tensor,
        T: int,
        d_model: int,
        basis_chunk: int,
        device: str,
    ) -> torch.Tensor:
        """Average Jacobian d h_final / d h_l over source positions, for one prompt."""
        J = torch.zeros(d_model, d_model, dtype=torch.float32, device=device)

        # Row t of the raw gradient is the sum over t' >= t (causality does the
        # masking for free). Dividing by the number of such t' turns that sum
        # into the mean the definition asks for. The last position has exactly
        # one t' (itself), the first has T.
        t_index = torch.arange(T, device=device, dtype=torch.float32)
        per_source_norm = (T - t_index).clamp(min=1.0).unsqueeze(1)  # [T, 1]

        for start in range(0, d_model, basis_chunk):
            end = min(start + basis_chunk, d_model)
            B = end - start

            # grad_outputs[b] selects component (start+b) of h_final at *every*
            # position t', which is exactly S_i = sum_t' (h_final,t')_i.
            grad_outputs = torch.zeros(B, T, d_model, device=device, dtype=h_final.dtype)
            for b in range(B):
                grad_outputs[b, :, start + b] = 1.0

            grads = torch.autograd.grad(
                outputs=h_final,
                inputs=h_l,
                grad_outputs=grad_outputs,
                retain_graph=True,
                is_grads_batched=True,
            )[0]  # [B, 1, T, d_model]

            grads = grads.squeeze(1).to(torch.float32)  # [B, T, d_model]
            grads = grads / per_source_norm             # mean over t' >= t
            J[start:end, :] = grads.mean(dim=1)         # mean over source positions t

        return J

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        import numpy as np

        np.savez_compressed(
            os.path.join(path, "jlens.npz"),
            **{f"layer_{l}": m.numpy() for l, m in self.matrices.items()},
        )
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved J-lens ({len(self.matrices)} layers) to {path}")

    @classmethod
    def load(cls, path: str, model: Any, tokenizer: Any) -> "JLens":
        import numpy as np

        npz_path = os.path.join(path, "jlens.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"No J-lens at {path} (expected {npz_path}). "
                f"Fit one with: pts fit-jlens --model <model> --output-path {path}"
            )

        data = np.load(npz_path)
        matrices = {
            int(key.split("_")[1]): torch.from_numpy(data[key]) for key in data.files
        }

        config = {}
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)

        fitted_for = config.get("model_id")
        current = getattr(model.config, "_name_or_path", None)
        if fitted_for and current and fitted_for != current:
            logger.warning(
                f"This J-lens was fitted on {fitted_for} but is being used with "
                f"{current}. J-lens matrices are model-specific; readouts will be "
                "meaningless across models."
            )

        logger.info(f"Loaded J-lens for layers {sorted(matrices)} from {path}")
        return cls(model, tokenizer, matrices=matrices, config=config)


def load_readout(
    model: Any,
    tokenizer: Any,
    method: str = READOUT_JLENS,
    jlens_path: Optional[str] = None,
) -> Readout:
    """Build the requested readout, or explain why it cannot be built.

    Falling back silently from J-lens to logit-lens would let a run claim
    workspace evidence it did not gather, so an unavailable J-lens is an error
    unless the caller explicitly asks for ``logit_lens``.
    """
    if method == READOUT_LOGIT_LENS:
        return LogitLens(model, tokenizer)

    if method == READOUT_JLENS:
        if not jlens_path:
            raise ValueError(
                "--readout-method jlens requires --jlens-path pointing at fitted "
                "matrices. Fit them with `pts fit-jlens`, or use "
                "`--readout-method logit_lens` for the zero-cost baseline "
                "(weaker evidence: it reads what the activation would say now, "
                "not what it pushes the model to say later)."
            )
        return JLens.load(jlens_path, model, tokenizer)

    raise ValueError(
        f"Unknown readout method {method!r}. Available: {', '.join(READOUT_METHODS)}."
    )
