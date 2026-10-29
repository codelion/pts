"""
Steering vectors: one on-disk format for every backend.

A steering vector set is a directory, or a single ``.safetensors`` file, holding
one direction per layer::

    steering_config.json          what the vectors are and where they apply
    steering_vectors.safetensors  tensors named "layers.<L>.<op>"

The hook point is fixed: the residual stream at the output of decoder block
``L``, which is what Hugging Face calls ``hidden_states[L + 1]``. Every backend
hooks exactly there, so a vector extracted on MLX means the same thing on
PyTorch. Changing that definition silently invalidates every published vector;
add a new ``hook`` value instead.

Two operations:

- ``add``: ``h <- h + alpha * v``. ``alpha = 1`` reproduces the vector as it
  was extracted (e.g. the raw mean difference).
- ``ablate``: ``h <- h - (h . u) u`` for the unit direction ``u``. Always full;
  ``alpha`` does not apply to ablations.

This module needs numpy and safetensors but not torch or MLX.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FORMAT_VERSION = "1"
HOOK_POINT = "block_output"
OPS = ("add", "ablate")
POSITIONS = ("all",)
CONFIG_NAME = "steering_config.json"
WEIGHTS_NAME = "steering_vectors.safetensors"
# safetensors metadata is str -> str, so the whole config rides along as JSON.
# That lets a lone .safetensors file be loaded without its sidecar.
METADATA_KEY = "pts_steering_config"
HUB_PREFIX = "hf://"


@dataclass
class LayerVector:
    """One direction applied at one layer."""

    layer: int
    op: str
    vector: np.ndarray
    # Mean residual-stream norm at this layer when the vector was extracted.
    # Used to warn when steering adds more than the activation itself carries.
    ref_norm: Optional[float] = None

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"Unknown op {self.op!r}; expected one of {OPS}")
        self.vector = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        if self.op == "ablate":
            norm = float(np.linalg.norm(self.vector))
            if norm == 0.0:
                raise ValueError(f"Cannot ablate a zero vector at layer {self.layer}")
            self.vector = self.vector / norm

    @property
    def tensor_name(self) -> str:
        return f"layers.{self.layer}.{self.op}"


@dataclass
class SteeringVector:
    """A set of per-layer directions plus everything needed to apply them safely."""

    entries: List[LayerVector]
    hidden_size: int
    num_layers: int
    base_model: Optional[str] = None
    compatible_models: List[str] = field(default_factory=list)
    default_alpha: float = 1.0
    positions: str = "all"
    behavior: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    hook: str = HOOK_POINT

    def __post_init__(self):
        if self.hook != HOOK_POINT:
            raise ValueError(f"Unsupported hook {self.hook!r}; this version only knows {HOOK_POINT!r}")
        if self.positions not in POSITIONS:
            raise ValueError(f"Unsupported positions {self.positions!r}; expected one of {POSITIONS}")
        seen = set()
        for e in self.entries:
            if not 0 <= e.layer < self.num_layers:
                raise ValueError(f"Layer {e.layer} is outside a {self.num_layers}-layer model")
            if e.vector.shape != (self.hidden_size,):
                raise ValueError(
                    f"{e.tensor_name} has shape {e.vector.shape}, expected ({self.hidden_size},)"
                )
            if e.tensor_name in seen:
                raise ValueError(f"Duplicate entry {e.tensor_name}")
            seen.add(e.tensor_name)

    @property
    def layers(self) -> List[int]:
        return sorted({e.layer for e in self.entries})

    # -- compatibility ------------------------------------------------------

    def check_compatible(
        self, hidden_size: int, num_layers: int, model_id: Optional[str] = None
    ) -> None:
        """Refuse a model of the wrong shape; warn on a model we cannot vouch for.

        Width and depth must match exactly. The model id only warns, because the
        same weights appear under many names (a 4-bit mlx-community copy has the
        same residual stream as the bf16 original).
        """
        if hidden_size != self.hidden_size or num_layers != self.num_layers:
            raise ValueError(
                f"Steering vector is for a model with hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}; this model has hidden_size={hidden_size}, "
                f"num_layers={num_layers}"
            )
        if model_id and self.base_model:
            known = [self.base_model, *self.compatible_models]
            if not any(_same_model(model_id, k) for k in known):
                logger.warning(
                    f"Steering vector was extracted from {self.base_model}; applying it to "
                    f"{model_id}. Shapes match, but the residual stream may not."
                )

    # -- serialisation ------------------------------------------------------

    def to_config(self) -> Dict[str, Any]:
        return {
            "format": "pts-steering",
            "format_version": FORMAT_VERSION,
            "hook": self.hook,
            "base_model": self.base_model,
            "compatible_models": list(self.compatible_models),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "default_alpha": self.default_alpha,
            "positions": self.positions,
            "behavior": self.behavior,
            "layers": [
                {"layer": e.layer, "op": e.op, "tensor": e.tensor_name, "ref_norm": e.ref_norm}
                for e in sorted(self.entries, key=lambda e: (e.layer, e.op))
            ],
            "provenance": self.provenance,
            "validation": self.validation,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], tensors: Dict[str, np.ndarray]) -> "SteeringVector":
        if config.get("format") != "pts-steering":
            raise ValueError("Not a PTS steering vector (missing format: pts-steering)")
        if str(config.get("format_version")) != FORMAT_VERSION:
            raise ValueError(
                f"Steering format version {config.get('format_version')!r} is not supported "
                f"(this PTS reads version {FORMAT_VERSION})"
            )
        entries = []
        for spec in config["layers"]:
            name = spec["tensor"]
            if name not in tensors:
                raise ValueError(f"Config names tensor {name!r} but the file does not contain it")
            entries.append(
                LayerVector(
                    layer=int(spec["layer"]), op=spec["op"], vector=tensors[name],
                    ref_norm=spec.get("ref_norm"),
                )
            )
        return cls(
            entries=entries,
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            base_model=config.get("base_model"),
            compatible_models=list(config.get("compatible_models") or []),
            default_alpha=float(config.get("default_alpha", 1.0)),
            positions=config.get("positions", "all"),
            behavior=config.get("behavior"),
            provenance=dict(config.get("provenance") or {}),
            validation=dict(config.get("validation") or {}),
            hook=config.get("hook", HOOK_POINT),
        )

    def save(self, path: str) -> str:
        """Write a directory (config + tensors), or a single file if ``path`` ends in .safetensors."""
        from safetensors.numpy import save_file

        config = self.to_config()
        tensors = {e.tensor_name: e.vector for e in self.entries}
        metadata = {METADATA_KEY: json.dumps(config)}
        if path.endswith(".safetensors"):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            save_file(tensors, path, metadata=metadata)
            return path
        os.makedirs(path, exist_ok=True)
        save_file(tensors, os.path.join(path, WEIGHTS_NAME), metadata=metadata)
        with open(os.path.join(path, CONFIG_NAME), "w") as f:
            json.dump(config, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "SteeringVector":
        """Load from a directory, a .safetensors file, or ``hf://org/repo[/subdir|/file.safetensors]``."""
        if path.startswith(HUB_PREFIX):
            path = _download_from_hub(path)
        from safetensors import safe_open
        from safetensors.numpy import load_file

        if os.path.isdir(path):
            weights = os.path.join(path, WEIGHTS_NAME)
            with open(os.path.join(path, CONFIG_NAME)) as f:
                config = json.load(f)
        else:
            weights = path
            with safe_open(weights, framework="numpy") as f:
                meta = f.metadata() or {}
            if METADATA_KEY not in meta:
                raise ValueError(
                    f"{path} has no embedded steering config; load its directory instead"
                )
            config = json.loads(meta[METADATA_KEY])
        return cls.from_config(config, load_file(weights))


def _same_model(a: str, b: str) -> bool:
    """Loose model-id match: compare repo names, ignoring org and quantisation suffixes."""

    def core(name: str) -> str:
        name = name.rstrip("/").split("/")[-1].lower()
        for suffix in ("-4bit", "-8bit", "-3bit", "-6bit", "-bf16", "-fp16", "-mlx"):
            name = name.replace(suffix, "")
        return name

    return core(a) == core(b)


def parse_hub_path(path: str) -> Tuple[str, Optional[str]]:
    """Split ``hf://org/repo/sub/path`` into ``("org/repo", "sub/path")``; the path may be empty."""
    parts = path[len(HUB_PREFIX):].split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Bad Hub steering path {path!r}: expected hf://<org>/<repo>[/<path>]")
    return f"{parts[0]}/{parts[1]}", (parts[2] if len(parts) == 3 and parts[2] else None)


def _download_from_hub(path: str) -> str:
    from huggingface_hub import hf_hub_download

    repo_id, sub = parse_hub_path(path)
    if sub and sub.endswith(".safetensors"):
        return hf_hub_download(repo_id, sub)
    prefix = f"{sub.rstrip('/')}/" if sub else ""
    config = hf_hub_download(repo_id, f"{prefix}{CONFIG_NAME}")
    hf_hub_download(repo_id, f"{prefix}{WEIGHTS_NAME}")
    return os.path.dirname(config)


def parse_vector_spec(spec: str) -> Tuple[str, Optional[float]]:
    """``"path:4.0"`` -> ``("path", 4.0)``; ``"hf://org/repo"`` -> ``("hf://org/repo", None)``."""
    head, sep, tail = spec.rpartition(":")
    if sep and head:
        try:
            return head, float(tail)
        except ValueError:
            pass
    return spec, None


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------

@dataclass
class LayerPlan:
    """What a backend applies at one layer: ablate first, then add."""

    ablate: Optional[np.ndarray] = None  # [k, d], orthonormal rows
    add: Optional[np.ndarray] = None     # [d]
    ref_norm: Optional[float] = None


def combine(
    vectors: Sequence[Tuple[SteeringVector, Optional[float]]],
    cosine_warning: float = 0.9,
    norm_warning: float = 1.0,
) -> Dict[int, LayerPlan]:
    """Merge several steering vectors into one plan per layer.

    Additions sum (``sum_i alpha_i v_i``), so stacking is linear, like LoRA
    adapters with weights. Ablation directions from every vector at a layer are
    orthonormalised together: removing them one at a time does not remove their
    span when they are not orthogonal. Ablations run before additions, so a
    vector cannot be added and then partly ablated away.

    Warns when two added vectors at a layer are nearly parallel (stacking them
    mostly doubles the strength) and when the total addition is larger than the
    residual stream it is added to, which is where output turns to noise.
    """
    adds: Dict[int, List[Tuple[np.ndarray, str]]] = {}
    ablates: Dict[int, List[np.ndarray]] = {}
    ref_norms: Dict[int, float] = {}

    for i, (vec, alpha) in enumerate(vectors):
        alpha = vec.default_alpha if alpha is None else alpha
        label = vec.behavior or f"vector {i}"
        for e in vec.entries:
            if e.ref_norm is not None:
                ref_norms.setdefault(e.layer, e.ref_norm)
            if e.op == "add":
                if alpha != 0.0:
                    adds.setdefault(e.layer, []).append((alpha * e.vector, label))
            else:
                ablates.setdefault(e.layer, []).append(e.vector)

    plans: Dict[int, LayerPlan] = {}
    for layer in sorted(set(adds) | set(ablates)):
        plan = LayerPlan(ref_norm=ref_norms.get(layer))
        if layer in ablates:
            q, r = np.linalg.qr(np.stack(ablates[layer]).T)
            keep = np.abs(np.diag(r)) > 1e-6 * max(1.0, float(np.abs(r).max()))
            plan.ablate = q[:, keep].T.astype(np.float32)
        if layer in adds:
            parts = adds[layer]
            for (a, la), (b, lb) in _pairs(parts):
                cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                if abs(cos) > cosine_warning:
                    logger.warning(
                        f"Layer {layer}: {la} and {lb} are nearly parallel (cos={cos:.2f}); "
                        "stacking them mostly changes the strength"
                    )
            plan.add = np.sum([p for p, _ in parts], axis=0).astype(np.float32)
            if plan.ref_norm:
                ratio = float(np.linalg.norm(plan.add)) / plan.ref_norm
                if ratio > norm_warning:
                    logger.warning(
                        f"Layer {layer}: steering adds {ratio:.1f}x the typical residual norm; "
                        "expect degraded output"
                    )
        plans[layer] = plan
    return plans


def _pairs(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def load_vectors(specs: Sequence[str]) -> List[Tuple[SteeringVector, Optional[float]]]:
    """Load ``path[:alpha]`` specs, as given to ``--vector``."""
    out = []
    for spec in specs:
        path, alpha = parse_vector_spec(spec)
        vec = SteeringVector.load(path)
        if alpha is not None and not math.isfinite(alpha):
            raise ValueError(f"Bad steering strength in {spec!r}")
        out.append((vec, alpha))
    return out
