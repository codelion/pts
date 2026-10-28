"""Model precision: which dtype gets picked where, and that half precision runs.

The selection logic is a pure function, so every (device, flag, purpose)
combination is checked here without the hardware. The device tests at the bottom
run on MPS or CUDA when present and are skipped otherwise.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pts.cli import parse_args  # noqa: E402
from pts.latent.jlens import JLens  # noqa: E402
from pts.oracle import Oracle  # noqa: E402
from pts.searchers import base as base_module  # noqa: E402
from pts.searchers.base import (  # noqa: E402
    BasePTSSearcher,
    ensure_dtype_runs,
    load_causal_lm,
    resolve_dtype,
)
from pts.searchers.token import TokenPTSSearcher  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"

HAS_MPS = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()


class AlwaysOracle(Oracle):
    def check_success(self, query, response):
        return True


# -- resolve_dtype ---------------------------------------------------------

@pytest.mark.parametrize("device,ampere,expected", [
    ("cpu", None, torch.float32),
    ("mps", None, torch.float32),
    ("cuda", True, torch.bfloat16),
    ("cuda", False, torch.float16),
    ("cuda:1", True, torch.bfloat16),
])
def test_auto_keeps_the_existing_defaults(monkeypatch, device, ampere, expected):
    monkeypatch.setattr(base_module, "_cuda_supports_bf16", lambda: ampere)
    assert resolve_dtype("auto", device) == expected
    assert resolve_dtype(None, device) == expected


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("flag,expected", [
    ("fp32", torch.float32),
    ("bf16", torch.bfloat16),
    ("fp16", torch.float16),
])
def test_an_explicit_dtype_is_honoured_on_every_device(device, flag, expected):
    assert resolve_dtype(flag, device) == expected


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("flag", ["auto", "fp32", "bf16", "fp16", None])
def test_jlens_fitting_is_always_fp32(device, flag):
    """Half-precision Jacobians underflow. Fitting must ignore any --dtype."""
    assert resolve_dtype(flag, device, purpose="fit_jlens") == torch.float32


def test_unknown_dtype_is_an_error():
    with pytest.raises(ValueError, match="Unknown dtype"):
        resolve_dtype("int8", "cpu")


def test_fit_jlens_has_no_dtype_flag():
    with pytest.raises(SystemExit):
        parse_args(["fit-jlens", "--model", TINY, "--output-path", "x", "--dtype", "bf16"])


@pytest.mark.parametrize("argv", [
    ["run", "--model", TINY],
    ["enrich", "--model", TINY, "--input-path", "a", "--output-path", "b"],
    ["export", "--input-path", "a", "--output-path", "b", "--format", "steering"],
])
def test_dtype_flag_defaults_to_auto(argv):
    args, _ = parse_args(argv)
    assert args.dtype == "auto"
    args, _ = parse_args(argv + ["--dtype", "bf16"])
    assert args.dtype == "bf16"


# -- loading and fallback (CPU) --------------------------------------------

def test_bf16_loads_and_runs_on_cpu():
    model = load_causal_lm(TINY, "cpu", dtype="bf16")
    assert model.dtype == torch.bfloat16
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]])).logits
    assert torch.isfinite(logits).all()


def test_a_dtype_that_cannot_run_falls_back_to_fp32(caplog):
    """MPS can lack a half-precision kernel; that must degrade, not crash mid-search."""
    model = load_causal_lm(TINY, "cpu", dtype="bf16")
    real_forward = model.forward

    def forward(*args, **kwargs):
        if model.dtype != torch.float32:
            raise RuntimeError("no bf16 kernel for this op")
        return real_forward(*args, **kwargs)

    model.forward = forward
    with caplog.at_level("WARNING"):
        model = ensure_dtype_runs(model, "mps")
    assert model.dtype == torch.float32
    assert "falling back to float32" in caplog.text


def test_searcher_records_its_dtype_on_events():
    s = TokenPTSSearcher(
        model_name=TINY, oracle=AlwaysOracle(), device="cpu", dtype="bf16",
        num_samples=2, batch_size=2, max_new_tokens=4,
    )
    assert s.model.dtype == torch.bfloat16
    assert s.dtype_name == "bfloat16"
    # AlwaysOracle saturates the baseline at 1.0, so search finds nothing;
    # check the metadata path through a shared-model searcher instead.
    shared = BasePTSSearcher(
        model_name=TINY, oracle=AlwaysOracle(), device="cpu",
        model=s.model, tokenizer=s.tokenizer,
    )
    assert shared.dtype_name == "bfloat16"
    p = shared.estimate_success_probability("What is 2+2?")
    assert 0.0 <= p <= 1.0


def test_readout_from_bf16_activations_is_a_float32_distribution():
    fp32 = load_causal_lm(TINY, "cpu", dtype="fp32")
    bf16 = load_causal_lm(TINY, "cpu", dtype="bf16")
    d = fp32.config.hidden_size
    tok = transformers.AutoTokenizer.from_pretrained(TINY)
    h = torch.randn(d)

    a = JLens(fp32, tok, matrices={0: torch.eye(d)}).read(h, 0, top_k=5)
    b = JLens(bf16, tok, matrices={0: torch.eye(d)}).read(h.to(torch.bfloat16), 0, top_k=5)

    assert all(isinstance(r.score, float) and 0.0 <= r.score <= 1.0 for r in b)
    # bf16 has ~3 significant digits; scores should agree to that order.
    assert abs(a[0].score - b[0].score) < 2e-2


# -- device tests (skipped without the hardware) ---------------------------

def _logits(model, device):
    with torch.no_grad():
        return model(torch.tensor([[1, 2, 3, 4, 5]], device=device)).logits.float().cpu()


@pytest.mark.parametrize("device,available", [("mps", HAS_MPS), ("cuda", HAS_CUDA)])
@pytest.mark.parametrize("flag", ["bf16", "fp16"])
def test_half_precision_matches_fp32_on_device(monkeypatch, device, available, flag):
    if not available:
        pytest.skip(f"{device} not available")
    # Without this, a missing MPS kernel silently runs on CPU instead of failing.
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    ref = _logits(load_causal_lm(TINY, device, dtype="fp32"), device)
    half = load_causal_lm(TINY, device, dtype=flag)
    assert half.dtype != torch.float32, f"{flag} fell back to fp32 on {device}"
    out = _logits(half, device)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=5e-2, rtol=5e-2)
