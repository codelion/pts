"""MLX and PyTorch agree: same weights, same vector, same numbers.

This is what makes the format backend-neutral rather than merely
backend-shared. Both backends load the same tiny random Llama checkpoint in
float32; they agree to ~1e-7 before steering, so tolerances here are tight.
Skipped unless torch, transformers, MLX and mlx-lm are all installed.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from pts.steering import capture, mlx_backend, steer, torch_backend  # noqa: E402
from pts.steering.vector import LayerVector, SteeringVector  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
IDS = [1, 5, 9, 2, 7, 11]
ATOL = 1e-5


@pytest.fixture(scope="module")
def models():
    tm = transformers.AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32).eval()
    mm, _ = mlx_lm.load(TINY)
    return tm, mm


def torch_logits(m):
    with torch.no_grad():
        return m(torch.tensor([IDS])).logits[0].numpy()


def mlx_logits(m):
    out = m(mx.array(IDS)[None]).astype(mx.float32)
    mx.eval(out)
    return np.array(out[0])


def test_both_backends_see_the_same_model(models):
    tm, mm = models
    assert torch_backend.model_shape(tm) == mlx_backend.model_shape(mm)
    np.testing.assert_allclose(torch_logits(tm), mlx_logits(mm), atol=ATOL)


def test_hook_points_agree_at_every_layer(models):
    tm, mm = models
    n = torch_backend.model_shape(tm)[1]
    a, b = capture(tm, IDS, range(n)), capture(mm, IDS, range(n))
    for i in range(n):
        np.testing.assert_allclose(a[i], b[i], atol=ATOL)


def test_steered_logits_agree(models):
    tm, mm = models
    d, n = torch_backend.model_shape(tm)
    rng = np.random.default_rng(0)
    vectors = [
        (SteeringVector([LayerVector(0, "add", rng.normal(size=d))], d, n), 3.0),
        (SteeringVector([LayerVector(0, "ablate", rng.normal(size=d)),
                         LayerVector(1, "add", rng.normal(size=d))], d, n), -2.0),
    ]
    with steer(tm, vectors):
        t = torch_logits(tm)
    with steer(mm, vectors):
        m = mlx_logits(mm)
    np.testing.assert_allclose(t, m, atol=ATOL)
    assert not np.allclose(t, torch_logits(tm), atol=1e-3), "steering changed nothing"


def test_vectors_built_on_either_backend_are_the_same(models, tmp_path):
    tm, mm = models
    pos, neg = [IDS, IDS[:4]], [IDS[::-1], IDS[1:]]
    a = torch_backend.mean_difference(tm, pos, neg, layers=[0, 1])
    b = mlx_backend.mean_difference(mm, pos, neg, layers=[0, 1])
    for x, y in zip(a.entries, b.entries):
        np.testing.assert_allclose(x.vector, y.vector, atol=ATOL)
        assert x.ref_norm == pytest.approx(y.ref_norm, rel=1e-5)


def test_a_vector_saved_from_mlx_steers_torch_identically(models, tmp_path):
    tm, mm = models
    vec = mlx_backend.mean_difference(mm, [IDS], [IDS[::-1]], layers=[1], base_model=TINY)
    path = vec.save(str(tmp_path / "v.safetensors"))
    loaded = SteeringVector.load(path)
    with loaded.apply(mm, alpha=40.0):
        m = mlx_logits(mm)
    with loaded.apply(tm, alpha=40.0):
        t = torch_logits(tm)
    np.testing.assert_allclose(t, m, atol=ATOL)
