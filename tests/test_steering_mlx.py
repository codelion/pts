"""Steering on MLX: the hook point, ablation, stacking, and restoring blocks.

A tiny random Llama is built in memory, so no download is needed. Skipped where
MLX is not installed (anything but Apple silicon). The weights are random, so
these assert exact math, never sensible outputs.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from pts.steering.vector import LayerVector, SteeringVector  # noqa: E402

D, L = 16, 4


def make_vector(seed=0, layers=(1,), op="add", **kw) -> SteeringVector:
    rng = np.random.default_rng(seed)
    entries = [LayerVector(layer=i, op=op, vector=rng.normal(size=D), ref_norm=10.0) for i in layers]
    return SteeringVector(entries=entries, hidden_size=D, num_layers=L, **kw)



@pytest.fixture(scope="module")
def tiny():
    from mlx_lm.models import llama

    mx.random.seed(0)
    args = llama.ModelArgs(
        model_type="llama", hidden_size=D, num_hidden_layers=L, intermediate_size=32,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=64,
    )
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


IDS = [1, 5, 9, 2, 7]


def logits(model, ids=IDS):
    out = model(mx.array(ids)[None]).astype(mx.float32)
    mx.eval(out)
    return np.array(out[0])


def test_capture_matches_the_models_own_output(tiny):
    """The hook point is the residual after block L: norm + unembed of the last one gives the logits."""
    from pts.steering.mlx_backend import capture

    h = capture(tiny, IDS, [L - 1])[L - 1]
    final = tiny.model.norm(mx.array(h)[None])
    np.testing.assert_allclose(
        np.array(tiny.model.embed_tokens.as_linear(final)[0]), logits(tiny), atol=1e-4
    )


def test_zero_alpha_is_an_exact_no_op_and_blocks_are_restored(tiny):
    from pts.steering.mlx_backend import get_layers, steer

    before_blocks = list(get_layers(tiny))
    base = logits(tiny)
    with steer(tiny, [(make_vector(layers=(1, 2)), 0.0)]):
        np.testing.assert_array_equal(logits(tiny), base)
    assert all(a is b for a, b in zip(get_layers(tiny), before_blocks))


def test_addition_lands_exactly_at_the_hook_point(tiny):
    from pts.steering.mlx_backend import capture, steer

    v = make_vector(layers=(1,))
    base = capture(tiny, IDS, [1])[1]
    with steer(tiny, [(v, 2.0)]):
        steered = capture(tiny, IDS, [1])[1]
    np.testing.assert_allclose(steered - base, np.tile(2.0 * v.entries[0].vector, (len(IDS), 1)), atol=1e-4)


def test_ablation_removes_the_direction(tiny):
    from pts.steering.mlx_backend import capture, steer

    v = make_vector(layers=(2,), op="ablate")
    u = v.entries[0].vector
    assert np.abs(capture(tiny, IDS, [2])[2] @ u).max() > 1e-3
    with steer(tiny, [(v, None)]):
        assert np.abs(capture(tiny, IDS, [2])[2] @ u).max() < 1e-4


def test_stacking_equals_the_summed_vector(tiny):
    from pts.steering.mlx_backend import steer

    a, b = make_vector(seed=1), make_vector(seed=2)
    summed = SteeringVector(
        [LayerVector(1, "add", 2 * a.entries[0].vector - b.entries[0].vector)], D, L
    )
    with steer(tiny, [(a, 2.0), (b, -1.0)]):
        stacked = logits(tiny)
    with steer(tiny, [(summed, 1.0)]):
        single = logits(tiny)
    np.testing.assert_allclose(stacked, single, atol=1e-4)


def test_blocks_are_restored_when_generation_raises(tiny):
    from pts.steering.mlx_backend import get_layers, steer

    before = list(get_layers(tiny))
    with pytest.raises(RuntimeError):
        with steer(tiny, [(make_vector(), 1.0)]):
            raise RuntimeError("boom")
    assert all(a is b for a, b in zip(get_layers(tiny), before))


def test_a_vector_for_another_model_is_refused(tiny):
    from pts.steering.mlx_backend import steer

    other = SteeringVector([LayerVector(0, "add", np.ones(D * 2))], D * 2, L)
    with pytest.raises(ValueError, match="hidden_size"):
        with steer(tiny, [(other, 1.0)]):
            pass


def test_steering_does_not_leak_into_parameters(tiny):
    """Vectors held by the wrapper must not show up as trainable weights."""
    from mlx.utils import tree_flatten

    from pts.steering.mlx_backend import steer

    n = len(tree_flatten(tiny.parameters()))
    with steer(tiny, [(make_vector(), 1.0)]):
        names = [k for k, _ in tree_flatten(tiny.parameters())]
    assert len(names) == n


def test_mean_difference_of_identical_sets_is_zero_and_round_trips(tiny, tmp_path):
    from pts.steering.mlx_backend import mean_difference

    vec = mean_difference(tiny, [IDS, IDS[:3]], [IDS, IDS[:3]], layers=[1, 2], base_model="tiny")
    assert all(np.abs(e.vector).max() < 1e-6 for e in vec.entries)
    assert all(e.ref_norm > 0 for e in vec.entries)

    real = mean_difference(tiny, [IDS], [IDS[::-1]], layers=[1])
    real.save(str(tmp_path / "v"))
    assert SteeringVector.load(str(tmp_path / "v")).provenance["method"] == "mean_difference"


def test_cli_builds_a_vector(tiny, tmp_path, monkeypatch, capsys):
    """End to end through the CLI, with model loading stubbed to the tiny model."""
    from pts import cli

    class Tok:
        def encode(self, text):
            return [ord(c) % 64 for c in text]

    monkeypatch.setattr(cli, "_load_mlx", lambda name: (tiny, Tok()))
    (tmp_path / "pos.txt").write_text("good\nfine\n")
    (tmp_path / "neg.txt").write_text("bad\nawful\n")
    out = str(tmp_path / "v.safetensors")
    cli.main(["build-vector", "--model", "tiny", "--positive", str(tmp_path / "pos.txt"),
              "--negative", str(tmp_path / "neg.txt"), "--layers", "1", "2",
              "--output-path", out])
    vec = SteeringVector.load(out)
    assert vec.layers == [1, 2] and vec.provenance["num_positive"] == 2
