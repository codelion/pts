"""Steering vectors: the on-disk format and stacking.

These need only numpy and safetensors -- no torch, no MLX -- because a vector
must be loadable and inspectable anywhere. The MLX hook is tested in
test_steering_mlx.py.
"""

import json
import logging

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")

from pts.steering.vector import (  # noqa: E402
    CONFIG_NAME,
    METADATA_KEY,
    LayerVector,
    SteeringVector,
    combine,
    parse_hub_path,
    parse_vector_spec,
)

D, L = 16, 4


def make_vector(seed=0, layers=(1,), op="add", **kw) -> SteeringVector:
    rng = np.random.default_rng(seed)
    entries = [LayerVector(layer=i, op=op, vector=rng.normal(size=D), ref_norm=10.0) for i in layers]
    return SteeringVector(entries=entries, hidden_size=D, num_layers=L, **kw)


# -- format ----------------------------------------------------------------

def test_directory_round_trip(tmp_path):
    v = make_vector(layers=(1, 2), base_model="org/model", behavior="sycophancy",
                    validation={"delta_p": 0.2})
    v.save(str(tmp_path / "vec"))
    w = SteeringVector.load(str(tmp_path / "vec"))
    assert w.to_config() == v.to_config()
    for a, b in zip(v.entries, w.entries):
        np.testing.assert_array_equal(a.vector, b.vector)


def test_single_file_carries_its_config(tmp_path):
    """A lone .safetensors file must be loadable without the sidecar JSON."""
    v = make_vector(base_model="org/model")
    path = v.save(str(tmp_path / "vec.safetensors"))
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as f:
        assert json.loads(f.metadata()[METADATA_KEY])["base_model"] == "org/model"
    assert SteeringVector.load(path).to_config() == v.to_config()


def test_directory_config_is_readable_json(tmp_path):
    make_vector(layers=(0, 3)).save(str(tmp_path / "vec"))
    config = json.loads((tmp_path / "vec" / CONFIG_NAME).read_text())
    assert config["format"] == "pts-steering"
    assert config["hook"] == "block_output"
    assert [x["tensor"] for x in config["layers"]] == ["layers.0.add", "layers.3.add"]


def test_unknown_format_version_is_refused(tmp_path):
    v = make_vector()
    config = v.to_config()
    config["format_version"] = "99"
    with pytest.raises(ValueError, match="not supported"):
        SteeringVector.from_config(config, {e.tensor_name: e.vector for e in v.entries})


def test_unknown_hook_point_is_refused():
    with pytest.raises(ValueError, match="Unsupported hook"):
        make_vector(hook="attention_output")


@pytest.mark.parametrize("bad", [
    dict(layers=(L,)),                      # layer out of range
])
def test_invalid_entries_are_refused(bad):
    with pytest.raises(ValueError):
        make_vector(**bad)


def test_wrong_width_is_refused():
    with pytest.raises(ValueError, match="shape"):
        SteeringVector(entries=[LayerVector(0, "add", np.ones(D + 1))], hidden_size=D, num_layers=L)


def test_shape_mismatch_raises_and_name_mismatch_warns(caplog):
    v = make_vector(base_model="Qwen/Qwen2.5-1.5B-Instruct")
    with pytest.raises(ValueError, match="hidden_size"):
        v.check_compatible(D * 2, L)
    with caplog.at_level(logging.WARNING):
        v.check_compatible(D, L, "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    assert not caplog.records, "a quantised copy of the same model should not warn"
    with caplog.at_level(logging.WARNING):
        v.check_compatible(D, L, "meta-llama/Llama-3.2-1B")
    assert "extracted from" in caplog.text


def test_ablation_directions_are_stored_unit_norm():
    v = make_vector(op="ablate")
    assert np.isclose(np.linalg.norm(v.entries[0].vector), 1.0)


@pytest.mark.parametrize("spec,expected", [
    ("vec", ("vec", None)),
    ("vec:4", ("vec", 4.0)),
    ("./a/vec.safetensors:-2.5", ("./a/vec.safetensors", -2.5)),
    ("hf://org/repo", ("hf://org/repo", None)),
    ("hf://org/repo:0.5", ("hf://org/repo", 0.5)),
    ("hf://org/repo/sub/v.safetensors:3", ("hf://org/repo/sub/v.safetensors", 3.0)),
])
def test_vector_spec_parsing(spec, expected):
    assert parse_vector_spec(spec) == expected


def test_hub_paths():
    assert parse_hub_path("hf://org/repo") == ("org/repo", None)
    assert parse_hub_path("hf://org/repo/sub/dir") == ("org/repo", "sub/dir")
    with pytest.raises(ValueError):
        parse_hub_path("hf://org")


# -- stacking --------------------------------------------------------------

def test_additions_sum_with_their_alphas():
    a, b = make_vector(seed=1), make_vector(seed=2)
    plan = combine([(a, 2.0), (b, -1.0)])[1]
    np.testing.assert_allclose(plan.add, 2 * a.entries[0].vector - b.entries[0].vector, rtol=1e-6)


def test_default_alpha_applies_when_none_given():
    a = make_vector(default_alpha=3.0)
    np.testing.assert_allclose(combine([(a, None)])[1].add, 3 * a.entries[0].vector, rtol=1e-6)


def test_ablations_are_orthonormalised_together():
    """Removing two non-orthogonal directions one at a time does not remove their span."""
    u = np.zeros(D); u[0] = 1
    w = np.zeros(D); w[0] = 1; w[1] = 1
    vecs = [
        (SteeringVector([LayerVector(1, "ablate", u)], D, L), None),
        (SteeringVector([LayerVector(1, "ablate", w)], D, L), None),
    ]
    q = combine(vecs)[1].ablate
    np.testing.assert_allclose(q @ q.T, np.eye(2), atol=1e-6)
    h = np.arange(D, dtype=np.float64) + 1
    projected = h - (h @ q.T) @ q
    assert abs(projected[0]) < 1e-5 and abs(projected[1]) < 1e-5
    np.testing.assert_allclose(projected[2:], h[2:], rtol=1e-6)


def test_parallel_vectors_and_oversized_steering_warn(caplog):
    a = make_vector(seed=1)
    with caplog.at_level(logging.WARNING):
        combine([(a, 1.0), (a, 1.0)])
    assert "nearly parallel" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        combine([(a, 100.0)])  # |v| ~ 4, ref_norm 10
    assert "typical residual norm" in caplog.text


def test_zero_alpha_contributes_nothing():
    assert combine([(make_vector(), 0.0)]) == {}
