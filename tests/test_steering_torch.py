"""Steering on PyTorch: the hook point, ablation, stacking, generate(), and hook removal.

Uses the same tiny random Llama as the other model tests; weights are random,
so these assert exact math, never sensible outputs.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pts.steering import capture, steer  # noqa: E402
from pts.steering.torch_backend import model_shape  # noqa: E402
from pts.steering.vector import LayerVector, SteeringVector  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
IDS = [1, 5, 9, 2, 7]


@pytest.fixture(scope="module")
def tiny():
    model = transformers.AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
    return model.eval()


def make_vector(model, seed=0, layers=(0,), op="add") -> SteeringVector:
    d, n = model_shape(model)
    rng = np.random.default_rng(seed)
    return SteeringVector(
        entries=[LayerVector(i, op, rng.normal(size=d), ref_norm=10.0) for i in layers],
        hidden_size=d, num_layers=n,
    )


def hook_ids(model):
    # transformers 5 installs its own hooks (e.g. for output_hidden_states), so
    # compare against what was there before rather than expecting none.
    return [set(b._forward_hooks) for b in model.model.layers]


def logits(model, ids=IDS):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].numpy()


def test_hook_point_is_hf_hidden_states_except_after_the_last_block(tiny):
    """The format's definition: block L's output is hidden_states[L + 1], except
    for the last block, where HF reports the activation after the final norm."""
    _, n = model_shape(tiny)
    with torch.no_grad():
        hs = tiny(torch.tensor([IDS]), output_hidden_states=True).hidden_states
    acts = capture(tiny, IDS, list(range(n)))
    for i in range(n - 1):
        np.testing.assert_allclose(acts[i], hs[i + 1][0].numpy(), atol=1e-6)
    with torch.no_grad():
        normed = tiny.model.norm(torch.from_numpy(acts[n - 1])[None])[0].numpy()
    np.testing.assert_allclose(normed, hs[n][0].numpy(), atol=1e-5)


def test_zero_alpha_is_an_exact_no_op_and_hooks_are_removed(tiny):
    base, before = logits(tiny), hook_ids(tiny)
    with steer(tiny, [(make_vector(tiny), 0.0)]):
        np.testing.assert_array_equal(logits(tiny), base)
    assert hook_ids(tiny) == before


def test_addition_lands_exactly_at_the_hook_point(tiny):
    v = make_vector(tiny, layers=(0,))
    base = capture(tiny, IDS, [0])[0]
    with steer(tiny, [(v, 2.0)]):
        steered = capture(tiny, IDS, [0])[0]
    np.testing.assert_allclose(steered - base, np.tile(2 * v.entries[0].vector, (len(IDS), 1)), atol=1e-5)


def test_ablation_removes_the_direction(tiny):
    v = make_vector(tiny, layers=(1,), op="ablate")
    u = v.entries[0].vector
    assert np.abs(capture(tiny, IDS, [1])[1] @ u).max() > 1e-3
    with v.apply(tiny):
        assert np.abs(capture(tiny, IDS, [1])[1] @ u).max() < 1e-5


def test_stacking_equals_the_summed_vector(tiny):
    a, b = make_vector(tiny, seed=1), make_vector(tiny, seed=2)
    d, n = model_shape(tiny)
    summed = SteeringVector([LayerVector(0, "add", 2 * a.entries[0].vector - b.entries[0].vector)], d, n)
    with steer(tiny, [(a, 2.0), (b, -1.0)]):
        stacked = logits(tiny)
    with steer(tiny, [(summed, 1.0)]):
        single = logits(tiny)
    np.testing.assert_allclose(stacked, single, atol=1e-5)


def test_steering_applies_through_cached_generation(tiny):
    """generate() runs one token at a time against a KV cache; the hook must fire on every step."""
    v = make_vector(tiny, seed=3)
    prompt = torch.tensor([IDS])
    kw = dict(max_new_tokens=8, do_sample=False, pad_token_id=0)
    with torch.no_grad():
        plain = tiny.generate(prompt, **kw)[0].tolist()
        with steer(tiny, [(v, 50.0)]):
            steered = tiny.generate(prompt, **kw)[0].tolist()
            # Re-score the steered continuation without the cache: same greedy tokens.
            full = tiny(torch.tensor([steered[:-1]])).logits[0].argmax(-1).tolist()
    assert steered != plain
    assert full[len(IDS) - 1:] == steered[len(IDS):]


def test_hooks_are_removed_when_generation_raises(tiny):
    before = hook_ids(tiny)
    with pytest.raises(RuntimeError):
        with steer(tiny, [(make_vector(tiny), 1.0)]):
            assert hook_ids(tiny) != before
            raise RuntimeError("boom")
    assert hook_ids(tiny) == before


def test_a_vector_for_another_model_is_refused(tiny):
    d, n = model_shape(tiny)
    other = SteeringVector([LayerVector(0, "add", np.ones(d * 2))], d * 2, n)
    with pytest.raises(ValueError, match="hidden_size"):
        with steer(tiny, [(other, 1.0)]):
            pass


def test_half_precision_model_is_steered_in_its_own_dtype(tiny):
    half = transformers.AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.bfloat16).eval()
    v = make_vector(half)
    with steer(half, [(v, 1.0)]):
        out = capture(half, IDS, [0])[0]
    assert np.isfinite(out).all()


def test_cli_builds_a_vector_on_torch(tmp_path):
    from pts import cli

    (tmp_path / "pos.txt").write_text("good\nfine\n")
    (tmp_path / "neg.txt").write_text("bad\nawful\n")
    out = str(tmp_path / "v.safetensors")
    cli.main(["build-vector", "--backend", "torch", "--device", "cpu", "--model", TINY,
              "--positive", str(tmp_path / "pos.txt"), "--negative", str(tmp_path / "neg.txt"),
              "--layers", "0", "1", "--output-path", out])
    vec = SteeringVector.load(out)
    assert vec.layers == [0, 1] and vec.provenance["backend"] == "torch"
