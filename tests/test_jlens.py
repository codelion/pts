"""J-lens correctness.

These need a model, so they are skipped when torch/transformers are absent --
which is itself part of the contract: the base package must work without them.

The central test is ``test_fast_jacobian_matches_brute_force``. The fitting code
takes a shortcut that is only valid if attention is causal, and if that shortcut
is wrong then every latent event PTS produces is garbage. So it is checked
against a brute-force autograd Jacobian rather than assumed.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pts.latent.activations import (  # noqa: E402
    ResidualCapture,
    get_layer_modules,
    get_unembedding,
    resolve_workspace_layers,
)
from pts.latent.jlens import (  # noqa: E402
    JLens,
    JLensCosine,
    LogitLens,
    load_readout,
    parse_hub_path,
)

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
    model.eval()
    return model, tok


# -- workspace layer selection --------------------------------------------

def test_auto_layers_follow_the_papers_depth_band():
    # The paper puts the workspace at roughly 38%-92% of depth. With 100 layers
    # that must land on ~L38..~L92, not on hardcoded numbers from some other model.
    assert resolve_workspace_layers(100) == [38, 56, 74, 92]


def test_explicit_layers_are_validated():
    with pytest.raises(ValueError, match="out of range"):
        resolve_workspace_layers(12, layers=[99])


# -- activation capture ----------------------------------------------------

def test_hooks_are_removed_even_when_the_forward_pass_raises(model_and_tokenizer):
    model, _ = model_and_tokenizer
    layers = get_layer_modules(model)
    before = len(layers[0]._forward_hooks)

    capture = ResidualCapture(model, layers)
    with pytest.raises(RuntimeError):
        with capture.capture([0]):
            raise RuntimeError("boom")

    # A leaked hook silently corrupts every later forward pass on this model.
    assert len(layers[0]._forward_hooks) == before


# -- the Jacobian ----------------------------------------------------------

def test_fast_jacobian_matches_brute_force(model_and_tokenizer):
    model, tok = model_and_tokenizer
    layers = get_layer_modules(model)
    final = len(layers) - 1
    L = 0
    d_model = model.config.hidden_size

    ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
    T = ids.shape[1]

    jl = JLens(model, tok)
    capture = ResidualCapture(model, layers)

    with capture.capture([L, final], keep_graph=True) as acts:
        with torch.enable_grad():
            model(input_ids=ids)
            J_fast = jl._jacobian_for_layer(
                acts[final][0], acts[L], T, d_model, basis_chunk=16, device="cpu"
            )

    # Brute force: the full [t', i, t, j] Jacobian, reduced by the definition.
    def run_from_activation(h_flat):
        h_new = h_flat.view(1, T, d_model)
        store = {}

        def replace(module, inputs, output):
            return (h_new,) + tuple(output[1:]) if isinstance(output, tuple) else h_new

        def grab(module, inputs, output):
            store["f"] = output[0] if isinstance(output, tuple) else output

        h1 = layers[L].register_forward_hook(replace)
        h2 = layers[final].register_forward_hook(grab)
        try:
            model(input_ids=ids)
        finally:
            h1.remove()
            h2.remove()
        return store["f"][0]

    with capture.capture([L]) as acts0:
        with torch.no_grad():
            model(input_ids=ids)
        h0 = acts0[L][0].clone()

    full = torch.autograd.functional.jacobian(
        run_from_activation, h0.reshape(-1), vectorize=True
    ).view(T, d_model, T, d_model)

    J_brute = torch.zeros(d_model, d_model)
    for t in range(T):
        J_brute += full[t:, :, t, :].sum(dim=0) / (T - t)
    J_brute /= T

    assert torch.allclose(J_fast, J_brute, atol=1e-4, rtol=1e-3)


def test_causal_mask_holds(model_and_tokenizer):
    """The fast path's whole justification: h_l,t cannot affect h_final,t' for t' < t.

    If this ever fails, the one-backward-pass-per-component shortcut is invalid.
    """
    model, tok = model_and_tokenizer
    layers = get_layer_modules(model)
    final = len(layers) - 1
    d_model = model.config.hidden_size

    ids = tok("The quick brown fox jumps.", return_tensors="pt").input_ids
    T = ids.shape[1]

    def run_from_activation(h_flat):
        h_new = h_flat.view(1, T, d_model)
        store = {}
        h1 = layers[0].register_forward_hook(
            lambda m, i, o: (h_new,) + tuple(o[1:]) if isinstance(o, tuple) else h_new
        )
        h2 = layers[final].register_forward_hook(
            lambda m, i, o: store.__setitem__("f", o[0] if isinstance(o, tuple) else o)
        )
        try:
            model(input_ids=ids)
        finally:
            h1.remove()
            h2.remove()
        return store["f"][0]

    capture = ResidualCapture(model, layers)
    with capture.capture([0]) as acts:
        with torch.no_grad():
            model(input_ids=ids)
        h0 = acts[0][0].clone()

    full = torch.autograd.functional.jacobian(
        run_from_activation, h0.reshape(-1), vectorize=True
    ).view(T, d_model, T, d_model)

    for t in range(1, T):
        assert full[:t, :, t, :].abs().max() < 1e-6, f"causality violated at source position {t}"


def test_identity_jacobian_is_exactly_the_logit_lens(model_and_tokenizer):
    """logit_lens is not a fallback hack -- it is J-lens with J = I."""
    model, tok = model_and_tokenizer
    d_model = model.config.hidden_size

    h = torch.randn(d_model)
    j_identity = JLens(model, tok, matrices={0: torch.eye(d_model)})
    logit = LogitLens(model, tok)

    a = j_identity.read(h, 0, top_k=5)
    b = logit.read(h, 0, top_k=5)

    assert [x.token_id for x in a] == [x.token_id for x in b]
    assert all(abs(x.score - y.score) < 1e-6 for x, y in zip(a, b))


def test_readout_scores_are_a_probability_distribution(model_and_tokenizer):
    model, tok = model_and_tokenizer
    h = torch.randn(model.config.hidden_size)
    results = LogitLens(model, tok).read(h, 0, top_k=10)

    assert len(results) == 10
    assert all(0.0 <= r.score <= 1.0 for r in results)
    # Ranked descending, ranks 1-indexed.
    assert [r.rank for r in results] == list(range(1, 11))
    assert results == sorted(results, key=lambda r: -r.score)


# -- fitting round trip ----------------------------------------------------

def test_fit_save_load_round_trip(model_and_tokenizer, tmp_path):
    model, tok = model_and_tokenizer
    texts = [
        "The cat sat on the mat and looked around.",
        "To solve this we first compute the product.",
        "Let me check the arithmetic: 12 times 12 is 144.",
    ]

    fitted = JLens(model, tok).fit(
        texts, layers=[0], seq_len=16, basis_chunk=8, device="cpu", show_progress=False
    )
    assert 0 in fitted.matrices
    assert fitted.matrices[0].shape == (model.config.hidden_size,) * 2
    assert fitted.config["num_sequences"] == 3

    fitted.save(str(tmp_path / "jl"))
    loaded = JLens.load(str(tmp_path / "jl"), model, tok)
    assert torch.allclose(loaded.matrices[0], fitted.matrices[0])


def test_jlens_without_a_path_is_an_error_not_a_silent_fallback(model_and_tokenizer):
    """Falling back to logit-lens would let a run claim evidence it never gathered."""
    model, tok = model_and_tokenizer
    with pytest.raises(ValueError, match="requires --jlens-path"):
        load_readout(model, tok, method="jlens", jlens_path=None)


def test_reading_an_unfitted_layer_is_an_error(model_and_tokenizer):
    model, tok = model_and_tokenizer
    jl = JLens(model, tok, matrices={0: torch.eye(model.config.hidden_size)})
    with pytest.raises(KeyError, match="No J-lens matrix fitted for layer 1"):
        jl.read(torch.randn(model.config.hidden_size), 1)


# -- reference (anthropics/jacobian-lens) lenses ---------------------------

def _save_reference_lens(path, matrices, n_prompts=7):
    """Write the layout ``jlens.JacobianLens.save`` produces: fp16, int layer keys."""
    d_model = next(iter(matrices.values())).shape[0]
    torch.save(
        {
            "J": {l: J.to(torch.float16) for l, J in matrices.items()},
            "n_prompts": n_prompts,
            "source_layers": sorted(matrices),
            "d_model": d_model,
        },
        path,
    )


def test_loads_the_reference_pt_layout(model_and_tokenizer, tmp_path):
    model, tok = model_and_tokenizer
    d_model = model.config.hidden_size
    J = torch.randn(d_model, d_model).to(torch.float16).float()  # exact in fp16
    _save_reference_lens(tmp_path / "lens.pt", {0: J})

    loaded = JLens.load(str(tmp_path / "lens.pt"), model, tok)
    assert loaded.layers == [0]
    assert loaded.matrices[0].dtype == torch.float32
    assert torch.equal(loaded.matrices[0], J)
    assert loaded.config["num_sequences"] == 7
    assert loaded.config["format"] == "reference_pt"

    # Same convention as our own fit: the loaded lens reads exactly like one
    # built from the same matrix.
    h = torch.randn(d_model)
    direct = JLens(model, tok, matrices={0: J}).read(h, 0, top_k=5)
    assert [r.token_id for r in loaded.read(h, 0, top_k=5)] == [r.token_id for r in direct]


def test_reference_lens_picks_up_the_model_from_its_sidecar(model_and_tokenizer, tmp_path):
    model, tok = model_and_tokenizer
    d_model = model.config.hidden_size
    _save_reference_lens(tmp_path / "lens.pt", {0: torch.eye(d_model)})
    (tmp_path / "config.yaml").write_text(
        '# Neuronpedia fit\nnp_model_id: "x"\nhf_model_name: "org/some-model"\n'
    )
    loaded = JLens.load(str(tmp_path / "lens.pt"), model, tok)
    assert loaded.config["model_id"] == "org/some-model"


def test_a_lens_for_another_model_is_refused(model_and_tokenizer, tmp_path):
    """A wrong-width lens would fail deep inside a matmul, or worse, not at all
    if the widths happened to line up with a transposed read."""
    model, tok = model_and_tokenizer
    d_model = model.config.hidden_size
    _save_reference_lens(tmp_path / "wide.pt", {0: torch.eye(d_model + 1)})
    with pytest.raises(ValueError, match="d_model"):
        JLens.load(str(tmp_path / "wide.pt"), model, tok)

    num_layers = len(get_layer_modules(model))
    _save_reference_lens(tmp_path / "deep.pt", {num_layers + 3: torch.eye(d_model)})
    with pytest.raises(ValueError, match="outside"):
        JLens.load(str(tmp_path / "deep.pt"), model, tok)


def test_a_non_lens_pt_file_is_refused(model_and_tokenizer, tmp_path):
    model, tok = model_and_tokenizer
    torch.save({"jacobian_sum": {}, "n_done": 0}, tmp_path / "ckpt.pt")
    with pytest.raises(ValueError, match="not a reference JacobianLens file"):
        JLens.load(str(tmp_path / "ckpt.pt"), model, tok)


def test_hub_paths_split_into_repo_and_file():
    assert parse_hub_path(
        "hf://neuronpedia/jacobian-lens/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
    ) == (
        "neuronpedia/jacobian-lens",
        "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
    )
    with pytest.raises(ValueError, match="hf://"):
        parse_hub_path("hf://neuronpedia/jacobian-lens")


# -- the cosine readout ----------------------------------------------------

def test_cosine_readout_matches_the_workspacebench_formula(model_and_tokenizer):
    """score_t = <J^T w_t, h> / (||J^T w_t|| ||h||): no final norm, no softmax."""
    model, tok = model_and_tokenizer
    d_model = model.config.hidden_size
    J = torch.randn(d_model, d_model)
    h = torch.randn(d_model)
    W = get_unembedding(model).float()

    expected = (W @ (J @ h)) / ((W @ J).norm(dim=1) * h.norm())
    top = torch.topk(expected, 5)

    results = JLensCosine(model, tok, matrices={0: J}).read(h, 0, top_k=5)
    assert [r.token_id for r in results] == top.indices.tolist()
    assert all(abs(r.score - e) < 1e-4 for r, e in zip(results, top.values.tolist()))
    assert all(-1.0 <= r.score <= 1.0 for r in results)
    assert all(r.readout_method == "jlens_cosine" for r in results)


def test_cosine_readout_needs_a_lens_too(model_and_tokenizer):
    model, tok = model_and_tokenizer
    with pytest.raises(ValueError, match="requires --jlens-path"):
        load_readout(model, tok, method="jlens_cosine", jlens_path=None)
