"""pts serve: mlx-lm's OpenAI-compatible server with steering, over real HTTP.

Runs the server on the tiny random Llama and checks every steered response
against generating directly with the same steering. Greedy decoding on a fixed
model is deterministic, so "same as direct" is an exact comparison. Skipped
where MLX is not installed.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

np = pytest.importorskip("numpy")
mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from pts.steering.mlx_backend import model_shape, steer  # noqa: E402
from pts.steering.mlx_server import (  # noqa: E402
    SteeringRegistry,
    decode_adapter,
    encode_adapter,
    make_server,
    mlx_cli_args,
    parse_preset,
)
from pts.steering.vector import LayerVector, SteeringVector  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPT = "hello world, this is"
MAX_TOKENS = 8


@pytest.fixture(scope="module")
def direct():
    return mlx_lm.load(TINY)


@pytest.fixture(scope="module")
def vectors(direct, tmp_path_factory):
    model, _ = direct
    d, n = model_shape(model)
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("vectors")
    paths = {}
    for name, seed in (("up", 1), ("down", 2)):
        vec = SteeringVector([LayerVector(0, "add", rng.normal(size=d) * 5)], d, n, base_model=TINY)
        paths[name] = vec.save(str(root / f"{name}.safetensors"))
    return paths


@pytest.fixture(scope="module")
def server(vectors):
    registry = SteeringRegistry(
        named=dict(vectors),
        presets=dict([parse_preset("upbeat=up:3"), parse_preset("both=up:3,down:-2")]),
    )
    cli_args = mlx_cli_args(["--model", TINY, "--host", "127.0.0.1", "--port", "0"])
    httpd, rg = make_server("127.0.0.1", 0, cli_args, registry)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    rg.stop_and_join()


def post(base, path, body):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def complete(base, **extra):
    status, body = post(base, "/v1/completions", {
        "prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0.0, **extra,
    })
    assert status == 200, body
    return body["choices"][0]["text"]


def generate_direct(direct, steering=()):
    model, tok = direct
    vecs = [(SteeringVector.load(p), a) for p, a in steering]
    run = lambda: mlx_lm.generate(model, tok, PROMPT, max_tokens=MAX_TOKENS)
    if not vecs:
        return run()
    with steer(model, vecs):
        return run()


def test_adapter_encoding_round_trips():
    steering = (("down", -2.0), ("up", 3.0))
    assert decode_adapter(encode_adapter("lora/x", steering)) == ("lora/x", steering)
    assert encode_adapter(None, ()) is None
    assert decode_adapter("plain/adapter") == ("plain/adapter", ())


def test_unsteered_requests_match_plain_mlx_lm(server, direct):
    assert complete(server) == generate_direct(direct)


def test_request_steering_matches_direct_steering(server, direct, vectors):
    steered = complete(server, steering=[{"vector": "up", "alpha": 3}])
    assert steered == generate_direct(direct, [(vectors["up"], 3.0)])
    assert steered != generate_direct(direct), "the vector should change greedy output"


def test_string_and_dict_forms_are_equivalent(server):
    assert complete(server, steering="up:3") == complete(server, steering=[{"vector": "up", "alpha": 3}])


def test_presets_select_steering_by_model_name(server, direct, vectors):
    assert complete(server, model="upbeat") == generate_direct(direct, [(vectors["up"], 3.0)])
    assert complete(server, model="both") == generate_direct(
        direct, [(vectors["up"], 3.0), (vectors["down"], -2.0)]
    )


def test_steering_does_not_leak_between_requests(server, direct):
    """Alternate steered and unsteered requests for the same prompt. A shared
    prompt cache or stale wrappers would make the second unsteered answer differ."""
    plain = generate_direct(direct)
    assert complete(server) == plain
    complete(server, steering="up:3")
    assert complete(server) == plain
    complete(server, model="both")
    assert complete(server) == plain


def test_concurrent_requests_with_different_steering(server, direct, vectors):
    expected = {
        None: generate_direct(direct),
        "up:3": generate_direct(direct, [(vectors["up"], 3.0)]),
        "down:-2": generate_direct(direct, [(vectors["down"], -2.0)]),
    }
    results = {}

    def worker(key, i):
        results[(key, i)] = complete(server, **({"steering": key} if key else {}))

    threads = [threading.Thread(target=worker, args=(k, i)) for k in expected for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for (key, _), text in results.items():
        assert text == expected[key], f"request with steering={key!r} got another request's output"


def test_unregistered_vectors_are_refused(server, vectors):
    status, _ = post(server, "/v1/completions", {
        "prompt": PROMPT, "max_tokens": 2, "steering": vectors["up"] + ":3",
    })
    assert status >= 400


def test_malformed_steering_is_refused(server):
    for bad in ([{"alpha": 3}], [{"vector": "up", "alpha": "big"}], 42):
        status, _ = post(server, "/v1/completions", {"prompt": PROMPT, "max_tokens": 2, "steering": bad})
        assert status >= 400, bad


def test_models_endpoint_lists_presets(server):
    with urllib.request.urlopen(server + "/v1/models", timeout=30) as r:
        data = json.loads(r.read())["data"]
    assert {m["id"] for m in data} == {"upbeat", "both"}
    both = next(m for m in data if m["id"] == "both")
    assert {s["vector"] for s in both["steering"]} == {"up", "down"}


def test_chat_completions_accept_steering(server):
    status, body = post(server, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0.0,
        "steering": "up:3",
    })
    # The tiny test tokenizer may lack a chat template; the point is that the
    # steering field is accepted on the chat route, not what it generates.
    assert status == 200 or "chat_template" in json.dumps(body), body
