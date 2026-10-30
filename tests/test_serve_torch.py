"""pts serve --backend torch: the same OpenAI-compatible API on transformers, over real HTTP.

Every steered response is checked against generating directly with the same
steering. Greedy decoding on a fixed CPU model is deterministic, so "same as
direct" is an exact comparison.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pts.steering.serving import SteeringRegistry, parse_preset  # noqa: E402
from pts.steering.torch_backend import model_shape, steer  # noqa: E402
from pts.steering.torch_server import TorchSteeringServer, make_server  # noqa: E402
from pts.steering.vector import LayerVector, SteeringVector  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPT = "hello world, this is"
MAX_TOKENS = 8


@pytest.fixture(scope="module")
def direct():
    model = transformers.AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32).eval()
    return model, transformers.AutoTokenizer.from_pretrained(TINY)


@pytest.fixture(scope="module")
def vectors(direct, tmp_path_factory):
    d, n = model_shape(direct[0])
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("vectors")
    paths = {}
    for name in ("up", "down"):
        vec = SteeringVector([LayerVector(0, "add", rng.normal(size=d) * 5)], d, n, base_model=TINY)
        paths[name] = vec.save(str(root / f"{name}.safetensors"))
    return paths


@pytest.fixture(scope="module")
def server(direct, vectors):
    registry = SteeringRegistry(
        named=dict(vectors),
        presets=dict([parse_preset("upbeat=up:3"), parse_preset("both=up:3,down:-2"),
                      parse_preset("elsewhere=up:3@some/other-model")]),
    )
    model, tok = direct
    httpd = make_server("127.0.0.1", 0, TorchSteeringServer(model, tok, TINY, registry))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def post(base, path, body, raw=False):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
            return r.status, (data.decode() if raw else json.loads(data))
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
    ids = tok(PROMPT, return_tensors="pt").input_ids
    vecs = [(SteeringVector.load(p), a) for p, a in steering]
    with torch.no_grad(), steer(model, vecs):
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_TOKENS,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def test_unsteered_requests_match_plain_generate(server, direct):
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


def test_streaming_sends_the_same_text(server):
    whole = complete(server, steering="up:3")
    status, body = post(server, "/v1/completions", {
        "prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0.0, "steering": "up:3", "stream": True,
    }, raw=True)
    assert status == 200
    events = [line[len("data: "):] for line in body.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e)["choices"][0] for e in events[:-1]]
    assert "".join(c["text"] for c in chunks) == whole
    assert chunks[-1]["finish_reason"] in ("length", "stop")


def test_stop_strings_cut_the_text_streamed_or_not(server):
    whole = complete(server)
    stop = whole.strip()[2:5]
    assert stop and stop in whole
    expected = whole[:whole.index(stop)]
    assert complete(server, stop=stop) == expected
    status, body = post(server, "/v1/completions", {
        "prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0.0, "stop": [stop], "stream": True,
    }, raw=True)
    events = [json.loads(line[6:]) for line in body.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert "".join(e["choices"][0]["text"] for e in events) == expected
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_unregistered_vectors_are_refused(server, vectors):
    status, _ = post(server, "/v1/completions", {
        "prompt": PROMPT, "max_tokens": 2, "steering": vectors["up"] + ":3",
    })
    assert status == 400


def test_malformed_requests_are_refused(server):
    for bad in ({"steering": [{"alpha": 3}]}, {"steering": [{"vector": "up", "alpha": "big"}]},
                {"steering": 42}, {"temperature": "hot"}, {"model": "not-this-one"},
                {"model": "elsewhere"}):
        status, body = post(server, "/v1/completions", {"prompt": PROMPT, "max_tokens": 2, **bad})
        assert status == 400 and "error" in body, bad


def test_models_endpoint_lists_the_model_and_presets(server):
    with urllib.request.urlopen(server + "/v1/models", timeout=30) as r:
        data = json.loads(r.read())["data"]
    assert {m["id"] for m in data} == {TINY, "upbeat", "both", "elsewhere"}
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
    if status == 200:
        assert body["choices"][0]["message"]["role"] == "assistant"


def test_cli_serves_with_the_torch_backend(monkeypatch, vectors):
    import pts.steering.torch_server as ts
    from pts import cli

    seen = {}
    monkeypatch.setattr(ts, "serve", lambda server, host, port: seen.update(server=server, host=host, port=port))
    cli.main(["serve", "--backend", "torch", "--model", TINY, "--device", "cpu", "--port", "9123",
              "--vector", f"up={vectors['up']}", "--preset", "upbeat=up:3", "--max-tokens", "64"])
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 9123)
    assert seen["server"].model_id == TINY and seen["server"].max_tokens == 64
    assert set(seen["server"].registry.presets) == {"upbeat"}
