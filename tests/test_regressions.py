"""Regressions for bugs an adversarial review found that the original suite missed.

The original tests were structurally incapable of catching these: they only
exercised >=19-layer models, single-source enrichment, and datasets containing
both scales. Each test here pins one of those blind spots open.
"""

import json

import pytest

from pts.event_storage import EventStorage
from pts.events import make_latent_event, make_sentence_event, make_token_event
from pts.latent.activations import resolve_workspace_layers
from pts.linking import EventLinker


# -- 1. latent event_id collisions ----------------------------------------

def test_latent_events_from_different_sources_do_not_collide():
    """Two pivotal tokens in one query, enriched at the same layer and offset,
    surfacing the same common meta-token -- the *normal* case for a real model's
    top-k readout. Without context/source in the id they collide, storage drops
    one, and the survivor keeps a link pointing at the dropped id.
    """
    a = make_latent_event(
        query="q", context="context A", metatoken=" the", token_id=5, score=0.9,
        layer=8, model_id="m", position=-2, source_event_id="token_evt_AAA",
    )
    b = make_latent_event(
        query="q", context="context B", metatoken=" the", token_id=5, score=0.9,
        layer=8, model_id="m", position=-2, source_event_id="token_evt_BBB",
    )
    assert a.event_id != b.event_id

    s = EventStorage()
    s.add_event(a)
    s.add_event(b)
    assert len(s) == 2, "one latent event was silently dropped by an id collision"


def test_latent_event_id_is_still_deterministic():
    kwargs = dict(
        query="q", context="c", metatoken=" verify", token_id=1, score=0.8,
        layer=8, model_id="m", position=-1, source_event_id="token_evt_A",
    )
    assert make_latent_event(**kwargs).event_id == make_latent_event(**kwargs).event_id


# -- 2. fit-jlens on small models ------------------------------------------

@pytest.mark.parametrize("num_layers,name", [
    (12, "gpt2"), (16, "Llama-3.2-1B"), (18, "boundary"), (24, "medium"), (28, "Qwen3-0.6B"),
])
def test_auto_workspace_layers_never_include_the_final_layer(num_layers, name):
    """round(0.92 * n) lands on n-1 for every n <= 18, so auto layer selection
    used to hand JLens.fit a layer it rejects. fit-jlens was unusable on GPT-2,
    Llama-3.2-1B, and every other small model -- and the only test covered a
    100-layer model, comfortably inside the safe regime.
    """
    layers = resolve_workspace_layers(num_layers)
    assert layers, f"{name}: no layers selected"
    assert max(layers) <= num_layers - 2, (
        f"{name} ({num_layers}L): selected {layers}, which includes the final "
        f"layer ({num_layers - 1}); JLens.fit rejects it"
    )
    assert min(layers) >= 0


def test_papers_depth_band_is_still_honoured_for_deep_models():
    assert resolve_workspace_layers(100) == [38, 56, 74, 92]


# -- 3. the score-scale invariant ------------------------------------------

@pytest.fixture
def mixed():
    """A real pivotal token, and a banal filler meta-token the lens loves."""
    s = EventStorage()
    s.add_event(make_token_event(
        query="q", context="c", token=" Wait", token_id=1,
        prob_before=0.30, prob_after=0.75, model_id="m",   # prob_delta = +0.45
    ))
    s.add_event(make_latent_event(
        query="q", context="c", metatoken=" the", token_id=2,
        score=0.92, layer=8, model_id="m", position=-1,    # readout score, not a delta
    ))
    return s


def test_most_important_does_not_rank_readouts_against_deltas(mixed):
    top = mixed.most_important(5)
    assert all(e.event_type != "latent_metatoken" for e in top), (
        "most_important() surfaced a latent readout; on an enriched dataset that "
        "makes the list a ranking of the model's most probable filler tokens"
    )
    assert top[0].label == " Wait"


def test_most_surfaced_ranks_latent_separately(mixed):
    surfaced = mixed.most_surfaced(5)
    assert [e.label for e in surfaced] == [" the"]


def test_summary_never_averages_the_two_scales(mixed):
    s = mixed.summary()
    # (0.45 + 0.92) / 2 = 0.685 is a number with no meaning.
    assert s["average_abs_prob_delta"] == pytest.approx(0.45)
    assert s["max_abs_prob_delta"] == pytest.approx(0.45)
    assert s["average_readout_score"] == pytest.approx(0.92)
    assert s["max_readout_score"] == pytest.approx(0.92)
    assert s["average_score"] == pytest.approx(0.45), "average_score must be emitted-only"


def test_filter_thresholds_each_scale_on_its_own_number(mixed):
    # A single 0.5 floor used to drop the +0.45 token and keep the " the" readout.
    kept = mixed.filter(min_prob_delta=0.5)
    assert [e.label for e in kept] == [" the"], "min_prob_delta must not touch latent events"

    kept = mixed.filter(min_readout_score=0.95)
    assert [e.label for e in kept] == [" Wait"], "min_readout_score must not touch emitted events"

    kept = mixed.filter(min_prob_delta=0.1, min_readout_score=0.1)
    assert len(kept) == 2


def test_filter_has_no_cross_scale_min_score():
    import inspect
    params = inspect.signature(EventStorage.filter).parameters
    assert "min_score" not in params, (
        "a single min_score across both scales is the bug; use min_prob_delta / "
        "min_readout_score"
    )


# -- 4. shuffle_control key consistency ------------------------------------

def test_shuffle_control_has_the_same_keys_on_every_path():
    """`pts link --shuffle-control` on a v1 token-only file -- the first thing
    anyone would run it on -- used to be a KeyError."""
    token_only = [make_token_event(
        query="q", context="c", token="t", token_id=1,
        prob_before=0.3, prob_after=0.7, model_id="m",
    )]
    control = EventLinker().shuffle_control(token_only)
    for key in ("observed_mean", "shuffled_mean", "observed_n", "shuffled_n"):
        assert key in control, f"missing {key} on the no-latent path"

    assert set(EventLinker().shuffle_control([])) >= {
        "observed_mean", "shuffled_mean", "observed_n", "shuffled_n"
    }


# -- 5. the linker misapplying an offset -----------------------------------

def test_offset_only_counts_against_the_event_it_was_measured_from():
    """An enrichment latent event's offset is measured against ONE source. Using
    it against every other event in the query hands near-maximal temporal
    proximity to every pair and inflates the link set with spurious edges."""
    src = make_token_event(
        query="q", context="ctx", token=" Wait", token_id=1,
        prob_before=0.3, prob_after=0.7, model_id="m", position=100,
    )
    other = make_token_event(
        query="q", context="ctx", token=" Hmm", token_id=2,
        prob_before=0.3, prob_after=0.7, model_id="m", position=400,
    )
    lat = make_latent_event(
        query="q", context="ctx", metatoken=" verify", token_id=3, score=0.8,
        layer=8, model_id="m", position=-1, source_event_id=src.event_id,
        metadata={"position_offset_from_linked_event": -1},
    )

    linker = EventLinker()
    to_source = linker.score_pair(lat, src)
    to_other = linker.score_pair(lat, other)

    assert to_source.evidence["temporal_proximity"] > 0.8
    assert to_other.evidence["temporal_proximity"] == 0.0, (
        "the offset was applied to an event it was never measured against"
    )
    assert to_source.score > to_other.score


def test_probe_latent_positions_are_comparable_to_token_positions():
    # A probe latent event carries an absolute, prompt-inclusive token index, so
    # it lives in the same frame as a token event and proximity is meaningful.
    lat = make_latent_event(
        query="q", context="c", metatoken=" verify", token_id=1, score=0.8,
        layer=8, model_id="m", position=98, search_method="latent_pts",
    )
    tok = make_token_event(
        query="q", context="c", token=" Wait", token_id=2,
        prob_before=0.3, prob_after=0.7, model_id="m", position=100,
    )
    link = EventLinker().score_pair(lat, tok)
    assert link.evidence["temporal_proximity"] > 0.5


# -- 9. save() must not silently drop events -------------------------------

def test_save_converts_numpy_scalars_instead_of_dropping_the_event(tmp_path):
    """save() used to catch TypeError per event, log, and carry on -- writing a
    short file and exiting 0 while reporting success."""
    np = pytest.importorskip("numpy")

    s = EventStorage()
    s.add_event(make_token_event(
        query="q", context="c", token="t", token_id=1,
        prob_before=0.1, prob_after=0.5, model_id="m",
    ))
    s.add_event(make_sentence_event(
        query="q", context="c", sentence="s", sentence_id=1,
        prob_before=0.1, prob_after=0.5, model_id="m",
        # The verification pass puts numpy/torch floats here.
        metadata={"attention_entropy": np.float32(0.5),
                  "arithmetic_errors": np.array([1.0, 2.0])},
    ))

    path = tmp_path / "e.jsonl"
    s.save(str(path))

    lines = [l for l in open(path) if l.strip()]
    assert len(lines) == 2, "an event was silently dropped on save"

    revived = json.loads(lines[1])
    assert revived["metadata"]["attention_entropy"] == pytest.approx(0.5)
    assert revived["metadata"]["arithmetic_errors"] == [1.0, 2.0]


def test_save_raises_on_a_genuinely_unserializable_value(tmp_path):
    s = EventStorage()
    e = make_token_event(query="q", context="c", token="t", token_id=1,
                         prob_before=0.1, prob_after=0.5, model_id="m")
    e.metadata["bad"] = object()
    s.add_event(e)

    with pytest.raises(TypeError, match="Cannot serialize"):
        s.save(str(tmp_path / "e.jsonl"))
