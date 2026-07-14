"""Storage, filtering, classification, and event linking. No model required."""

import json

import pytest

from pts.classification import (
    COMPUTATION,
    CONSOLIDATION,
    SELF_CORRECTION,
    VERIFICATION,
    get_classifier,
    map_v1_category,
)
from pts.event_storage import EventStorage, TokenStorage
from pts.events import EVENT_LATENT, EVENT_TOKEN, make_latent_event, make_sentence_event, make_token_event
from pts.linking import EventLinker, context_overlap, temporal_proximity


@pytest.fixture
def events():
    tok = make_token_event(
        query="What is 17 * 24?", context="Let me compute this carefully.",
        token=" Wait", token_id=1, prob_before=0.35, prob_after=0.67,
        model_id="m", category=VERIFICATION, position=10,
    )
    sent = make_sentence_event(
        query="What is 17 * 24?", context="Let me compute this carefully.",
        sentence="Wait, let me verify the arithmetic.", sentence_id=2,
        prob_before=0.38, prob_after=0.72, model_id="m", category=VERIFICATION,
    )
    lat = make_latent_event(
        query="What is 17 * 24?", context="Let me compute this carefully.",
        metatoken=" verify", token_id=9, score=0.83, layer=18,
        model_id="m", position=-2, category=VERIFICATION,
        metadata={"position_offset_from_linked_event": -2},
    )
    return [lat, tok, sent]


# -- storage ---------------------------------------------------------------

def test_storage_dedupes_by_event_id():
    # the legacy format's searcher and CLI each wrote every token, silently doubling datasets.
    e = make_token_event(query="q", context="c", token="t", token_id=1,
                         prob_before=0.1, prob_after=0.5, model_id="m")
    s = EventStorage()
    s.add_event(e)
    s.add_event(e)
    s.add_event(e.to_dict())
    assert len(s) == 1


def test_storage_round_trips_jsonl(tmp_path, events):
    path = tmp_path / "events.jsonl"
    s = EventStorage()
    s.add_events(events)
    s.save(str(path))

    reloaded = EventStorage(filepath=str(path))
    assert len(reloaded) == 3
    assert {e.event_type for e in reloaded} == {EVENT_LATENT, EVENT_TOKEN, "thought_anchor"}


def test_storage_loads_v1_files_directly(tmp_path):
    path = tmp_path / "v1.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({
            "model_id": "m", "query": "q", "pivot_context": "c", "pivot_token": "t",
            "pivot_token_id": 1, "prob_before": 0.3, "prob_after": 0.7,
            "prob_delta": 0.4, "task_type": "math",
        }) + "\n")

    s = EventStorage(filepath=str(path))
    assert len(s) == 1
    assert s[0].event_type == EVENT_TOKEN
    assert "migrated" in s[0].tags


def test_storage_skips_bad_lines_without_dying(tmp_path):
    path = tmp_path / "mixed.jsonl"
    with open(path, "w") as f:
        f.write("not json\n")
        f.write(json.dumps({"nothing": "useful"}) + "\n")
        f.write(json.dumps({
            "model_id": "m", "query": "q", "pivot_context": "c", "pivot_token": "t",
            "pivot_token_id": 1, "prob_before": 0.3, "prob_after": 0.7, "task_type": "math",
        }) + "\n")

    s = EventStorage(filepath=str(path))
    assert len(s) == 1


def test_storage_filters(events):
    s = EventStorage()
    s.add_events(events)

    assert len(s.filter(granularity="latent")) == 1
    assert len(s.filter(event_type=EVENT_TOKEN)) == 1
    assert len(s.filter(category=VERIFICATION)) == 3
    assert len(s.filter(is_positive=True)) == 2  # latent has is_positive=None
    assert len(s.filter(layer_range=(10, 20))) == 1

    # Each scale is thresholded on its own number. min_prob_delta leaves the
    # latent event alone; min_readout_score leaves the emitted ones alone.
    assert len(s.filter(min_prob_delta=0.5)) == 1        # only the latent event survives
    assert len(s.filter(min_readout_score=0.9)) == 2     # both emitted events survive
    assert len(s.filter(min_prob_delta=0.1, min_readout_score=0.1)) == 3


def test_summary_counts_each_scale(events):
    s = EventStorage()
    s.add_events(events)
    summary = s.summary()
    assert summary["total_events"] == 3
    assert summary["latent_events"] == 1
    assert summary["token_events"] == 1
    assert summary["sentence_events"] == 1
    # Latent events have no valence and must not be counted as positive.
    assert summary["positive_events"] == 2


def test_token_storage_v1_surface_still_works():
    s = TokenStorage()
    s.add_token({
        "model_id": "m", "query": "q", "pivot_context": "c", "pivot_token": "t",
        "pivot_token_id": 1, "prob_before": 0.3, "prob_after": 0.7, "task_type": "math",
    })
    assert len(s) == 1
    assert s.tokens[0]["label"] == "t"


# -- classification --------------------------------------------------------

def test_verification_wording_beats_arithmetic():
    # v1 classified ANY sentence containing "=" as computation, so this landed
    # in the wrong bucket despite being plainly a self-correction.
    assert get_classifier().classify("Wait, let me check 2 + 2 = 4") == SELF_CORRECTION


def test_plain_arithmetic_is_still_computation():
    assert get_classifier().classify("2 + 2 = 4") == COMPUTATION


def test_consolidation():
    assert get_classifier().classify("Therefore the total is 408.") == CONSOLIDATION


def test_metatoken_subword_matching():
    clf = get_classifier()
    assert clf.classify_metatoken(" verify") == VERIFICATION
    assert clf.classify_metatoken("Ġverif") == VERIFICATION
    assert clf.classify_metatoken("   ") is None


def test_v1_categories_map_onto_the_unified_taxonomy():
    assert map_v1_category("self_checking") == VERIFICATION
    assert map_v1_category("active_computation") == COMPUTATION
    assert map_v1_category(None) is None


# -- linking ---------------------------------------------------------------

def test_linker_connects_latent_to_emitted(events):
    links = EventLinker(threshold=0.4).link(events)
    assert links
    latent, tok, sent = events
    assert tok.event_id in latent.precedes_event_ids
    assert latent.event_id in tok.follows_event_ids


def test_linker_never_crosses_queries():
    a = make_latent_event(query="query A", context="ctx", metatoken=" verify",
                          token_id=1, score=0.9, layer=8, model_id="m", position=-1)
    b = make_token_event(query="query B", context="ctx", token=" Wait", token_id=2,
                         prob_before=0.3, prob_after=0.7, model_id="m")
    assert EventLinker(threshold=0.0).link([a, b]) == []


def test_latent_after_the_event_is_not_evidence():
    # The claim under test is that workspace activity PRECEDES emission. A
    # latent event later in time must score zero on proximity.
    assert temporal_proximity(latent_pos=10, emitted_pos=5) == 0.0
    assert temporal_proximity(latent_pos=5, emitted_pos=5) == 1.0
    assert temporal_proximity(latent_pos=4, emitted_pos=5, window=8) > 0.8


def test_context_overlap_is_symmetric_and_bounded():
    a, b = "the cat sat on the mat", "the dog sat on the mat"
    assert context_overlap(a, b) == context_overlap(b, a)
    assert 0.0 <= context_overlap(a, b) <= 1.0
    assert context_overlap(a, a) == 1.0
    assert context_overlap("", "anything") == 0.0


def test_shuffle_control_reports_both_means(events):
    other = make_token_event(query="a different question", context="different ctx",
                             token=" x", token_id=3, prob_before=0.2, prob_after=0.6,
                             model_id="m")
    control = EventLinker().shuffle_control(events + [other])
    assert "observed_mean" in control and "shuffled_mean" in control
    # Same-query pairs should beat mismatched ones, or the whole linking premise
    # is unsupported.
    assert control["observed_mean"] > control["shuffled_mean"]
