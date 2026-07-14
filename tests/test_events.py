"""Schema, migration, and round-trip tests. No model required."""

import json

import pytest

from pts.events import (
    EVENT_LATENT,
    EVENT_SENTENCE,
    EVENT_TOKEN,
    CausalReasoningEvent,
    from_any_record,
    from_legacy_pivotal_token,
    from_legacy_thought_anchor,
    make_latent_event,
    make_sentence_event,
    make_token_event,
    to_legacy_pivotal_token,
    to_legacy_thought_anchor,
)


@pytest.fixture
def v1_token():
    return {
        "model_id": "Qwen/Qwen3-0.6B",
        "query": "What is 17 * 24?",
        "pivot_context": "Let me compute 17 * 24.",
        "pivot_token": " Wait",
        "pivot_token_id": 13824,
        "prob_before": 0.35,
        "prob_after": 0.67,
        "prob_delta": 0.32,
        "is_positive": True,
        "task_type": "math",
        "dataset_id": "codelion/optillmbench",
        "dataset_item_id": "42",
        "timestamp": "2026-01-01T00:00:00",
    }


@pytest.fixture
def v1_anchor():
    return {
        "model_id": "Qwen/Qwen3-0.6B",
        "query": "What is 17 * 24?",
        "sentence": "Wait, I should verify the arithmetic.",
        "sentence_id": 5,
        "prefix_context": "Let me compute this.",
        "prob_with_sentence": 0.72,
        "prob_without_sentence": 0.38,
        "prob_delta": 0.34,
        "task_type": "math",
        "sentence_category": "self_checking",
        "suffix_context": "So the answer is 408.",
        "alternatives_tested": ["I'll just guess."],
        "causal_dependencies": [3, 4],
        "dataset_id": "codelion/optillmbench",
        "dataset_item_id": "42",
        "timestamp": "2026-01-01T00:00:00",
    }


def test_v1_token_migrates(v1_token):
    e = from_legacy_pivotal_token(v1_token)
    assert e.event_type == EVENT_TOKEN
    assert e.granularity == "token"
    assert e.visibility == "emitted"
    assert e.label == " Wait"
    assert e.context == "Let me compute 17 * 24."
    assert e.token_id == 13824
    assert e.prob_delta == pytest.approx(0.32)
    assert e.score == pytest.approx(0.32)
    assert e.is_positive is True
    assert e.timestamp == "2026-01-01T00:00:00"


def test_v1_anchor_migrates(v1_anchor):
    e = from_legacy_thought_anchor(v1_anchor)
    assert e.event_type == EVENT_SENTENCE
    assert e.granularity == "sentence"
    # the legacy format's prob_with/prob_without map onto prob_after/prob_before, not the
    # other way round. Getting this backwards would flip every sign.
    assert e.prob_after == pytest.approx(0.72)
    assert e.prob_before == pytest.approx(0.38)
    assert e.prob_delta == pytest.approx(0.34)
    assert e.is_positive is True
    assert e.position == 5


def test_v1_anchor_keeps_extra_fields_in_metadata(v1_anchor):
    e = from_legacy_thought_anchor(v1_anchor)
    assert e.metadata["suffix_context"] == "So the answer is 408."
    assert e.metadata["alternatives_tested"] == ["I'll just guess."]
    assert e.metadata["causal_dependencies"] == [3, 4]


def test_event_id_is_deterministic(v1_token):
    # Re-migrating the same record must produce the same id, or links break
    # across re-runs.
    assert from_any_record(v1_token).event_id == from_any_record(v1_token).event_id


def test_event_ids_differ_across_events(v1_token, v1_anchor):
    assert from_any_record(v1_token).event_id != from_any_record(v1_anchor).event_id


def test_v1_token_round_trips(v1_token):
    back = to_legacy_pivotal_token(from_legacy_pivotal_token(v1_token))
    for key in ("query", "pivot_context", "pivot_token", "pivot_token_id",
                "prob_before", "prob_after", "prob_delta", "is_positive",
                "model_id", "task_type", "dataset_id", "dataset_item_id"):
        assert back[key] == v1_token[key], key


def test_v1_anchor_round_trips(v1_anchor):
    back = to_legacy_thought_anchor(from_legacy_thought_anchor(v1_anchor))
    for key in ("query", "sentence", "sentence_id", "prefix_context",
                "prob_with_sentence", "prob_without_sentence", "prob_delta",
                "suffix_context", "alternatives_tested", "causal_dependencies"):
        assert back[key] == v1_anchor[key], key


def test_v2_round_trips_through_jsonl(v1_token):
    e = from_any_record(v1_token)
    revived = from_any_record(json.loads(json.dumps(e.to_dict())))
    assert revived.to_dict() == e.to_dict()


def test_unknown_fields_survive_a_round_trip():
    # A a future version reader must not silently drop fields a 1.0 reader doesn't know.
    raw = from_any_record({
        "model_id": "m", "query": "q", "pivot_token": "t", "pivot_context": "c",
        "pivot_token_id": 1, "prob_before": 0.1, "prob_after": 0.5,
        "task_type": "math",
    }).to_dict()
    raw["some_future_field"] = "keep me"
    assert CausalReasoningEvent.from_dict(raw).metadata["some_future_field"] == "keep me"


def test_latent_events_make_no_causal_claim():
    e = make_latent_event(
        query="q", context="c", metatoken=" verify", token_id=1,
        score=0.83, layer=18, model_id="m", position=-2,
    )
    assert e.event_type == EVENT_LATENT
    assert e.visibility == "latent"
    # These must stay None. A latent readout score is not a probability delta,
    # and claiming otherwise is the single easiest way to overstate the result.
    assert e.prob_delta is None
    assert e.prob_before is None
    assert e.prob_after is None
    assert e.is_positive is None
    assert e.score == pytest.approx(0.83)


def test_token_event_score_is_abs_delta():
    e = make_token_event(
        query="q", context="c", token="t", token_id=1,
        prob_before=0.8, prob_after=0.3, model_id="m",
    )
    assert e.prob_delta == pytest.approx(-0.5)
    assert e.score == pytest.approx(0.5)
    assert e.is_positive is False


def test_sentence_event_argument_order():
    e = make_sentence_event(
        query="q", context="c", sentence="s", sentence_id=2,
        prob_before=0.3, prob_after=0.7, model_id="m",
    )
    assert e.prob_delta == pytest.approx(0.4)
    assert e.is_positive is True


def test_from_any_record_rejects_junk():
    with pytest.raises(ValueError, match="Unrecognized PTS record"):
        from_any_record({"nothing": "useful"})


def test_from_any_record_accepts_an_event(v1_token):
    e = from_any_record(v1_token)
    assert from_any_record(e) is e


def test_link_to_is_idempotent():
    e = make_token_event(query="q", context="c", token="t", token_id=1,
                         prob_before=0.1, prob_after=0.5, model_id="m")
    e.link_to("other", precedes=True)
    e.link_to("other", precedes=True)
    assert e.linked_event_ids == ["other"]
    assert e.precedes_event_ids == ["other"]
