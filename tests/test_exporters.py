"""Export behaviour, including the two claim-correctness rules.

No model required: these test the parts that decide what gets written, not the
parts that run a forward pass.
"""

import json

import pytest

from pts.event_storage import EventStorage
from pts.events import make_latent_event, make_sentence_event, make_token_event
from pts.exporters import EventExporter, detect_file_type, generate_dataset_card


@pytest.fixture
def storage():
    s = EventStorage()
    s.add_event(make_token_event(
        query="q", context="ctx", token=" Wait", token_id=1,
        prob_before=0.35, prob_after=0.67, model_id="m", category="verification",
    ))
    s.add_event(make_sentence_event(
        query="q", context="ctx", sentence="Let me check.", sentence_id=1,
        prob_before=0.38, prob_after=0.72, model_id="m", category="verification",
    ))
    # A realistic readout score. Vocabulary softmax mass is spread thin, so a
    # strong meta-token still scores far below a 0.1 probability-delta floor.
    s.add_event(make_latent_event(
        query="q", context="ctx", metatoken=" verify", token_id=9,
        score=0.02, layer=18, model_id="m", position=-2, category="verification",
    ))
    return s


def test_causal_events_does_not_filter_latent_by_prob_delta(tmp_path, storage):
    """The bug this guards against deleted every latent event from the export.

    Latent scores are readout probabilities; emitted scores are probability
    deltas. One threshold applied to both looks principled and silently drops
    the entire latent scale.
    """
    out = tmp_path / "events.jsonl"
    EventExporter(storage).export_causal_events(
        str(out), min_prob_delta=0.1, min_score=0.0
    )
    records = [json.loads(l) for l in open(out)]
    types = {r["event_type"] for r in records}

    assert "latent_metatoken" in types, "latent events were dropped by a prob-delta filter"
    assert len(records) == 3


def test_causal_events_still_filters_emitted_by_prob_delta(tmp_path, storage):
    out = tmp_path / "events.jsonl"
    EventExporter(storage).export_causal_events(
        str(out), min_prob_delta=0.5, min_score=0.0
    )
    records = [json.loads(l) for l in open(out)]
    # Both emitted events move probability by ~0.32-0.34, below 0.5.
    assert all(r["event_type"] == "latent_metatoken" for r in records)


def test_metatokens_export_keeps_nulls(tmp_path, storage):
    out = tmp_path / "latent.jsonl"
    EventExporter(storage).export_metatokens(str(out))
    records = [json.loads(l) for l in open(out)]
    assert len(records) == 1
    assert records[0]["prob_delta"] is None
    assert records[0]["is_positive"] is None


def test_v1_views_round_trip(tmp_path, storage):
    exporter = EventExporter(storage)

    tok_path = tmp_path / "v1_tokens.jsonl"
    exporter.export_pivotal_tokens(str(tok_path))
    tokens = [json.loads(l) for l in open(tok_path)]
    assert len(tokens) == 1
    assert tokens[0]["pivot_token"] == " Wait"
    assert tokens[0]["prob_delta"] == pytest.approx(0.32)

    anc_path = tmp_path / "v1_anchors.jsonl"
    exporter.export_thought_anchors(str(anc_path))
    anchors = [json.loads(l) for l in open(anc_path)]
    assert len(anchors) == 1
    assert anchors[0]["prob_with_sentence"] == pytest.approx(0.72)
    assert anchors[0]["prob_without_sentence"] == pytest.approx(0.38)


def test_dpo_refuses_to_fabricate_without_an_oracle(tmp_path, storage, caplog):
    """v1 emitted DPO pairs whose 'rejected' token was never measured.

    With no oracle available, the correct behaviour is to write nothing and say
    why -- not to fall back to a likelihood heuristic and present it as a
    measured preference.
    """
    out = tmp_path / "dpo.jsonl"
    EventExporter(storage).export_dpo(
        str(out), model_name=None, dataset=None, find_rejected_tokens=True
    )
    assert not out.exists(), "DPO export invented pairs with no oracle"
    assert any("no usable oracle" in r.message.lower() or "no dpo pairs" in r.message.lower()
               for r in caplog.records)


def test_dpo_uses_precomputed_rejected_tokens(tmp_path):
    s = EventStorage()
    s.add_event(make_token_event(
        query="q", context="ctx", token=" Wait", token_id=1,
        prob_before=0.35, prob_after=0.67, model_id="m",
        metadata={"rejected_token": " Guess", "rejected_prob": 0.1},
    ))
    out = tmp_path / "dpo.jsonl"
    EventExporter(s).export_dpo(str(out), find_rejected_tokens=False)

    pairs = [json.loads(l) for l in open(out)]
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == " Wait"
    assert pairs[0]["rejected"] == " Guess"
    assert pairs[0]["metadata"]["counterpart_verified"] is True


def test_detect_file_type(tmp_path, storage):
    events = tmp_path / "e.jsonl"
    EventExporter(storage).export_causal_events(str(events))
    assert detect_file_type(str(events)) == "causal_events"

    latent = tmp_path / "l.jsonl"
    EventExporter(storage).export_metatokens(str(latent))
    assert detect_file_type(str(latent)) == "metatokens"

    v1 = tmp_path / "v1.jsonl"
    EventExporter(storage).export_pivotal_tokens(str(v1))
    assert detect_file_type(str(v1)) == "pivotal_tokens"


def test_dataset_card_states_the_latent_caveats(tmp_path, storage):
    path = tmp_path / "e.jsonl"
    EventExporter(storage).export_causal_events(str(path))
    card = generate_dataset_card(str(path), model_name="m")

    # A card describing latent events without these is actively misleading.
    assert "hypotheses, not measurements" in card
    assert "observational" in card
    assert "meta-token" in card.lower()
    assert "not the paper's" in card
    # Counts must be read from the file, not asserted.
    assert "`latent_metatoken`: 1" in card


def test_dataset_card_omits_latent_caveats_when_no_latent_events(tmp_path):
    s = EventStorage()
    s.add_event(make_token_event(query="q", context="c", token="t", token_id=1,
                                 prob_before=0.1, prob_after=0.5, model_id="m"))
    path = tmp_path / "e.jsonl"
    EventExporter(s).export_causal_events(str(path))
    card = generate_dataset_card(str(path))
    assert "hypotheses, not measurements" not in card


def test_optillm_reasoning_patterns_are_preserved():
    from pts.exporters import OPTILLM_REASONING_PATTERNS

    # OptiLLM's autothink reads `reasoning_pattern` off steering records and
    # expects exactly these five values. Renaming them breaks that integration
    # silently, so pin them.
    assert set(OPTILLM_REASONING_PATTERNS) == {
        "depth_and_thoroughness", "numerical_accuracy", "self_correction",
        "exploration", "organization",
    }
