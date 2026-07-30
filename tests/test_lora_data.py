import json

import pytest

from conformance.workloads.lora_data import (
    build_messages,
    build_system_prompt,
    build_training_record,
    parse_prediction,
    target_completion,
)
from conformance.workloads.lora_eval import score

LABELS = ["card_arrival", "lost_or_stolen_card", "top_up_failed"]


def test_system_prompt_lists_every_label() -> None:
    prompt = build_system_prompt(LABELS)
    for label in LABELS:
        assert f"- {label}" in prompt


def test_messages_are_system_then_user() -> None:
    msgs = build_messages("  where is my card?  ", LABELS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "where is my card?"


def test_training_record_appends_assistant_turn() -> None:
    rec = build_training_record("where is my card?", "card_arrival", LABELS)
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert json.loads(rec["messages"][-1]["content"]) == {"intent": "card_arrival"}


def test_target_completion_is_compact_json() -> None:
    assert target_completion("card_arrival") == '{"intent":"card_arrival"}'


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"intent": "card_arrival"}', "card_arrival"),
        ('{"intent":"card_arrival"}', "card_arrival"),
        ('```json\n{"intent": "card_arrival"}\n```', "card_arrival"),
        ('Sure! Here you go: {"intent": "card_arrival"} Hope that helps.', "card_arrival"),
        ('  \n {"intent": " card_arrival "} \n', "card_arrival"),
    ],
)
def test_parse_prediction_recovers_valid_json(raw: str, expected: str) -> None:
    assert parse_prediction(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "card_arrival",
        "The intent is card arrival.",
        '{"label": "card_arrival"}',
        '{"intent": 42}',
        '{"intent": ""}',
        '{"intent": null}',
        "{not json at all}",
    ],
)
def test_parse_prediction_rejects_unusable_output(raw: str) -> None:
    assert parse_prediction(raw) is None


def test_score_counts_unparseable_as_wrong_but_tracks_it_separately() -> None:
    preds = [
        '{"intent": "card_arrival"}',  # valid + correct
        '{"intent": "top_up_failed"}',  # valid + wrong
        "I think it is card_arrival",  # unparseable
        '{"intent": "not_a_real_label"}',  # valid, correct-format, out of label space
    ]
    gold = ["card_arrival", "card_arrival", "card_arrival", "card_arrival"]
    s = score(preds, gold, LABELS)

    assert s.n == 4
    assert s.json_valid == 3
    assert s.correct == 1
    assert s.in_label_space == 2
    assert s.accuracy == pytest.approx(0.25)
    assert s.accuracy_on_parsed == pytest.approx(1 / 3)
    assert s.json_valid_rate == pytest.approx(0.75)


def test_score_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        score(["a"], ["x", "y"], LABELS)


def test_score_handles_empty_input() -> None:
    s = score([], [], LABELS)
    assert s.accuracy == 0.0
    assert s.json_valid_rate == 0.0
    assert s.to_dict()["n"] == 0
