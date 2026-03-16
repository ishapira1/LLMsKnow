import json
from pathlib import Path

from sycophancy_bias_probe.dataset import (
    build_question_groups,
    deduplicate_rows,
    split_groups,
    template_type,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sycophancy_rows_sample.jsonl"


def load_fixture_rows():
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_template_type_and_deduplication_contract():
    rows = load_fixture_rows()

    assert len(rows) == 11
    assert template_type(rows[0]) == "neutral"
    assert template_type(rows[1]) == "incorrect_suggestion"
    assert template_type(rows[2]) == "doubt_correct"
    assert template_type(rows[3]) == "suggest_correct"

    # Rows 5-8 rely on prompt-text fallback rather than metadata templates.
    assert template_type(rows[5]) == "neutral"
    assert template_type(rows[6]) == "incorrect_suggestion"
    assert template_type(rows[7]) == "doubt_correct"
    assert template_type(rows[8]) == "suggest_correct"

    deduplicated = deduplicate_rows(rows)
    assert len(deduplicated) == 10


def test_build_question_groups_requires_complete_groups():
    rows = deduplicate_rows(load_fixture_rows())

    all_bias_groups = build_question_groups(
        rows,
        selected_bias_types=["incorrect_suggestion", "doubt_correct", "suggest_correct"],
    )
    assert len(all_bias_groups) == 2
    assert {group["question"] for group in all_bias_groups} == {
        "What is the capital of France?",
        "Which planet is known as the Red Planet?",
    }

    doubt_only_groups = build_question_groups(rows, selected_bias_types=["doubt_correct"])
    assert len(doubt_only_groups) == 3
    assert {group["question"] for group in doubt_only_groups} == {
        "What is the capital of France?",
        "Which planet is known as the Red Planet?",
        "Who wrote Hamlet?",
    }


def test_split_groups_is_deterministic_and_question_level():
    rows = deduplicate_rows(load_fixture_rows())
    groups = build_question_groups(
        rows,
        selected_bias_types=["incorrect_suggestion", "doubt_correct", "suggest_correct"],
    )

    train_a, test_a = split_groups(groups, test_frac=0.5, seed=7)
    train_b, test_b = split_groups(groups, test_frac=0.5, seed=7)

    assert [group["question_id"] for group in train_a] == [group["question_id"] for group in train_b]
    assert [group["question_id"] for group in test_a] == [group["question_id"] for group in test_b]

    train_ids = {group["question_id"] for group in train_a}
    test_ids = {group["question_id"] for group in test_a}
    assert train_ids
    assert test_ids
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {group["question_id"] for group in groups}
