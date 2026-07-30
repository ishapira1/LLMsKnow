from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llmssycoph import fixed_development_cohort as cohort


def _source(dataset: str, example_id: str) -> dict:
    correct_letter = "A"
    incorrect_letter = "B"
    question = f"Question {dataset} {example_id}\n(A) correct\n(B) wrong"
    prompt = f"{question}\nAnswer:"
    return {
        "dataset": dataset,
        "source_dataset": f"source/{dataset}",
        "source_split": "train",
        "source_example_id": example_id,
        "question_id": f"q_{example_id}",
        "question": question,
        "answers_list": ["correct", "wrong"],
        "letters": "AB",
        "correct_letter": correct_letter,
        "correct_answer": "correct",
        "incorrect_letter": incorrect_letter,
        "incorrect_option_text": "wrong",
        "neutral_prompt": prompt,
    }


def _neutral(source: dict) -> dict:
    prompt_hash = cohort._sha256_text(source["neutral_prompt"])
    return {
        **{key: value for key, value in source.items() if key != "neutral_prompt"},
        "prompt": source["neutral_prompt"],
        "correctness": 1,
        "response_letter": "A",
        "response_text": "A",
        "openai_model": cohort.MODEL_SNAPSHOT,
        "prompt_sha256": prompt_hash,
        "messages_sha256": "messages-hash",
        "openai_request_id": "request-id",
        "result_source": "test",
        "choice_probabilities": {"A": 0.9, "B": 0.1},
        "choice_probability_correct": 0.9,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class FixedDevelopmentCohortTests(unittest.TestCase):
    def test_freeze_is_deterministic_and_neutral_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            sources = [
                _source("commonsense_qa", "c1"),
                _source("commonsense_qa", "c2"),
                _source("arc_challenge", "a1"),
                _source("arc_challenge", "a2"),
            ]
            _write_jsonl(source_root / "selected_questions.jsonl", sources)
            _write_jsonl(
                source_root / "reused_neutral_records.jsonl",
                [_neutral(sources[0]), _neutral(sources[2])],
            )
            _write_jsonl(
                source_root / "records" / "neutral_results.jsonl",
                [_neutral(sources[1]), _neutral(sources[3])],
            )
            (source_root / "experiment_config.json").write_text(
                json.dumps({"request_settings": {"temperature": 1.0}}),
                encoding="utf-8",
            )
            manifest = root / "cohort.jsonl"
            spec = root / "cohort.json"
            target = {"commonsense_qa": 1, "arc_challenge": 2}

            with mock.patch.object(cohort, "TARGET_COUNTS", target):
                frozen = cohort.freeze_development_cohort(
                    source_root=source_root,
                    manifest_path=manifest,
                    spec_path=spec,
                    target_counts=target,
                    expected_available_counts={
                        "commonsense_qa": 2,
                        "arc_challenge": 2,
                    },
                )
                audit = cohort.audit_development_cohort(
                    manifest_path=manifest,
                    spec_path=spec,
                )

            self.assertEqual(
                frozen["selected_questions_by_dataset"],
                target,
            )
            self.assertEqual(audit["status"], "passed")
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["neutral_correctness"] == 1 for row in rows))
            self.assertTrue(all(row["source_split"] == "train" for row in rows))
            first_hash = frozen["manifest_sha256"]

            with mock.patch.object(cohort, "TARGET_COUNTS", target):
                frozen_again = cohort.freeze_development_cohort(
                    source_root=source_root,
                    manifest_path=manifest,
                    spec_path=spec,
                    target_counts=target,
                    expected_available_counts={
                        "commonsense_qa": 2,
                        "arc_challenge": 2,
                    },
                )
            self.assertEqual(first_hash, frozen_again["manifest_sha256"])

    def test_rejects_neutral_incorrect_record(self) -> None:
        source = _source("commonsense_qa", "c1")
        neutral = _neutral(source)
        neutral["correctness"] = 0
        with self.assertRaisesRegex(cohort.CohortError, "Neutral-incorrect"):
            cohort._validate_pair(source, neutral)


if __name__ == "__main__":
    unittest.main()
