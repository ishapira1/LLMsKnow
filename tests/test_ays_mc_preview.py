from __future__ import annotations

import json
import unittest
from pathlib import Path

from sycophancy_bias_probe.constants import ALL_AYS_MC_DATASETS, ALL_BIAS_TYPES
from sycophancy_bias_probe.dataset import (
    as_prompt_text,
    materialize_ays_mc_single_turn_rows,
    template_type,
)


AYS_PATH = Path(__file__).resolve().parents[1] / "data" / "sycophancy-eval" / "are_you_sure.jsonl"


def load_first_ays_mc_rows():
    rows_by_dataset = {}
    with AYS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            dataset = str((row.get("base", {}) or {}).get("dataset", "") or "")
            if dataset not in ALL_AYS_MC_DATASETS or dataset in rows_by_dataset:
                continue
            rows_by_dataset[dataset] = row
            if len(rows_by_dataset) == len(ALL_AYS_MC_DATASETS):
                break
    return rows_by_dataset


class AYSMultipleChoicePreviewTests(unittest.TestCase):
    def test_print_materialized_prompt_examples_for_each_ays_mc_dataset(self):
        self.assertTrue(AYS_PATH.exists(), msg=f"Missing AYS source file: {AYS_PATH}")

        rows_by_dataset = load_first_ays_mc_rows()
        self.assertEqual(set(rows_by_dataset), set(ALL_AYS_MC_DATASETS))

        print("\n=== AYS MC Materialization Preview ===")
        for dataset in ALL_AYS_MC_DATASETS:
            with self.subTest(dataset=dataset):
                materialized = materialize_ays_mc_single_turn_rows(
                    [rows_by_dataset[dataset]],
                    selected_bias_types=list(ALL_BIAS_TYPES),
                    selected_ays_mc_datasets=[dataset],
                )
                self.assertEqual(len(materialized), 1 + len(ALL_BIAS_TYPES))

                rows_by_type = {}
                for row in materialized:
                    row_type = template_type(row)
                    self.assertIsNotNone(row_type)
                    rows_by_type[row_type] = row

                self.assertEqual(set(rows_by_type), {"neutral", *ALL_BIAS_TYPES})

                neutral_base = rows_by_type["neutral"]["base"]
                neutral_prompt = as_prompt_text(rows_by_type["neutral"]["prompt"])
                incorrect_prompt = as_prompt_text(rows_by_type["incorrect_suggestion"]["prompt"])
                doubt_prompt = as_prompt_text(rows_by_type["doubt_correct"]["prompt"])
                suggest_prompt = as_prompt_text(rows_by_type["suggest_correct"]["prompt"])

                self.assertEqual(neutral_prompt, neutral_base["question"])
                self.assertIn(neutral_base["incorrect_answer"], incorrect_prompt)
                self.assertIn(neutral_base["correct_answer"], doubt_prompt)
                self.assertIn(neutral_base["correct_answer"], suggest_prompt)

                print(f"\n--- Dataset: {dataset} ---")
                print("Question:")
                print(neutral_base["question"])
                print(f"Correct answer text: {neutral_base['correct_answer']}")
                print(f"Incorrect answer text: {neutral_base['incorrect_answer']}")
                print(f"Correct letter: {neutral_base.get('correct_letter', '')}")
                print("Prompt variants:")
                for prompt_type in ["neutral", "incorrect_suggestion", "doubt_correct", "suggest_correct"]:
                    prompt_text = as_prompt_text(rows_by_type[prompt_type]["prompt"])
                    print(f"[{prompt_type}]")
                    print(prompt_text)


if __name__ == "__main__":
    unittest.main()
