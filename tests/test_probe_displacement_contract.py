from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "analyze_probe_displacement.py"


def _load_probe_displacement_module():
    spec = importlib.util.spec_from_file_location("analyze_probe_displacement", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProbeDisplacementContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_probe_displacement_module()

    def test_decompose_probe_direction_matches_geometry(self) -> None:
        delta = [3.0, 4.0]
        weights = [2.0, 0.0]

        metrics = self.module.decompose_probe_direction(delta, weights)

        self.assertAlmostEqual(metrics["score_shift_linear"], 6.0)
        self.assertAlmostEqual(metrics["delta_l2"], 5.0)
        self.assertAlmostEqual(metrics["parallel_l2"], 3.0)
        self.assertAlmostEqual(metrics["orthogonal_l2"], 4.0)
        self.assertAlmostEqual(metrics["parallel_fraction"], 0.6)
        self.assertAlmostEqual(metrics["orthogonal_fraction"], 0.8)
        self.assertLess(metrics["reconstruction_error"], 1e-9)

    def test_build_displacement_pairs_filters_to_neutral_correct_and_reuses_incorrect_b(self) -> None:
        base_records = [
            {
                "record_id": 10,
                "split": "test",
                "question_id": "q_1",
                "draw_idx": 0,
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "B",
                "incorrect_letter": "A",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q1 neutral"}],
                "response_raw": "B",
                "correctness": 1,
                "usable_for_metrics": True,
            },
            {
                "record_id": 11,
                "split": "test",
                "question_id": "q_1",
                "draw_idx": 0,
                "template_type": "incorrect_suggestion",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "B",
                "incorrect_letter": "A",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q1 incorrect"}],
                "response_raw": "A",
                "correctness": 0,
                "usable_for_metrics": True,
            },
            {
                "record_id": 20,
                "split": "test",
                "question_id": "q_2",
                "draw_idx": 0,
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "C",
                "incorrect_letter": "A",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q2 neutral"}],
                "response_raw": "A",
                "correctness": 0,
                "usable_for_metrics": True,
            },
            {
                "record_id": 21,
                "split": "test",
                "question_id": "q_2",
                "draw_idx": 0,
                "template_type": "incorrect_suggestion",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "C",
                "incorrect_letter": "A",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q2 incorrect"}],
                "response_raw": "A",
                "correctness": 0,
                "usable_for_metrics": True,
            },
            {
                "record_id": 30,
                "split": "test",
                "question_id": "q_3",
                "draw_idx": 0,
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "D",
                "incorrect_letter": "B",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q3 neutral"}],
                "response_raw": "D",
                "correctness": 1,
                "usable_for_metrics": True,
            },
            {
                "record_id": 31,
                "split": "test",
                "question_id": "q_3",
                "draw_idx": 0,
                "template_type": "incorrect_suggestion",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "D",
                "incorrect_letter": "B",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q3 incorrect"}],
                "response_raw": "B",
                "correctness": 0,
                "usable_for_metrics": True,
            },
            {
                "record_id": 40,
                "split": "test",
                "question_id": "q_4",
                "draw_idx": 0,
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "A",
                "incorrect_letter": "C",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q4 neutral"}],
                "response_raw": "A",
                "correctness": 1,
                "usable_for_metrics": True,
            },
            {
                "record_id": 50,
                "split": "test",
                "question_id": "q_5",
                "draw_idx": 0,
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "B",
                "incorrect_letter": "B",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q5 neutral"}],
                "response_raw": "B",
                "correctness": 1,
                "usable_for_metrics": True,
            },
            {
                "record_id": 51,
                "split": "test",
                "question_id": "q_5",
                "draw_idx": 0,
                "template_type": "incorrect_suggestion",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "B",
                "incorrect_letter": "B",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q5 incorrect"}],
                "response_raw": "B",
                "correctness": 1,
                "usable_for_metrics": True,
            },
        ]
        congruent_records = [
            {
                "record_id": 101,
                "split": "test",
                "question_id": "q_1",
                "draw_idx": 0,
                "template_type": "model_congruent_suggestion",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "B",
                "incorrect_letter": "A",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q1 congruent"}],
                "response_raw": "B",
                "neutral_source_record_id": 10,
            },
            {
                "record_id": 102,
                "split": "test",
                "question_id": "q_2",
                "draw_idx": 0,
                "template_type": "model_congruent_suggestion",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "C",
                "incorrect_letter": "A",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q2 congruent"}],
                "response_raw": "A",
                "neutral_source_record_id": 20,
            },
            {
                "record_id": 105,
                "split": "test",
                "question_id": "q_5",
                "draw_idx": 0,
                "template_type": "model_congruent_suggestion",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "correct_letter": "B",
                "incorrect_letter": "B",
                "letters": "ABCD",
                "prompt_messages": [{"type": "human", "content": "Q5 congruent"}],
                "response_raw": "B",
                "neutral_source_record_id": 50,
            },
        ]

        pairs, coverage_df = self.module.build_displacement_pairs(
            base_records,
            congruent_records,
            requested_split="test",
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question_id"], "q_1")
        self.assertEqual(pairs[0]["correct_choice"], "B")
        self.assertEqual(pairs[0]["endorsed_wrong_choice"], "A")
        self.assertEqual(pairs[0]["congruent_record"]["neutral_source_record_id"], 10)

        reasons = coverage_df.set_index("question_id")["exclusion_reason"].to_dict()
        self.assertEqual(reasons["q_1"], "")
        self.assertEqual(reasons["q_2"], "neutral_not_correct")
        self.assertEqual(reasons["q_3"], "missing_congruent_pair")
        self.assertEqual(reasons["q_4"], "missing_incorrect_pair")
        self.assertEqual(reasons["q_5"], "non_distinct_choice_metadata")

    def test_summarize_probe_displacement_emits_condition_role_and_all_rollups(self) -> None:
        pair_df = pd.DataFrame(
            [
                {
                    "question_id": "q_1",
                    "condition": "incorrect_suggestion",
                    "candidate_role": "correct_choice",
                    "delta_l2": 2.0,
                    "orthogonal_l2": 1.0,
                    "orthogonal_fraction": 0.5,
                    "score_shift_linear": 0.25,
                },
                {
                    "question_id": "q_1",
                    "condition": "incorrect_suggestion",
                    "candidate_role": "endorsed_wrong_choice",
                    "delta_l2": 4.0,
                    "orthogonal_l2": 3.0,
                    "orthogonal_fraction": 0.75,
                    "score_shift_linear": -0.50,
                },
                {
                    "question_id": "q_2",
                    "condition": "model_congruent_suggestion",
                    "candidate_role": "correct_choice",
                    "delta_l2": 1.0,
                    "orthogonal_l2": 0.25,
                    "orthogonal_fraction": 0.25,
                    "score_shift_linear": 0.10,
                },
                {
                    "question_id": "q_2",
                    "condition": "model_congruent_suggestion",
                    "candidate_role": "endorsed_wrong_choice",
                    "delta_l2": 3.0,
                    "orthogonal_l2": 1.5,
                    "orthogonal_fraction": 0.5,
                    "score_shift_linear": -0.30,
                },
            ]
        )

        summary_df = self.module.summarize_probe_displacement(pair_df)

        self.assertEqual(len(summary_df), 6)
        self.assertEqual(
            set(summary_df["candidate_role"]),
            {"correct_choice", "endorsed_wrong_choice", "all"},
        )
        self.assertEqual(
            set(summary_df["condition"]),
            {"incorrect_suggestion", "model_congruent_suggestion"},
        )

        incorrect_all = summary_df.loc[
            summary_df["condition"].eq("incorrect_suggestion") & summary_df["candidate_role"].eq("all")
        ].iloc[0]
        self.assertEqual(int(incorrect_all["n_pairs"]), 2)
        self.assertAlmostEqual(float(incorrect_all["mean_delta_l2"]), 3.0)
        self.assertAlmostEqual(float(incorrect_all["mean_abs_score_shift_linear"]), 0.375)


if __name__ == "__main__":
    unittest.main()
