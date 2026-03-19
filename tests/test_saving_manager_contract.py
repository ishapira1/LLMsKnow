from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from llmssycoph.saving_manager import (
    MODEL_SUMMARY_BY_BIAS_COLUMNS,
    MODEL_SUMMARY_BY_TEMPLATE_COLUMNS,
    PROBE_SUMMARY_COLUMNS,
    build_mc_probe_scores_by_prompt_df,
    build_model_summary_by_bias_df,
    build_model_summary_by_template_df,
    build_model_summary_payload,
    build_probe_summary_df,
    build_probe_summary_payload,
    PROBE_CANDIDATE_SCORE_COLUMNS,
    SAMPLED_RESPONSE_COLUMNS,
    build_run_summary_payload,
    to_probe_candidate_scores_df,
    to_samples_df,
)


class SavingManagerContractTests(unittest.TestCase):
    def test_samples_and_candidate_scores_use_compact_column_contract(self):
        sample_records = [
            {
                "record_id": 1,
                "split": "test",
                "question_id": "q_1",
                "prompt_id": "q_1__neutral",
                "dataset": "aqua_mc",
                "template_type": "neutral",
                "draw_idx": 0,
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "answer_channel": "letter",
                "question": "Question 1",
                "correct_answer": "A",
                "incorrect_answer": "B",
                "correct_letter": "A",
                "incorrect_letter": "B",
                "prompt_template": "{question}",
                "prompt_text": "Prompt text",
                "response_raw": "A",
                "response": "A",
                "committed_answer": "A",
                "commitment_kind": "letter",
                "commitment_source": "standalone_answer_line",
                "starts_with_answer_prefix": True,
                "strict_format_exact": True,
                "commitment_line": "A",
                "answer_marker_count": 0,
                "multiple_answer_markers": False,
                "correctness": 1,
                "grading_status": "correct",
                "grading_reason": "single_letter_match",
                "usable_for_metrics": True,
                "completion_token_count": 1,
                "hit_max_new_tokens": False,
                "stopped_on_eos": False,
                "finish_reason": "choice_probabilities",
                "sampling_mode": "choice_probabilities",
                "choice_probabilities": {"A": 0.7, "B": 0.3},
                "choice_probability_correct": 0.7,
                "choice_probability_selected": 0.7,
                "T_prompt": 0.7,
                "probe_x": 0.2,
                "probe_xprime": 0.3,
            }
        ]
        samples_df = to_samples_df(sample_records, model_name="test/model")
        self.assertEqual(list(samples_df.columns), SAMPLED_RESPONSE_COLUMNS)
        self.assertNotIn("choice_probabilities", samples_df.columns)
        self.assertNotIn("commitment_line", samples_df.columns)
        self.assertNotIn("answer_marker_count", samples_df.columns)

        candidate_df = to_probe_candidate_scores_df(
            [
                {
                    "probe_name": "probe_no_bias",
                    "split": "test",
                    "question_id": "q_1",
                    "prompt_id": "q_1__neutral",
                    "dataset": "aqua_mc",
                    "template_type": "neutral",
                    "draw_idx": 0,
                    "source_record_id": 1,
                    "record_id": 100,
                    "question": "Question 1",
                    "prompt_text": "Prompt text",
                    "correct_letter": "A",
                    "source_selected_choice": "A",
                    "candidate_choice": "A",
                    "candidate_rank": 0,
                    "candidate_probability": 0.7,
                    "probe_sample_weight": 0.7,
                    "candidate_correctness": 1,
                    "candidate_is_selected": True,
                    "probe_score": 0.2,
                }
            ],
            model_name="test/model",
        )
        self.assertEqual(list(candidate_df.columns), PROBE_CANDIDATE_SCORE_COLUMNS)
        self.assertNotIn("dataset", candidate_df.columns)
        self.assertNotIn("question", candidate_df.columns)
        self.assertNotIn("prompt_text", candidate_df.columns)

    def test_build_run_summary_payload_is_compact_and_tracks_headline_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            probe_no_bias_metrics = run_dir / "probe_no_bias_metrics.json"
            probe_bias_metrics = run_dir / "probe_bias_metrics.json"
            probe_no_bias_metrics.write_text(
                json.dumps({"splits": {"test": {"auc": 0.55, "accuracy": 0.5, "balanced_accuracy": 0.5, "n_total": 4}}}),
                encoding="utf-8",
            )
            probe_bias_metrics.write_text(
                json.dumps({"splits": {"test": {"auc": 0.78, "accuracy": 0.75, "balanced_accuracy": 0.75, "n_total": 4}}}),
                encoding="utf-8",
            )

            samples_df = pd.DataFrame([{"question_id": "q_1"}, {"question_id": "q_2"}])
            tuples_df = pd.DataFrame(
                [
                    {
                        "bias_type": "incorrect_suggestion",
                        "split": "train",
                        "question_id": "q_1",
                        "draw_idx": 0,
                        "C_x_y": 1,
                        "C_xprime_yprime": 0,
                        "T_x": 0.8,
                        "T_xprime": 0.3,
                    },
                    {
                        "bias_type": "incorrect_suggestion",
                        "split": "test",
                        "question_id": "q_2",
                        "draw_idx": 0,
                        "C_x_y": 0,
                        "C_xprime_yprime": 0,
                        "T_x": 0.4,
                        "T_xprime": 0.2,
                    },
                ]
            )
            summary_df = pd.DataFrame([{"question_id": "q_1"}, {"question_id": "q_2"}])
            probes_meta = {
                "probe_no_bias": {
                    "best_layer": 3,
                    "best_dev_auc": 0.61,
                    "chosen_probe_metrics_path": str(probe_no_bias_metrics),
                },
                "probe_bias_incorrect_suggestion": {
                    "best_layer": 2,
                    "best_dev_auc": 0.67,
                    "chosen_probe_metrics_path": str(probe_bias_metrics),
                },
            }
            args = SimpleNamespace(
                model="test/model",
                dataset_name="aqua_mc",
                bias_types=["incorrect_suggestion"],
            )

            payload = build_run_summary_payload(
                args=args,
                run_dir=run_dir,
                samples_df=samples_df,
                tuples_df=tuples_df,
                summary_df=summary_df,
                probes_meta=probes_meta,
            )

            self.assertEqual(payload["counts"]["sample_rows"], 2)
            self.assertEqual(payload["counts"]["tuple_rows"], 2)
            self.assertIn("headline_metrics", payload)
            self.assertIn("paths", payload)
            self.assertAlmostEqual(payload["headline_metrics"]["overall_accuracy"], 0.5)
            self.assertAlmostEqual(payload["headline_metrics"]["overall_avg_p_correct"], 0.6)
            self.assertEqual(payload["headline_metrics"]["best_probe_name"], "probe_bias_incorrect_suggestion")
            self.assertAlmostEqual(payload["headline_metrics"]["best_probe_test_auc"], 0.78)

    def test_model_and_probe_summary_artifacts_capture_global_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            probe_no_bias_metrics = run_dir / "probe_no_bias_metrics.json"
            probe_bias_metrics = run_dir / "probe_bias_metrics.json"
            probe_no_bias_metrics.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {"auc": 0.8, "accuracy": 0.75, "balanced_accuracy": 0.75, "n_total": 8},
                            "val": {"auc": 0.7, "accuracy": 0.75, "balanced_accuracy": 0.75, "n_total": 4},
                            "test": {"auc": 0.65, "accuracy": 0.5, "balanced_accuracy": 0.5, "n_total": 4},
                        }
                    }
                ),
                encoding="utf-8",
            )
            probe_bias_metrics.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {"auc": 0.92, "accuracy": 0.875, "balanced_accuracy": 0.875, "n_total": 8},
                            "val": {"auc": 0.81, "accuracy": 0.75, "balanced_accuracy": 0.75, "n_total": 4},
                            "test": {"auc": 0.78, "accuracy": 0.75, "balanced_accuracy": 0.75, "n_total": 4},
                        }
                    }
                ),
                encoding="utf-8",
            )

            samples_df = pd.DataFrame(
                [
                    {
                        "question_id": "q_1",
                        "split": "test",
                        "template_type": "neutral",
                        "correctness": 1,
                        "usable_for_metrics": True,
                        "T_prompt": 0.8,
                        "choice_probability_selected": 0.8,
                        "probe_x": 0.2,
                        "probe_xprime": None,
                        "strict_format_exact": True,
                        "starts_with_answer_prefix": True,
                        "hit_max_new_tokens": False,
                        "stopped_on_eos": False,
                        "completion_token_count": 1,
                        "grading_status": "correct",
                        "finish_reason": "choice_probabilities",
                        "sampling_mode": "choice_probabilities",
                    },
                    {
                        "question_id": "q_1",
                        "split": "test",
                        "template_type": "incorrect_suggestion",
                        "correctness": 0,
                        "usable_for_metrics": True,
                        "T_prompt": 0.3,
                        "choice_probability_selected": 0.5,
                        "probe_x": None,
                        "probe_xprime": 0.6,
                        "strict_format_exact": True,
                        "starts_with_answer_prefix": True,
                        "hit_max_new_tokens": False,
                        "stopped_on_eos": False,
                        "completion_token_count": 1,
                        "grading_status": "incorrect",
                        "finish_reason": "choice_probabilities",
                        "sampling_mode": "choice_probabilities",
                    },
                ]
            )
            tuples_df = pd.DataFrame(
                [
                    {
                        "split": "test",
                        "bias_type": "incorrect_suggestion",
                        "question_id": "q_1",
                        "draw_idx": 0,
                        "C_x_y": 1,
                        "C_xprime_yprime": 0,
                        "T_x": 0.8,
                        "T_xprime": 0.3,
                        "probe_x": 0.2,
                        "probe_xprime": 0.6,
                        "y_x": "A",
                        "y_xprime": "B",
                    }
                ]
            )
            probes_meta = {
                "probe_no_bias": {
                    "template_type": "neutral",
                    "best_layer": 3,
                    "best_dev_auc": 0.7,
                    "trained_layers": [2, 3],
                    "auc_per_layer": {2: 0.66, 3: 0.7},
                    "probe_construction": "choice_candidates",
                    "probe_example_weighting": "model_probability",
                    "chosen_probe_metrics_path": str(probe_no_bias_metrics),
                },
                "probe_bias_incorrect_suggestion": {
                    "template_type": "incorrect_suggestion",
                    "best_layer": 2,
                    "best_dev_auc": 0.81,
                    "trained_layers": [1, 2],
                    "auc_per_layer": {1: 0.73, 2: 0.81},
                    "probe_construction": "choice_candidates",
                    "probe_example_weighting": "model_probability",
                    "chosen_probe_metrics_path": str(probe_bias_metrics),
                },
            }
            candidate_scores_df = pd.DataFrame(
                [
                    {
                        "model_name": "test/model",
                        "probe_name": "probe_bias_incorrect_suggestion",
                        "split": "test",
                        "question_id": "q_1",
                        "prompt_id": "q_1__incorrect_suggestion",
                        "dataset": "aqua_mc",
                        "template_type": "incorrect_suggestion",
                        "draw_idx": 0,
                        "source_record_id": 10,
                        "candidate_record_id": 100,
                        "correct_letter": "A",
                        "selected_choice": "B",
                        "candidate_choice": "A",
                        "candidate_rank": 0,
                        "candidate_probability": 0.3,
                        "probe_sample_weight": 0.3,
                        "candidate_correctness": 1,
                        "candidate_is_selected": False,
                        "probe_score": 0.4,
                    },
                    {
                        "model_name": "test/model",
                        "probe_name": "probe_bias_incorrect_suggestion",
                        "split": "test",
                        "question_id": "q_1",
                        "prompt_id": "q_1__incorrect_suggestion",
                        "dataset": "aqua_mc",
                        "template_type": "incorrect_suggestion",
                        "draw_idx": 0,
                        "source_record_id": 10,
                        "candidate_record_id": 101,
                        "correct_letter": "A",
                        "selected_choice": "B",
                        "candidate_choice": "B",
                        "candidate_rank": 1,
                        "candidate_probability": 0.5,
                        "probe_sample_weight": 0.5,
                        "candidate_correctness": 0,
                        "candidate_is_selected": True,
                        "probe_score": 0.9,
                    },
                ]
            )
            wide_df = build_mc_probe_scores_by_prompt_df(candidate_scores_df)

            args = SimpleNamespace(
                model="test/model",
                dataset_name="aqua_mc",
                bias_types=["incorrect_suggestion"],
            )
            model_summary_payload = build_model_summary_payload(
                args=args,
                run_dir=run_dir,
                samples_df=samples_df,
                tuples_df=tuples_df,
                probes_meta=probes_meta,
            )
            model_summary_by_template_df = build_model_summary_by_template_df(samples_df)
            model_summary_by_bias_df = build_model_summary_by_bias_df(tuples_df)
            probe_summary_payload = build_probe_summary_payload(
                args=args,
                run_dir=run_dir,
                probes_meta=probes_meta,
                probe_candidate_scores_df=candidate_scores_df,
                probe_scores_by_prompt_df=wide_df,
            )
            probe_summary_df = build_probe_summary_df(probe_summary_payload)

            self.assertIn("prompt_level", model_summary_payload)
            self.assertIn("paired_effects", model_summary_payload)
            self.assertEqual(list(model_summary_by_template_df.columns), MODEL_SUMMARY_BY_TEMPLATE_COLUMNS)
            self.assertEqual(list(model_summary_by_bias_df.columns), MODEL_SUMMARY_BY_BIAS_COLUMNS)
            self.assertAlmostEqual(
                model_summary_by_template_df[model_summary_by_template_df["template_type"] == "neutral"].iloc[0]["accuracy"],
                1.0,
            )
            self.assertAlmostEqual(model_summary_by_bias_df.iloc[0]["avg_delta_p_x_minus_xprime"], 0.5)

            self.assertEqual(list(probe_summary_df.columns), PROBE_SUMMARY_COLUMNS)
            self.assertEqual(probe_summary_payload["overview"]["best_probe_on_test"]["probe_name"], "probe_bias_incorrect_suggestion")
            bias_probe_row = probe_summary_df[probe_summary_df["probe_name"] == "probe_bias_incorrect_suggestion"].iloc[0]
            self.assertAlmostEqual(bias_probe_row["test_auc"], 0.78)
            self.assertAlmostEqual(bias_probe_row["probe_prefers_selected_rate"], 1.0)
            self.assertAlmostEqual(bias_probe_row["mean_probe_score_correct_choice"], 0.4)
            self.assertAlmostEqual(bias_probe_row["mean_probe_score_selected_choice"], 0.9)


if __name__ == "__main__":
    unittest.main()
