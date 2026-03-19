from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llmssycoph.integrity import check_run_integrity
from llmssycoph.runtime import preferred_run_artifact_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class IntegrityContractTests(unittest.TestCase):
    def _build_smoke_run(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        samples_path = preferred_run_artifact_path(run_dir, "sampled_responses")
        sampling_records_path = preferred_run_artifact_path(run_dir, "sampling_records")
        tuples_path = preferred_run_artifact_path(run_dir, "final_tuples")
        summary_path = preferred_run_artifact_path(run_dir, "summary_by_question")
        model_summary_by_template_path = preferred_run_artifact_path(run_dir, "model_summary_by_template")
        model_summary_by_bias_path = preferred_run_artifact_path(run_dir, "model_summary_by_bias")
        probe_candidate_scores_path = preferred_run_artifact_path(run_dir, "probe_candidate_scores")
        probe_scores_by_prompt_path = preferred_run_artifact_path(run_dir, "probe_scores_by_prompt")
        probe_summary_csv_path = preferred_run_artifact_path(run_dir, "probe_summary_csv")
        executive_summary_path = preferred_run_artifact_path(run_dir, "executive_summary")
        for path in (
            samples_path,
            sampling_records_path,
            tuples_path,
            summary_path,
            model_summary_by_template_path,
            model_summary_by_bias_path,
            probe_candidate_scores_path,
            probe_scores_by_prompt_path,
            probe_summary_csv_path,
            executive_summary_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        question_splits = {"train": ["q_1"], "val": ["q_2"], "test": ["q_3"]}
        bias_types = ["incorrect_suggestion"]
        draw_count = 1
        templates = ["neutral", *bias_types]
        rows = []
        record_id = 0
        for split_name, question_ids in question_splits.items():
            for question_id in question_ids:
                for template_type in templates:
                    rows.append(
                        {
                            "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                            "record_id": record_id,
                            "split": split_name,
                            "question_id": question_id,
                            "prompt_id": f"{question_id}__{template_type}",
                            "dataset": "aqua_mc",
                            "template_type": template_type,
                            "draw_idx": 0,
                            "question": f"Question {question_id}",
                            "correct_answer": "A",
                            "incorrect_answer": "B",
                            "incorrect_answer_source": "",
                            "task_format": "multiple_choice",
                            "mc_mode": "strict_mc",
                            "answer_channel": "letter",
                            "prompt_spec_version": 1,
                            "grading_spec_version": 1,
                            "correct_letter": "A",
                            "incorrect_letter": "B",
                            "letters": "ABCD",
                            "answer_options": "(A) one\n(B) two",
                            "answers_list": json.dumps(["one", "two"]),
                            "gold_answers": json.dumps(["A"]),
                            "prompt_template": "{question}",
                            "prompt_text": f"Prompt {question_id} {template_type}",
                            "response_raw": "Answer: A",
                            "response": "A",
                            "committed_answer": "A",
                            "commitment_kind": "letter",
                            "commitment_source": "explicit_answer_line",
                            "starts_with_answer_prefix": True,
                            "strict_format_exact": True,
                            "commitment_line": "Answer: A",
                            "answer_marker_count": 1,
                            "multiple_answer_markers": False,
                            "correctness": 1,
                            "grading_status": "correct",
                            "grading_reason": "single_letter_match",
                            "usable_for_metrics": True,
                            "completion_token_count": 3,
                            "hit_max_new_tokens": False,
                            "stopped_on_eos": False,
                            "finish_reason": "answer_commitment",
                            "sampling_mode": "generation",
                            "choice_probabilities": "{}",
                            "choice_probability_correct": None,
                            "choice_probability_selected": None,
                            "T_prompt": 1.0,
                            "probe_x": 0.3 if template_type == "neutral" else float("nan"),
                            "probe_xprime": 0.7 if template_type != "neutral" else float("nan"),
                        }
                    )
                    record_id += 1

        sampled_df = pd.DataFrame(rows)
        sampled_df.to_csv(samples_path, index=False)
        sampling_records_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

        tuples_df = pd.DataFrame(
            [
                {
                    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                    "split": split_name,
                    "question_id": question_id,
                    "prompt_id_x": f"{question_id}__neutral",
                    "prompt_id_xprime": f"{question_id}__incorrect_suggestion",
                    "dataset": "aqua_mc",
                    "bias_type": "incorrect_suggestion",
                    "draw_idx": 0,
                    "question": f"Question {question_id}",
                    "correct_answer": "A",
                    "incorrect_answer": "B",
                    "gold_answers": json.dumps(["A"]),
                    "prompt_x": f"Prompt {question_id} neutral",
                    "prompt_with_bias": f"Prompt {question_id} incorrect_suggestion",
                    "prompt_template_x": "{question}",
                    "prompt_template_xprime": "{question} bias",
                    "y_x": "A",
                    "y_xprime": "A",
                    "C_x_y": 1,
                    "C_xprime_yprime": 1,
                    "T_x": 1.0,
                    "T_xprime": 1.0,
                    "probe_x_name": "probe_no_bias",
                    "probe_xprime_name": "probe_bias_incorrect_suggestion",
                    "probe_x": 0.3,
                    "probe_xprime": 0.7,
                }
                for split_name, question_ids in question_splits.items()
                for question_id in question_ids
            ]
        )
        tuples_df.to_csv(tuples_path, index=False)

        summary_df = pd.DataFrame(
            [
                {
                    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                    "split": split_name,
                    "question_id": question_id,
                    "prompt_id_x": f"{question_id}__neutral",
                    "prompt_id_xprime": f"{question_id}__incorrect_suggestion",
                    "dataset": "aqua_mc",
                    "bias_type": "incorrect_suggestion",
                    "question": f"Question {question_id}",
                    "correct_answer": "A",
                    "incorrect_answer": "B",
                    "prompt_template_x": "{question}",
                    "prompt_template_xprime": "{question} bias",
                    "prompt_x": f"Prompt {question_id} neutral",
                    "prompt_with_bias": f"Prompt {question_id} incorrect_suggestion",
                    "T_x": 1.0,
                    "T_xprime": 1.0,
                    "mean_C_x": 1.0,
                    "mean_C_xprime": 1.0,
                    "mean_probe_x": 0.3,
                    "mean_probe_xprime": 0.7,
                    "n_draws": 1,
                }
                for split_name, question_ids in question_splits.items()
                for question_id in question_ids
            ]
        )
        summary_df.to_csv(summary_path, index=False)
        pd.DataFrame(
            [
                {
                    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                    "probe_name": "probe_no_bias" if template_type == "neutral" else "probe_bias_incorrect_suggestion",
                    "split": split_name,
                    "question_id": question_id,
                    "prompt_id": f"{question_id}__{template_type}",
                    "dataset": "aqua_mc",
                    "template_type": template_type,
                    "draw_idx": 0,
                    "source_record_id": idx + 1,
                    "candidate_record_id": (idx + 1) * 100,
                    "question": f"Question {question_id}",
                    "prompt_text": f"Prompt {question_id} {template_type}",
                    "correct_letter": "A",
                    "selected_choice": "A",
                    "candidate_choice": "A",
                    "candidate_rank": 0,
                    "candidate_probability": 1.0,
                    "probe_sample_weight": 1.0,
                    "candidate_correctness": 1,
                    "candidate_is_selected": True,
                    "probe_score": 0.8,
                }
                for idx, (split_name, question_ids) in enumerate(question_splits.items())
                for question_id in question_ids
                for template_type in templates
            ]
        ).to_csv(probe_candidate_scores_path, index=False)
        pd.DataFrame(
            [
                {
                    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                    "probe_name": "probe_no_bias" if template_type == "neutral" else "probe_bias_incorrect_suggestion",
                    "split": split_name,
                    "question_id": question_id,
                    "prompt_id": f"{question_id}__{template_type}",
                    "dataset": "aqua_mc",
                    "template_type": template_type,
                    "draw_idx": 0,
                    "source_record_id": idx + 1,
                    "correct_letter": "A",
                    "selected_choice": "A",
                    "selected_choice_is_correct": True,
                    "probe_score_correct_choice": 0.8,
                    "probe_score_selected_choice": 0.8,
                    "correct_choice_probability": 1.0,
                    "selected_choice_probability": 1.0,
                    "probe_argmax_choice": "A",
                    "probe_argmax_score": 0.8,
                    "probe_prefers_correct": True,
                    "probe_prefers_selected": True,
                    "probe_score_gap_correct_minus_selected": 0.0,
                    "score_A": 0.8,
                }
                for idx, (split_name, question_ids) in enumerate(question_splits.items())
                for question_id in question_ids
                for template_type in templates
            ]
        ).to_csv(probe_scores_by_prompt_path, index=False)
        pd.DataFrame(
            [
                {
                    "template_type": template_type,
                    "n_rows": 3,
                    "n_questions": 3,
                    "n_usable_rows": 3,
                    "usable_rate": 1.0,
                    "ambiguous_rate": 0.0,
                    "accuracy": 1.0,
                    "avg_p_correct": 1.0,
                    "avg_p_selected": 1.0,
                    "avg_selected_minus_correct_probability_gap": 0.0,
                    "avg_probe_score_selected_prompt": 0.3 if template_type == "neutral" else 0.7,
                    "exact_format_rate": 1.0,
                    "starts_with_answer_prefix_rate": 1.0,
                    "cap_hit_rate": 0.0,
                    "stopped_on_eos_rate": 0.0,
                    "avg_completion_token_count": 3.0,
                }
                for template_type in templates
            ]
        ).to_csv(model_summary_by_template_path, index=False)
        pd.DataFrame(
            [
                {
                    "bias_type": "incorrect_suggestion",
                    "n_pairs": 3,
                    "n_questions": 3,
                    "accuracy_x": 1.0,
                    "accuracy_xprime": 1.0,
                    "delta_accuracy_x_minus_xprime": 0.0,
                    "avg_p_x": 1.0,
                    "avg_p_xprime": 1.0,
                    "avg_delta_p_x_minus_xprime": 0.0,
                    "avg_probe_x": 0.3,
                    "avg_probe_xprime": 0.7,
                    "avg_delta_probe_x_minus_xprime": -0.4,
                    "harmful_flip_rate": 0.0,
                    "helpful_flip_rate": 0.0,
                    "unchanged_correctness_rate": 1.0,
                    "answer_change_rate": 0.0,
                }
            ]
        ).to_csv(model_summary_by_bias_path, index=False)
        pd.DataFrame(
            [
                {
                    "probe_name": "probe_no_bias",
                    "template_type": "neutral",
                    "probe_construction": "choice_candidates",
                    "probe_example_weighting": "model_probability",
                    "best_layer": 1,
                    "best_dev_auc": 0.8,
                    "train_auc": 0.82,
                    "val_auc": 0.8,
                    "test_auc": 0.75,
                    "train_accuracy": 1.0,
                    "val_accuracy": 1.0,
                    "test_accuracy": 1.0,
                    "train_balanced_accuracy": 1.0,
                    "val_balanced_accuracy": 1.0,
                    "test_balanced_accuracy": 1.0,
                    "train_n_total": 3,
                    "val_n_total": 3,
                    "test_n_total": 3,
                    "train_minus_val_auc": 0.02,
                    "val_minus_test_auc": 0.05,
                    "train_minus_test_auc": 0.07,
                    "probe_prefers_correct_rate": 1.0,
                    "probe_prefers_selected_rate": 1.0,
                    "mean_probe_score_correct_candidate": 0.8,
                    "mean_probe_score_incorrect_candidate": 0.2,
                    "mean_probe_score_selected_candidate": 0.8,
                    "mean_probe_score_non_selected_candidate": 0.2,
                    "mean_probe_score_correct_choice": 0.8,
                    "mean_probe_score_selected_choice": 0.8,
                    "mean_correct_minus_selected_probe_gap": 0.0,
                },
                {
                    "probe_name": "probe_bias_incorrect_suggestion",
                    "template_type": "incorrect_suggestion",
                    "probe_construction": "choice_candidates",
                    "probe_example_weighting": "model_probability",
                    "best_layer": 1,
                    "best_dev_auc": 0.82,
                    "train_auc": 0.84,
                    "val_auc": 0.82,
                    "test_auc": 0.8,
                    "train_accuracy": 1.0,
                    "val_accuracy": 1.0,
                    "test_accuracy": 1.0,
                    "train_balanced_accuracy": 1.0,
                    "val_balanced_accuracy": 1.0,
                    "test_balanced_accuracy": 1.0,
                    "train_n_total": 3,
                    "val_n_total": 3,
                    "test_n_total": 3,
                    "train_minus_val_auc": 0.02,
                    "val_minus_test_auc": 0.02,
                    "train_minus_test_auc": 0.04,
                    "probe_prefers_correct_rate": 1.0,
                    "probe_prefers_selected_rate": 1.0,
                    "mean_probe_score_correct_candidate": 0.8,
                    "mean_probe_score_incorrect_candidate": 0.2,
                    "mean_probe_score_selected_candidate": 0.8,
                    "mean_probe_score_non_selected_candidate": 0.2,
                    "mean_probe_score_correct_choice": 0.8,
                    "mean_probe_score_selected_choice": 0.8,
                    "mean_correct_minus_selected_probe_gap": 0.0,
                },
            ]
        ).to_csv(probe_summary_csv_path, index=False)
        executive_summary_path.write_text(
            "# Executive Summary\n\n## Model Overview\n\nA compact human summary.\n",
            encoding="utf-8",
        )

        _write_json(
            preferred_run_artifact_path(run_dir, "run_config"),
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "model_slug": "HuggingFaceTB_SmolLM2_135M_Instruct",
                "device": "cpu",
                "benchmark_source": "ays_mc_single_turn",
                "input_jsonl": "are_you_sure.jsonl",
                "dataset_name": "aqua_mc",
                "ays_mc_datasets": ["aqua_mc"],
                "mc_mode": "strict_mc",
                "bias_types": bias_types,
                "smoke_test": True,
                "smoke_questions": 3,
                "max_questions": 3,
                "n_draws": draw_count,
                "sample_batch_size": 1,
                "temperature": 1.0,
                "max_new_tokens": 256,
                "probe_construction": "auto",
                "probe_example_weighting": "model_probability",
                "strict_mc_choice_scoring": True,
                "model_summary_by_template_path": str(model_summary_by_template_path),
                "model_summary_by_bias_path": str(model_summary_by_bias_path),
                "probe_scores_by_prompt_path": str(probe_scores_by_prompt_path),
                "probe_summary_csv_path": str(probe_summary_csv_path),
                "probe_candidate_scores_path": str(probe_candidate_scores_path),
                "executive_summary_path": str(executive_summary_path),
            },
        )
        _write_json(
            preferred_run_artifact_path(run_dir, "status"),
            {
                "status": "completed",
                "run_dir": str(run_dir),
            },
        )
        _write_json(
            preferred_run_artifact_path(run_dir, "sampling_manifest"),
            {
                "expected_records": len(rows),
                "n_records": len(rows),
                "is_complete": True,
                "sampling_spec": {
                    "train_question_ids": question_splits["train"],
                    "val_question_ids": question_splits["val"],
                    "test_question_ids": question_splits["test"],
                },
                "split_stats": {
                    split_name: {
                        "expected_records": len(question_ids) * len(templates) * draw_count,
                    }
                    for split_name, question_ids in question_splits.items()
                },
            },
        )
        _write_json(
            preferred_run_artifact_path(run_dir, "sampling_integrity_summary"),
            {
                "sampling_integrity_version": 1,
                "total_records": len(rows),
                "sampling_modes_present": ["generation"],
                "by_sampling_mode": {
                    "generation": {
                        "total": len(rows),
                        "buckets": {
                            "exact_compliance": {"count": len(rows), "rate": 1.0},
                            "minor_format_deviation_still_scoreable": {"count": 0, "rate": 0.0},
                            "format_failure": {"count": 0, "rate": 0.0},
                        },
                    }
                },
            },
        )
        _write_json(
            preferred_run_artifact_path(run_dir, "probe_metadata"),
            {
                "strict_mc_quality": {
                    "status": "passed",
                    "issues": [],
                    "summary": {
                        "commitment_rate": 1.0,
                        "starts_with_answer_rate": 1.0,
                        "cap_hit_rate": 0.0,
                        "explicit_parse_failures": 0,
                        "exact_format_rate": 1.0,
                        "multiple_answer_marker_rows": 0,
                        "max_neutral_bias_answer_gap": 0.0,
                        "by_template": {
                            "neutral": {
                                "total": 3,
                                "committed_rate": 1.0,
                                "starts_with_answer_rate": 1.0,
                                "cap_hit_rate": 0.0,
                                "exact_format_rate": 1.0,
                                "multiple_answer_marker_rows": 0,
                            },
                            "incorrect_suggestion": {
                                "total": 3,
                                "committed_rate": 1.0,
                                "starts_with_answer_rate": 1.0,
                                "cap_hit_rate": 0.0,
                                "exact_format_rate": 1.0,
                                "multiple_answer_marker_rows": 0,
                            },
                        },
                    },
                },
                "probe_no_bias": {"best_layer": 1},
                "probe_bias_incorrect_suggestion": {"best_layer": 1},
            },
        )
        all_probes_dir = preferred_run_artifact_path(run_dir, "all_probes_dir")
        chosen_probe_dir = preferred_run_artifact_path(run_dir, "chosen_probe_dir")
        (all_probes_dir / "probe_no_bias").mkdir(parents=True, exist_ok=True)
        (all_probes_dir / "probe_bias_incorrect_suggestion").mkdir(parents=True, exist_ok=True)
        (chosen_probe_dir / "probe_no_bias").mkdir(parents=True, exist_ok=True)
        (chosen_probe_dir / "probe_bias_incorrect_suggestion").mkdir(parents=True, exist_ok=True)
        _write_json(
            all_probes_dir / "manifest.json",
            {
                "artifact_group": "all_probes",
                "probe_names": ["probe_bias_incorrect_suggestion", "probe_no_bias"],
            },
        )
        _write_json(
            chosen_probe_dir / "manifest.json",
            {
                "artifact_group": "chosen_probe",
                "probe_names": ["probe_bias_incorrect_suggestion", "probe_no_bias"],
            },
        )
        _write_json(all_probes_dir / "probe_no_bias" / "manifest.json", {"probe_name": "probe_no_bias"})
        _write_json(
            all_probes_dir / "probe_bias_incorrect_suggestion" / "manifest.json",
            {"probe_name": "probe_bias_incorrect_suggestion"},
        )
        _write_json(chosen_probe_dir / "probe_no_bias" / "manifest.json", {"probe_name": "probe_no_bias"})
        _write_json(
            chosen_probe_dir / "probe_bias_incorrect_suggestion" / "manifest.json",
            {"probe_name": "probe_bias_incorrect_suggestion"},
        )
        run_log_path = preferred_run_artifact_path(run_dir, "run_log")
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text("ok\n", encoding="utf-8")
        _write_json(
            preferred_run_artifact_path(run_dir, "run_summary"),
            {
                "headline_metrics": {
                    "overall_accuracy": 1.0,
                    "overall_avg_p_correct": 1.0,
                    "avg_delta_p_x_minus_xprime": 0.0,
                    "best_probe_name": "probe_bias_incorrect_suggestion",
                    "best_probe_test_auc": 0.8,
                },
                "paths": {
                    "model_summary_by_template": str(model_summary_by_template_path),
                    "model_summary_by_bias": str(model_summary_by_bias_path),
                    "probe_summary_csv": str(probe_summary_csv_path),
                    "probe_scores_by_prompt": str(probe_scores_by_prompt_path),
                    "executive_summary": str(executive_summary_path),
                },
            },
        )

    def test_check_run_integrity_accepts_complete_smoke_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir)

            report = check_run_integrity(run_dir)

            self.assertEqual(report["sample_count"], 6)
            self.assertEqual(report["tuple_count"], 3)
            self.assertEqual(report["question_count"], 3)

    def test_check_run_integrity_rejects_duplicate_sample_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir)
            sampled_df = pd.read_csv(preferred_run_artifact_path(run_dir, "sampled_responses"))
            sampled_df = pd.concat([sampled_df, sampled_df.iloc[[0]]], ignore_index=True)
            sampled_df.to_csv(preferred_run_artifact_path(run_dir, "sampled_responses"), index=False)

            with self.assertRaises(RuntimeError):
                check_run_integrity(run_dir)

    def test_check_run_integrity_rejects_missing_prompt_id_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir)
            sampled_df = pd.read_csv(preferred_run_artifact_path(run_dir, "sampled_responses")).drop(
                columns=["prompt_id"]
            )
            sampled_df.to_csv(preferred_run_artifact_path(run_dir, "sampled_responses"), index=False)

            with self.assertRaises(RuntimeError):
                check_run_integrity(run_dir)


if __name__ == "__main__":
    unittest.main()
