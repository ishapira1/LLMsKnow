from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from llmssycoph.integrity import check_run_integrity, main
from llmssycoph.runtime import preferred_run_artifact_path, write_csv_atomic, write_json_atomic, write_text_atomic
from llmssycoph.saving_manager import (
    build_executive_summary_markdown,
    build_mc_probe_scores_by_prompt_df,
    build_reports_summary_payload,
    build_tuple_rows,
    to_probe_candidate_scores_df,
    to_samples_df,
    to_tuples_df,
)


class IntegrityContractTests(unittest.TestCase):
    def _build_smoke_run(
        self,
        run_dir: Path,
        *,
        sampling_only: bool = False,
        dataset_name: str = "aqua_mc",
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        question_splits = {"train": ["q_1"], "val": ["q_2"], "test": ["q_3"]}
        bias_types = ["incorrect_suggestion"]
        templates = ["neutral", *bias_types]
        model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

        samples_path = preferred_run_artifact_path(run_dir, "sampled_responses")
        sampling_records_path = preferred_run_artifact_path(run_dir, "sampling_records")
        probe_scores_by_prompt_path = preferred_run_artifact_path(run_dir, "probe_scores_by_prompt")
        reports_summary_path = preferred_run_artifact_path(run_dir, "reports_summary")
        reports_summary_csv_path = preferred_run_artifact_path(run_dir, "reports_summary_csv")
        run_summary_path = preferred_run_artifact_path(run_dir, "run_summary")
        executive_summary_path = preferred_run_artifact_path(run_dir, "executive_summary")
        mc_confusion_matrix_path = preferred_run_artifact_path(run_dir, "mc_confusion_matrix")

        all_records = []
        probe_candidate_rows = []
        record_id = 1
        candidate_record_id = 100
        for split_name, question_ids in question_splits.items():
            for question_id in question_ids:
                for template_type in templates:
                    is_neutral = template_type == "neutral"
                    prompt_id = f"{question_id}__{template_type}"
                    t_prompt = 0.9 if is_neutral else 0.7
                    probe_score = None if sampling_only else (0.25 if is_neutral else 0.65)
                    all_records.append(
                        {
                            "record_id": record_id,
                            "split": split_name,
                            "question_id": question_id,
                            "prompt_id": prompt_id,
                            "dataset": dataset_name,
                            "template_type": template_type,
                            "draw_idx": 0,
                            "task_format": "multiple_choice",
                            "mc_mode": "strict_mc",
                            "answer_channel": "letter",
                            "question": f"Question {question_id}",
                            "correct_answer": "Option A",
                            "incorrect_answer": "Option B",
                            "correct_letter": "A",
                            "incorrect_letter": "B",
                            "gold_answers": ["A"],
                            "prompt_template": "{question}",
                            "prompt_text": f"Prompt {question_id} {template_type}",
                            "response_raw": "A",
                            "response": "A",
                            "starts_with_answer_prefix": True,
                            "strict_format_exact": True,
                            "correctness": 1,
                            "grading_status": "correct",
                            "grading_reason": "single_letter_match",
                            "usable_for_metrics": True,
                            "completion_token_count": 1,
                            "hit_max_new_tokens": False,
                            "stopped_on_eos": False,
                            "finish_reason": "choice_probabilities",
                            "sampling_mode": "choice_probabilities",
                            "choice_probabilities": {"A": t_prompt, "B": 1.0 - t_prompt},
                            "choice_probability_correct": t_prompt,
                            "choice_probability_selected": t_prompt,
                            "T_prompt": t_prompt,
                            "probe_x": probe_score if is_neutral else None,
                            "probe_xprime": probe_score if not is_neutral else None,
                        }
                    )
                    probe_name = "probe_no_bias" if is_neutral else "probe_bias_incorrect_suggestion"
                    if not sampling_only:
                        for candidate_choice, candidate_probability, candidate_correctness, candidate_is_selected, score in (
                            ("A", t_prompt, 1, True, probe_score),
                            ("B", 1.0 - t_prompt, 0, False, probe_score - 0.2),
                        ):
                            probe_candidate_rows.append(
                                {
                                    "probe_name": probe_name,
                                    "split": split_name,
                                    "question_id": question_id,
                                    "prompt_id": prompt_id,
                                    "dataset": dataset_name,
                                    "template_type": template_type,
                                    "draw_idx": 0,
                                    "source_record_id": record_id,
                                    "record_id": candidate_record_id,
                                    "correct_letter": "A",
                                    "source_selected_choice": "A",
                                    "candidate_choice": candidate_choice,
                                    "candidate_rank": 0 if candidate_choice == "A" else 1,
                                    "candidate_probability": candidate_probability,
                                    "probe_sample_weight": candidate_probability,
                                    "candidate_correctness": candidate_correctness,
                                    "candidate_is_selected": candidate_is_selected,
                                    "probe_score": score,
                                }
                            )
                            candidate_record_id += 1
                    record_id += 1

        samples_df = to_samples_df(all_records, model_name=model_name)
        tuple_rows = build_tuple_rows(all_records, model_name=model_name, bias_types=bias_types)
        tuples_df = to_tuples_df(tuple_rows)
        probe_candidate_scores_df = to_probe_candidate_scores_df(probe_candidate_rows, model_name=model_name)
        probe_scores_by_prompt_df = build_mc_probe_scores_by_prompt_df(probe_candidate_scores_df)

        probes_meta = {
            "probe_training_status": "skipped_by_sampling_only" if sampling_only else "completed",
            "strict_mc_quality": {
                "status": "passed",
                "issues": [],
                "summary": {
                    "commitment_rate": 1.0,
                    "starts_with_answer_rate": 1.0,
                    "exact_format_rate": 1.0,
                    "cap_hit_rate": 0.0,
                    "explicit_parse_failures": 0,
                    "max_neutral_bias_answer_gap": 0.0,
                    "by_template": {
                        "neutral": {"total": 3},
                        "incorrect_suggestion": {"total": 3},
                    },
                },
            },
        }
        if not sampling_only:
            probe_no_bias_metrics = run_dir / "probe_no_bias_metrics.json"
            probe_bias_metrics = run_dir / "probe_bias_metrics.json"
            write_text_atomic(
                probe_no_bias_metrics,
                json.dumps(
                    {
                        "splits": {
                            "train": {"auc": 0.72, "accuracy": 1.0, "balanced_accuracy": 1.0, "n_total": 2},
                            "val": {"auc": 0.7, "accuracy": 1.0, "balanced_accuracy": 1.0, "n_total": 2},
                            "test": {"auc": 0.68, "accuracy": 1.0, "balanced_accuracy": 1.0, "n_total": 2},
                        }
                    }
                ),
            )
            write_text_atomic(
                probe_bias_metrics,
                json.dumps(
                    {
                        "splits": {
                            "train": {"auc": 0.84, "accuracy": 1.0, "balanced_accuracy": 1.0, "n_total": 2},
                            "val": {"auc": 0.82, "accuracy": 1.0, "balanced_accuracy": 1.0, "n_total": 2},
                            "test": {"auc": 0.8, "accuracy": 1.0, "balanced_accuracy": 1.0, "n_total": 2},
                        }
                    }
                ),
            )
            probes_meta.update(
                {
                    "probe_no_bias": {
                        "template_type": "neutral",
                        "best_layer": 1,
                        "best_dev_auc": 0.7,
                        "trained_layers": [1],
                        "auc_per_layer": {1: 0.7},
                        "probe_construction": "choice_candidates",
                        "probe_example_weighting": "model_probability",
                        "chosen_probe_metrics_path": str(probe_no_bias_metrics),
                    },
                    "probe_bias_incorrect_suggestion": {
                        "template_type": "incorrect_suggestion",
                        "best_layer": 1,
                        "best_dev_auc": 0.82,
                        "trained_layers": [1],
                        "auc_per_layer": {1: 0.82},
                        "probe_construction": "choice_candidates",
                        "probe_example_weighting": "model_probability",
                        "chosen_probe_metrics_path": str(probe_bias_metrics),
                    },
                }
            )

        args = SimpleNamespace(
            model=model_name,
            dataset_name=dataset_name,
            bias_types=bias_types,
            sampling_only=sampling_only,
        )
        reports_summary = build_reports_summary_payload(
            args=args,
            run_dir=run_dir,
            samples_df=samples_df,
            tuples_df=tuples_df,
            probe_scores_by_prompt_df=probe_scores_by_prompt_df,
            probes_meta=probes_meta,
            probe_candidate_scores_df=probe_candidate_scores_df,
        )
        executive_summary = build_executive_summary_markdown(reports_summary)

        write_csv_atomic(samples_path, samples_df)
        write_csv_atomic(reports_summary_csv_path, pd.DataFrame(reports_summary["summary_rows"]))
        write_csv_atomic(
            mc_confusion_matrix_path,
            pd.DataFrame(
                reports_summary.get("mc_confusion_matrix", {}).get("summary_rows", []),
                columns=["predicted_letter", *reports_summary.get("mc_confusion_matrix", {}).get("choice_labels", [])],
            ),
        )
        sampling_records_path.parent.mkdir(parents=True, exist_ok=True)
        sampling_records_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in all_records) + "\n",
            encoding="utf-8",
        )
        write_csv_atomic(probe_scores_by_prompt_path, probe_scores_by_prompt_df)
        write_json_atomic(reports_summary_path, reports_summary["summary_rows"])
        write_json_atomic(run_summary_path, reports_summary)
        write_text_atomic(executive_summary_path, executive_summary)

        write_json_atomic(
            preferred_run_artifact_path(run_dir, "run_config"),
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "model": model_name,
                "device": "cpu",
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "benchmark_source": "ays_mc_single_turn",
                "input_jsonl": "are_you_sure.jsonl",
                "dataset_name": dataset_name,
                "ays_mc_datasets": [dataset_name],
                "mc_mode": "strict_mc",
                "bias_types": bias_types,
                "smoke_test": True,
                "smoke_questions": 3,
                "max_questions": 3,
                "n_draws": 1,
                "sample_batch_size": 1,
                "temperature": 1.0,
                "max_new_tokens": 32,
                "sampling_only": sampling_only,
                "probe_construction": "auto",
                "probe_example_weighting": "model_probability",
                "strict_mc_choice_scoring": True,
                "sampling_records_path": str(sampling_records_path),
                "sampling_manifest_path": str(preferred_run_artifact_path(run_dir, "sampling_manifest")),
                "sampling_integrity_summary_path": str(
                    preferred_run_artifact_path(run_dir, "sampling_integrity_summary")
                ),
                "sampled_responses_path": str(samples_path),
                "reports_summary_path": str(reports_summary_path),
                "reports_summary_csv_path": str(reports_summary_csv_path),
                "run_summary_path": str(run_summary_path),
                "probe_scores_by_prompt_path": str(probe_scores_by_prompt_path),
                "executive_summary_path": str(executive_summary_path),
            },
        )
        write_json_atomic(
            preferred_run_artifact_path(run_dir, "status"),
            {
                "status": "completed",
                "run_dir": str(run_dir),
            },
        )
        write_json_atomic(
            preferred_run_artifact_path(run_dir, "sampling_manifest"),
            {
                "expected_records": len(all_records),
                "n_records": len(all_records),
                "is_complete": True,
                "sampling_spec": {
                    "train_question_ids": question_splits["train"],
                    "val_question_ids": question_splits["val"],
                    "test_question_ids": question_splits["test"],
                },
                "split_stats": {
                    split_name: {"expected_records": len(question_ids) * len(templates)}
                    for split_name, question_ids in question_splits.items()
                },
            },
        )
        write_json_atomic(
            preferred_run_artifact_path(run_dir, "sampling_integrity_summary"),
            {
                "sampling_integrity_version": 1,
                "total_records": len(all_records),
                "sampling_modes_present": ["choice_probabilities"],
                "by_sampling_mode": {
                    "choice_probabilities": {
                        "total": len(all_records),
                        "buckets": {
                            "exact_compliance": {"count": len(all_records), "rate": 1.0},
                            "minor_format_deviation_still_scoreable": {"count": 0, "rate": 0.0},
                            "format_failure": {"count": 0, "rate": 0.0},
                        },
                    }
                },
            },
        )

        if not sampling_only:
            all_probes_dir = preferred_run_artifact_path(run_dir, "all_probes_dir")
            chosen_probe_dir = preferred_run_artifact_path(run_dir, "chosen_probe_dir")
            write_json_atomic(
                all_probes_dir / "manifest.json",
                {
                    "artifact_group": "all_probes",
                    "probe_names": ["probe_bias_incorrect_suggestion", "probe_no_bias"],
                },
            )
            write_json_atomic(
                chosen_probe_dir / "manifest.json",
                {
                    "artifact_group": "chosen_probe",
                    "probe_names": ["probe_bias_incorrect_suggestion", "probe_no_bias"],
                },
            )
            write_json_atomic(all_probes_dir / "probe_no_bias" / "manifest.json", {"probe_name": "probe_no_bias"})
            write_json_atomic(
                all_probes_dir / "probe_bias_incorrect_suggestion" / "manifest.json",
                {"probe_name": "probe_bias_incorrect_suggestion"},
            )
            write_json_atomic(chosen_probe_dir / "probe_no_bias" / "manifest.json", {"probe_name": "probe_no_bias"})
            write_json_atomic(
                chosen_probe_dir / "probe_bias_incorrect_suggestion" / "manifest.json",
                {"probe_name": "probe_bias_incorrect_suggestion"},
            )

        run_log_path = preferred_run_artifact_path(run_dir, "run_log")
        write_text_atomic(run_log_path, "ok\n")

    def test_check_run_integrity_accepts_complete_smoke_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir)

            report = check_run_integrity(run_dir)

            self.assertEqual(report["sample_count"], 6)
            self.assertEqual(report["tuple_count"], 3)
            self.assertEqual(report["question_count"], 3)

    def test_check_run_integrity_accepts_sampling_only_smoke_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir, sampling_only=True)

            report = check_run_integrity(run_dir)

            self.assertEqual(report["sample_count"], 6)
            self.assertEqual(report["tuple_count"], 3)
            self.assertEqual(report["question_count"], 3)

    def test_check_run_integrity_accepts_non_aqua_ays_smoke_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir, dataset_name="commonsense_qa")

            report = check_run_integrity(run_dir)

            self.assertEqual(report["sample_count"], 6)
            self.assertEqual(report["tuple_count"], 3)
            self.assertEqual(report["question_count"], 3)

    def test_check_run_integrity_accepts_auto_requested_cuda_resolved_device(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir)
            run_config_path = preferred_run_artifact_path(run_dir, "run_config")
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            run_config["device"] = "auto"
            run_config["requested_device"] = "auto"
            run_config["resolved_device"] = "cuda"
            write_json_atomic(run_config_path, run_config)

            report = check_run_integrity(run_dir)

            self.assertEqual(report["requested_device"], "auto")
            self.assertEqual(report["resolved_device"], "cuda")

    def test_check_run_integrity_rejects_duplicate_sample_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir)
            sampled_df = pd.read_csv(preferred_run_artifact_path(run_dir, "sampled_responses"))
            sampled_df = pd.concat([sampled_df, sampled_df.iloc[[0]]], ignore_index=True)
            sampled_df.to_csv(preferred_run_artifact_path(run_dir, "sampled_responses"), index=False)

            with self.assertRaises(RuntimeError):
                check_run_integrity(run_dir)

    def test_integrity_main_warns_and_returns_zero_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir)
            sampled_df = pd.read_csv(preferred_run_artifact_path(run_dir, "sampled_responses")).drop(
                columns=["prompt_id"]
            )
            sampled_df.to_csv(preferred_run_artifact_path(run_dir, "sampled_responses"), index=False)

            exit_code = main(["--run_dir", str(run_dir)])

            self.assertEqual(exit_code, 0)

    def test_integrity_main_strict_returns_nonzero_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            self._build_smoke_run(run_dir)
            sampled_df = pd.read_csv(preferred_run_artifact_path(run_dir, "sampled_responses")).drop(
                columns=["prompt_id"]
            )
            sampled_df.to_csv(preferred_run_artifact_path(run_dir, "sampled_responses"), index=False)

            exit_code = main(["--run_dir", str(run_dir), "--strict"])

            self.assertEqual(exit_code, 1)

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
