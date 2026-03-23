from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

from llmssycoph.analysis import (
    build_candidate_probability_long_df,
    get_analysis_function_spec,
    load_analysis_context,
    run_analysis_operation,
    safe_generate_analysis_notebook,
    safe_run_analysis_operation,
)
from llmssycoph.analysis.core import AnalysisNotSupportedError
from llmssycoph.analysis.dataframes import build_paired_probe_df, build_probe_scores_df


matplotlib.use("Agg")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _mc_samples_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question_id": "q1",
                "template_type": "neutral",
                "response": "A",
                "P(selected)": 0.8,
                "P(correct)": 0.8,
                "task_format": "multiple_choice",
                "dataset": "commonsense_qa",
                "P(A)": 0.8,
                "P(B)": 0.1,
                "P(C)": 0.05,
                "P(D)": 0.03,
                "P(E)": 0.02,
            },
            {
                "question_id": "q1",
                "template_type": "incorrect_suggestion",
                "response": "B",
                "P(selected)": 0.6,
                "P(correct)": 0.2,
                "task_format": "multiple_choice",
                "dataset": "commonsense_qa",
                "P(A)": 0.2,
                "P(B)": 0.6,
                "P(C)": 0.08,
                "P(D)": 0.07,
                "P(E)": 0.05,
            },
        ]
    )


def _probe_scores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_name": "probe_no_bias",
                "question_id": "q1",
                "prompt_id": "q1__neutral",
                "template_type": "neutral",
                "correct_letter": "A",
                "selected_choice": "A",
                "probe_score_correct_choice": 0.9,
                "probe_score_selected_choice": 0.9,
                "probe_argmax_choice": "A",
            }
        ]
    )


def _make_run_dir(base: Path, *, task_format: str = "multiple_choice", include_probe_rows: bool = True) -> Path:
    run_dir = base / "results" / "sycophancy_bias_probe" / "dummy_model" / "commonsense_qa" / "dummy_run"
    _write_json(
        run_dir / "run_config.json",
        {
            "model": "dummy/model",
            "dataset_name": "commonsense_qa",
            "sampling_only": not include_probe_rows,
        },
    )
    _write_json(
        run_dir / "run_summary.json",
        {
            "model_name": "dummy/model",
            "dataset_name": "commonsense_qa",
            "sampling_only": not include_probe_rows,
        },
    )
    samples = _mc_samples_frame()
    if task_format != "multiple_choice":
        samples["task_format"] = task_format
    _write_csv(run_dir / "sampling" / "sampled_responses.csv", samples)
    probe_scores = _probe_scores_frame() if include_probe_rows else _probe_scores_frame().iloc[0:0].copy()
    _write_csv(run_dir / "probes" / "probe_scores_by_prompt.csv", probe_scores)
    return run_dir


class AnalysisContractTests(unittest.TestCase):
    def test_load_analysis_context_accepts_mc_run_and_creates_output_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp))
            ctx = load_analysis_context(run_dir)
            self.assertEqual(ctx.run_dir, run_dir.resolve())
            self.assertTrue(ctx.analysis_dir.exists())
            self.assertTrue(ctx.plots_dir.exists())
            self.assertTrue(ctx.tables_dir.exists())

    def test_load_analysis_context_rejects_non_mc_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp), task_format="short_answer")
            with self.assertRaises(AnalysisNotSupportedError):
                load_analysis_context(run_dir)

    def test_run_analysis_operation_saves_pdf_and_csv_to_run_subdirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp))
            ctx = load_analysis_context(run_dir)
            frame = run_analysis_operation(ctx, "table_template_overview", output_stem="template_overview")
            fig = run_analysis_operation(ctx, "plot_neutral_option_selection", output_stem="neutral_option_selection")
            self.assertIsInstance(frame, pd.DataFrame)
            self.assertTrue((ctx.tables_dir / "template_overview.csv").exists())
            self.assertTrue((ctx.plots_dir / "neutral_option_selection.pdf").exists())
            fig.clf()

    def test_analysis_function_metadata_and_canonical_dataframe_builder_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp))
            ctx = load_analysis_context(run_dir)
            spec = get_analysis_function_spec("plot_neutral_option_selection")
            self.assertEqual(spec.output_kind, "plot")
            long_df = build_candidate_probability_long_df(ctx)
            self.assertIn("candidate_option", long_df.columns)
            self.assertIn("p_option", long_df.columns)

    def test_build_probe_scores_df_backfills_probe_semantics_for_legacy_probe_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp))
            ctx = load_analysis_context(run_dir)
            probe_df = build_probe_scores_df(ctx)
            self.assertIn("probe_training_template_type", probe_df.columns)
            self.assertIn("probe_evaluated_on_template_type", probe_df.columns)
            self.assertIn("probe_is_neutral_family", probe_df.columns)
            self.assertIn("probe_matches_evaluated_template", probe_df.columns)
            row = probe_df.iloc[0]
            self.assertEqual(row["probe_training_template_type"], "neutral")
            self.assertEqual(row["probe_evaluated_on_template_type"], "neutral")
            self.assertTrue(bool(row["probe_is_neutral_family"]))
            self.assertTrue(bool(row["probe_matches_evaluated_template"]))

    def test_build_paired_probe_df_labels_cross_family_semantics_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp))
            probe_scores = pd.DataFrame(
                [
                    {
                        "probe_name": "probe_no_bias",
                        "question_id": "q1",
                        "prompt_id": "q1__neutral",
                        "template_type": "neutral",
                        "split": "test",
                        "draw_idx": 0,
                        "correct_letter": "A",
                        "incorrect_letter": "B",
                        "selected_choice": "A",
                        "probe_score_correct_choice": 0.9,
                        "probe_score_selected_choice": 0.9,
                        "probe_argmax_choice": "A",
                        "score_A": 0.9,
                        "score_B": 0.1,
                    },
                    {
                        "probe_name": "probe_bias_incorrect_suggestion",
                        "question_id": "q1",
                        "prompt_id": "q1__incorrect_suggestion",
                        "template_type": "incorrect_suggestion",
                        "split": "test",
                        "draw_idx": 0,
                        "correct_letter": "A",
                        "incorrect_letter": "B",
                        "selected_choice": "B",
                        "probe_score_correct_choice": 0.4,
                        "probe_score_selected_choice": 0.8,
                        "probe_argmax_choice": "B",
                        "score_A": 0.4,
                        "score_B": 0.8,
                    },
                ]
            )
            _write_csv(run_dir / "probes" / "probe_scores_by_prompt.csv", probe_scores)
            ctx = load_analysis_context(run_dir)
            paired_df = build_paired_probe_df(ctx)
            self.assertEqual(len(paired_df), 1)
            row = paired_df.iloc[0]
            self.assertFalse(bool(row["same_probe_name_across_conditions"]))
            self.assertFalse(bool(row["same_probe_training_template_across_conditions"]))
            self.assertEqual(row["probe_training_template_type_x"], "neutral")
            self.assertEqual(row["probe_training_template_type_xprime"], "incorrect_suggestion")
            self.assertEqual(row["probe_pairing_semantics"], "neutral_on_x__matched_template_on_xprime")

    def test_safe_run_analysis_operation_records_cell_failures_in_tables_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp))
            ctx = load_analysis_context(run_dir)
            status = safe_run_analysis_operation(
                ctx,
                "does_not_exist",
                cell_id="cell_test_missing",
                output_stem="missing_output",
            )
            self.assertFalse(status["ok"])
            failure_path = ctx.tables_dir / "analysis_cell_failures.csv"
            self.assertTrue(failure_path.exists())
            failure_df = pd.read_csv(failure_path)
            self.assertIn("cell_test_missing", failure_df["cell_id"].tolist())

    def test_safe_generate_analysis_notebook_writes_notebook_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp))
            status = safe_generate_analysis_notebook(run_dir)
            self.assertEqual(status["status"], "completed")
            self.assertTrue((run_dir / "analysis" / "analysis_full_mc_report.ipynb").exists())
            self.assertTrue((run_dir / "analysis" / "analysis_notebook_status.json").exists())
            notebook_text = (run_dir / "analysis" / "analysis_full_mc_report.ipynb").read_text(encoding="utf-8")
            self.assertIn("safe_display_analysis_operation", notebook_text)
            self.assertIn("_ = safe_display_analysis_operation", notebook_text)
            self.assertIn("sys.path.insert", notebook_text)
            self.assertIn("src/llmssycoph", notebook_text)
            self.assertIn("- Run: `dummy_run`", notebook_text)
            self.assertIn("- Model: `dummy/model`", notebook_text)
            self.assertIn("- Dataset: `commonsense_qa`", notebook_text)
            self.assertIn("## Model Overview", notebook_text)
            self.assertIn("## Summary by Bias", notebook_text)
            self.assertIn("## Runtime", notebook_text)
            self.assertIn("## Notebook Guide", notebook_text)
            self.assertIn("## Section 1: External", notebook_text)
            self.assertIn("### Section 1.1: Neutral Signals", notebook_text)
            self.assertIn("## Section 2: Probe Analysis", notebook_text)

    def test_safe_generate_analysis_notebook_omits_probe_section_when_probe_rows_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp), include_probe_rows=False)
            status = safe_generate_analysis_notebook(run_dir)
            self.assertEqual(status["status"], "completed")
            notebook_text = (run_dir / "analysis" / "analysis_full_mc_report.ipynb").read_text(encoding="utf-8")
            self.assertIn("## Probe Analysis Note", notebook_text)
            self.assertIn("sampling_only", notebook_text)
            self.assertNotIn("This section uses probe artifacts", notebook_text)
            self.assertNotIn("plot_probe_layerwise_performance", notebook_text)
            self.assertIn("## Section 1: External", notebook_text)
            self.assertIn("### Section 1.1: Neutral Signals", notebook_text)
            self.assertIn("table_external_summary_statistics", notebook_text)

    def test_safe_generate_analysis_notebook_catches_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(Path(tmp), task_format="short_answer")
            status = safe_generate_analysis_notebook(run_dir)
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["error_type"], "AnalysisNotSupportedError")
            self.assertTrue((run_dir / "analysis" / "analysis_notebook_status.json").exists())
