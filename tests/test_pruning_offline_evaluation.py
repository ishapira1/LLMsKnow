from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from llmssycoph.pruning.global_selection import select_global_configuration
from llmssycoph.pruning.offline_evaluation import (
    GLOBAL_SELECTION_COLUMNS,
    aggregate_offline_evaluation,
    main,
    pair_item_tables,
)


STRONG = "incorrect_suggestion_strong"
WEAK = "incorrect_suggestion"
AGREEMENT = "suggest_correct"


def _tables(*, all_recover: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_rows = []
    candidate_rows = []
    outcomes = [
        ("A", "valid"),
        ("C", "valid"),
        ("", "refusal"),
        ("", "malformed"),
        ("B", "valid"),
        ("Z", "invalid"),
    ]
    if all_recover:
        outcomes = [("A", "valid")] * len(outcomes)

    for index, (strong_choice, strong_status) in enumerate(outcomes, start=1):
        question_id = f"q{index}"
        common = {
            "dataset": "arc_challenge",
            "split": "val",
            "question_id": question_id,
            "correct_letter": "A",
            "choice_letters": ["A", "B", "C"],
            "neutral_choice": "A",
            "neutral_status": "valid",
            "p_neutral_c": 0.8,
            "p_neutral_b": 0.1,
        }
        family_specs = {
            STRONG: {
                "suggested_letter": "B",
                "baseline_choice": "B",
                "baseline_status": "valid",
                "baseline_p_c": 0.2,
                "baseline_p_b": 0.7,
                "candidate_choice": strong_choice,
                "candidate_status": strong_status,
                "candidate_p_c": 0.5,
                "candidate_p_b": 0.2,
            },
            WEAK: {
                "suggested_letter": "B",
                "baseline_choice": "A",
                "baseline_status": "valid",
                "baseline_p_c": 0.5,
                "baseline_p_b": 0.4,
                "candidate_choice": "A",
                "candidate_status": "valid",
                "candidate_p_c": 0.6,
                "candidate_p_b": 0.3,
            },
            AGREEMENT: {
                "suggested_letter": "A",
                "baseline_choice": "A",
                "baseline_status": "valid",
                "baseline_p_c": 0.9,
                "baseline_p_b": 0.05,
                "candidate_choice": "A",
                "candidate_status": "valid",
                "candidate_p_c": 0.9,
                "candidate_p_b": 0.05,
            },
        }
        for condition, spec in family_specs.items():
            identity = {
                **common,
                "condition": condition,
                "suggested_letter": spec["suggested_letter"],
            }
            baseline_rows.append(
                {
                    **identity,
                    "biased_choice": spec["baseline_choice"],
                    "biased_status": spec["baseline_status"],
                    "p_biased_c": spec["baseline_p_c"],
                    "p_biased_b": spec["baseline_p_b"],
                    "preservation_loss": 1.0,
                    "wikitext_perplexity": 10.0,
                }
            )
            candidate_rows.append(
                {
                    **identity,
                    "biased_choice": spec["candidate_choice"],
                    "biased_status": spec["candidate_status"],
                    "p_biased_c": spec["candidate_p_c"],
                    "p_biased_b": spec["candidate_p_b"],
                    "preservation_loss": 1.02,
                    "wikitext_perplexity": 10.2,
                    "p": 1e-5,
                    "q": 3e-6,
                    "calibration_seed": 5,
                    "actual_mask_count": 120,
                }
            )
    return pd.DataFrame(baseline_rows), pd.DataFrame(candidate_rows)


class PruningOfflineEvaluationTests(unittest.TestCase):
    def test_analytic_probability_and_transition_metrics(self):
        baseline, candidate = _tables()
        paired = pair_item_tables(baseline, candidate, actual_mask_count=120)
        result = aggregate_offline_evaluation(paired, n_bootstrap=200, bootstrap_seed=17)

        strong = result.family_summary[result.family_summary["family"].eq(STRONG)].iloc[0]
        weak = result.family_summary[result.family_summary["family"].eq(WEAK)].iloc[0]
        self.assertAlmostEqual(float(strong["p_b_decrease"]), 0.5)
        self.assertAlmostEqual(float(strong["p_c_recovery"]), 0.3)
        self.assertAlmostEqual(float(strong["c_minus_b_margin_recovery"]), 0.8)
        self.assertAlmostEqual(float(weak["p_b_decrease"]), 0.1)
        self.assertAlmostEqual(float(weak["p_c_recovery"]), 0.1)
        self.assertAlmostEqual(float(weak["c_minus_b_margin_recovery"]), 0.2)

        self.assertEqual(int(strong["n_baseline_strict_flips"]), 6)
        self.assertAlmostEqual(float(strong["b_to_c_recovery_rate"]), 1 / 6)
        self.assertAlmostEqual(float(strong["b_to_other_wrong_rate"]), 1 / 6)
        self.assertAlmostEqual(float(strong["b_to_invalid_rate"]), 1 / 6)
        self.assertAlmostEqual(float(strong["b_to_refusal_rate"]), 1 / 6)
        self.assertAlmostEqual(float(strong["b_to_malformed_rate"]), 1 / 6)
        self.assertAlmostEqual(float(strong["remains_suggested_rate"]), 1 / 6)

        metric = result.metric_summary.set_index("metric")
        self.assertAlmostEqual(float(metric.loc["strong_p_b_decrease", "estimate"]), 0.5)
        self.assertLessEqual(
            float(metric.loc["strong_p_b_decrease", "ci_low"]),
            float(metric.loc["strong_p_b_decrease", "estimate"]),
        )
        self.assertGreaterEqual(
            float(metric.loc["strong_p_b_decrease", "ci_high"]),
            float(metric.loc["strong_p_b_decrease", "estimate"]),
        )
        self.assertAlmostEqual(float(metric.loc["neutral_accuracy_change", "estimate"]), 0.0)
        self.assertAlmostEqual(
            float(metric.loc["correct_suggestion_agreement_change", "estimate"]), 0.0
        )

    def test_every_candidate_configuration_must_use_identical_held_out_items(self):
        baseline, candidate = _tables()
        candidate = candidate.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "fixed baseline held-out set"):
            pair_item_tables(baseline, candidate, actual_mask_count=120)

    def test_multiple_draws_are_validated_per_draw_and_clustered_by_question(self):
        baseline_zero, candidate_zero = _tables()
        baseline_zero["draw_idx"] = 0
        candidate_zero["draw_idx"] = 0
        baseline_one = baseline_zero.copy()
        candidate_one = candidate_zero.copy()
        baseline_one["draw_idx"] = 1
        candidate_one["draw_idx"] = 1
        candidate_one["neutral_choice"] = "B"
        candidate_one["p_neutral_c"] = 0.4
        baseline = pd.concat([baseline_zero, baseline_one], ignore_index=True)
        candidate = pd.concat([candidate_zero, candidate_one], ignore_index=True)

        result = aggregate_offline_evaluation(
            pair_item_tables(baseline, candidate, actual_mask_count=120),
            n_bootstrap=0,
        )
        self.assertEqual(len(result.paired_items), 2 * len(candidate_zero))
        strong = result.family_summary[result.family_summary["family"].eq(STRONG)].iloc[0]
        self.assertEqual(int(strong["n_questions"]), 6)
        selection = result.selection_summary[
            result.selection_summary["actual_mask_count"].eq(120)
        ].iloc[0]
        self.assertAlmostEqual(float(selection["neutral_accuracy"]), 0.5)
        neutral_metric = result.metric_summary.set_index("metric").loc[
            "neutral_accuracy_candidate", "estimate"
        ]
        self.assertAlmostEqual(float(neutral_metric), 0.5)

    def test_selection_summary_is_directly_accepted_by_global_selector(self):
        baseline, candidate = _tables(all_recover=True)
        result = aggregate_offline_evaluation(
            pair_item_tables(baseline, candidate, actual_mask_count=120),
            n_bootstrap=0,
        )
        self.assertTrue(set(GLOBAL_SELECTION_COLUMNS).issubset(result.selection_summary.columns))
        selection, audit = select_global_configuration(result.selection_summary)
        self.assertEqual(selection.status, "selected")
        self.assertEqual(selection.selected_p, 1e-5)
        self.assertEqual(selection.selected_q, 3e-6)
        self.assertEqual(selection.actual_mask_count, 120)
        self.assertTrue(bool(audit.iloc[0]["feasible"]))

    def test_cli_reads_csv_and_jsonl_and_writes_all_outputs(self):
        baseline, candidate = _tables(all_recover=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.csv"
            candidate_path = root / "candidate.jsonl"
            baseline_evaluation_path = root / "baseline_evaluation.json"
            candidate_evaluation_path = root / "candidate_evaluation.json"
            output_dir = root / "out"
            baseline.to_csv(baseline_path, index=False)
            candidate.to_json(candidate_path, orient="records", lines=True)
            baseline_evaluation_path.write_text(
                json.dumps(
                    {
                        "preservation_loss": float(baseline.iloc[0]["preservation_loss"]),
                        "wikitext_perplexity": float(
                            baseline.iloc[0]["wikitext_perplexity"]
                        ),
                    }
                ),
                encoding="utf-8",
            )
            candidate_evaluation_path.write_text(
                json.dumps(
                    {
                        "preservation_loss": float(candidate.iloc[0]["preservation_loss"]),
                        "wikitext_perplexity": float(
                            candidate.iloc[0]["wikitext_perplexity"]
                        ),
                    }
                ),
                encoding="utf-8",
            )
            status = main(
                [
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(candidate_path),
                    "--output-dir",
                    str(output_dir),
                    "--baseline-evaluation-artifact",
                    str(baseline_evaluation_path),
                    "--candidate-evaluation-artifact",
                    str(candidate_evaluation_path),
                    "--n-bootstrap",
                    "20",
                ]
            )
            self.assertEqual(status, 0)
            for filename in (
                "paired_items.csv",
                "family_summary.csv",
                "metric_summary.csv",
                "selection_summary.csv",
                "offline_evaluation_manifest.json",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)
            manifest = json.loads(
                (output_dir / "offline_evaluation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["n_paired_rows"], len(candidate))
            self.assertEqual(manifest["n_configurations"], 1)
            self.assertEqual(
                manifest["metadata"]["baseline_sha256"],
                hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["metadata"]["candidate_sha256"],
                hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            )
            for name, path in manifest["outputs"].items():
                self.assertEqual(
                    manifest["output_sha256"][name],
                    hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                )
            self.assertEqual(
                manifest["metadata"]["guardrail_sources"]["baseline"][
                    "original_sha256_at_capture"
                ],
                hashlib.sha256(baseline_evaluation_path.read_bytes()).hexdigest(),
            )
            snapshot = Path(
                manifest["metadata"]["guardrail_sources"]["baseline"]["path"]
            )
            self.assertTrue(snapshot.is_file())
            self.assertEqual(
                manifest["metadata"]["guardrail_sources"]["baseline"]["sha256"],
                hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
