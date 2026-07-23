from __future__ import annotations

import unittest

import pandas as pd

from llmssycoph.pruning.global_selection import (
    P_GRID,
    Q_GRID,
    select_global_configuration,
    transition_counts,
)


def _row(p: float, q: float, **overrides):
    row = {
        "p": p,
        "q": q,
        "split": "val",
        "calibration_seed": 5,
        "actual_mask_count": 0 if q == 0 else 100,
        "wrong_probability_uplift": 0.40,
        "biased_correct_probability": 0.20,
        "neutral_accuracy": 0.90,
        "neutral_correct_probability": 0.80,
        "correction_accuracy": 0.90,
        "agreement_accuracy": 0.90,
        "preservation_loss": 1.0,
        "wikitext_perplexity": 10.0,
        "other_wrong_invalid_rate": 0.01,
        "b_to_c_recovery_rate": 0.0,
    }
    row.update(overrides)
    return row


class GlobalPruningSelectionTests(unittest.TestCase):
    def test_grid_contains_paper_anchor_and_larger_q(self):
        self.assertIn(1e-5, P_GRID)
        self.assertIn(5e-5, Q_GRID)
        self.assertIn(1e-4, Q_GRID)

    def test_selects_highest_recovery_then_smallest_mask(self):
        summary = pd.DataFrame(
            [
                _row(0.0, 0.0),
                _row(
                    1e-5,
                    5e-5,
                    actual_mask_count=200,
                    wrong_probability_uplift=0.20,
                    biased_correct_probability=0.30,
                    b_to_c_recovery_rate=0.60,
                ),
                _row(
                    5e-5,
                    5e-5,
                    actual_mask_count=150,
                    wrong_probability_uplift=0.20,
                    biased_correct_probability=0.30,
                    b_to_c_recovery_rate=0.60,
                ),
            ]
        )
        result, audit = select_global_configuration(summary)
        self.assertEqual(result.status, "selected")
        self.assertEqual(result.selected_p, 5e-5)
        self.assertEqual(result.actual_mask_count, 150)
        self.assertTrue(audit["feasible"].all())

    def test_no_feasible_mask_is_explicit(self):
        summary = pd.DataFrame(
            [
                _row(0.0, 0.0),
                _row(
                    1e-5,
                    1e-4,
                    wrong_probability_uplift=0.39,
                    biased_correct_probability=0.205,
                    neutral_accuracy=0.70,
                ),
            ]
        )
        result, audit = select_global_configuration(summary)
        self.assertEqual(result.status, "no_feasible_mask")
        self.assertIsNone(result.selected_p)
        self.assertFalse(bool(audit.iloc[0]["feasible"]))

    def test_transition_counts_distinguish_recovery_other_wrong_and_invalid(self):
        rows = [
            {
                "correct_letter": "A",
                "suggested_letter": "B",
                "choice_letters": ["A", "B", "C"],
                "baseline_neutral_choice": "A",
                "baseline_biased_choice": "B",
                "candidate_biased_choice": "A",
            },
            {
                "correct_letter": "A",
                "suggested_letter": "B",
                "choice_letters": ["A", "B", "C"],
                "baseline_neutral_choice": "A",
                "baseline_biased_choice": "B",
                "candidate_biased_choice": "C",
            },
            {
                "correct_letter": "A",
                "suggested_letter": "B",
                "choice_letters": ["A", "B", "C"],
                "baseline_neutral_choice": "A",
                "baseline_biased_choice": "B",
                "candidate_biased_choice": "",
            },
            {
                "correct_letter": "A",
                "suggested_letter": "B",
                "choice_letters": ["A", "B", "C"],
                "baseline_neutral_choice": "A",
                "baseline_biased_choice": "B",
                "candidate_biased_choice": "B",
            },
            {
                "correct_letter": "A",
                "suggested_letter": "B",
                "choice_letters": ["A", "B", "C"],
                "baseline_neutral_choice": "C",
                "baseline_biased_choice": "B",
                "candidate_biased_choice": "A",
            },
        ]
        summary = transition_counts(rows)
        self.assertEqual(summary["n_baseline_strict_flips"], 4.0)
        self.assertEqual(summary["b_to_c_recovery_rate"], 0.25)
        self.assertEqual(summary["b_to_other_wrong_rate"], 0.25)
        self.assertEqual(summary["b_to_invalid_rate"], 0.25)
        self.assertEqual(summary["remains_suggested_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
