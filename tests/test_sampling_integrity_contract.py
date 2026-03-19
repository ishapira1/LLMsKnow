from __future__ import annotations

import unittest
from unittest.mock import patch

from llmssycoph.sampling_integrity import build_sampling_integrity_summary, log_sampling_integrity_summary


class SamplingIntegrityContractTests(unittest.TestCase):
    def test_generation_summary_distinguishes_exact_minor_and_failure(self):
        records = [
            {
                "sampling_mode": "generation",
                "template_type": "neutral",
                "usable_for_metrics": True,
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "strict_format_exact": True,
                "grading_reason": "single_letter_match",
                "correctness": 1,
            },
            {
                "sampling_mode": "generation",
                "template_type": "neutral",
                "usable_for_metrics": True,
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "strict_format_exact": False,
                "grading_reason": "single_letter_non_match",
                "correctness": 0,
            },
            {
                "sampling_mode": "generation",
                "template_type": "incorrect_suggestion",
                "usable_for_metrics": False,
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "strict_format_exact": False,
                "grading_reason": "no_committed_answer",
                "correctness": None,
            },
        ]

        summary = build_sampling_integrity_summary(records)
        generation = summary["by_sampling_mode"]["generation"]

        self.assertEqual(generation["total"], 3)
        self.assertEqual(generation["buckets"]["exact_compliance"]["count"], 1)
        self.assertEqual(generation["buckets"]["minor_format_deviation_still_scoreable"]["count"], 1)
        self.assertEqual(generation["buckets"]["format_failure"]["count"], 1)
        self.assertIn("Exact compliance: 33.33%", generation["human_summary"])

    def test_choice_probability_summary_validates_distribution_consistency(self):
        records = [
            {
                "sampling_mode": "choice_probabilities",
                "template_type": "neutral",
                "letters": "ABCD",
                "correct_letter": "C",
                "response_raw": "C",
                "choice_probabilities": {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.1},
                "choice_probability_correct": 0.6,
                "choice_probability_selected": 0.6,
                "completion_token_count": 1,
                "hit_max_new_tokens": False,
                "stopped_on_eos": False,
                "finish_reason": "choice_probabilities",
                "usable_for_metrics": True,
                "correctness": 1,
            },
            {
                "sampling_mode": "choice_probabilities",
                "template_type": "incorrect_suggestion",
                "letters": "ABCD",
                "correct_letter": "C",
                "response_raw": "A",
                "choice_probabilities": {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.1},
                "choice_probability_correct": 0.1,
                "choice_probability_selected": 0.1,
                "completion_token_count": 2,
                "hit_max_new_tokens": False,
                "stopped_on_eos": False,
                "finish_reason": "choice_probabilities",
                "usable_for_metrics": True,
                "correctness": 0,
            },
        ]

        summary = build_sampling_integrity_summary(records)
        choice_summary = summary["by_sampling_mode"]["choice_probabilities"]

        self.assertEqual(choice_summary["total"], 2)
        self.assertEqual(choice_summary["buckets"]["exact_compliance"]["count"], 1)
        self.assertEqual(choice_summary["buckets"]["integrity_failure"]["count"], 1)
        self.assertEqual(choice_summary["selected_choice_counts"], {"A": 1, "C": 1})
        self.assertTrue(
            any("selected_choice_not_argmax" in key for key in choice_summary["reason_counts"])
        )

    def test_choice_probability_logging_warns_when_all_rows_select_same_choice(self):
        records = [
            {
                "sampling_mode": "choice_probabilities",
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "letters": "ABCD",
                "correct_letter": "B",
                "response_raw": "A",
                "choice_probabilities": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                "choice_probability_correct": 0.1,
                "choice_probability_selected": 0.7,
                "completion_token_count": 1,
                "hit_max_new_tokens": False,
                "stopped_on_eos": False,
                "finish_reason": "choice_probabilities",
                "usable_for_metrics": True,
                "correctness": 0,
            },
            {
                "sampling_mode": "choice_probabilities",
                "template_type": "incorrect_suggestion",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "letters": "ABCD",
                "correct_letter": "C",
                "response_raw": "A",
                "choice_probabilities": {"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05},
                "choice_probability_correct": 0.05,
                "choice_probability_selected": 0.8,
                "completion_token_count": 1,
                "hit_max_new_tokens": False,
                "stopped_on_eos": False,
                "finish_reason": "choice_probabilities",
                "usable_for_metrics": True,
                "correctness": 0,
            },
        ]

        summary = build_sampling_integrity_summary(records)
        choice_summary = summary["by_sampling_mode"]["choice_probabilities"]

        self.assertEqual(choice_summary["selected_choice_counts"], {"A": 2})

        with patch("llmssycoph.sampling_integrity.warn_status") as mock_warn:
            log_sampling_integrity_summary(summary)

        mock_warn.assert_called_once_with(
            "sampling_integrity.py",
            "choice_probability_single_selected_choice",
            "all 2/2 choice-probability rows selected the same highest-probability option (A). "
            "Check answer ordering, prompt construction, and whether the model collapsed to one label.",
        )

    def test_clean_sampling_integrity_summary_uses_ok_status(self):
        records = [
            {
                "sampling_mode": "choice_probabilities",
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "letters": "ABCD",
                "correct_letter": "B",
                "response_raw": "A",
                "choice_probabilities": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                "choice_probability_correct": 0.1,
                "choice_probability_selected": 0.7,
                "completion_token_count": 1,
                "hit_max_new_tokens": False,
                "stopped_on_eos": False,
                "finish_reason": "choice_probabilities",
                "usable_for_metrics": True,
                "correctness": 0,
            }
        ]

        summary = build_sampling_integrity_summary(records)

        with patch("llmssycoph.sampling_integrity.ok_status") as mock_ok, patch(
            "llmssycoph.sampling_integrity.log_status"
        ) as mock_log:
            log_sampling_integrity_summary(summary)

        mock_ok.assert_any_call(
            "sampling_integrity.py",
            "sampling integrity mode=choice_probabilities: Exact compliance: 100.00% | Integrity failure: 0.00%",
        )
        mock_ok.assert_any_call(
            "sampling_integrity.py",
            "sampling integrity mode=choice_probabilities template=neutral: Exact compliance: 100.00% | Integrity failure: 0.00%",
        )
        mock_log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
