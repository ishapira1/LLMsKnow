from __future__ import annotations

import unittest

from llmssycoph.sampling_integrity import build_sampling_integrity_summary


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
        self.assertTrue(
            any("selected_choice_not_argmax" in key for key in choice_summary["reason_counts"])
        )


if __name__ == "__main__":
    unittest.main()
