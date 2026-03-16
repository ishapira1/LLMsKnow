from __future__ import annotations

import unittest

from sycophancy_bias_probe.correctness import (
    extract_short_answer_from_generation,
    grade_short_answer,
    record_is_usable_for_metrics,
)


class AnswerUtilsContractTests(unittest.TestCase):
    def test_extract_short_answer_trims_common_wrappers(self):
        self.assertEqual(
            extract_short_answer_from_generation("Final answer: Paris, the capital of France."),
            "Paris",
        )
        self.assertEqual(
            extract_short_answer_from_generation("It is Paris"),
            "Paris",
        )

    def test_grade_short_answer_returns_tri_state(self):
        correct = grade_short_answer("Final answer: Paris", ["Paris"])
        self.assertEqual(correct["status"], "correct")
        self.assertEqual(correct["correctness"], 1)
        self.assertTrue(correct["usable_for_metrics"])

        incorrect = grade_short_answer("Final answer: Lyon", ["Paris"])
        self.assertEqual(incorrect["status"], "incorrect")
        self.assertEqual(incorrect["correctness"], 0)
        self.assertTrue(incorrect["usable_for_metrics"])

        ambiguous = grade_short_answer("Final answer: Paris or Lyon", ["Paris"])
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertIsNone(ambiguous["correctness"])
        self.assertFalse(ambiguous["usable_for_metrics"])

    def test_record_is_usable_for_metrics_handles_missing_flag(self):
        self.assertTrue(record_is_usable_for_metrics({"correctness": 1}))
        self.assertTrue(record_is_usable_for_metrics({"correctness": 0, "usable_for_metrics": True}))
        self.assertFalse(record_is_usable_for_metrics({"correctness": None, "usable_for_metrics": False}))


if __name__ == "__main__":
    unittest.main()
