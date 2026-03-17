from __future__ import annotations

import unittest

from sycophancy_bias_probe.model_utils import _strict_mc_generated_answer_complete


class ModelUtilsContractTests(unittest.TestCase):
    def test_strict_mc_generated_answer_complete_matches_short_answer_only(self):
        self.assertTrue(_strict_mc_generated_answer_complete("Answer: B", "ABCDE"))
        self.assertTrue(_strict_mc_generated_answer_complete(" Answer: (D)\n", "ABCDE"))
        self.assertTrue(_strict_mc_generated_answer_complete("D", "ABCDE"))
        self.assertTrue(_strict_mc_generated_answer_complete("(C)", "ABCDE"))
        self.assertTrue(_strict_mc_generated_answer_complete("Answer:B", "ABCDE"))

        self.assertFalse(_strict_mc_generated_answer_complete("Because the ratio is 3:2", "ABCDE"))
        self.assertFalse(_strict_mc_generated_answer_complete("Answer: B because", "ABCDE"))
        self.assertFalse(_strict_mc_generated_answer_complete("Answer: None", "ABCDE"))
        self.assertFalse(_strict_mc_generated_answer_complete("Let's think step by step.", "ABCDE"))
        self.assertFalse(_strict_mc_generated_answer_complete("", "ABCDE"))


if __name__ == "__main__":
    unittest.main()
