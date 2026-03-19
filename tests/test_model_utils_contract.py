from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from llmssycoph.model_utils import (
    _resolve_model_inputs,
    _strict_mc_generated_answer_complete,
)


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

    def test_resolve_model_inputs_accepts_batch_encoding_like_objects(self):
        class FakeBatchEncoding(SimpleNamespace):
            def to(self, device):
                return FakeBatchEncoding(
                    input_ids=self.input_ids.to(device),
                    attention_mask=self.attention_mask.to(device),
                )

        class FakeTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return FakeBatchEncoding(
                    input_ids=torch.tensor([[1, 2, 3]]),
                    attention_mask=torch.tensor([[1, 1, 1]]),
                )

        input_ids, attention_mask = _resolve_model_inputs(
            FakeTokenizer(),
            [{"type": "human", "content": "Question"}],
            torch.device("cpu"),
        )
        self.assertTrue(torch.equal(input_ids, torch.tensor([[1, 2, 3]])))
        self.assertTrue(torch.equal(attention_mask, torch.tensor([[1, 1, 1]])))


if __name__ == "__main__":
    unittest.main()
