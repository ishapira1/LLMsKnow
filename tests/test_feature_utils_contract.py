from __future__ import annotations

from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from sycophancy_bias_probe.feature_utils import (
    _assistant_text_last_token_index,
    get_hidden_feature_for_completion,
    score_logprob_answer,
)
from sycophancy_bias_probe.probes import get_hidden_feature_all_layers_for_completion


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        if text == "Paris":
            return SimpleNamespace(input_ids=[7])
        raise AssertionError(f"Unexpected tokenization request: {text!r}")


class FakeTensor:
    def __init__(self, array):
        self.array = np.array(array)

    def to(self, device):
        return self

    def __getitem__(self, key):
        return FakeTensor(self.array[key])

    @property
    def shape(self):
        return self.array.shape

    def tolist(self):
        return self.array.tolist()

    def detach(self):
        return self

    def float(self):
        return FakeTensor(self.array.astype(float))

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.array)

    def item(self):
        return float(np.array(self.array).item())


class FakeEncoding:
    def __init__(self, ids):
        self.ids = list(ids)


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorchModule:
    @staticmethod
    def no_grad():
        return FakeNoGrad()

    @staticmethod
    def tensor(data, device=None):
        return FakeTensor(data)

    @staticmethod
    def log_softmax(tensor, dim=-1):
        arr = np.array(tensor.array, dtype=float)
        shifted = arr - np.max(arr, axis=dim, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
        return FakeTensor(log_probs)


class FakeHiddenStateModel:
    device = "cpu"

    def __call__(self, input_tensor, use_cache=False, output_hidden_states=True, return_dict=True):
        seq_len = input_tensor.shape[1]
        hidden_states = []
        for layer in range(3):
            hs = np.zeros((1, seq_len, 2), dtype=np.float32)
            for idx in range(seq_len):
                hs[0, idx] = np.array([float(layer), float(idx)], dtype=np.float32)
            hidden_states.append(FakeTensor(hs))
        return SimpleNamespace(hidden_states=tuple(hidden_states))


class FakeLogitModel:
    device = "cpu"

    def __call__(self, input_tensor, use_cache=False, output_hidden_states=False, return_dict=True):
        seq_len = input_tensor.shape[1]
        logits = np.zeros((1, seq_len, 16), dtype=np.float32)
        logits[0, 0, 7] = -5.0
        logits[0, 2, 7] = 5.0
        return SimpleNamespace(logits=FakeTensor(logits))


class FeatureUtilsContractTests(unittest.TestCase):
    def test_assistant_text_last_token_index_prefers_last_occurrence(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(_assistant_text_last_token_index(tokenizer, [11, 7, 22, 7], "Paris"), 3)

    def test_hidden_feature_helpers_use_assistant_completion_occurrence(self):
        encoded = FakeTensor([[11, 7, 22, 7]])
        tokenizer = FakeTokenizer()
        model = FakeHiddenStateModel()

        with patch.dict(sys.modules, {"torch": FakeTorchModule()}):
            with patch("sycophancy_bias_probe.feature_utils.encode_chat", return_value=encoded):
                vec = get_hidden_feature_for_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{"type": "human", "content": "I think the answer is Paris"}],
                    completion="Paris",
                    layer=1,
                )

            with patch("sycophancy_bias_probe.probes._encode_chat", return_value=encoded):
                mat = get_hidden_feature_all_layers_for_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{"type": "human", "content": "I think the answer is Paris"}],
                    completion="Paris",
                    layer_grid=[1, 2],
                )

        np.testing.assert_array_equal(vec, np.array([1.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(
            mat,
            np.array([[1.0, 3.0], [2.0, 3.0]], dtype=np.float32),
        )

    def test_hidden_feature_helpers_accept_tokenizers_encoding_style_objects(self):
        encoded = FakeEncoding([11, 7, 22, 7])
        tokenizer = FakeTokenizer()
        model = FakeHiddenStateModel()

        with patch.dict(sys.modules, {"torch": FakeTorchModule()}):
            with patch("sycophancy_bias_probe.feature_utils.encode_chat", return_value=encoded):
                vec = get_hidden_feature_for_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{"type": "human", "content": "I think the answer is Paris"}],
                    completion="Paris",
                    layer=1,
                )

            with patch("sycophancy_bias_probe.probes._encode_chat", return_value=encoded):
                mat = get_hidden_feature_all_layers_for_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{"type": "human", "content": "I think the answer is Paris"}],
                    completion="Paris",
                    layer_grid=[1, 2],
                )

        np.testing.assert_array_equal(vec, np.array([1.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(
            mat,
            np.array([[1.0, 3.0], [2.0, 3.0]], dtype=np.float32),
        )

    def test_score_logprob_answer_uses_assistant_completion_occurrence(self):
        encoded = FakeTensor([[11, 7, 22, 7]])
        tokenizer = FakeTokenizer()
        model = FakeLogitModel()

        with patch.dict(sys.modules, {"torch": FakeTorchModule()}):
            with patch("sycophancy_bias_probe.feature_utils.encode_chat", return_value=encoded):
                total_logp, mean_logp = score_logprob_answer(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{"type": "human", "content": "I think the answer is Paris"}],
                    answer="Paris",
                )

        self.assertGreater(total_logp, -1.0)
        self.assertGreater(mean_logp, -1.0)

    def test_score_logprob_answer_accepts_tokenizers_encoding_style_objects(self):
        encoded = FakeEncoding([11, 7, 22, 7])
        tokenizer = FakeTokenizer()
        model = FakeLogitModel()

        with patch.dict(sys.modules, {"torch": FakeTorchModule()}):
            with patch("sycophancy_bias_probe.feature_utils.encode_chat", return_value=encoded):
                total_logp, mean_logp = score_logprob_answer(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{"type": "human", "content": "I think the answer is Paris"}],
                    answer="Paris",
                )

        self.assertGreater(total_logp, -1.0)
        self.assertGreater(mean_logp, -1.0)


if __name__ == "__main__":
    unittest.main()
