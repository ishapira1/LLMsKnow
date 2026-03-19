from __future__ import annotations

from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from llmssycoph.llm import score_choices, score_logprob_answer
from llmssycoph.probes import get_hidden_feature_for_completion
from llmssycoph.probes.features import _assistant_text_last_token_index
from llmssycoph.probes import get_hidden_feature_all_layers_for_completion


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        if text == 'Paris':
            return SimpleNamespace(input_ids=[7])
        raise AssertionError(f'Unexpected tokenization request: {text!r}')


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

    @staticmethod
    def softmax(tensor, dim=-1):
        arr = np.array(tensor.array, dtype=float)
        shifted = arr - np.max(arr, axis=dim, keepdims=True)
        probs = np.exp(shifted)
        probs = probs / np.sum(probs, axis=dim, keepdims=True)
        return FakeTensor(probs)


class FakeHiddenStateModel:
    device = 'cpu'

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
    device = 'cpu'

    def __call__(self, input_tensor, use_cache=False, output_hidden_states=False, return_dict=True):
        seq_len = input_tensor.shape[1]
        logits = np.zeros((1, seq_len, 16), dtype=np.float32)
        logits[0, 0, 7] = -5.0
        logits[0, 2, 7] = 5.0
        return SimpleNamespace(logits=FakeTensor(logits))


class FakeChoiceTokenizer:
    def __call__(self, text, add_special_tokens=False):
        token_map = {
            "A": [1],
            " A": [4],
            " B": [2],
            "B": [3],
        }
        if text not in token_map:
            raise AssertionError(f"Unexpected tokenization request: {text!r}")
        return SimpleNamespace(input_ids=token_map[text])


class FakeChoiceLogitModel:
    device = 'cpu'

    def __call__(self, input_ids=None, attention_mask=None, use_cache=False, output_hidden_states=False, return_dict=True):
        seq_len = input_ids.shape[1]
        logits = np.zeros((1, seq_len, 8), dtype=np.float32)
        logits[0, -1, 1] = 0.0
        logits[0, -1, 4] = -1.0
        logits[0, -1, 2] = 2.0
        logits[0, -1, 3] = -2.0
        return SimpleNamespace(logits=FakeTensor(logits))


class FeatureUtilsContractTests(unittest.TestCase):
    def test_assistant_text_last_token_index_prefers_last_occurrence(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(_assistant_text_last_token_index(tokenizer, [11, 7, 22, 7], 'Paris'), 3)

    def test_hidden_feature_helpers_use_assistant_completion_occurrence(self):
        encoded = FakeTensor([[11, 7, 22, 7]])
        tokenizer = FakeTokenizer()
        model = FakeHiddenStateModel()

        with patch.dict(sys.modules, {'torch': FakeTorchModule()}):
            with patch('llmssycoph.probes.features.encode_chat', return_value=encoded):
                vec = get_hidden_feature_for_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{'type': 'human', 'content': 'I think the answer is Paris'}],
                    completion='Paris',
                    layer=1,
                )
                mat = get_hidden_feature_all_layers_for_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{'type': 'human', 'content': 'I think the answer is Paris'}],
                    completion='Paris',
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

        with patch.dict(sys.modules, {'torch': FakeTorchModule()}):
            with patch('llmssycoph.probes.features.encode_chat', return_value=encoded):
                vec = get_hidden_feature_for_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{'type': 'human', 'content': 'I think the answer is Paris'}],
                    completion='Paris',
                    layer=1,
                )
                mat = get_hidden_feature_all_layers_for_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{'type': 'human', 'content': 'I think the answer is Paris'}],
                    completion='Paris',
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

        with patch.dict(sys.modules, {'torch': FakeTorchModule()}):
            with patch('llmssycoph.llm.scoring.encode_chat', return_value=encoded):
                total_logp, mean_logp = score_logprob_answer(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{'type': 'human', 'content': 'I think the answer is Paris'}],
                    answer='Paris',
                )

        self.assertGreater(total_logp, -1.0)
        self.assertGreater(mean_logp, -1.0)

    def test_score_logprob_answer_accepts_tokenizers_encoding_style_objects(self):
        encoded = FakeEncoding([11, 7, 22, 7])
        tokenizer = FakeTokenizer()
        model = FakeLogitModel()

        with patch.dict(sys.modules, {'torch': FakeTorchModule()}):
            with patch('llmssycoph.llm.scoring.encode_chat', return_value=encoded):
                total_logp, mean_logp = score_logprob_answer(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{'type': 'human', 'content': 'I think the answer is Paris'}],
                    answer='Paris',
                )

        self.assertGreater(total_logp, -1.0)
        self.assertGreater(mean_logp, -1.0)

    def test_score_choices_normalizes_choice_probabilities(self):
        encoded = FakeTensor([[11, 22, 33]])
        tokenizer = FakeChoiceTokenizer()
        model = FakeChoiceLogitModel()

        with patch.dict(sys.modules, {'torch': FakeTorchModule()}):
            with patch('llmssycoph.llm.scoring._resolve_model_inputs', return_value=(encoded, None)):
                probs = score_choices(
                    model=model,
                    tokenizer=tokenizer,
                    messages=[{'type': 'human', 'content': 'Question\n\nAnswer:'}],
                    choices=['A', 'B'],
                )

        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertGreater(probs['B'], probs['A'])


if __name__ == '__main__':
    unittest.main()
