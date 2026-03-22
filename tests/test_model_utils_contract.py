from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llmssycoph.llm import (
    BaseLLM,
    GenerationResult,
    get_registered_llm_factory,
    get_registered_llm_capabilities,
    load_llm,
    register_llm,
    registered_llm_names,
    resolve_llm_backend,
    unregister_llm,
)
from llmssycoph.llm.huggingface import _device_uses_gpu, _warn_if_not_using_gpu
from llmssycoph.llm.generation import (
    _resolve_model_inputs,
    _strict_mc_generated_answer_complete,
    encode_chat,
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

    def test_encode_chat_requires_apply_chat_template(self):
        class TokenizerWithoutChatTemplate:
            name_or_path = "no-chat-template-tokenizer"

        with self.assertRaisesRegex(TypeError, "apply_chat_template"):
            encode_chat(
                TokenizerWithoutChatTemplate(),
                [{"type": "human", "content": "Question"}],
            )

    def test_encode_chat_uses_model_native_chat_template_with_generation_prompt(self):
        class RecordingTokenizer:
            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
                self.calls.append(
                    {
                        "messages": messages,
                        "tokenize": tokenize,
                        "add_generation_prompt": add_generation_prompt,
                        "return_tensors": return_tensors,
                    }
                )
                return "encoded"

        tokenizer = RecordingTokenizer()
        encoded = encode_chat(
            tokenizer,
            [{"type": "human", "content": "Which option is correct?\n\nAnswer:"}],
            add_generation_prompt=True,
        )

        self.assertEqual(encoded, "encoded")
        self.assertEqual(
            tokenizer.calls,
            [
                {
                    "messages": [{"role": "user", "content": "Which option is correct?\n\nAnswer:"}],
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "return_tensors": "pt",
                }
            ],
        )

    def test_load_llm_falls_back_to_huggingface_for_unregistered_name(self):
        with patch("llmssycoph.llm.registry.HuggingFaceLLM", autospec=True) as mock_hf_llm:
            instance = mock_hf_llm.return_value
            llm = load_llm(
                "mistralai/Mistral-7B-Instruct-v0.2",
                device="cpu",
                device_map_auto=False,
                hf_cache_dir=None,
            )

        self.assertIs(llm, instance)
        mock_hf_llm.assert_called_once_with(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device="cpu",
            device_map_auto=False,
            hf_cache_dir=None,
        )

    def test_gpu_device_detection_accepts_cuda_and_mps(self):
        self.assertTrue(_device_uses_gpu("cuda"))
        self.assertTrue(_device_uses_gpu("cuda:0"))
        self.assertTrue(_device_uses_gpu("mps"))
        self.assertFalse(_device_uses_gpu("cpu"))

    def test_warn_if_not_using_gpu_emits_warning_for_cpu(self):
        with patch("llmssycoph.llm.huggingface.warn_status") as mock_warn:
            _warn_if_not_using_gpu(
                model_name="mistralai/Mistral-7B-Instruct-v0.2",
                device="cpu",
            )

        mock_warn.assert_called_once_with(
            "llm/huggingface.py",
            "model_loading_without_gpu",
            "loading model=mistralai/Mistral-7B-Instruct-v0.2 without GPU acceleration "
            "(resolved device=cpu). This run may be much slower.",
        )

    def test_load_llm_prefers_registered_backend(self):
        class DummyLLM(BaseLLM):
            def __init__(self, model_name: str, **kwargs):
                super().__init__(model_name=model_name)
                self.kwargs = kwargs

            def generate(
                self,
                messages,
                *,
                n,
                max_new_tokens=64,
                temperature=0.0,
                top_p=1.0,
                batch_size=1,
                safe_fallback=True,
                strict_mc_letters="",
            ):
                return [GenerationResult(response_raw="dummy") for _ in range(n)]

            def score_choices(self, messages, choices):
                total = max(1, len(choices))
                return {str(choice): 1.0 / total for choice in choices}

        register_llm("dummy-backend", lambda **kwargs: DummyLLM(**kwargs))
        try:
            self.assertIsNotNone(get_registered_llm_factory("dummy-backend"))
            self.assertIn("dummy-backend", registered_llm_names())

            llm = load_llm(
                "dummy-backend",
                device="cpu",
                device_map_auto=False,
                hf_cache_dir=None,
            )
            self.assertIsInstance(llm, DummyLLM)
            self.assertEqual(llm.model_name, "dummy-backend")
            self.assertEqual(llm.kwargs["device"], "cpu")
        finally:
            unregister_llm("dummy-backend")

        self.assertNotIn("dummy-backend", registered_llm_names())

    def test_explicit_openai_model_ids_are_registered_and_resolve_to_openai_backend(self):
        self.assertEqual(resolve_llm_backend("gpt-5.4-nano"), "openai")
        capabilities = get_registered_llm_capabilities("gpt-5.4-nano")
        self.assertIsNotNone(capabilities)
        self.assertFalse(capabilities.supports_hidden_state_probes)
        self.assertTrue(capabilities.supports_choice_scoring)

        with patch("llmssycoph.llm.registry.OpenAILLM", autospec=True) as mock_openai_llm:
            instance = mock_openai_llm.return_value
            llm = load_llm(
                "gpt-5.4-nano",
                device="auto",
                device_map_auto=False,
                hf_cache_dir=None,
            )

        self.assertIs(llm, instance)
        mock_openai_llm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
