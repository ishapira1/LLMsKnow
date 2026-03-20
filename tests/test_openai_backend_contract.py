from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llmssycoph.llm.openai_backend import OpenAILLM
from llmssycoph.llm.openai_models import OpenAIModelSpec


class _FakeResponsesAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.responses = _FakeResponsesAPI(responses)


class _FakeChatCompletionsAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _FakeChatNamespace:
    def __init__(self, responses):
        self.completions = _FakeChatCompletionsAPI(responses)


class _FakeLegacyClient:
    def __init__(self, responses):
        self.chat = _FakeChatNamespace(responses)


class OpenAIBackendContractTests(unittest.TestCase):
    def test_client_kwargs_use_project_key_and_org_id(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY_FOR_PROJECT": "sk-project",
                "OPENAI_ORG_ID": "org-123",
            },
            clear=False,
        ):
            kwargs = OpenAILLM._client_kwargs_from_env()

        self.assertEqual(kwargs["api_key"], "sk-project")
        self.assertEqual(kwargs["organization"], "org-123")

    def test_generate_uses_responses_api_and_normalizes_output(self):
        fake_response = SimpleNamespace(
            output_text="C",
            usage=SimpleNamespace(output_tokens=1),
            status="completed",
            incomplete_details=None,
        )
        fake_client = _FakeClient([fake_response, fake_response])

        with patch.dict(os.environ, {"OPENAI_API_KEY_FOR_PROJECT": "sk-project"}, clear=False):
            with patch.object(OpenAILLM, "_build_client", return_value=fake_client):
                llm = OpenAILLM(
                    model_name="gpt-5.4-nano",
                    device="auto",
                    device_map_auto=False,
                    hf_cache_dir=None,
                    model_spec=OpenAIModelSpec(model_name="gpt-5.4-nano", reasoning_effort="none"),
                )

        outputs = llm.generate(
            [{"type": "human", "content": "Question"}],
            n=2,
            max_new_tokens=32,
            temperature=0.1,
            top_p=1.0,
            batch_size=1,
            safe_fallback=True,
            strict_mc_letters="ABCD",
        )

        self.assertEqual([output.response_raw for output in outputs], ["C", "C"])
        self.assertEqual([output.completion_token_count for output in outputs], [1, 1])
        self.assertEqual(len(fake_client.responses.calls), 2)
        first_call = fake_client.responses.calls[0]
        self.assertEqual(first_call["model"], "gpt-5.4-nano")
        self.assertEqual(first_call["max_output_tokens"], 32)
        self.assertEqual(first_call["temperature"], 0.1)
        self.assertEqual(first_call["reasoning"], {"effort": "none"})
        self.assertEqual(first_call["input"][0]["role"], "user")
        self.assertEqual(first_call["input"][0]["content"][0]["text"], "Question")

    def test_score_choices_uses_chat_logprobs(self):
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token="C",
                                logprob=-0.2,
                                top_logprobs=[
                                    SimpleNamespace(token="A", logprob=-2.0),
                                    SimpleNamespace(token="B", logprob=-1.5),
                                    SimpleNamespace(token="C", logprob=-0.2),
                                    SimpleNamespace(token="D", logprob=-2.5),
                                ],
                            )
                        ]
                    )
                )
            ]
        )
        fake_client = _FakeLegacyClient([fake_response])

        with patch.dict(os.environ, {"OPENAI_API_KEY_FOR_PROJECT": "sk-project"}, clear=False):
            with patch.object(OpenAILLM, "_build_client", return_value=fake_client):
                llm = OpenAILLM(
                    model_name="gpt-5.4-nano",
                    device="auto",
                    device_map_auto=False,
                    hf_cache_dir=None,
                    model_spec=OpenAIModelSpec(model_name="gpt-5.4-nano", reasoning_effort="none"),
                )

        probabilities = llm.score_choices([{"type": "human", "content": "Question"}], ["A", "B", "C", "D"])

        self.assertGreater(probabilities["C"], probabilities["B"])
        self.assertGreater(probabilities["B"], probabilities["A"])
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertEqual(len(fake_client.chat.completions.calls), 1)
        first_call = fake_client.chat.completions.calls[0]
        self.assertTrue(first_call["logprobs"])
        self.assertEqual(first_call["top_logprobs"], 5)
        self.assertEqual(first_call["extra_body"], {"max_completion_tokens": 32, "reasoning_effort": "none"})

    def test_score_choices_rejects_more_than_five_options(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY_FOR_PROJECT": "sk-project"}, clear=False):
            with patch.object(OpenAILLM, "_build_client", return_value=_FakeLegacyClient([])):
                llm = OpenAILLM(
                    model_name="gpt-5.4-nano",
                    device="auto",
                    device_map_auto=False,
                    hf_cache_dir=None,
                    model_spec=OpenAIModelSpec(model_name="gpt-5.4-nano", reasoning_effort="none"),
                )

        with self.assertRaisesRegex(ValueError, "at most 5 answer options"):
            llm.score_choices(
                [{"type": "human", "content": "Question"}],
                ["A", "B", "C", "D", "E", "F"],
            )

    def test_generate_falls_back_to_chat_completions_when_responses_api_is_unavailable(self):
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="C"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(completion_tokens=1),
        )
        fake_client = _FakeLegacyClient([fake_response])

        with patch.dict(os.environ, {"OPENAI_API_KEY_FOR_PROJECT": "sk-project"}, clear=False):
            with patch.object(OpenAILLM, "_build_client", return_value=fake_client):
                llm = OpenAILLM(
                    model_name="gpt-5.4-nano",
                    device="auto",
                    device_map_auto=False,
                    hf_cache_dir=None,
                    model_spec=OpenAIModelSpec(model_name="gpt-5.4-nano", reasoning_effort="none"),
                )

        outputs = llm.generate(
            [{"type": "human", "content": "Question"}],
            n=1,
            max_new_tokens=32,
            temperature=0.1,
            top_p=1.0,
            batch_size=1,
            safe_fallback=True,
            strict_mc_letters="ABCD",
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].response_raw, "C")
        self.assertEqual(outputs[0].finish_reason, "stop")
        self.assertEqual(len(fake_client.chat.completions.calls), 1)
        first_call = fake_client.chat.completions.calls[0]
        self.assertEqual(first_call["model"], "gpt-5.4-nano")
        self.assertEqual(first_call["extra_body"], {"max_completion_tokens": 32, "reasoning_effort": "none"})
        self.assertEqual(first_call["messages"][0]["role"], "user")
        self.assertEqual(first_call["messages"][0]["content"], "Question")


if __name__ == "__main__":
    unittest.main()
