from __future__ import annotations

import math
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

from ..logging_utils import log_status
from .base import BaseLLM, GenerationResult, LLMCapabilities
from .openai_models import OpenAIModelSpec


def _deep_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def _text_from_content_parts(content_parts: Any) -> str:
    texts: List[str] = []
    for part in list(content_parts or []):
        if isinstance(part, dict):
            part_type = str(part.get("type", "") or "")
            if part_type in {"output_text", "text", "input_text"}:
                text = part.get("text")
                if text:
                    texts.append(str(text))
            continue
        part_type = str(getattr(part, "type", "") or "")
        if part_type in {"output_text", "text", "input_text"}:
            text = getattr(part, "text", None)
            if text:
                texts.append(str(text))
    return "".join(texts).strip()


def _response_output_text(response: Any) -> str:
    output_text = _deep_get(response, "output_text")
    if output_text:
        return str(output_text).strip()

    texts: List[str] = []
    for item in list(_deep_get(response, "output") or []):
        if isinstance(item, dict):
            text = _text_from_content_parts(item.get("content", []))
        else:
            text = _text_from_content_parts(getattr(item, "content", []))
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


_CHOICE_VARIANT_TEXT_TEMPLATES = ("{choice}", " {choice}", "\n{choice}")


def _normalize_choices(choices: List[str]) -> List[str]:
    return [str(choice or "").strip() for choice in choices if str(choice or "").strip()]


def _choice_variant_texts(choice: str) -> List[str]:
    normalized_choice = str(choice or "").strip()
    if not normalized_choice:
        return []
    return [template.format(choice=normalized_choice) for template in _CHOICE_VARIANT_TEXT_TEMPLATES]


def _token_logprob_pairs_from_chat_choice(choice: Any) -> Dict[str, float]:
    pairs: Dict[str, float] = {}
    content = list(_deep_get(choice, "logprobs.content") or [])
    if not content:
        return pairs
    first_token = content[0]
    token = str(getattr(first_token, "token", "") or "")
    logprob = getattr(first_token, "logprob", None)
    if token and logprob is not None:
        pairs[token] = float(logprob)
    for candidate in list(getattr(first_token, "top_logprobs", []) or []):
        candidate_token = str(getattr(candidate, "token", "") or "")
        candidate_logprob = getattr(candidate, "logprob", None)
        if not candidate_token or candidate_logprob is None:
            continue
        candidate_value = float(candidate_logprob)
        previous = pairs.get(candidate_token)
        if previous is None or candidate_value > previous:
            pairs[candidate_token] = candidate_value
    return pairs


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        device: str,
        device_map_auto: bool,
        hf_cache_dir: Optional[str],
        *,
        model_spec: Optional[OpenAIModelSpec] = None,
    ):
        super().__init__(model_name=model_name)
        self.device = device
        self.device_map_auto = bool(device_map_auto)
        self.hf_cache_dir = hf_cache_dir
        self.model_spec = model_spec or OpenAIModelSpec(model_name=model_name)
        self._client_local = threading.local()
        self._default_client = self._build_client()
        self._client_local.client = self._default_client
        self._uses_responses_api = hasattr(self._default_client, "responses")

    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            backend_name="openai",
            supports_hidden_state_probes=False,
            supports_choice_scoring=True,
            exposes_model_and_tokenizer=False,
        )

    @staticmethod
    def _client_kwargs_from_env() -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY_FOR_PROJECT") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI backend requires OPENAI_API_KEY_FOR_PROJECT or OPENAI_API_KEY in the environment."
            )

        kwargs: Dict[str, Any] = {"api_key": api_key}
        org_id = str(os.getenv("OPENAI_ORG_ID", "") or "").strip()
        if org_id:
            kwargs["organization"] = org_id
        return kwargs

    @classmethod
    def _build_client(cls):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI backend requires the 'openai' Python package. Install project dependencies again."
            ) from exc

        kwargs = cls._client_kwargs_from_env()
        try:
            return OpenAI(**kwargs)
        except TypeError:
            org_id = kwargs.pop("organization", None)
            if org_id:
                kwargs["default_headers"] = {"OpenAI-Organization": org_id}
            return OpenAI(**kwargs)

    def _client_for_current_thread(self):
        client = getattr(self._client_local, "client", None)
        if client is None:
            client = self._build_client()
            self._client_local.client = client
        return client

    @staticmethod
    def _should_retry_openai_exception(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True
        code = str(getattr(exc, "code", "") or "").lower()
        if code in {"rate_limit_exceeded", "server_error", "timeout"}:
            return True
        name = exc.__class__.__name__
        return name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}

    def _request_with_retries(self, request_fn):
        max_attempts = 6
        for attempt_idx in range(max_attempts):
            try:
                return request_fn()
            except Exception as exc:
                if attempt_idx >= max_attempts - 1 or not self._should_retry_openai_exception(exc):
                    raise
                sleep_seconds = min(8.0, 0.5 * (2**attempt_idx)) + random.uniform(0.0, 0.25)
                log_status(
                    "llm/openai_backend.py",
                    f"OpenAI request retry {attempt_idx + 1}/{max_attempts - 1} after "
                    f"{exc.__class__.__name__}: sleeping {sleep_seconds:.2f}s",
                )
                time.sleep(sleep_seconds)

    @staticmethod
    def _to_openai_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for message in messages:
            raw_role = str(message.get("type", "") or message.get("role", "") or "user").strip().lower()
            role = {
                "human": "user",
                "user": "user",
                "ai": "assistant",
                "assistant": "assistant",
                "system": "system",
                "developer": "developer",
            }.get(raw_role, "user")
            content = str(message.get("content", "") or "")
            converted.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        return converted

    def _response_to_generation_result(self, response: Any, max_new_tokens: int) -> GenerationResult:
        response_text = _response_output_text(response)
        completion_token_count = _deep_get(response, "usage.output_tokens")
        incomplete_reason = str(_deep_get(response, "incomplete_details.reason") or "")
        finish_reason = (
            incomplete_reason
            or str(_deep_get(response, "finish_reason") or "")
            or str(_deep_get(response, "status") or "")
        )
        hit_max_new_tokens = incomplete_reason in {"max_output_tokens", "length"}
        if not hit_max_new_tokens and completion_token_count is not None:
            try:
                hit_max_new_tokens = int(completion_token_count) >= int(max_new_tokens)
            except Exception:
                hit_max_new_tokens = False

        stopped_on_eos = finish_reason in {"completed", "stop", "end_turn"}
        return GenerationResult(
            response_raw=response_text,
            completion_token_count=completion_token_count,
            hit_max_new_tokens=hit_max_new_tokens,
            stopped_on_eos=stopped_on_eos,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _to_chat_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        converted: List[Dict[str, str]] = []
        for message in messages:
            raw_role = str(message.get("type", "") or message.get("role", "") or "user").strip().lower()
            role = {
                "human": "user",
                "user": "user",
                "ai": "assistant",
                "assistant": "assistant",
                "system": "system",
                "developer": "developer",
            }.get(raw_role, "user")
            converted.append(
                {
                    "role": role,
                    "content": str(message.get("content", "") or ""),
                }
            )
        return converted

    def _chat_completion_to_generation_result(self, response: Any, max_new_tokens: int) -> GenerationResult:
        choices = list(getattr(response, "choices", []) or [])
        first_choice = choices[0] if choices else None
        message = getattr(first_choice, "message", None)
        response_text = str(getattr(message, "content", "") or "").strip()
        completion_token_count = _deep_get(response, "usage.completion_tokens")
        finish_reason = str(getattr(first_choice, "finish_reason", "") or "")
        hit_max_new_tokens = finish_reason == "length"
        if not hit_max_new_tokens and completion_token_count is not None:
            try:
                hit_max_new_tokens = int(completion_token_count) >= int(max_new_tokens)
            except Exception:
                hit_max_new_tokens = False

        stopped_on_eos = finish_reason in {"stop", "end_turn"}
        return GenerationResult(
            response_raw=response_text,
            completion_token_count=completion_token_count,
            hit_max_new_tokens=hit_max_new_tokens,
            stopped_on_eos=stopped_on_eos,
            finish_reason=finish_reason,
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        n: int,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 1,
        safe_fallback: bool = True,
        strict_mc_letters: str = "",
    ) -> List[GenerationResult]:
        del batch_size, safe_fallback, strict_mc_letters

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

        outputs: List[GenerationResult] = []
        if self._uses_responses_api:
            payload["input"] = self._to_openai_input(messages)
            payload["max_output_tokens"] = int(max_new_tokens)
            if self.model_spec.reasoning_effort:
                payload["reasoning"] = {"effort": str(self.model_spec.reasoning_effort)}
            for _ in range(int(n)):
                log_status("llm/openai_backend.py", f"requesting OpenAI response for model={self.model_name}")
                client = self._client_for_current_thread()
                response = self._request_with_retries(lambda: client.responses.create(**payload))
                outputs.append(self._response_to_generation_result(response, max_new_tokens=max_new_tokens))
            return outputs

        chat_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._to_chat_messages(messages),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        extra_body: Dict[str, Any] = {"max_completion_tokens": int(max_new_tokens)}
        if self.model_spec.reasoning_effort:
            extra_body["reasoning_effort"] = str(self.model_spec.reasoning_effort)
        chat_payload["extra_body"] = extra_body
        for _ in range(int(n)):
            log_status(
                "llm/openai_backend.py",
                f"requesting OpenAI chat completion for model={self.model_name} "
                "(installed SDK does not expose Responses API)",
            )
            client = self._client_for_current_thread()
            response = self._request_with_retries(lambda: client.chat.completions.create(**chat_payload))
            outputs.append(self._chat_completion_to_generation_result(response, max_new_tokens=max_new_tokens))
        return outputs

    def score_choices(
        self,
        messages: List[Dict[str, Any]],
        choices: List[str],
    ) -> Dict[str, float]:
        normalized_choices = _normalize_choices(choices)
        if not normalized_choices:
            raise ValueError("score_choices requires at least one non-empty choice.")
        if len(normalized_choices) > 5:
            raise ValueError(
                "OpenAI strict-MC choice scoring currently supports at most 5 answer options "
                "because the Chat Completions API caps top_logprobs at 5."
            )

        chat_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._to_chat_messages(messages),
            "temperature": 1.0,
            "top_p": 1.0,
            "logprobs": True,
            "top_logprobs": 5,
        }
        extra_body: Dict[str, Any] = {"max_completion_tokens": 1}
        if self.model_spec.reasoning_effort:
            extra_body["reasoning_effort"] = str(self.model_spec.reasoning_effort)
        chat_payload["extra_body"] = extra_body

        log_status("llm/openai_backend.py", f"requesting OpenAI choice logprobs for model={self.model_name}")
        client = self._client_for_current_thread()
        response = self._request_with_retries(lambda: client.chat.completions.create(**chat_payload))
        choices_payload = list(getattr(response, "choices", []) or [])
        if not choices_payload:
            raise RuntimeError("OpenAI choice scoring returned no choices.")

        token_logprobs = _token_logprob_pairs_from_chat_choice(choices_payload[0])
        raw_choice_probs: Dict[str, float] = {}
        for choice in normalized_choices:
            raw_probability = 0.0
            counted_tokens = set()
            for variant_text in _choice_variant_texts(choice):
                if variant_text in counted_tokens:
                    continue
                variant_logprob = token_logprobs.get(variant_text)
                if variant_logprob is None:
                    continue
                raw_probability += math.exp(float(variant_logprob))
                counted_tokens.add(variant_text)
            raw_choice_probs[choice] = raw_probability

        total_mass = float(sum(raw_choice_probs.values()))
        if total_mass <= 0.0:
            raise RuntimeError(
                "OpenAI choice scoring produced zero probability mass over the provided choices. "
                f"Observed first-token candidates: {sorted(token_logprobs)}"
            )

        return {
            choice: raw_choice_probs[choice] / total_mass
            for choice in normalized_choices
        }


__all__ = ["OpenAILLM"]
