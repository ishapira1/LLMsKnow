from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import log_status, warn_status
from .base import BaseLLM, GenerationResult, LLMCapabilities
from .generation import generate_many as _generate_many
from .scoring import score_choices as _score_choices


def _device_uses_gpu(device: str) -> bool:
    normalized = str(device or "").strip().lower()
    return normalized == "mps" or normalized.startswith("cuda")


def _warn_if_not_using_gpu(model_name: str, device: str) -> None:
    if _device_uses_gpu(device):
        return
    warn_status(
        "llm/huggingface.py",
        "model_loading_without_gpu",
        f"loading model={model_name} without GPU acceleration (resolved device={device}). "
        "This run may be much slower.",
    )


def _hf_load_kwargs(hf_cache_dir: Optional[str]) -> Dict[str, Any]:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    load_kwargs: Dict[str, Any] = {"cache_dir": hf_cache_dir}
    if hf_token:
        load_kwargs["token"] = hf_token
    return load_kwargs


def _is_gated_repo_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "gated repo" in lowered or "cannot access gated repo" in lowered or "401 client error" in lowered


def _raise_helpful_hf_auth_error(model_name: str, exc: Exception) -> None:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        raise RuntimeError(
            f"Hugging Face model {model_name!r} appears to require gated-repo access, and the request "
            "failed even though HF_TOKEN/HUGGINGFACE_TOKEN is set. Make sure the Hugging Face account "
            "behind that token has been granted access to the model."
        ) from exc
    raise RuntimeError(
        f"Hugging Face model {model_name!r} appears to require gated-repo access, but no Hugging Face "
        "token was found. Add HF_TOKEN or HUGGINGFACE_TOKEN to your .env file or shell environment "
        "and retry."
    ) from exc


class HuggingFaceLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        device: str,
        device_map_auto: bool,
        hf_cache_dir: Optional[str],
    ):
        super().__init__(model_name=model_name)
        self.device = device
        self.device_map_auto = bool(device_map_auto)
        self.hf_cache_dir = hf_cache_dir
        self.model, self.tokenizer = self._load_model_and_tokenizer(
            model_name=model_name,
            device=device,
            device_map_auto=device_map_auto,
            hf_cache_dir=hf_cache_dir,
        )

    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            backend_name="huggingface",
            supports_hidden_state_probes=True,
            supports_choice_scoring=True,
            exposes_model_and_tokenizer=True,
        )

    @staticmethod
    def _load_model_and_tokenizer(
        model_name: str,
        device: str,
        device_map_auto: bool,
        hf_cache_dir: Optional[str],
    ) -> Tuple[Any, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log_status("llm/huggingface.py", f"loading model={model_name} on device={device}")
        _warn_if_not_using_gpu(model_name=model_name, device=device)
        load_kwargs = _hf_load_kwargs(hf_cache_dir)
        if "token" in load_kwargs:
            log_status("llm/huggingface.py", f"using Hugging Face auth token for model={model_name}")
        try:
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if device_map_auto else None,
                    **load_kwargs,
                )
                if not device_map_auto:
                    model = model.to("cuda")
            elif device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    **load_kwargs,
                )
                model = model.to("mps")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    **load_kwargs,
                )
                model = model.to("cpu")

            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **load_kwargs)
        except Exception as exc:
            if _is_gated_repo_error(exc):
                _raise_helpful_hf_auth_error(model_name, exc)
            raise
        model.eval()
        return model, tokenizer

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
        outputs = _generate_many(
            self.model,
            self.tokenizer,
            messages,
            n=n,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
            safe_fallback=safe_fallback,
            return_metadata=True,
            strict_mc_letters=strict_mc_letters,
        )
        return [GenerationResult.from_output(output) for output in outputs]

    def score_choices(
        self,
        messages: List[Dict[str, Any]],
        choices: List[str],
    ) -> Dict[str, float]:
        return _score_choices(
            self.model,
            self.tokenizer,
            messages,
            choices,
        )

    def get_model_and_tokenizer(self) -> Tuple[Any, Any]:
        return self.model, self.tokenizer


__all__ = [
    "HuggingFaceLLM",
    "_hf_load_kwargs",
    "_is_gated_repo_error",
    "_raise_helpful_hf_auth_error",
]
