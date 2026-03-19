from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import log_status, warn_status
from .base import BaseLLM, GenerationResult
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
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if device_map_auto else None,
                cache_dir=hf_cache_dir,
            )
            if not device_map_auto:
                model = model.to("cuda")
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                cache_dir=hf_cache_dir,
            )
            model = model.to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                cache_dir=hf_cache_dir,
            )
            model = model.to("cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=hf_cache_dir)
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


__all__ = ["HuggingFaceLLM"]
