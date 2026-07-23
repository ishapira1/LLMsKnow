from __future__ import annotations

from typing import Optional

from .huggingface import HuggingFaceLLM
from .registry import load_llm


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
    torch_dtype: Optional[str] = None,
    revision: Optional[str] = None,
):
    llm = HuggingFaceLLM(
        model_name=model_name,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
        revision=revision,
    )
    return llm.get_model_and_tokenizer()


__all__ = ["load_llm", "load_model_and_tokenizer"]
