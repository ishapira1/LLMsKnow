from __future__ import annotations

from typing import Optional

from ..logging_utils import log_status


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log_status("llm/loading.py", f"loading model={model_name} on device={device}")
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


__all__ = ["load_model_and_tokenizer"]
