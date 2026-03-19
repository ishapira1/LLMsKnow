from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..logging_utils import log_status
from .base import BaseLLM
from .huggingface import HuggingFaceLLM


LLMFactory = Callable[..., BaseLLM]

_LLM_REGISTRY: Dict[str, LLMFactory] = {}


def register_llm(name: str, factory: LLMFactory) -> None:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("LLM registry name must be a non-empty string.")
    _LLM_REGISTRY[normalized] = factory


def unregister_llm(name: str) -> None:
    _LLM_REGISTRY.pop(str(name).strip(), None)


def get_registered_llm_factory(name: str) -> Optional[LLMFactory]:
    return _LLM_REGISTRY.get(str(name).strip())


def registered_llm_names() -> List[str]:
    return sorted(_LLM_REGISTRY)


def load_llm(
    model_name: str,
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
) -> BaseLLM:
    factory = get_registered_llm_factory(model_name)
    if factory is None:
        log_status(
            "llm/registry.py",
            f"model={model_name} not found in registry; loading as a Hugging Face model identifier",
        )
        return HuggingFaceLLM(
            model_name=model_name,
            device=device,
            device_map_auto=device_map_auto,
            hf_cache_dir=hf_cache_dir,
        )

    log_status("llm/registry.py", f"model={model_name} resolved via registered backend")
    return factory(
        model_name=model_name,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
    )


__all__ = [
    "LLMFactory",
    "get_registered_llm_factory",
    "load_llm",
    "register_llm",
    "registered_llm_names",
    "unregister_llm",
]
