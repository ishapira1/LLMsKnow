from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..logging_utils import log_status
from .base import BaseLLM, LLMCapabilities
from .huggingface import HuggingFaceLLM
from .openai_backend import OpenAILLM
from .openai_models import OPENAI_MODEL_SPECS, OpenAIModelSpec


LLMFactory = Callable[..., BaseLLM]

_LLM_REGISTRY: Dict[str, LLMFactory] = {}
_LLM_CAPABILITIES: Dict[str, LLMCapabilities] = {}

HUGGINGFACE_CAPABILITIES = LLMCapabilities(
    backend_name="huggingface",
    supports_hidden_state_probes=True,
    supports_choice_scoring=True,
    exposes_model_and_tokenizer=True,
)


def register_llm(
    name: str,
    factory: LLMFactory,
    *,
    capabilities: Optional[LLMCapabilities] = None,
) -> None:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("LLM registry name must be a non-empty string.")
    _LLM_REGISTRY[normalized] = factory
    if capabilities is not None:
        _LLM_CAPABILITIES[normalized] = capabilities


def unregister_llm(name: str) -> None:
    normalized = str(name).strip()
    _LLM_REGISTRY.pop(normalized, None)
    _LLM_CAPABILITIES.pop(normalized, None)


def get_registered_llm_factory(name: str) -> Optional[LLMFactory]:
    return _LLM_REGISTRY.get(str(name).strip())


def get_registered_llm_capabilities(name: str) -> Optional[LLMCapabilities]:
    return _LLM_CAPABILITIES.get(str(name).strip())


def registered_llm_names() -> List[str]:
    return sorted(_LLM_REGISTRY)


def resolve_llm_capabilities(model_name: str) -> LLMCapabilities:
    registered_capabilities = get_registered_llm_capabilities(model_name)
    if registered_capabilities is not None:
        return registered_capabilities
    if get_registered_llm_factory(model_name) is not None:
        return LLMCapabilities(backend_name="registered")
    return HUGGINGFACE_CAPABILITIES


def resolve_llm_backend(model_name: str) -> str:
    return resolve_llm_capabilities(model_name).backend_name


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


def _make_openai_factory(spec: OpenAIModelSpec) -> LLMFactory:
    def _factory(**kwargs):
        return OpenAILLM(model_spec=spec, **kwargs)

    return _factory


for _openai_model_name, _openai_spec in OPENAI_MODEL_SPECS.items():
    register_llm(
        _openai_model_name,
        _make_openai_factory(_openai_spec),
        capabilities=LLMCapabilities(
            backend_name="openai",
            supports_hidden_state_probes=False,
            supports_choice_scoring=True,
            exposes_model_and_tokenizer=False,
        ),
    )


__all__ = [
    "LLMFactory",
    "get_registered_llm_factory",
    "get_registered_llm_capabilities",
    "load_llm",
    "register_llm",
    "registered_llm_names",
    "resolve_llm_backend",
    "resolve_llm_capabilities",
    "unregister_llm",
]
