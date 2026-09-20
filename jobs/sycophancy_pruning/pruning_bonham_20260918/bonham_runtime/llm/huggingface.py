from __future__ import annotations

import os
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

from llmssycoph.logging_utils import log_status, warn_status
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


def _auto_device_max_memory(
    *, device_map_auto: bool, cuda_device_count: int
) -> Optional[Dict[int, str]]:
    """Resolve an opt-in per-device cap used to force safe model sharding.

    A model can fit its parameters on one MIG slice while leaving too little
    workspace for long-sequence attention.  Capping parameter placement below
    physical capacity makes Accelerate split the unchanged model across the
    requested visible devices, reserving workspace on each slice.  The default
    remains unchanged when the environment variable is absent.
    """

    raw = str(os.getenv("LLMSSYCOPH_DEVICE_MAX_MEMORY_GIB", "") or "").strip()
    if not raw:
        return None
    if not device_map_auto:
        raise ValueError(
            "LLMSSYCOPH_DEVICE_MAX_MEMORY_GIB requires device_map_auto so the "
            "model can be distributed across visible CUDA devices."
        )
    try:
        gib = int(raw)
    except ValueError as exc:
        raise ValueError(
            "LLMSSYCOPH_DEVICE_MAX_MEMORY_GIB must be a positive integer."
        ) from exc
    if gib <= 0:
        raise ValueError(
            "LLMSSYCOPH_DEVICE_MAX_MEMORY_GIB must be a positive integer."
        )
    count = int(cuda_device_count)
    if count <= 0:
        raise ValueError(
            "Per-device max memory was requested, but no visible CUDA device exists."
        )
    return {index: f"{gib}GiB" for index in range(count)}


def _qwen_prefers_bfloat16(model_name: str) -> bool:
    normalized = str(model_name or "").lower()
    return "qwen" in normalized


def _is_gemma4(model_name: str) -> bool:
    normalized = str(model_name or "").lower()
    return "gemma-4" in normalized or "gemma4" in normalized


def _install_chat_template_defaults(tokenizer: Any, model_name: str) -> Any:
    """Bind model-specific native-template defaults without changing call sites."""

    if not _is_gemma4(model_name):
        return tokenizer
    if bool(getattr(tokenizer, "_llmsknow_gemma4_template_defaults", False)):
        return tokenizer
    original = tokenizer.apply_chat_template

    def apply_chat_template_with_defaults(
        _tokenizer: Any, conversation: Any, *args: Any, **kwargs: Any
    ) -> Any:
        kwargs.setdefault("enable_thinking", False)
        return original(conversation, *args, **kwargs)

    tokenizer.apply_chat_template = MethodType(apply_chat_template_with_defaults, tokenizer)
    tokenizer._llmsknow_gemma4_template_defaults = True
    tokenizer._llmsknow_chat_template_kwargs = {"enable_thinking": False}
    return tokenizer


def _auto_model_class(model_name: str) -> Any:
    if _is_gemma4(model_name):
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def _resolve_torch_dtype(model_name: str, device: str, torch_dtype: Optional[str]):
    import torch

    requested = str(torch_dtype or "auto").strip().lower()
    if requested in {"", "auto"}:
        if str(device or "").strip().lower() == "cuda":
            return torch.bfloat16 if _qwen_prefers_bfloat16(model_name) else torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if requested not in mapping:
        raise ValueError(
            f"Unsupported torch_dtype={torch_dtype!r}. Use auto, float16, bfloat16, or float32."
        )
    return mapping[requested]


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


def _record_pinned_tokenizer_revision(tokenizer: Any, revision: Optional[str]) -> None:
    """Retain the exact requested tokenizer commit when Transformers omits it."""

    pinned = str(revision or "")
    if not pinned:
        return
    init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    observed = str(init_kwargs.get("_commit_hash", "") or "")
    if observed and observed != pinned:
        raise RuntimeError(
            "Tokenizer revision mismatch: "
            f"requested={pinned!r} observed={observed!r}."
        )
    init_kwargs["_commit_hash"] = pinned
    tokenizer.init_kwargs = init_kwargs
    tokenizer._llmsknow_revision_source = "requested_exact_commit"


class HuggingFaceLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        device: str,
        device_map_auto: bool,
        hf_cache_dir: Optional[str],
        torch_dtype: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        super().__init__(model_name=model_name)
        self.device = device
        self.device_map_auto = bool(device_map_auto)
        self.hf_cache_dir = hf_cache_dir
        self.revision = revision
        self.model, self.tokenizer = self._load_model_and_tokenizer(
            model_name=model_name,
            device=device,
            device_map_auto=device_map_auto,
            hf_cache_dir=hf_cache_dir,
            torch_dtype=torch_dtype,
            revision=revision,
        )

    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            backend_name="huggingface",
            supports_hidden_state_probes=True,
            supports_choice_scoring=True,
            exposes_model_and_tokenizer=True,
            supports_structured_tool_transcripts=True,
        )

    @staticmethod
    def _load_model_and_tokenizer(
        model_name: str,
        device: str,
        device_map_auto: bool,
        hf_cache_dir: Optional[str],
        torch_dtype: Optional[str] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
    ) -> Tuple[Any, Any]:
        import torch
        from transformers import AutoTokenizer

        log_status("llm/huggingface.py", f"loading model={model_name} on device={device}")
        _warn_if_not_using_gpu(model_name=model_name, device=device)
        load_kwargs = _hf_load_kwargs(hf_cache_dir)
        if revision:
            load_kwargs["revision"] = revision
        if local_files_only:
            load_kwargs["local_files_only"] = True
        resolved_torch_dtype = _resolve_torch_dtype(model_name, device, torch_dtype)
        auto_model = _auto_model_class(model_name)
        log_status(
            "llm/huggingface.py",
            f"resolved torch_dtype={str(resolved_torch_dtype).replace('torch.', '')} for model={model_name}",
        )
        if "token" in load_kwargs:
            log_status("llm/huggingface.py", f"using Hugging Face auth token for model={model_name}")
        try:
            if device == "cuda":
                max_memory = _auto_device_max_memory(
                    device_map_auto=bool(device_map_auto),
                    cuda_device_count=int(torch.cuda.device_count()),
                )
                placement_kwargs: Dict[str, Any] = {
                    "device_map": "auto" if device_map_auto else None,
                }
                if max_memory is not None:
                    placement_kwargs["max_memory"] = max_memory
                    log_status(
                        "llm/huggingface.py",
                        f"using device_map=auto max_memory={max_memory}",
                    )
                model = auto_model.from_pretrained(
                    model_name,
                    torch_dtype=resolved_torch_dtype,
                    **placement_kwargs,
                    **load_kwargs,
                )
                if not device_map_auto:
                    model = model.to("cuda")
            elif device == "mps":
                model = auto_model.from_pretrained(
                    model_name,
                    torch_dtype=resolved_torch_dtype,
                    **load_kwargs,
                )
                model = model.to("mps")
            else:
                model = auto_model.from_pretrained(
                    model_name,
                    torch_dtype=resolved_torch_dtype,
                    **load_kwargs,
                )
                model = model.to("cpu")

            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **load_kwargs)
            _record_pinned_tokenizer_revision(tokenizer, revision)
            tokenizer = _install_chat_template_defaults(tokenizer, model_name)
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
        tools: Optional[List[Dict[str, Any]]] = None,
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
            tools=tools,
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
    "_resolve_torch_dtype",
    "_raise_helpful_hf_auth_error",
]
