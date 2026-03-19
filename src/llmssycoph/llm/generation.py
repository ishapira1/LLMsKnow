from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..logging_utils import log_status, warn_status


def _import_torch():
    import torch

    return torch


def to_hf_chat(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_type = message.get("type")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if message_type == "human":
            out.append({"role": "user", "content": content})
        elif message_type == "assistant":
            out.append({"role": "assistant", "content": content})
        elif message_type == "system":
            out.append({"role": "system", "content": content})
        else:
            out.append({"role": "user", "content": content})
    return out


def encode_chat(tokenizer, messages: List[Dict[str, Any]], add_generation_prompt: bool = True):
    hf_messages = to_hf_chat(messages)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        tokenizer_name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
        raise TypeError(
            f"Tokenizer {tokenizer_name!r} does not expose apply_chat_template(). "
            "This pipeline now relies on model-native chat templates instead of a manual fallback format."
        )

    return apply_chat_template(
        hf_messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )


def _token_id_list_from_encoded(encoded: Any, device: Any = None) -> List[int]:
    target = encoded
    if hasattr(target, "input_ids"):
        if device is not None and hasattr(target, "to"):
            target = target.to(device)
        return _token_id_list_from_encoded(target.input_ids, device=None)

    if hasattr(target, "ids"):
        return [int(token_id) for token_id in target.ids]

    if device is not None and hasattr(target, "to"):
        target = target.to(device)

    if hasattr(target, "tolist"):
        values = target.tolist()
    else:
        values = list(target)

    if values and isinstance(values[0], list):
        values = values[0]
    return [int(token_id) for token_id in values]


def _resolve_model_inputs(
    tokenizer,
    messages: List[Dict[str, Any]],
    device: Any,
    *,
    add_generation_prompt: bool = True,
) -> Tuple[Any, Any]:
    torch = _import_torch()
    encoded = encode_chat(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.to(device)
        input_ids = encoded.input_ids
        attention_mask = getattr(encoded, "attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        return input_ids, attention_mask

    input_ids = encoded.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


def _eos_token_ids(tokenizer) -> List[int]:
    eos_value = getattr(tokenizer, "eos_token_id", None)
    if eos_value is None:
        return []
    if isinstance(eos_value, (list, tuple)):
        return [int(token_id) for token_id in eos_value]
    return [int(eos_value)]


def _strict_mc_generated_answer_complete(text: str, letters: str) -> bool:
    allowed = set(str(letters or "").strip().upper())
    candidate = str(text or "")
    if not candidate.strip() or not allowed:
        return False

    match = re.fullmatch(
        r"\s*(?:answer\s*:\s*)?\(?([A-Za-z])(?:\)|\])?[\s\].,:;\-]*",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return False
    return match.group(1).upper() in allowed


def _strict_mc_stopping_criteria(tokenizer, input_len: int, letters: str):
    normalized_letters = str(letters or "").strip().upper()
    if not normalized_letters:
        return None

    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StrictMCStopper(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
            if input_ids.shape[0] != 1:
                return False
            generated_ids = input_ids[0, input_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return _strict_mc_generated_answer_complete(text, normalized_letters)

    return StoppingCriteriaList([_StrictMCStopper()])


def _decode_generation_metadata(
    tokenizer,
    gen_ids,
    max_new_tokens: int,
    strict_mc_letters: str = "",
) -> Dict[str, Any]:
    token_ids = gen_ids.detach().cpu().tolist()
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    while token_ids and pad_token_id is not None and token_ids[-1] == pad_token_id:
        token_ids.pop()

    completion_token_count = len(token_ids)
    eos_token_ids = set(_eos_token_ids(tokenizer))
    stopped_on_eos = bool(token_ids and eos_token_ids and token_ids[-1] in eos_token_ids)
    hit_max_new_tokens = completion_token_count >= int(max_new_tokens) and not stopped_on_eos
    response_raw = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if (
        strict_mc_letters
        and response_raw
        and _strict_mc_generated_answer_complete(response_raw, strict_mc_letters)
        and not stopped_on_eos
        and not hit_max_new_tokens
    ):
        finish_reason = "answer_commitment"
    elif stopped_on_eos:
        finish_reason = "eos_token"
    elif hit_max_new_tokens:
        finish_reason = "length"
    elif completion_token_count <= 0:
        finish_reason = "empty"
    else:
        finish_reason = "unknown"

    return {
        "response_raw": response_raw,
        "completion_token_count": completion_token_count,
        "hit_max_new_tokens": hit_max_new_tokens,
        "stopped_on_eos": stopped_on_eos,
        "finish_reason": finish_reason,
    }


def generate_one(
    model,
    tokenizer,
    messages,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    return_metadata: bool = False,
    strict_mc_letters: str = "",
):
    torch = _import_torch()
    with torch.no_grad():
        input_ids, attention_mask = _resolve_model_inputs(
            tokenizer,
            messages,
            model.device,
            add_generation_prompt=True,
        )
        stopping_criteria = _strict_mc_stopping_criteria(
            tokenizer,
            input_ids.shape[1],
            strict_mc_letters,
        )

        do_sample = temperature > 0
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria,
        )
        gen_ids = out.sequences[0, input_ids.shape[1] :]
        metadata = _decode_generation_metadata(
            tokenizer,
            gen_ids,
            max_new_tokens=max_new_tokens,
            strict_mc_letters=strict_mc_letters,
        )
        if return_metadata:
            return metadata
        return metadata["response_raw"]


def _clear_device_cache(device: Any) -> None:
    torch = _import_torch()
    dtype = getattr(device, "type", str(device))
    if dtype == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dtype == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _should_fallback_to_sequential(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "out of memory",
        "cuda error",
        "cublas",
        "cudnn",
        "mps",
        "resource exhausted",
        "hip",
    )
    return isinstance(exc, RuntimeError) and any(marker in msg for marker in markers)


def generate_many(
    model,
    tokenizer,
    messages,
    n: int,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    safe_fallback: bool = True,
    return_metadata: bool = False,
    strict_mc_letters: str = "",
) -> List[Any]:
    if n <= 0:
        return []

    batch_size = max(1, int(batch_size))
    if strict_mc_letters and batch_size != 1:
        log_status(
            "llm/generation.py",
            "strict MC generation uses sequential decoding so each sample can stop at the first committed answer.",
        )
        batch_size = 1
    if batch_size == 1:
        return [
            generate_one(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                return_metadata=return_metadata,
                strict_mc_letters=strict_mc_letters,
            )
            for _ in range(n)
        ]

    torch = _import_torch()
    with torch.no_grad():
        do_sample = temperature > 0
        input_ids_base, attention_mask_base = _resolve_model_inputs(
            tokenizer,
            messages,
            model.device,
            add_generation_prompt=True,
        )
        input_len = input_ids_base.shape[1]

        outputs: List[str] = []
        remaining = n
        while remaining > 0:
            chunk = min(batch_size, remaining)
            input_ids = input_ids_base.expand(chunk, -1).contiguous()
            attention_mask = attention_mask_base.expand(chunk, -1).contiguous()
            try:
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    return_dict_in_generate=True,
                )
                for row in range(chunk):
                    gen_ids = out.sequences[row, input_len:]
                    metadata = _decode_generation_metadata(
                        tokenizer,
                        gen_ids,
                        max_new_tokens=max_new_tokens,
                        strict_mc_letters=strict_mc_letters,
                    )
                    outputs.append(metadata if return_metadata else metadata["response_raw"])
            except Exception as exc:
                if not safe_fallback or not _should_fallback_to_sequential(exc):
                    raise
                warn_status(
                    "llm/generation.py",
                    "batched_generation_fallback",
                    "batched generation failed; falling back to sequential generation "
                    f"for this prompt chunk={chunk}. {type(exc).__name__}: {exc}",
                )
                _clear_device_cache(model.device)
                for _ in range(chunk):
                    outputs.append(
                        generate_one(
                            model,
                            tokenizer,
                            messages,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            return_metadata=return_metadata,
                            strict_mc_letters=strict_mc_letters,
                        )
                    )

            remaining -= chunk

        return outputs


__all__ = [
    "_resolve_model_inputs",
    "_strict_mc_generated_answer_complete",
    "_token_id_list_from_encoded",
    "to_hf_chat",
    "encode_chat",
    "generate_one",
    "_clear_device_cache",
    "_should_fallback_to_sequential",
    "generate_many",
]
