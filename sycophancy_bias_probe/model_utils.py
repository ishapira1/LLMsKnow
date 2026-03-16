from __future__ import annotations

from typing import Any, Dict, List

from .logging_utils import log_status


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
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )

    text = ""
    for message in hf_messages:
        text += f"{message['role'].upper()}: {message['content']}\n"
    if add_generation_prompt:
        text += "ASSISTANT: "
    return tokenizer(text, return_tensors="pt").input_ids


def _eos_token_ids(tokenizer) -> List[int]:
    eos_value = getattr(tokenizer, "eos_token_id", None)
    if eos_value is None:
        return []
    if isinstance(eos_value, (list, tuple)):
        return [int(token_id) for token_id in eos_value]
    return [int(eos_value)]


def _decode_generation_metadata(tokenizer, gen_ids, max_new_tokens: int) -> Dict[str, Any]:
    token_ids = gen_ids.detach().cpu().tolist()
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    while token_ids and pad_token_id is not None and token_ids[-1] == pad_token_id:
        token_ids.pop()

    completion_token_count = len(token_ids)
    eos_token_ids = set(_eos_token_ids(tokenizer))
    stopped_on_eos = bool(token_ids and eos_token_ids and token_ids[-1] in eos_token_ids)
    hit_max_new_tokens = completion_token_count >= int(max_new_tokens) and not stopped_on_eos
    if stopped_on_eos:
        finish_reason = "eos_token"
    elif hit_max_new_tokens:
        finish_reason = "length"
    elif completion_token_count <= 0:
        finish_reason = "empty"
    else:
        finish_reason = "unknown"

    return {
        "response_raw": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
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
):
    torch = _import_torch()
    with torch.no_grad():
        input_ids = encode_chat(tokenizer, messages, add_generation_prompt=True).to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)

        do_sample = temperature > 0
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            return_dict_in_generate=True,
        )
        gen_ids = out.sequences[0, input_ids.shape[1] :]
        metadata = _decode_generation_metadata(tokenizer, gen_ids, max_new_tokens=max_new_tokens)
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
) -> List[Any]:
    if n <= 0:
        return []

    batch_size = max(1, int(batch_size))
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
            )
            for _ in range(n)
        ]

    torch = _import_torch()
    with torch.no_grad():
        do_sample = temperature > 0
        input_ids_base = encode_chat(tokenizer, messages, add_generation_prompt=True).to(model.device)
        attention_mask_base = torch.ones_like(input_ids_base, device=model.device)
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
                    )
                    outputs.append(metadata if return_metadata else metadata["response_raw"])
            except Exception as exc:
                if not safe_fallback or not _should_fallback_to_sequential(exc):
                    raise
                log_status(
                    "model_utils.py",
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
                        )
                    )

            remaining -= chunk

        return outputs


__all__ = [
    "to_hf_chat",
    "encode_chat",
    "generate_one",
    "_clear_device_cache",
    "_should_fallback_to_sequential",
    "generate_many",
]
