from __future__ import annotations

from typing import Any, Dict, List


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


def generate_one(
    model,
    tokenizer,
    messages,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
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
        )
        gen_ids = out[0, input_ids.shape[1] :]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


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
) -> List[str]:
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
                )
                for row in range(chunk):
                    gen_ids = out[row, input_len:]
                    outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
            except Exception as exc:
                if not safe_fallback or not _should_fallback_to_sequential(exc):
                    raise
                print(
                    "[warn] batched generation failed; falling back to sequential generation "
                    f"for this prompt (chunk={chunk}). {type(exc).__name__}: {exc}"
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
