from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

from ..llm.generation import _resolve_model_inputs, _token_id_list_from_encoded, encode_chat
from ..probes.features import _assistant_text_span


_CHOICE_VARIANT_TEXT_TEMPLATES = ("{choice}", " {choice}", "\n{choice}")


def _import_torch():
    import torch

    return torch


def _model_device(model: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _choice_token_ids(tokenizer: Any, choice: str) -> List[int]:
    token_ids: List[int] = []
    seen = set()
    normalized = str(choice or "").strip()
    if not normalized:
        return []
    for template in _CHOICE_VARIANT_TEXT_TEMPLATES:
        text = template.format(choice=normalized)
        encoded = tokenizer(text, add_special_tokens=False)
        ids = getattr(encoded, "input_ids", encoded)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) != 1:
            continue
        token_id = int(ids[0])
        if token_id not in seen:
            token_ids.append(token_id)
            seen.add(token_id)
    return token_ids


def _choice_logmass(log_probs: Any, token_ids: Sequence[int]):
    torch = _import_torch()
    if not token_ids:
        return None
    ids = torch.tensor(list(token_ids), dtype=torch.long, device=log_probs.device)
    return torch.logsumexp(log_probs.index_select(0, ids), dim=0)


def choice_token_loss(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    choices: Sequence[str],
    target_choice: str,
):
    """Differentiable strict-MC loss matching renormalized `score_choices` semantics."""
    torch = _import_torch()
    normalized_choices = [str(choice or "").strip().upper() for choice in choices if str(choice or "").strip()]
    target = str(target_choice or "").strip().upper()
    if target not in normalized_choices:
        raise ValueError(f"target_choice={target_choice!r} is not in choices={normalized_choices!r}.")

    input_ids, attention_mask = _resolve_model_inputs(
        tokenizer,
        messages,
        _model_device(model),
        add_generation_prompt=True,
    )
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    log_probs = torch.log_softmax(out.logits[0, -1], dim=-1)
    log_masses = {}
    for choice in normalized_choices:
        token_ids = _choice_token_ids(tokenizer, choice)
        log_mass = _choice_logmass(log_probs, token_ids)
        if log_mass is None:
            raise ValueError(f"Choice {choice!r} has no single-token realization for this tokenizer.")
        log_masses[choice] = log_mass
    denominator = torch.logsumexp(torch.stack([log_masses[choice] for choice in normalized_choices]), dim=0)
    return -(log_masses[target] - denominator)


def choice_token_probabilities(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    choices: Sequence[str],
) -> Dict[str, float]:
    torch = _import_torch()
    with torch.no_grad():
        normalized_choices = [str(choice or "").strip().upper() for choice in choices if str(choice or "").strip()]
        input_ids, attention_mask = _resolve_model_inputs(
            tokenizer,
            messages,
            _model_device(model),
            add_generation_prompt=True,
        )
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        probs = torch.softmax(out.logits[0, -1], dim=-1)
        raw: Dict[str, float] = {}
        for choice in normalized_choices:
            mass = 0.0
            for token_id in _choice_token_ids(tokenizer, choice):
                mass += float(probs[token_id].item())
            raw[choice] = mass
        total = float(sum(raw.values()))
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("choice_token_probabilities produced zero candidate mass.")
        return {choice: raw[choice] / total for choice in normalized_choices}


def completion_nll_loss(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    completion: str,
):
    torch = _import_torch()
    completion_text = str(completion or "").strip()
    if not completion_text:
        raise ValueError("completion_nll_loss requires a non-empty completion.")
    msgs = list(messages) + [{"type": "assistant", "content": completion_text}]
    input_ids_list = _token_id_list_from_encoded(
        encode_chat(tokenizer, msgs, add_generation_prompt=False),
        device=_model_device(model),
    )
    start, end = _assistant_text_span(tokenizer, input_ids_list, completion_text)
    if end <= start:
        raise ValueError("Could not locate assistant completion span in encoded chat.")
    input_tensor = torch.tensor([input_ids_list], device=_model_device(model))
    attention_mask = torch.ones_like(input_tensor)
    out = model(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    losses = []
    for idx in range(start, end):
        if idx == 0:
            continue
        token_id = int(input_ids_list[idx])
        log_probs = torch.log_softmax(out.logits[0, idx - 1], dim=-1)
        losses.append(-log_probs[token_id])
    if not losses:
        raise ValueError("Completion span has no scored tokens.")
    return torch.stack(losses).mean()


def loss_for_example(model: Any, tokenizer: Any, example: Mapping[str, Any]):
    loss_type = str(example.get("loss_type", "choice_token") or "choice_token")
    if loss_type == "choice_token":
        return choice_token_loss(
            model,
            tokenizer,
            list(example["messages"]),
            choices=list(example["choices"]),
            target_choice=str(example["target_choice"]),
        )
    if loss_type == "completion_nll":
        return completion_nll_loss(
            model,
            tokenizer,
            list(example["messages"]),
            str(example["completion"]),
        )
    raise ValueError(f"Unsupported pruning loss_type={loss_type!r}.")


__all__ = [
    "choice_token_loss",
    "choice_token_probabilities",
    "completion_nll_loss",
    "loss_for_example",
]
