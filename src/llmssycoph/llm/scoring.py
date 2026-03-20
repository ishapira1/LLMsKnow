from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..probes.features import _assistant_text_span
from .generation import _resolve_model_inputs, _token_id_list_from_encoded, encode_chat


_CHOICE_VARIANT_TEXT_TEMPLATES = ("{choice}", " {choice}", "\n{choice}")


def _import_torch():
    import torch

    return torch


def score_logprob_answer(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    answer: str,
) -> Tuple[float, float]:
    torch = _import_torch()
    with torch.no_grad():
        msgs = list(messages) + [{"type": "assistant", "content": answer}]
        input_ids = _token_id_list_from_encoded(
            encode_chat(tokenizer, msgs, add_generation_prompt=False),
            device=model.device,
        )

        start, end = _assistant_text_span(tokenizer, input_ids, answer)
        ans_ids = input_ids[start:end]

        input_tensor = torch.tensor([input_ids], device=model.device)
        out = model(input_tensor, use_cache=False, output_hidden_states=False, return_dict=True)
        logits = out.logits[0]

        logp = 0.0
        count = 0
        for idx in range(start, start + len(ans_ids)):
            if idx == 0:
                continue
            token_id = input_ids[idx]
            token_logp = torch.log_softmax(logits[idx - 1], dim=-1)[token_id].item()
            logp += token_logp
            count += 1

        if count == 0:
            return float("-inf"), float("-inf")

        return logp, logp / count


def _normalize_choices(choices: List[str]) -> List[str]:
    return [str(choice or "").strip() for choice in choices if str(choice or "").strip()]


def _choice_variant_texts(choice: str) -> List[str]:
    normalized_choice = str(choice or "").strip()
    if not normalized_choice:
        return []
    return [
        template.format(choice=normalized_choice)
        for template in _CHOICE_VARIANT_TEXT_TEMPLATES
    ]


def _decode_token_ids(tokenizer, token_ids: List[int]) -> Optional[str]:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode) or not token_ids:
        return None

    try:
        return str(decode(token_ids, skip_special_tokens=False))
    except TypeError:
        try:
            return str(decode(token_ids))
        except Exception:
            return None
    except Exception:
        return None


def _choice_variant_metadata(tokenizer, choice: str) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for variant_text in _choice_variant_texts(choice):
        ids = tokenizer(variant_text, add_special_tokens=False).input_ids
        token_ids = [int(token_id) for token_id in ids]
        variant: Dict[str, Any] = {
            "variant_text": variant_text,
            "token_ids": token_ids,
            "single_token": len(token_ids) == 1,
        }
        if len(token_ids) == 1:
            variant["token_id"] = token_ids[0]
            decoded_token = _decode_token_ids(tokenizer, token_ids)
            if decoded_token is not None:
                variant["decoded_token"] = decoded_token
        variants.append(variant)
    return variants


def score_p_true(
    model,
    tokenizer,
    question: str,
    proposed_answer: str,
) -> float:
    verify_messages = [
        {"type": "human", "content": "Your job is to evaluate if a proposed answer to a question is correct."},
        {
            "type": "human",
            "content": (
                f"Question: {question}\nProposed Answer: {proposed_answer}\n\n"
                "Is the proposed answer:\nA: CORRECT\nB: INCORRECT\n"
                "Just return the letter A or B."
            ),
        },
    ]
    try:
        return float(
            audit_choice_tokenization(
                model=model,
                tokenizer=tokenizer,
                messages=verify_messages,
                choices=["A", "B"],
            )["choice_probabilities"]["A"]
        )
    except (RuntimeError, ValueError):
        return float("nan")


def _choice_scoring_details(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    choices: List[str],
) -> Dict[str, Any]:
    torch = _import_torch()
    normalized_choices = _normalize_choices(choices)
    if not normalized_choices:
        raise ValueError("score_choices requires at least one non-empty choice.")

    choice_variants = {
        choice: _choice_variant_metadata(tokenizer, choice)
        for choice in normalized_choices
    }
    missing = [
        choice
        for choice, variants in choice_variants.items()
        if not any(bool(variant.get("single_token")) for variant in variants)
    ]
    if missing:
        raise ValueError(
            "score_choices requires every choice to have at least one single-token realization. "
            f"Missing choices: {missing}"
        )

    with torch.no_grad():
        input_ids, attention_mask = _resolve_model_inputs(
            tokenizer,
            messages,
            model.device,
            add_generation_prompt=True,
        )
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        next_logits = out.logits[0, -1]
        next_probs = torch.softmax(next_logits, dim=-1)

        raw_choice_probs: Dict[str, float] = {}
        choice_audit: Dict[str, Dict[str, Any]] = {}
        for choice in normalized_choices:
            counted_token_ids = set()
            raw_probability = 0.0
            variants: List[Dict[str, Any]] = []
            for variant in choice_variants[choice]:
                variant_entry = dict(variant)
                token_id = variant_entry.get("token_id")
                if variant_entry.get("single_token") and token_id is not None:
                    token_probability = float(next_probs[token_id].item())
                    variant_entry["logit"] = float(next_logits[token_id].item())
                    variant_entry["probability"] = token_probability
                    counted = int(token_id) not in counted_token_ids
                    variant_entry["counted_in_choice_probability"] = counted
                    if counted:
                        raw_probability += token_probability
                        counted_token_ids.add(int(token_id))
                else:
                    variant_entry["counted_in_choice_probability"] = False
                variants.append(variant_entry)

            raw_choice_probs[choice] = raw_probability
            choice_audit[choice] = {
                "variants": variants,
                "counted_token_ids": sorted(counted_token_ids),
                "raw_probability": raw_probability,
            }

    total_mass = float(sum(raw_choice_probs.values()))
    if total_mass <= 0.0:
        raise RuntimeError("score_choices produced zero probability mass over the provided choices.")

    choice_probabilities = {
        choice: raw_choice_probs[choice] / total_mass
        for choice in normalized_choices
    }
    for choice in normalized_choices:
        choice_audit[choice]["renormalized_probability"] = choice_probabilities[choice]

    prompt_token_ids = _token_id_list_from_encoded(input_ids[0])
    return {
        "choice_probabilities": choice_probabilities,
        "choices": choice_audit,
        "prompt_token_count": len(prompt_token_ids),
        "prompt_last_token_id": prompt_token_ids[-1] if prompt_token_ids else None,
        "prompt_tail_text": _decode_token_ids(tokenizer, prompt_token_ids[-8:]) if prompt_token_ids else None,
    }


def audit_choice_tokenization(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    choices: List[str],
) -> Dict[str, Any]:
    return _choice_scoring_details(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        choices=choices,
    )


def score_choices(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    choices: List[str],
) -> Dict[str, float]:
    return dict(
        _choice_scoring_details(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            choices=choices,
        )["choice_probabilities"]
    )


__all__ = [
    "audit_choice_tokenization",
    "score_choices",
    "score_logprob_answer",
    "score_p_true",
]
