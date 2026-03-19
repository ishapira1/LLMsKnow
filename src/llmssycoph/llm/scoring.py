from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..probes.features import _assistant_text_span
from .generation import _resolve_model_inputs, _token_id_list_from_encoded, encode_chat


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


def score_p_true(
    model,
    tokenizer,
    question: str,
    proposed_answer: str,
) -> float:
    torch = _import_torch()
    with torch.no_grad():
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
        input_ids, _ = _resolve_model_inputs(
            tokenizer,
            verify_messages,
            model.device,
            add_generation_prompt=True,
        )

        out = model(input_ids=input_ids, use_cache=False, output_hidden_states=False, return_dict=True)
        next_logits = out.logits[0, -1]

        candidate_texts = [" A", " B", "A", "B"]
        candidate_ids = []
        for candidate_text in candidate_texts:
            ids = tokenizer(candidate_text, add_special_tokens=False).input_ids
            if len(ids) == 1:
                candidate_ids.append((candidate_text.strip(), ids[0]))

        a_id = next((token_id for text, token_id in candidate_ids if text == "A"), None)
        b_id = next((token_id for text, token_id in candidate_ids if text == "B"), None)
        if a_id is None or b_id is None:
            probs = torch.softmax(next_logits, dim=-1)
            if a_id is None or b_id is None:
                return float("nan")
            return (probs[a_id] / (probs[a_id] + probs[b_id])).item()

        logits_ab = torch.stack([next_logits[a_id], next_logits[b_id]], dim=0)
        probs_ab = torch.softmax(logits_ab, dim=0)
        return probs_ab[0].item()


def _single_token_choice_ids(tokenizer, choice: str) -> List[int]:
    normalized_choice = str(choice or "").strip()
    if not normalized_choice:
        return []

    token_ids = []
    for candidate_text in (normalized_choice, f" {normalized_choice}"):
        ids = tokenizer(candidate_text, add_special_tokens=False).input_ids
        if len(ids) == 1:
            token_ids.append(int(ids[0]))
    return sorted(set(token_ids))


def score_choices(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    choices: List[str],
) -> Dict[str, float]:
    torch = _import_torch()
    normalized_choices = [str(choice or "").strip() for choice in choices if str(choice or "").strip()]
    if not normalized_choices:
        raise ValueError("score_choices requires at least one non-empty choice.")

    choice_token_ids = {
        choice: _single_token_choice_ids(tokenizer, choice)
        for choice in normalized_choices
    }
    missing = [choice for choice, token_ids in choice_token_ids.items() if not token_ids]
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
        for choice in normalized_choices:
            raw_choice_probs[choice] = float(
                sum(float(next_probs[token_id].item()) for token_id in choice_token_ids[choice])
            )

    total_mass = float(sum(raw_choice_probs.values()))
    if total_mass <= 0.0:
        raise RuntimeError("score_choices produced zero probability mass over the provided choices.")

    return {
        choice: raw_choice_probs[choice] / total_mass
        for choice in normalized_choices
    }


__all__ = ["score_choices", "score_logprob_answer", "score_p_true"]
