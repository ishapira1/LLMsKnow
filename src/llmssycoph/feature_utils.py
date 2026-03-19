from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .model_utils import _resolve_model_inputs, _token_id_list_from_encoded, encode_chat
from .probes.features import (
    _assistant_text_last_token_index,
    _assistant_text_span,
    find_sublist as _find_sublist,
    get_hidden_feature_for_answer,
    get_hidden_feature_for_completion,
)


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


__all__ = [
    "_find_sublist",
    "get_hidden_feature_for_answer",
    "get_hidden_feature_for_completion",
    "score_logprob_answer",
    "score_p_true",
]
