from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model_utils import encode_chat


def _import_torch():
    import torch

    return torch


def _find_sublist(hay: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(hay):
        return None
    for idx in range(len(hay) - len(needle) + 1):
        if hay[idx : idx + len(needle)] == needle:
            return idx
    return None


def _find_last_sublist(hay: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(hay):
        return None
    for idx in range(len(hay) - len(needle), -1, -1):
        if hay[idx : idx + len(needle)] == needle:
            return idx
    return None


def _assistant_text_span(
    tokenizer,
    full_ids: List[int],
    assistant_text: str,
) -> Tuple[int, int]:
    text_ids = tokenizer(assistant_text, add_special_tokens=False).input_ids
    start = _find_last_sublist(full_ids, text_ids)
    if start is None:
        last_idx = len(full_ids) - 1
        return last_idx, last_idx + 1
    return start, start + len(text_ids)


def _assistant_text_last_token_index(
    tokenizer,
    full_ids: List[int],
    assistant_text: str,
) -> int:
    _, end = _assistant_text_span(tokenizer, full_ids, assistant_text)
    return end - 1


def score_logprob_answer(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    answer: str,
) -> Tuple[float, float]:
    torch = _import_torch()
    with torch.no_grad():
        msgs = list(messages) + [{"type": "assistant", "content": answer}]
        input_ids = encode_chat(tokenizer, msgs, add_generation_prompt=False).to(model.device)[0].tolist()

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
        input_ids = encode_chat(tokenizer, verify_messages, add_generation_prompt=True).to(model.device)

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


def get_hidden_feature_for_answer(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    answer: str,
    layer: int,
) -> np.ndarray:
    """Return the hidden state at the last token of the provided answer span."""
    return get_hidden_feature_for_completion(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        completion=answer,
        layer=layer,
    )


def get_hidden_feature_for_completion(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    completion: str,
    layer: int,
) -> np.ndarray:
    """Return the hidden state at the last token of the provided full assistant completion."""
    torch = _import_torch()
    with torch.no_grad():
        msgs = list(messages) + [{"type": "assistant", "content": completion}]
        ids = encode_chat(tokenizer, msgs, add_generation_prompt=False).to(model.device)[0].tolist()
        last_idx = _assistant_text_last_token_index(tokenizer, ids, completion)

        input_tensor = torch.tensor([ids], device=model.device)
        out = model(input_tensor, use_cache=False, output_hidden_states=True, return_dict=True)

        hidden_states = out.hidden_states[layer]
        return hidden_states[0, last_idx].detach().float().cpu().numpy()


__all__ = [
    "_find_sublist",
    "get_hidden_feature_for_completion",
    "score_logprob_answer",
    "score_p_true",
    "get_hidden_feature_for_answer",
]
