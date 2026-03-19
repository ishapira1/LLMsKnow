from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..llm.generation import _token_id_list_from_encoded, encode_chat


def _import_torch():
    import torch

    return torch


def find_sublist(hay: List[int], needle: List[int]) -> Optional[int]:
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


def get_hidden_feature_for_answer(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    answer: str,
    layer: int,
) -> np.ndarray:
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
    torch = _import_torch()
    with torch.no_grad():
        msgs = list(messages) + [{'type': 'assistant', 'content': completion}]
        ids = _token_id_list_from_encoded(
            encode_chat(tokenizer, msgs, add_generation_prompt=False),
            device=model.device,
        )
        last_idx = _assistant_text_last_token_index(tokenizer, ids, completion)

        input_tensor = torch.tensor([ids], device=model.device)
        out = model(input_tensor, use_cache=False, output_hidden_states=True, return_dict=True)

        hidden_states = out.hidden_states[layer]
        return hidden_states[0, last_idx].detach().float().cpu().numpy()


def get_hidden_feature_all_layers_for_completion(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    completion: str,
    layer_grid: Sequence[int],
) -> np.ndarray:
    torch = _import_torch()
    with torch.no_grad():
        msgs = list(messages) + [{'type': 'assistant', 'content': completion}]
        ids = _token_id_list_from_encoded(
            encode_chat(tokenizer, msgs, add_generation_prompt=False),
            device=model.device,
        )
        last_idx = _assistant_text_last_token_index(tokenizer, ids, completion)

        input_tensor = torch.tensor([ids], device=model.device)
        out = model(input_tensor, use_cache=False, output_hidden_states=True, return_dict=True)

        vecs = []
        for layer in layer_grid:
            hidden_states = out.hidden_states[layer]
            vecs.append(hidden_states[0, last_idx].detach().float().cpu().numpy())
        return np.stack(vecs, axis=0)


def get_hidden_feature_all_layers_for_answer(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    answer: str,
    layer_grid: Sequence[int],
) -> np.ndarray:
    return get_hidden_feature_all_layers_for_completion(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        completion=answer,
        layer_grid=layer_grid,
    )


__all__ = [
    'find_sublist',
    'get_hidden_feature_all_layers_for_answer',
    'get_hidden_feature_all_layers_for_completion',
    'get_hidden_feature_for_answer',
    'get_hidden_feature_for_completion',
]
