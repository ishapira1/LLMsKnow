from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Sequence

import numpy as np

from ..llm.generation import _resolve_model_inputs
from ..llm.scoring import _choice_variant_metadata


@dataclass(frozen=True)
class PromptState:
    """Pre-answer residual states and next-choice probabilities for one prompt."""

    hidden_by_layer: Dict[int, np.ndarray]
    choice_probabilities: Dict[str, float]
    choice_log_scores: Dict[str, float]
    prompt_token_count: int


def model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError("Could not infer the model device.") from exc


def resolve_transformer_blocks(model: Any) -> Sequence[Any]:
    """Resolve decoder blocks for the HF causal-LM families used by this project."""

    candidates = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
        except AttributeError:
            continue
        if hasattr(value, "__len__") and hasattr(value, "__getitem__") and len(value) > 0:
            return value
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        decoder = get_decoder()
        value = getattr(decoder, "layers", None)
        if value is not None and hasattr(value, "__len__") and len(value) > 0:
            return value
    raise TypeError(
        "Could not resolve transformer decoder blocks. Expected one of "
        "model.layers, model.decoder.layers, transformer.h, or gpt_neox.layers."
    )


def residual_layer_count(model: Any) -> int:
    """Number of post-block residual states (hidden_states indices 1..N)."""

    return int(len(resolve_transformer_blocks(model)))


def resolve_final_norm(model: Any) -> Any:
    """Resolve the final decoder normalization used for hidden_states[-1]."""

    candidates = (
        ("model", "norm"),
        ("model", "decoder", "final_layer_norm"),
        ("model", "decoder", "layer_norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    )
    for path in candidates:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
        except AttributeError:
            continue
        if hasattr(value, "register_forward_hook"):
            return value
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        decoder = get_decoder()
        for name in ("norm", "final_layer_norm", "layer_norm"):
            value = getattr(decoder, name, None)
            if value is not None and hasattr(value, "register_forward_hook"):
                return value
    raise TypeError(
        "Could not resolve the model's final decoder normalization. The final "
        "hidden-state layer cannot be intervened on safely for this architecture."
    )


def block_for_residual_layer(model: Any, residual_layer: int) -> Any:
    """Map an HF ``hidden_states`` index to its exact producing module.

    The probes use ``outputs.hidden_states[layer]``. In Hugging Face decoder-only
    models, hidden state 0 is the embedding output, indices 1..N-1 are decoder
    block outputs, and index N is the final-normalized last-block output. Using
    the final norm for N avoids silently applying a vector at a different site.
    """

    blocks = resolve_transformer_blocks(model)
    layer = int(residual_layer)
    if layer < 1 or layer > len(blocks):
        raise ValueError(f"Residual layer must be in [1, {len(blocks)}], got {layer}.")
    if layer == len(blocks):
        return resolve_final_norm(model)
    return blocks[layer - 1]


def _replace_first_output(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    if hasattr(output, "last_hidden_state"):
        raise TypeError(
            "Decoder block returned a model-output object rather than a tensor/tuple; "
            "add an explicit adapter before using this model for interventions."
        )
    return hidden


@contextmanager
def residual_addition_hook(
    model: Any,
    *,
    residual_layer: int,
    addition_vectors: Any,
    token_index: int = -1,
) -> Iterator[None]:
    """Add one residual vector per batch row at a specific sequence position."""

    block = block_for_residual_layer(model, residual_layer)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if getattr(hidden, "ndim", None) != 3:
            raise ValueError(
                "Expected decoder hidden state with shape [batch, sequence, hidden], "
                f"got {getattr(hidden, 'shape', None)}."
            )
        vectors = addition_vectors.to(device=hidden.device, dtype=hidden.dtype)
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.shape[0] == 1 and hidden.shape[0] > 1:
            vectors = vectors.expand(hidden.shape[0], -1)
        if vectors.shape != (hidden.shape[0], hidden.shape[-1]):
            raise ValueError(
                "Residual additions must have shape [batch, hidden]; "
                f"got {tuple(vectors.shape)} for hidden {tuple(hidden.shape)}."
            )
        modified = hidden.clone()
        modified[:, int(token_index), :] = modified[:, int(token_index), :] + vectors
        return _replace_first_output(output, modified)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def residual_replacement_hook(
    model: Any,
    *,
    residual_layer: int,
    replacement_vectors: Any,
    token_index: int = -1,
) -> Iterator[None]:
    """Replace one residual vector per batch row at a specific sequence position."""

    module = block_for_residual_layer(model, residual_layer)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if getattr(hidden, "ndim", None) != 3:
            raise ValueError(
                "Expected decoder hidden state with shape [batch, sequence, hidden], "
                f"got {getattr(hidden, 'shape', None)}."
            )
        vectors = replacement_vectors.to(device=hidden.device, dtype=hidden.dtype)
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.shape[0] == 1 and hidden.shape[0] > 1:
            vectors = vectors.expand(hidden.shape[0], -1)
        if vectors.shape != (hidden.shape[0], hidden.shape[-1]):
            raise ValueError(
                "Residual replacements must have shape [batch, hidden]; "
                f"got {tuple(vectors.shape)} for hidden {tuple(hidden.shape)}."
            )
        modified = hidden.clone()
        modified[:, int(token_index), :] = vectors
        return _replace_first_output(output, modified)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _choice_token_groups(tokenizer: Any, choices: Sequence[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for raw_choice in choices:
        choice = str(raw_choice or "").strip()
        if not choice:
            continue
        token_ids: List[int] = []
        for variant in _choice_variant_metadata(tokenizer, choice):
            if not bool(variant.get("single_token")):
                continue
            token_id = variant.get("token_id")
            if token_id is not None and int(token_id) not in token_ids:
                token_ids.append(int(token_id))
        if not token_ids:
            raise ValueError(
                f"Choice {choice!r} has no single-token realization under the tokenizer."
            )
        groups[choice] = token_ids
    if not groups:
        raise ValueError("At least one non-empty answer choice is required.")
    return groups


def choice_distributions_from_logits(
    next_token_logits: Any,
    *,
    tokenizer: Any,
    choices: Sequence[str],
) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Renormalize next-token logits over allowed strict-MC choice realizations."""

    import torch

    logits = next_token_logits
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    groups = _choice_token_groups(tokenizer, choices)
    ordered_choices = list(groups)
    per_choice_scores = []
    for choice in ordered_choices:
        token_index = torch.tensor(groups[choice], dtype=torch.long, device=logits.device)
        per_choice_scores.append(torch.logsumexp(logits.index_select(-1, token_index).float(), dim=-1))
    score_matrix = torch.stack(per_choice_scores, dim=-1)
    probability_matrix = torch.softmax(score_matrix, dim=-1)

    probability_rows: List[Dict[str, float]] = []
    log_score_rows: List[Dict[str, float]] = []
    for row_idx in range(score_matrix.shape[0]):
        probability_rows.append(
            {
                choice: float(probability_matrix[row_idx, col_idx].item())
                for col_idx, choice in enumerate(ordered_choices)
            }
        )
        log_score_rows.append(
            {
                choice: float(score_matrix[row_idx, col_idx].item())
                for col_idx, choice in enumerate(ordered_choices)
            }
        )
    return probability_rows, log_score_rows


def extract_prompt_state(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    choices: Sequence[str],
    residual_layers: Sequence[int],
) -> PromptState:
    """Read the final-prompt-token state before the first answer token."""

    import torch

    requested_layers = sorted({int(layer) for layer in residual_layers})
    n_layers = residual_layer_count(model)
    invalid = [layer for layer in requested_layers if layer < 1 or layer > n_layers]
    if invalid:
        raise ValueError(f"Invalid residual layers {invalid}; model has layers 1..{n_layers}.")

    with torch.no_grad():
        input_ids, attention_mask = _resolve_model_inputs(
            tokenizer,
            messages,
            model_device(model),
            add_generation_prompt=True,
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        probability_rows, log_score_rows = choice_distributions_from_logits(
            outputs.logits[0, -1],
            tokenizer=tokenizer,
            choices=choices,
        )
        hidden_by_layer = {
            layer: outputs.hidden_states[layer][0, -1].detach().float().cpu().numpy()
            for layer in requested_layers
        }
        return PromptState(
            hidden_by_layer=hidden_by_layer,
            choice_probabilities=probability_rows[0],
            choice_log_scores=log_score_rows[0],
            prompt_token_count=int(input_ids.shape[1]),
        )


def score_with_residual_additions(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    choices: Sequence[str],
    residual_layer: int,
    addition_vectors: np.ndarray,
    max_batch_size: int | None = None,
) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Score one prompt under a batch of pre-answer residual additions."""

    import torch

    vectors_np = np.asarray(addition_vectors, dtype=np.float32)
    if vectors_np.ndim == 1:
        vectors_np = vectors_np[None, :]
    if vectors_np.ndim != 2 or vectors_np.shape[0] <= 0:
        raise ValueError("addition_vectors must have shape [n_interventions, hidden].")

    input_ids_base, attention_mask_base = _resolve_model_inputs(
        tokenizer,
        messages,
        model_device(model),
        add_generation_prompt=True,
    )
    chunk_size = max(1, int(max_batch_size or vectors_np.shape[0]))
    all_probabilities: List[Dict[str, float]] = []
    all_log_scores: List[Dict[str, float]] = []
    with torch.no_grad():
        for start in range(0, vectors_np.shape[0], chunk_size):
            stop = min(vectors_np.shape[0], start + chunk_size)
            chunk = torch.as_tensor(vectors_np[start:stop], dtype=torch.float32)
            batch_size = int(chunk.shape[0])
            input_ids = input_ids_base.expand(batch_size, -1).contiguous()
            attention_mask = attention_mask_base.expand(batch_size, -1).contiguous()
            with residual_addition_hook(
                model,
                residual_layer=int(residual_layer),
                addition_vectors=chunk,
                token_index=-1,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            probabilities, log_scores = choice_distributions_from_logits(
                outputs.logits[:, -1],
                tokenizer=tokenizer,
                choices=choices,
            )
            all_probabilities.extend(probabilities)
            all_log_scores.extend(log_scores)
    return all_probabilities, all_log_scores


def score_with_residual_replacements(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    choices: Sequence[str],
    residual_layer: int,
    replacement_vectors: np.ndarray,
    max_batch_size: int | None = None,
) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Score one prompt under exact final-token residual replacements."""

    import torch

    vectors_np = np.asarray(replacement_vectors, dtype=np.float32)
    if vectors_np.ndim == 1:
        vectors_np = vectors_np[None, :]
    if vectors_np.ndim != 2 or vectors_np.shape[0] <= 0:
        raise ValueError("replacement_vectors must have shape [n_interventions, hidden].")
    input_ids_base, attention_mask_base = _resolve_model_inputs(
        tokenizer,
        messages,
        model_device(model),
        add_generation_prompt=True,
    )
    chunk_size = max(1, int(max_batch_size or vectors_np.shape[0]))
    all_probabilities: List[Dict[str, float]] = []
    all_log_scores: List[Dict[str, float]] = []
    with torch.no_grad():
        for start in range(0, vectors_np.shape[0], chunk_size):
            stop = min(vectors_np.shape[0], start + chunk_size)
            chunk = torch.as_tensor(vectors_np[start:stop], dtype=torch.float32)
            batch_size = int(chunk.shape[0])
            input_ids = input_ids_base.expand(batch_size, -1).contiguous()
            attention_mask = attention_mask_base.expand(batch_size, -1).contiguous()
            with residual_replacement_hook(
                model,
                residual_layer=int(residual_layer),
                replacement_vectors=chunk,
                token_index=-1,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            probabilities, log_scores = choice_distributions_from_logits(
                outputs.logits[:, -1],
                tokenizer=tokenizer,
                choices=choices,
            )
            all_probabilities.extend(probabilities)
            all_log_scores.extend(log_scores)
    return all_probabilities, all_log_scores


def top_choice(probabilities: Mapping[str, float]) -> str:
    if not probabilities:
        return ""
    return max(probabilities, key=lambda choice: float(probabilities[choice]))


__all__ = [
    "PromptState",
    "block_for_residual_layer",
    "choice_distributions_from_logits",
    "extract_prompt_state",
    "model_device",
    "residual_addition_hook",
    "residual_layer_count",
    "residual_replacement_hook",
    "resolve_transformer_blocks",
    "resolve_final_norm",
    "score_with_residual_additions",
    "score_with_residual_replacements",
    "top_choice",
]
