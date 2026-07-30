from __future__ import annotations

from contextlib import ExitStack, contextmanager
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
    choice_token_ids: Dict[str, tuple[int, ...]]
    prompt_token_count: int
    prompt_token_ids: tuple[int, ...] = ()
    final_token_id: int | None = None
    final_token_text: str = ""


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
    token_index: int | Sequence[int] = -1,
    token_mask: Any | None = None,
) -> Iterator[None]:
    """Add one vector per batch row at one token or a weighted token mask."""

    if token_mask is not None and token_index != -1:
        raise ValueError("Specify token_index or token_mask, not both.")

    block = block_for_residual_layer(model, residual_layer)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        import torch

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
        # An enabled zero hook must be an exact no-op: return the original output
        # object rather than cloning or adding representable zeros.
        if bool((vectors == 0).all().item()):
            return output
        if token_mask is not None:
            mask = torch.as_tensor(
                token_mask,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and hidden.shape[0] > 1:
                mask = mask.expand(hidden.shape[0], -1)
            if mask.shape != hidden.shape[:2]:
                raise ValueError(
                    "Residual token masks must have shape [batch, sequence]; "
                    f"got {tuple(mask.shape)} for hidden {tuple(hidden.shape)}."
                )
            if not bool(torch.isfinite(mask).all().item()):
                raise FloatingPointError("Nonfinite residual token-mask weights.")
            if bool((mask == 0).all().item()):
                return output
            modified = hidden.clone()
            modified = modified + mask.unsqueeze(-1) * vectors.unsqueeze(1)
            return _replace_first_output(output, modified)
        if isinstance(token_index, Sequence) and not isinstance(token_index, (str, bytes)):
            indices = torch.as_tensor(
                [int(value) for value in token_index],
                dtype=torch.long,
                device=hidden.device,
            )
            if indices.shape != (hidden.shape[0],):
                raise ValueError(
                    "Per-example token indices must have shape [batch], "
                    f"got {tuple(indices.shape)} for batch={hidden.shape[0]}."
                )
        else:
            indices = torch.full(
                (hidden.shape[0],),
                int(token_index),
                dtype=torch.long,
                device=hidden.device,
            )
        indices = torch.where(indices < 0, indices + hidden.shape[1], indices)
        if bool(((indices < 0) | (indices >= hidden.shape[1])).any().item()):
            raise IndexError(
                f"Residual token indices {indices.tolist()} outside sequence length={hidden.shape[1]}."
            )
        modified = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        modified[rows, indices, :] = modified[rows, indices, :] + vectors
        return _replace_first_output(output, modified)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def residual_additions_hooks(
    model: Any,
    *,
    additions_by_layer: Mapping[int, Any],
    token_index: int | Sequence[int] = -1,
    token_mask: Any | None = None,
) -> Iterator[None]:
    """Install residual additions at several layers for the same forward pass.

    Each layer receives its own vector, while the selected prompt position(s)
    are shared. ``ExitStack`` guarantees that every hook is removed even when
    model execution fails partway through the forward pass.
    """

    layers = sorted(int(layer) for layer in additions_by_layer)
    if not layers:
        raise ValueError("additions_by_layer must contain at least one layer.")
    if len(layers) != len(additions_by_layer):
        raise ValueError("Residual layer keys must be unique integers.")
    with ExitStack() as stack:
        for layer in layers:
            stack.enter_context(
                residual_addition_hook(
                    model,
                    residual_layer=layer,
                    addition_vectors=additions_by_layer[layer],
                    token_index=token_index,
                    token_mask=token_mask,
                )
            )
        yield


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


@contextmanager
def residual_generation_addition_hook(
    model: Any,
    *,
    residual_layer: int,
    addition_vector: Any,
    mode: str,
) -> Iterator[None]:
    """Steer only the initial prompt boundary or every autoregressive step."""

    if mode not in {"final_prompt_only", "all_generation_tokens"}:
        raise ValueError(f"Unknown generation steering mode {mode!r}.")
    block = block_for_residual_layer(model, residual_layer)
    calls = 0

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        nonlocal calls
        calls += 1
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if getattr(hidden, "ndim", None) != 3:
            raise ValueError(
                "Expected generation hidden state [batch, sequence, hidden], "
                f"got {getattr(hidden, 'shape', None)}."
            )
        if mode == "final_prompt_only" and calls > 1:
            return output
        vectors = addition_vector.to(device=hidden.device, dtype=hidden.dtype)
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.shape[0] == 1 and hidden.shape[0] > 1:
            vectors = vectors.expand(hidden.shape[0], -1)
        if vectors.shape != (hidden.shape[0], hidden.shape[-1]):
            raise ValueError(
                f"Generation vector shape={tuple(vectors.shape)} hidden={tuple(hidden.shape)}."
            )
        if bool((vectors == 0).all().item()):
            return output
        modified = hidden.clone()
        modified[:, -1, :] = modified[:, -1, :] + vectors
        return _replace_first_output(output, modified)

    handle = block.register_forward_hook(hook)
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
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("Non-finite next-token logits in strict-choice scoring.")
    groups = _choice_token_groups(tokenizer, choices)
    ordered_choices = list(groups)
    per_choice_scores = []
    for choice in ordered_choices:
        token_index = torch.tensor(groups[choice], dtype=torch.long, device=logits.device)
        per_choice_scores.append(torch.logsumexp(logits.index_select(-1, token_index).float(), dim=-1))
    score_matrix = torch.stack(per_choice_scores, dim=-1)
    probability_matrix = torch.softmax(score_matrix, dim=-1)
    if not bool(torch.isfinite(score_matrix).all().item()):
        raise FloatingPointError("Non-finite option log scores in strict-choice scoring.")
    if not bool(torch.isfinite(probability_matrix).all().item()):
        raise FloatingPointError("Non-finite option probabilities in strict-choice scoring.")

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
            choice_token_ids={
                choice: tuple(token_ids)
                for choice, token_ids in _choice_token_groups(
                    tokenizer,
                    choices,
                ).items()
            },
            prompt_token_count=int(input_ids.shape[1]),
            prompt_token_ids=tuple(int(value) for value in input_ids[0].detach().cpu().tolist()),
            final_token_id=int(input_ids[0, -1].item()),
            final_token_text=str(
                tokenizer.decode([int(input_ids[0, -1].item())], skip_special_tokens=False)
            ),
        )


def score_repeated_prompt_without_hook(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    choices: Sequence[str],
    batch_size: int,
) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Score identical prompts in one batch without registering any hook."""

    import torch

    size = int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be at least one.")
    input_ids_base, attention_mask_base = _resolve_model_inputs(
        tokenizer,
        messages,
        model_device(model),
        add_generation_prompt=True,
    )
    input_ids = input_ids_base.expand(size, -1).contiguous()
    attention_mask = attention_mask_base.expand(size, -1).contiguous()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    return choice_distributions_from_logits(
        outputs.logits[:, -1],
        tokenizer=tokenizer,
        choices=choices,
    )


def resolve_prompt_suffix_mask(
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    neutral_messages: List[Dict[str, Any]],
    condition: str,
) -> Dict[str, Any]:
    """Resolve the framing/instruction suffix through exact rendered offsets."""

    from ..llm.generation import _token_id_list_from_encoded, to_hf_chat

    def user_content(items: List[Dict[str, Any]]) -> str:
        human = [
            str(item.get("content", ""))
            for item in items
            if str(item.get("type", "")) == "human"
        ]
        if len(human) != 1 or not human[0]:
            raise ValueError("Suffix steering requires exactly one nonempty human message.")
        return human[0]

    content = user_content(messages)
    neutral_content = user_content(neutral_messages)
    instruction_marker = "Use plain text answer-only, with no JSON and no tool schema."
    neutral_instruction_start = neutral_content.find(instruction_marker)
    if neutral_instruction_start < 0:
        raise ValueError("Could not locate the frozen answer-only instruction.")
    if str(condition) == "neutral":
        content_start = neutral_instruction_start
    else:
        question_prefix = neutral_content[:neutral_instruction_start].rstrip()
        expected_prefix = question_prefix + "\n\n"
        if not content.startswith(expected_prefix):
            raise ValueError(
                "Condition prompt does not preserve the neutral question prefix."
            )
        content_instruction_start = content.find(instruction_marker)
        if content_instruction_start <= len(expected_prefix):
            raise ValueError("Condition prompt has no framing insertion before instruction.")
        if content[content_instruction_start:] != neutral_content[neutral_instruction_start:]:
            raise ValueError(
                "Condition and neutral answer-only instruction suffixes differ."
            )
        content_start = len(expected_prefix)

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("Tokenizer does not expose apply_chat_template().")
    hf_messages = to_hf_chat(messages)
    rendered = str(
        apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    rendered_ids = _token_id_list_from_encoded(
        apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
    )
    encoded_offsets = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offset_ids = _token_id_list_from_encoded(encoded_offsets["input_ids"])
    offsets_raw = encoded_offsets["offset_mapping"]
    if hasattr(offsets_raw, "tolist"):
        offsets_raw = offsets_raw.tolist()
    if offsets_raw and isinstance(offsets_raw[0], list) and len(offsets_raw) == 1:
        offsets_raw = offsets_raw[0]
    offsets = [(int(start), int(end)) for start, end in offsets_raw]
    if offset_ids != rendered_ids:
        raise ValueError(
            "Rendered-text offset tokenization does not exactly reproduce chat-template "
            "token IDs."
        )
    if len(offsets) != len(rendered_ids):
        raise ValueError("Offset count does not equal rendered prompt token count.")
    content_occurrences = [
        index
        for index in range(len(rendered))
        if rendered.startswith(content, index)
    ]
    if len(content_occurrences) != 1:
        raise ValueError(
            f"Expected the human content once in rendered chat, found {len(content_occurrences)}."
        )
    rendered_start = content_occurrences[0] + content_start
    candidates = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > rendered_start and start <= rendered_start
    ]
    if not candidates:
        candidates = [
            index for index, (_start, end) in enumerate(offsets) if end > rendered_start
        ]
    if not candidates:
        raise ValueError("No rendered token overlaps the requested suffix start.")
    suffix_start = int(candidates[0])
    suffix_end = len(rendered_ids) - 1
    if suffix_start > suffix_end:
        raise ValueError("Suffix start occurs after the assistant boundary.")
    mask = np.zeros(len(rendered_ids), dtype=np.float32)
    mask[suffix_start : suffix_end + 1] = 1.0
    return {
        "token_mask": mask,
        "suffix_start_index": suffix_start,
        "suffix_end_index": suffix_end,
        "suffix_token_count": int(mask.sum()),
        "prompt_token_count": len(rendered_ids),
        "prompt_token_ids": rendered_ids,
        "rendered_suffix_start_offset": int(rendered_start),
        "rendered_text": rendered,
    }


def candidate_feature_with_prompt_steering(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    completion: str,
    feature_layer: int,
    steering_layer: int,
    addition_vector: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Read a teacher-forced candidate feature while steering the prompt boundary.

    The prompt encoding with ``add_generation_prompt=True`` must be an exact
    prefix of the prompt-plus-assistant encoding. This makes the intervention
    token auditable and prevents accidentally steering the candidate token.
    """

    import torch

    from ..llm.generation import _token_id_list_from_encoded, encode_chat
    from ..probes.features import _assistant_text_last_token_index

    prompt_ids, _ = _resolve_model_inputs(
        tokenizer,
        messages,
        model_device(model),
        add_generation_prompt=True,
    )
    prompt_token_ids = [int(value) for value in prompt_ids[0].detach().cpu().tolist()]
    full_messages = list(messages) + [{"type": "assistant", "content": str(completion)}]
    full_token_ids = _token_id_list_from_encoded(
        encode_chat(tokenizer, full_messages, add_generation_prompt=False),
        device=model_device(model),
    )
    if full_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise ValueError(
            "Prompt-plus-candidate chat encoding does not preserve the exact generation-prompt "
            "prefix; prompt-boundary steering would be ambiguous."
        )
    prompt_boundary_index = len(prompt_token_ids) - 1
    candidate_index = _assistant_text_last_token_index(
        tokenizer,
        full_token_ids,
        str(completion),
    )
    if candidate_index <= prompt_boundary_index:
        raise ValueError(
            "Candidate feature token does not occur after the prompt boundary: "
            f"boundary={prompt_boundary_index} candidate={candidate_index}."
        )
    input_ids = torch.tensor([full_token_ids], dtype=torch.long, device=model_device(model))
    attention_mask = torch.ones_like(input_ids)
    vector = torch.as_tensor(np.asarray(addition_vector, dtype=np.float32))
    with torch.no_grad():
        with residual_addition_hook(
            model,
            residual_layer=int(steering_layer),
            addition_vectors=vector,
            token_index=prompt_boundary_index,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
    feature = (
        outputs.hidden_states[int(feature_layer)][0, candidate_index]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    return feature, {
        "prompt_token_count": len(prompt_token_ids),
        "full_token_count": len(full_token_ids),
        "prompt_boundary_index": int(prompt_boundary_index),
        "prompt_boundary_token_id": int(full_token_ids[prompt_boundary_index]),
        "candidate_feature_index": int(candidate_index),
        "candidate_feature_token_id": int(full_token_ids[candidate_index]),
    }


def generate_with_residual_addition(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    choices: Sequence[str],
    residual_layer: int,
    addition_vector: np.ndarray,
    mode: str,
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    """Greedily generate text under an auditable prompt/generation-token hook."""

    import re
    import torch

    input_ids, attention_mask = _resolve_model_inputs(
        tokenizer,
        messages,
        model_device(model),
        add_generation_prompt=True,
    )
    vector = torch.as_tensor(np.asarray(addition_vector, dtype=np.float32))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": pad_token_id,
    }
    with torch.no_grad():
        with residual_generation_addition_hook(
            model,
            residual_layer=int(residual_layer),
            addition_vector=vector,
            mode=mode,
        ):
            generated = model.generate(**generation_kwargs)
    sequences = generated.sequences
    generated_ids = sequences[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = str(tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
    allowed = "".join(str(choice) for choice in choices)
    match = re.fullmatch(
        rf"\s*(?:answer\s*:\s*)?\(?([{re.escape(allowed)}])(?:\)|\])?[\s\].,:;\-]*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    normalized_choice = match.group(1).upper() if match else ""
    token_count = len(generated_ids)
    unique_fraction = (
        len(set(int(value) for value in generated_ids)) / token_count
        if token_count
        else 0.0
    )
    repeated_bigram_fraction = 0.0
    if token_count >= 2:
        bigrams = list(zip(generated_ids[:-1], generated_ids[1:]))
        repeated_bigram_fraction = 1.0 - len(set(bigrams)) / len(bigrams)
    score_tensors = list(getattr(generated, "scores", ()) or ())
    nonfinite = any(
        not bool(torch.isfinite(score).all().item()) for score in score_tensors
    )
    return {
        "scoring_mode": "free_generation",
        "generation_steering_mode": mode,
        "generated_text": text,
        "generated_token_ids": [int(value) for value in generated_ids],
        "generated_token_count": token_count,
        "parsed_option": normalized_choice,
        "valid_answer": bool(normalized_choice),
        "answer_format_failure": not bool(normalized_choice),
        "unique_token_fraction": float(unique_fraction),
        "repeated_bigram_fraction": float(repeated_bigram_fraction),
        "repetition_failure": bool(repeated_bigram_fraction > 0.5),
        "collapse_failure": bool(
            token_count >= 4
            and (unique_fraction < 0.25 or repeated_bigram_fraction > 0.75)
        ),
        "nonfinite_failure": bool(nonfinite),
        "hit_max_new_tokens": token_count >= int(max_new_tokens),
    }


def completion_nll_with_prompt_steering(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    completion: str,
    residual_layer: int,
    addition_vector: np.ndarray,
) -> Dict[str, Any]:
    """Teacher-forced completion NLL with steering only at the prompt boundary."""

    import torch
    import torch.nn.functional as functional

    from ..llm.generation import _token_id_list_from_encoded, encode_chat
    from ..probes.features import _assistant_text_span

    prompt_ids, _ = _resolve_model_inputs(
        tokenizer,
        messages,
        model_device(model),
        add_generation_prompt=True,
    )
    prompt_token_ids = [int(value) for value in prompt_ids[0].detach().cpu().tolist()]
    full_messages = list(messages) + [{"type": "assistant", "content": str(completion)}]
    full_token_ids = _token_id_list_from_encoded(
        encode_chat(tokenizer, full_messages, add_generation_prompt=False),
        device=model_device(model),
    )
    if full_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise ValueError(
            "Completion scoring does not preserve the exact generation-prompt prefix."
        )
    start, end = _assistant_text_span(tokenizer, full_token_ids, str(completion))
    if start < len(prompt_token_ids) or end <= start:
        raise ValueError(
            f"Invalid assistant target span start={start} end={end} "
            f"prompt_tokens={len(prompt_token_ids)}."
        )
    input_ids = torch.tensor([full_token_ids], dtype=torch.long, device=model_device(model))
    attention_mask = torch.ones_like(input_ids)
    vector = torch.as_tensor(np.asarray(addition_vector, dtype=np.float32))
    with torch.no_grad():
        with residual_addition_hook(
            model,
            residual_layer=int(residual_layer),
            addition_vectors=vector,
            token_index=len(prompt_token_ids) - 1,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    prediction_logits = outputs.logits[0, start - 1 : end - 1].float()
    target_ids = input_ids[0, start:end]
    if not bool(torch.isfinite(prediction_logits).all().item()):
        raise FloatingPointError("Non-finite logits in steered completion NLL.")
    losses = functional.cross_entropy(
        prediction_logits,
        target_ids,
        reduction="none",
    )
    mean_nll = float(losses.mean().item())
    return {
        "target_token_count": int(target_ids.numel()),
        "target_mean_nll": mean_nll,
        "target_perplexity": float(np.exp(min(mean_nll, 80.0))),
        "prompt_boundary_index": len(prompt_token_ids) - 1,
        "target_start_index": int(start),
        "target_end_index": int(end),
    }


def score_with_residual_additions(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    choices: Sequence[str],
    residual_layer: int,
    addition_vectors: np.ndarray,
    token_masks: np.ndarray | None = None,
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
    masks_np: np.ndarray | None = None
    if token_masks is not None:
        masks_np = np.asarray(token_masks, dtype=np.float32)
        if masks_np.ndim == 1:
            masks_np = masks_np[None, :]
        if masks_np.shape[0] == 1 and vectors_np.shape[0] > 1:
            masks_np = np.broadcast_to(
                masks_np, (vectors_np.shape[0], masks_np.shape[1])
            ).copy()
        expected = (vectors_np.shape[0], int(input_ids_base.shape[1]))
        if masks_np.shape != expected:
            raise ValueError(
                f"token_masks must have shape {expected}, got {masks_np.shape}."
            )
        if not np.isfinite(masks_np).all():
            raise FloatingPointError("Nonfinite token-mask weights.")
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
                token_index=-1 if masks_np is None else -1,
                token_mask=(
                    None
                    if masks_np is None
                    else torch.as_tensor(masks_np[start:stop], dtype=torch.float32)
                ),
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


def score_with_multilayer_residual_additions(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    choices: Sequence[str],
    addition_vectors_by_layer: Mapping[int, np.ndarray],
    token_masks: np.ndarray | None = None,
    max_batch_size: int | None = None,
) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Score one prompt while adding layer-specific vectors simultaneously."""

    import torch

    if not addition_vectors_by_layer:
        raise ValueError("addition_vectors_by_layer must not be empty.")
    vectors_by_layer: Dict[int, np.ndarray] = {}
    intervention_count: int | None = None
    for raw_layer, raw_vectors in addition_vectors_by_layer.items():
        layer = int(raw_layer)
        vectors = np.asarray(raw_vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or vectors.shape[0] <= 0:
            raise ValueError(
                f"Layer {layer} additions must have shape [n_interventions, hidden]."
            )
        if not np.isfinite(vectors).all():
            raise FloatingPointError(f"Nonfinite residual additions at layer {layer}.")
        if intervention_count is None:
            intervention_count = int(vectors.shape[0])
        elif int(vectors.shape[0]) != intervention_count:
            raise ValueError("Every layer must provide the same intervention count.")
        vectors_by_layer[layer] = vectors
    assert intervention_count is not None

    input_ids_base, attention_mask_base = _resolve_model_inputs(
        tokenizer,
        messages,
        model_device(model),
        add_generation_prompt=True,
    )
    masks_np: np.ndarray | None = None
    if token_masks is not None:
        masks_np = np.asarray(token_masks, dtype=np.float32)
        if masks_np.ndim == 1:
            masks_np = masks_np[None, :]
        if masks_np.shape[0] == 1 and intervention_count > 1:
            masks_np = np.broadcast_to(
                masks_np, (intervention_count, masks_np.shape[1])
            ).copy()
        expected = (intervention_count, int(input_ids_base.shape[1]))
        if masks_np.shape != expected:
            raise ValueError(
                f"token_masks must have shape {expected}, got {masks_np.shape}."
            )
        if not np.isfinite(masks_np).all():
            raise FloatingPointError("Nonfinite token-mask weights.")

    chunk_size = max(1, int(max_batch_size or intervention_count))
    all_probabilities: List[Dict[str, float]] = []
    all_log_scores: List[Dict[str, float]] = []
    with torch.no_grad():
        for start in range(0, intervention_count, chunk_size):
            stop = min(intervention_count, start + chunk_size)
            additions = {
                layer: torch.as_tensor(vectors[start:stop], dtype=torch.float32)
                for layer, vectors in vectors_by_layer.items()
            }
            batch_size = stop - start
            input_ids = input_ids_base.expand(batch_size, -1).contiguous()
            attention_mask = attention_mask_base.expand(batch_size, -1).contiguous()
            mask = (
                None
                if masks_np is None
                else torch.as_tensor(masks_np[start:stop], dtype=torch.float32)
            )
            with residual_additions_hooks(
                model,
                additions_by_layer=additions,
                token_index=-1,
                token_mask=mask,
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
