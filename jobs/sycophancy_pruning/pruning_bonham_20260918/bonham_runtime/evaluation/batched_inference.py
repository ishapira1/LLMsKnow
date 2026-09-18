"""Deterministic multi-prompt generation and full-option sequence batching."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ..llm.base import GenerationResult
from ..llm.generation import (
    _decode_generation_metadata,
    _resolve_model_inputs,
)

from .choice_scoring import (
    ChoiceSequenceScoringError,
    encode_candidate_sequence,
    normalized_probabilities_from_log_likelihoods,
)


BATCHED_INFERENCE_VERSION = "deterministic_multi_prompt_full_sequence_v1"


class BatchedInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationRequest:
    messages: Sequence[Mapping[str, Any]]
    max_new_tokens: int
    tools: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class ChoiceRequest:
    messages: Sequence[Mapping[str, Any]]
    candidates: Sequence[str]


def _model_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise BatchedInferenceError("Could not resolve a non-meta model input device")


def generate_message_batch(
    llm: Any,
    requests: Sequence[GenerationRequest],
    *,
    batch_size: int,
    residual_layer: Optional[int] = None,
    addition_vector: Any = None,
    require_batched: bool = True,
) -> Tuple[Sequence[GenerationResult], Mapping[str, Any]]:
    """Greedily generate for distinct prompts using left-padded batches."""

    rows = tuple(requests)
    if not rows:
        return (), {
            "version": BATCHED_INFERENCE_VERSION,
            "request_count": 0,
            "batched_forward_count": 0,
            "fallback_count": 0,
        }
    width = int(batch_size)
    if width <= 1 and require_batched and len(rows) > 1:
        raise BatchedInferenceError("Production multi-prompt generation requires batch_size > 1")
    if (residual_layer is None) != (addition_vector is None):
        raise BatchedInferenceError("Steering layer/vector must be supplied together")
    device = _model_input_device(llm.model)
    grouped: Dict[int, List[Tuple[int, GenerationRequest]]] = {}
    for index, request in enumerate(rows):
        maximum = int(request.max_new_tokens)
        if maximum <= 0:
            raise BatchedInferenceError("max_new_tokens must be positive")
        grouped.setdefault(maximum, []).append((index, request))
    results: List[Optional[GenerationResult]] = [None] * len(rows)
    batched_forwards = 0
    max_batch_observed = 0
    started = time.monotonic()
    for max_new_tokens, group in sorted(grouped.items()):
        for offset in range(0, len(group), max(1, width)):
            chunk = group[offset : offset + max(1, width)]
            encoded = [
                _resolve_model_inputs(
                    llm.tokenizer,
                    request.messages,
                    device,
                    add_generation_prompt=True,
                    tools=request.tools or None,
                )
                for _index, request in chunk
            ]
            lengths = [int(ids.shape[1]) for ids, _mask in encoded]
            maximum_prompt = max(lengths)
            pad_id = getattr(llm.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(llm.tokenizer, "eos_token_id", None)
            if pad_id is None:
                raise BatchedInferenceError("Tokenizer needs pad_token_id or eos_token_id")
            input_ids = torch.full(
                (len(chunk), maximum_prompt),
                int(pad_id),
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, ((ids, mask), length) in enumerate(zip(encoded, lengths)):
                input_ids[row_index, maximum_prompt - length :] = ids[0]
                attention_mask[row_index, maximum_prompt - length :] = mask[0]
            hook = nullcontext()
            if residual_layer is not None:
                from ..interventions.activations import residual_generation_addition_hook

                hook = residual_generation_addition_hook(
                    llm.model,
                    residual_layer=int(residual_layer),
                    addition_vector=torch.as_tensor(addition_vector, dtype=torch.float32),
                    mode="final_prompt_only",
                )
            try:
                with torch.no_grad(), hook:
                    output = llm.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        return_dict_in_generate=True,
                        pad_token_id=int(pad_id),
                    )
            except Exception as exc:
                if require_batched:
                    raise BatchedInferenceError(
                        f"Required multi-prompt generation batch failed: {type(exc).__name__}: {exc}"
                    ) from exc
                raise
            if int(output.sequences.shape[0]) != len(chunk):
                raise BatchedInferenceError("Generation returned the wrong batch dimension")
            for row_index, (original_index, _request) in enumerate(chunk):
                generated_ids = output.sequences[row_index, maximum_prompt:]
                metadata = _decode_generation_metadata(
                    llm.tokenizer,
                    generated_ids,
                    max_new_tokens=max_new_tokens,
                    strict_mc_letters="",
                )
                results[original_index] = GenerationResult.from_output(metadata)
            batched_forwards += 1
            max_batch_observed = max(max_batch_observed, len(chunk))
    if any(value is None for value in results):
        raise BatchedInferenceError("Generation batch omitted a request")
    audit = {
        "version": BATCHED_INFERENCE_VERSION,
        "request_count": len(rows),
        "configured_batch_size": width,
        "max_batch_observed": max_batch_observed,
        "batched_forward_count": batched_forwards,
        "fallback_count": 0,
        "padding": "left",
        "greedy": True,
        "steering_prompt_boundary": "final_nonpadding_prompt_token",
        "wall_seconds": time.monotonic() - started,
    }
    return tuple(value for value in results if value is not None), audit


def score_option_sequence_batch(
    model: Any,
    tokenizer: Any,
    requests: Sequence[ChoiceRequest],
    *,
    batch_size: int,
    max_length: int = 4096,
    residual_layer: Optional[int] = None,
    addition_vector: Any = None,
    require_batched: bool = True,
) -> Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]:
    """Flatten every request's complete candidates into padded forward batches."""

    rows = tuple(requests)
    if not rows:
        return (), {
            "version": BATCHED_INFERENCE_VERSION,
            "request_count": 0,
            "candidate_count": 0,
            "batched_forward_count": 0,
            "fallback_count": 0,
        }
    width = int(batch_size)
    flattened = []
    labels_by_request = []
    for request_index, request in enumerate(rows):
        labels = tuple(str(value) for value in request.candidates)
        if len(labels) < 2 or len(set(labels)) != len(labels) or any(not value for value in labels):
            raise ChoiceSequenceScoringError(
                "Candidates must be at least two unique non-empty strings"
            )
        labels_by_request.append(labels)
        for label in labels:
            flattened.append(
                (
                    request_index,
                    label,
                    encode_candidate_sequence(
                        tokenizer, request.messages, label, max_length=max_length
                    ),
                )
            )
    if width <= 1 and require_batched and len(flattened) > 1:
        raise BatchedInferenceError("Production option scoring requires batch_size > 1")
    if (residual_layer is None) != (addition_vector is None):
        raise ChoiceSequenceScoringError(
            "residual_layer and addition_vector must either both be set or both be absent"
        )
    device = _model_input_device(model)
    logps: Dict[Tuple[int, str], Tuple[float, Sequence[float], int]] = {}
    forwards = 0
    max_batch_observed = 0
    started = time.monotonic()
    for offset in range(0, len(flattened), max(1, width)):
        chunk = flattened[offset : offset + max(1, width)]
        lengths = [len(encoded.input_ids) for _index, _label, encoded in chunk]
        maximum = max(lengths)
        input_ids = torch.zeros((len(chunk), maximum), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        boundary_indices = []
        for row_index, ((_request_index, _label, encoded), length) in enumerate(
            zip(chunk, lengths)
        ):
            input_ids[row_index, :length] = torch.tensor(
                encoded.input_ids, dtype=torch.long, device=device
            )
            attention_mask[row_index, :length] = 1
            boundary_indices.append(int(encoded.response_start) - 1)
        hook = nullcontext()
        if residual_layer is not None:
            from ..interventions.activations import residual_addition_hook

            vector = torch.as_tensor(addition_vector, dtype=torch.float32)
            hook = residual_addition_hook(
                model,
                residual_layer=int(residual_layer),
                addition_vectors=vector,
                token_index=boundary_indices,
            )
        try:
            with torch.no_grad(), hook:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        except Exception as exc:
            if require_batched:
                raise BatchedInferenceError(
                    f"Required option-sequence batch failed: {type(exc).__name__}: {exc}"
                ) from exc
            raise
        logits = output.logits.float()
        for row_index, (request_index, label, encoded) in enumerate(chunk):
            token_values = []
            for position in range(encoded.response_start, len(encoded.input_ids)):
                token_id = int(encoded.input_ids[position])
                value = float(
                    torch.log_softmax(logits[row_index, position - 1], dim=-1)[
                        token_id
                    ].item()
                )
                if not math.isfinite(value):
                    raise ChoiceSequenceScoringError(
                        "Model produced a non-finite candidate log likelihood"
                    )
                token_values.append(value)
            logps[(request_index, label)] = (
                float(sum(token_values)),
                token_values,
                encoded.response_token_count,
            )
        forwards += 1
        max_batch_observed = max(max_batch_observed, len(chunk))
    results = []
    for request_index, labels in enumerate(labels_by_request):
        likelihoods = {label: logps[(request_index, label)][0] for label in labels}
        candidate_audit = {
            label: {
                "response_token_count": logps[(request_index, label)][2],
                "log_likelihood": logps[(request_index, label)][0],
                "mean_token_log_likelihood": (
                    logps[(request_index, label)][0] / logps[(request_index, label)][2]
                ),
                "token_log_likelihoods": list(logps[(request_index, label)][1]),
            }
            for label in labels
        }
        results.append(
            {
                "method": "complete_candidate_sequence_log_likelihood",
                "intervention_site": (
                    "post_block_residual_final_rendered_prompt_token"
                    if residual_layer is not None
                    else "none"
                ),
                "sequence_length_normalization": "sum",
                "choice_probabilities": normalized_probabilities_from_log_likelihoods(
                    likelihoods
                ),
                "candidate_log_likelihoods": likelihoods,
                "candidate_audit": candidate_audit,
            }
        )
    audit = {
        "version": BATCHED_INFERENCE_VERSION,
        "request_count": len(rows),
        "candidate_count": len(flattened),
        "configured_batch_size": width,
        "max_batch_observed": max_batch_observed,
        "batched_forward_count": forwards,
        "fallback_count": 0,
        "padding": "right_with_attention_mask",
        "full_option_sequences": True,
        "per_row_steering_prompt_boundaries": residual_layer is not None,
        "wall_seconds": time.monotonic() - started,
    }
    return tuple(results), audit


__all__ = [
    "BATCHED_INFERENCE_VERSION",
    "BatchedInferenceError",
    "ChoiceRequest",
    "GenerationRequest",
    "generate_message_batch",
    "score_option_sequence_batch",
]
