from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..llm.generation import _token_id_list_from_encoded, to_hf_chat


class ChoiceSequenceScoringError(RuntimeError):
    """Raised when exact candidate-sequence scoring cannot be authenticated."""


def _import_torch():
    import torch

    return torch


def normalized_probabilities_from_log_likelihoods(
    log_likelihoods: Mapping[str, float],
) -> Dict[str, float]:
    if len(log_likelihoods) < 2:
        raise ChoiceSequenceScoringError("At least two candidate sequences are required")
    values = {str(key): float(value) for key, value in log_likelihoods.items()}
    if len(values) != len(log_likelihoods) or any(not math.isfinite(value) for value in values.values()):
        raise ChoiceSequenceScoringError("Candidate log likelihoods must be unique and finite")
    maximum = max(values.values())
    masses = {key: math.exp(value - maximum) for key, value in values.items()}
    denominator = sum(masses.values())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ChoiceSequenceScoringError("Candidate likelihood normalization failed")
    return {key: value / denominator for key, value in masses.items()}


def _offsets(value: Any) -> Sequence[Tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], (list, tuple)):
        value = value[0]
    return [(int(start), int(end)) for start, end in value]


@dataclass(frozen=True)
class EncodedCandidate:
    candidate: str
    input_ids: Tuple[int, ...]
    response_start: int
    prompt_text: str

    @property
    def response_token_count(self) -> int:
        return len(self.input_ids) - self.response_start


def encode_candidate_sequence(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    candidate: str,
    *,
    max_length: int = 4096,
) -> EncodedCandidate:
    candidate = str(candidate)
    if not candidate:
        raise ChoiceSequenceScoringError("Candidate sequence cannot be empty")
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise ChoiceSequenceScoringError("A model-native chat template is required")
    hf_messages = to_hf_chat(messages)
    try:
        prompt_text = apply_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_template_ids = _token_id_list_from_encoded(
            apply_template(
                hf_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        )
    except Exception as exc:
        raise ChoiceSequenceScoringError(f"Failed to render deployed chat template: {exc}") from exc
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ChoiceSequenceScoringError("Chat template returned an empty/non-text prompt")
    full_text = prompt_text + candidate
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception as exc:
        raise ChoiceSequenceScoringError(
            "Exact sequence scoring requires a fast tokenizer with offset mappings"
        ) from exc
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise ChoiceSequenceScoringError("Tokenizer omitted input IDs or offset mappings")
    input_ids = tuple(_token_id_list_from_encoded(encoded["input_ids"]))
    offsets = _offsets(encoded["offset_mapping"])
    if len(input_ids) != len(offsets):
        raise ChoiceSequenceScoringError("Tokenizer returned inconsistent IDs and offsets")
    boundary = len(prompt_text)
    crossing = [index for index, (start, end) in enumerate(offsets) if start < boundary < end]
    if crossing:
        raise ChoiceSequenceScoringError(
            "A token crosses the prompt/candidate boundary; candidate spacing must be explicit"
        )
    response_indices = [
        index for index, (start, end) in enumerate(offsets) if start >= boundary and end > boundary
    ]
    if not response_indices or response_indices != list(range(response_indices[0], len(input_ids))):
        raise ChoiceSequenceScoringError("Candidate tokens are not a non-empty contiguous suffix")
    response_start = response_indices[0]
    # The tokenize=True and tokenize=False routes must represent the identical
    # deployed prompt before we trust offsets from the text route.
    if list(input_ids[:response_start]) != list(prompt_template_ids):
        raise ChoiceSequenceScoringError(
            "Chat-template text/token serialization parity failed before the candidate boundary"
        )
    if len(input_ids) > int(max_length):
        raise ChoiceSequenceScoringError(
            f"Sequence length {len(input_ids)} exceeds max_length={max_length}; truncation is forbidden"
        )
    special_ids = {int(value) for value in (getattr(tokenizer, "all_special_ids", None) or [])}
    scored_specials = sorted(set(input_ids[response_start:]).intersection(special_ids))
    if scored_specials:
        raise ChoiceSequenceScoringError(
            f"Candidate response contains special/control token IDs {scored_specials}"
        )
    return EncodedCandidate(
        candidate=candidate,
        input_ids=input_ids,
        response_start=response_start,
        prompt_text=prompt_text,
    )


def score_option_sequences(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    *,
    max_length: int = 4096,
    residual_layer: Optional[int] = None,
    addition_vector: Any = None,
) -> Mapping[str, Any]:
    """Score every complete label sequence and renormalize across candidates."""

    labels = tuple(str(value) for value in candidates)
    if len(labels) < 2 or len(set(labels)) != len(labels) or any(not value for value in labels):
        raise ChoiceSequenceScoringError("Candidates must be at least two unique non-empty strings")
    torch = _import_torch()
    if (residual_layer is None) != (addition_vector is None):
        raise ChoiceSequenceScoringError(
            "residual_layer and addition_vector must either both be set or both be absent"
        )
    audits: Dict[str, Any] = {}
    log_likelihoods: Dict[str, float] = {}
    for label in labels:
        encoded = encode_candidate_sequence(
            tokenizer,
            messages,
            label,
            max_length=max_length,
        )
        input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids)
        hook = nullcontext()
        if residual_layer is not None:
            from ..interventions.activations import residual_addition_hook

            vector = torch.as_tensor(addition_vector, dtype=torch.float32)
            hook = residual_addition_hook(
                model,
                residual_layer=int(residual_layer),
                addition_vectors=vector,
                token_index=encoded.response_start - 1,
            )
        with torch.no_grad(), hook:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = output.logits[0].float()
            total = 0.0
            token_log_likelihoods = []
            for position in range(encoded.response_start, len(encoded.input_ids)):
                if position <= 0:
                    raise ChoiceSequenceScoringError("Candidate cannot begin at the first token")
                token_id = int(encoded.input_ids[position])
                token_logp = float(torch.log_softmax(logits[position - 1], dim=-1)[token_id].item())
                if not math.isfinite(token_logp):
                    raise ChoiceSequenceScoringError("Model produced a non-finite candidate log likelihood")
                total += token_logp
                token_log_likelihoods.append(token_logp)
        log_likelihoods[label] = total
        audits[label] = {
            "response_token_count": encoded.response_token_count,
            "log_likelihood": total,
            "mean_token_log_likelihood": total / encoded.response_token_count,
            "token_log_likelihoods": token_log_likelihoods,
        }
    probabilities = normalized_probabilities_from_log_likelihoods(log_likelihoods)
    return {
        "method": "complete_candidate_sequence_log_likelihood",
        "intervention_site": (
            "post_block_residual_final_rendered_prompt_token"
            if residual_layer is not None
            else "none"
        ),
        "sequence_length_normalization": "sum",
        "choice_probabilities": probabilities,
        "candidate_log_likelihoods": log_likelihoods,
        "candidate_audit": audits,
    }


__all__ = [
    "ChoiceSequenceScoringError",
    "EncodedCandidate",
    "encode_candidate_sequence",
    "normalized_probabilities_from_log_likelihoods",
    "score_option_sequences",
]
