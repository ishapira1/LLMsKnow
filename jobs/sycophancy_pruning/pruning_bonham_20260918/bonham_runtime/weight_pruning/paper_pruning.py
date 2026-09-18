"""Paper-faithful, manifest-driven attribution pruning.

This module intentionally lives beside the released pruning implementation rather
than silently changing it.  The ``attribution_score_set_difference_global`` CLI
path opts into this implementation whenever explicit JSONL manifests are given.

The important invariants are:

* one scalar, response-token-mean loss per manifest row;
* an equal-weight average over rows;
* FP32 accumulation of signed gradients;
* ``abs(w * mean_gradient)`` for preservation (abs *after* averaging); and
* exact global, rather than per-matrix, top-k selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


MANIFEST_ALIASES = {
    "raw_prompt": ("raw_prompt", "prompt", "prompt_text", "clean_prompt"),
    "messages": ("messages", "prompt_messages"),
    "target_text": ("target_text", "completion", "target", "response", "clean_response"),
    "target_letter": ("target_letter", "correct_letter", "answer_letter"),
    "choice_letters": ("choice_letters", "answer_letters", "allowed_letters"),
}


class ManifestError(ValueError):
    """Raised when a manifest cannot satisfy the scoring contract."""


@dataclass(frozen=True)
class EncodedCompletion:
    input_ids: torch.Tensor
    response_start: int
    rendered_prompt: str
    target_text: str


@dataclass(frozen=True)
class PreparedExample:
    record: Mapping[str, Any]
    completion: Optional[EncodedCompletion] = None
    choices: tuple[EncodedCompletion, ...] = ()
    target_choice_index: Optional[int] = None
    reference_choices: tuple[EncodedCompletion, ...] = ()
    choice_letters: tuple[str, ...] = ()
    target_choice_indices: tuple[int, ...] = ()
    correct_choice_index: Optional[int] = None
    objective_id: Optional[str] = None
    objective_scope: Optional[str] = None


CAUSAL_CHOICE_OBJECTIVES = (
    "target_vs_rest_log_odds",
    "correct_vs_rest_log_odds",
    "wrong_mass_vs_correct_log_odds",
)
CAUSAL_OBJECTIVE_SCOPES = ("raw_margin", "contextual_uptake")
CAUSAL_CHOICE_MARGIN_VERSION = "causal_choice_margin_v1"


def causal_manifest_objective_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a content-bound audit of every causal objective in a score shard.

    This deliberately hashes the *selected rows*, not just the manifest file.  A
    cache made with a different ``nsamples`` prefix, reference prompt, formula,
    or sign convention therefore cannot be reused accidentally.
    """

    objective_rows: list[dict[str, Any]] = []
    for row in rows:
        messages = _normalize_messages(
            _first(row, "messages"),
            path_hint=f"manifest line {row.get('_manifest_line', '?')}",
        )
        scope = str(row.get("objective_scope", "")).strip()
        reference_messages: list[dict[str, str]] = []
        if scope == "contextual_uptake":
            reference_messages = _normalize_messages(
                row.get("reference_messages"),
                path_hint=f"manifest line {row.get('_manifest_line', '?')} reference",
            )
        objective_rows.append(
            {
                "example_id": str(row.get("example_id", "")),
                "causal_pair_id": str(row.get("causal_pair_id", "")),
                "condition_id": str(row.get("condition_id", "")),
                "score_role": str(row.get("score_role", "")),
                "objective_id": str(row.get("objective_id", "")),
                "objective_scope": scope,
                "objective_formula": str(row.get("objective_formula", "")),
                "score_sign_convention": str(row.get("score_sign_convention", "")),
                "correct_letter": str(row.get("correct_letter", "")),
                "target_letter": row.get("target_letter"),
                "target_letters": list(row.get("target_letters", [])),
                "prompt_sha256": _canonical_json_sha256(messages),
                "reference_prompt_sha256": (
                    _canonical_json_sha256(reference_messages)
                    if reference_messages
                    else ""
                ),
                "reference_pair_id": str(row.get("reference_pair_id", "")),
            }
        )
    payload = {
        "version": CAUSAL_CHOICE_MARGIN_VERSION,
        "num_examples": len(objective_rows),
        "objective_ids": sorted({row["objective_id"] for row in objective_rows}),
        "objective_scopes": sorted({row["objective_scope"] for row in objective_rows}),
        "rows_sha256": _canonical_json_sha256(objective_rows),
    }
    payload["audit_sha256"] = _canonical_json_sha256(payload)
    return payload


def _first(record: Mapping[str, Any], canonical: str, default: Any = None) -> Any:
    for key in MANIFEST_ALIASES[canonical]:
        value = record.get(key)
        if value is not None:
            return value
    return default


def sha256_file(path: Union[str, os.PathLike[str]]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(
    path: Union[str, os.PathLike[str]],
    *,
    nsamples: Optional[int],
    expected_model: Optional[str] = None,
    expected_revision: Optional[str] = None,
    expected_tokenizer_revision: Optional[str] = None,
    expected_calibration_seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Load an ordered JSONL manifest and fail rather than silently shrinking it."""

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{path}:{line_number}: each line must be an object")
            row.setdefault("_manifest_line", line_number)
            rows.append(row)

    if nsamples is not None:
        if len(rows) < nsamples:
            raise ManifestError(
                f"{path}: requested {nsamples} examples but manifest contains only {len(rows)}"
            )
        rows = rows[:nsamples]
    if not rows:
        raise ManifestError(f"{path}: manifest is empty")

    for row in rows:
        line = row["_manifest_line"]
        model_id = row.get("model_id")
        if expected_model and model_id and model_id != expected_model:
            raise ManifestError(
                f"{path}:{line}: model_id={model_id!r} does not match --model={expected_model!r}"
            )
        revision_value = row.get("revision")
        legacy_revision = row.get("model_revision")
        if (
            revision_value is not None
            and legacy_revision is not None
            and revision_value != legacy_revision
        ):
            raise ManifestError(
                f"{path}:{line}: revision and legacy model_revision disagree"
            )
        model_revision = revision_value if revision_value is not None else legacy_revision
        if expected_revision and model_revision and model_revision != expected_revision:
            raise ManifestError(
                f"{path}:{line}: model_revision={model_revision!r} does not match "
                f"--revision={expected_revision!r}"
            )
        tokenizer_revision = row.get("tokenizer_revision")
        if (
            expected_tokenizer_revision
            and tokenizer_revision
            and tokenizer_revision != expected_tokenizer_revision
        ):
            raise ManifestError(
                f"{path}:{line}: tokenizer_revision={tokenizer_revision!r} does not match "
                f"the pinned tokenizer revision {expected_tokenizer_revision!r}"
            )
        if expected_calibration_seed is not None:
            calibration_seed = row.get("calibration_seed")
            if calibration_seed is None or int(calibration_seed) != int(
                expected_calibration_seed
            ):
                raise ManifestError(
                    f"{path}:{line}: calibration_seed={calibration_seed!r} does not match "
                    f"--seed={expected_calibration_seed}"
                )
    return rows


def _normalize_messages(value: Any, *, path_hint: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{path_hint}: messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ManifestError(f"{path_hint}: messages[{index}] must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ManifestError(
                f"{path_hint}: messages[{index}] needs a valid role and string content"
            )
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] == "assistant":
        raise ManifestError(
            f"{path_hint}: messages must contain prompt turns only; put the assistant answer in target_text"
        )
    return messages


def render_prompt(record: Mapping[str, Any], tokenizer: Any, score_format: str) -> str:
    hint = f"manifest line {record.get('_manifest_line', '?')}"
    if score_format == "raw":
        prompt = _first(record, "raw_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ManifestError(f"{hint}: raw scoring requires non-empty raw_prompt")
        return prompt
    if score_format != "chat":
        raise ValueError(f"unknown score_format: {score_format}")

    value = _first(record, "messages")
    if value is None:
        raw_prompt = _first(record, "raw_prompt")
        if not isinstance(raw_prompt, str) or not raw_prompt:
            raise ManifestError(f"{hint}: chat scoring requires messages or raw_prompt")
        value = [{"role": "user", "content": raw_prompt}]
    messages = _normalize_messages(value, path_hint=hint)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ManifestError(f"{hint}: tokenizer has no chat template support")
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise ManifestError(f"{hint}: failed to render chat template: {exc}") from exc
    if not isinstance(rendered, str) or not rendered:
        raise ManifestError(f"{hint}: chat template returned an empty prompt")
    return rendered


def _as_1d(values: Any) -> list[int]:
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _as_offsets(values: Any) -> list[tuple[int, int]]:
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if values and isinstance(values[0], list) and values[0] and isinstance(values[0][0], list):
        values = values[0]
    return [(int(start), int(end)) for start, end in values]


def encode_completion(
    record: Mapping[str, Any],
    tokenizer: Any,
    score_format: str,
    *,
    target_override: Optional[str] = None,
    max_length: Optional[int] = None,
    tokenization_mode: str = "full_string_offsets",
) -> EncodedCompletion:
    """Tokenize a completion and recover the response-token suffix.

    ``full_string_offsets`` is the paper-adapter contract: tokenize the complete
    serialized string once and locate the response with fast-tokenizer offsets.
    ``separate`` deliberately reproduces the released Hadas loader: tokenize the
    prompt and response in two independent calls and concatenate their IDs.
    """

    prompt = render_prompt(record, tokenizer, score_format)
    target = target_override if target_override is not None else _first(record, "target_text")
    if target is None:
        target = _first(record, "target_letter")
    if not isinstance(target, str) or not target:
        raise ManifestError(
            f"manifest line {record.get('_manifest_line', '?')}: missing non-empty target_text"
        )
    if tokenization_mode == "separate":
        separate_prompt = record.get("separate_prompt", prompt)
        separate_target = (
            target
            if target_override is not None
            else record.get("separate_target_text", target)
        )
        if not isinstance(separate_prompt, str) or not isinstance(
            separate_target, str
        ):
            raise ManifestError(
                f"manifest line {record.get('_manifest_line', '?')}: "
                "separate_prompt and separate_target_text must be strings"
            )
        if separate_prompt + separate_target != prompt + target:
            raise ManifestError(
                f"manifest line {record.get('_manifest_line', '?')}: separate and "
                "full-string tokenization must serialize identical complete text"
            )
        prompt_encoded = tokenizer(separate_prompt, add_special_tokens=False)
        target_encoded = tokenizer(separate_target, add_special_tokens=False)
        prompt_ids = _as_1d(prompt_encoded["input_ids"])
        target_ids = _as_1d(target_encoded["input_ids"])
        if not prompt_ids:
            raise ManifestError("prompt must contain at least one token before the response")
        if not target_ids:
            raise ManifestError(
                f"manifest line {record.get('_manifest_line', '?')}: target produced no response tokens"
            )
        special_ids = {
            int(token_id)
            for token_id in (getattr(tokenizer, "all_special_ids", None) or [])
        }
        scored_specials = [token_id for token_id in target_ids if token_id in special_ids]
        if scored_specials:
            raise ManifestError(
                f"manifest line {record.get('_manifest_line', '?')}: response span contains "
                f"special/control token IDs {sorted(set(scored_specials))}"
            )
        ids = prompt_ids + target_ids
        if max_length is not None and len(ids) > max_length:
            raise ManifestError(
                f"manifest line {record.get('_manifest_line', '?')}: sequence has {len(ids)} tokens, "
                f"exceeding --max_score_length={max_length}; truncation is not allowed"
            )
        return EncodedCompletion(
            input_ids=torch.tensor(ids, dtype=torch.long),
            response_start=len(prompt_ids),
            rendered_prompt=separate_prompt,
            target_text=separate_target,
        )
    if tokenization_mode != "full_string_offsets":
        raise ValueError(f"unknown tokenization_mode: {tokenization_mode}")

    full_text = prompt + target
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception as exc:
        raise ManifestError(
            "scoring requires a fast tokenizer with return_offsets_mapping support"
        ) from exc
    if "offset_mapping" not in encoded:
        raise ManifestError("tokenizer did not return offset_mapping; fail-closed response masking")
    ids = _as_1d(encoded["input_ids"])
    offsets = _as_offsets(encoded["offset_mapping"])
    if len(ids) != len(offsets):
        raise ManifestError("tokenizer returned inconsistent input_ids and offset_mapping lengths")
    boundary = len(prompt)
    response_indices = [
        index for index, (start, end) in enumerate(offsets) if end > boundary and start >= boundary
    ]
    crossing = [
        index for index, (start, end) in enumerate(offsets) if start < boundary < end
    ]
    if crossing:
        raise ManifestError(
            f"manifest line {record.get('_manifest_line', '?')}: a token crosses the prompt/target "
            "boundary; make spacing explicit in raw_prompt or target_text"
        )
    if not response_indices:
        raise ManifestError(
            f"manifest line {record.get('_manifest_line', '?')}: target produced no response tokens"
        )
    if response_indices != list(range(response_indices[0], len(ids))):
        raise ManifestError("response tokens are not a contiguous suffix")
    special_ids = {
        int(token_id) for token_id in (getattr(tokenizer, "all_special_ids", None) or [])
    }
    scored_specials = [
        ids[index] for index in response_indices if ids[index] in special_ids
    ]
    if scored_specials:
        raise ManifestError(
            f"manifest line {record.get('_manifest_line', '?')}: response span contains "
            f"special/control token IDs {sorted(set(scored_specials))}"
        )
    response_start = response_indices[0]
    if response_start == 0:
        raise ManifestError("prompt must contain at least one token before the response")
    if max_length is not None and len(ids) > max_length:
        raise ManifestError(
            f"manifest line {record.get('_manifest_line', '?')}: sequence has {len(ids)} tokens, "
            f"exceeding --max_score_length={max_length}; truncation is not allowed"
        )
    return EncodedCompletion(
        input_ids=torch.tensor(ids, dtype=torch.long),
        response_start=response_start,
        rendered_prompt=prompt,
        target_text=target,
    )


def _choice_letters(record: Mapping[str, Any]) -> list[str]:
    letters = _first(record, "choice_letters", [])
    if isinstance(letters, Mapping):
        letters = list(letters.keys())
    if not letters and isinstance(record.get("choices"), Mapping):
        letters = list(record["choices"].keys())
    if not isinstance(letters, Sequence) or isinstance(letters, (str, bytes)):
        raise ManifestError("choice_letters must be a list")
    result = [str(letter) for letter in letters]
    if len(result) < 2 or len(set(result)) != len(result):
        raise ManifestError("choice_token loss requires at least two unique choice_letters")
    return result


def _choice_target(record: Mapping[str, Any], letter: str, target_letter: str) -> str:
    mapping = record.get("choice_target_texts")
    if isinstance(mapping, Mapping) and letter in mapping:
        return str(mapping[letter])
    target_text = _first(record, "target_text")
    if isinstance(target_text, str) and target_text.endswith(target_letter):
        return target_text[: -len(target_letter)] + letter
    return letter


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _causal_choice_example(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    score_format: str,
    max_length: Optional[int],
    tokenization_mode: str,
) -> PreparedExample:
    hint = f"manifest line {record.get('_manifest_line', '?')}"
    if score_format != "chat":
        raise ManifestError(f"{hint}: causal_choice_margin requires score_format=chat")
    letters = _choice_letters(record)
    correct_letter = str(record.get("correct_letter", "")).strip().upper()
    objective_id = str(record.get("objective_id", "")).strip()
    objective_scope = str(record.get("objective_scope", "")).strip()
    objective_formula = str(record.get("objective_formula", "")).strip()
    if objective_id not in CAUSAL_CHOICE_OBJECTIVES:
        raise ManifestError(f"{hint}: unsupported causal objective_id={objective_id!r}")
    if objective_scope not in CAUSAL_OBJECTIVE_SCOPES:
        raise ManifestError(f"{hint}: invalid objective_scope={objective_scope!r}")
    if not objective_formula:
        raise ManifestError(f"{hint}: causal objective_formula must be explicit")
    if correct_letter not in letters:
        raise ManifestError(f"{hint}: correct_letter is absent from choice_letters")
    raw_targets = record.get("target_letters")
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
        raise ManifestError(f"{hint}: target_letters must be an explicit list")
    target_letters = tuple(str(value).strip().upper() for value in raw_targets)
    if not target_letters or len(set(target_letters)) != len(target_letters):
        raise ManifestError(f"{hint}: target_letters must be unique and non-empty")
    if any(value not in letters for value in target_letters):
        raise ManifestError(f"{hint}: target_letters contains an unknown option")
    target_letter = str(record.get("target_letter") or "").strip().upper() or None
    if objective_id in {"target_vs_rest_log_odds", "correct_vs_rest_log_odds"}:
        expected = correct_letter if objective_id == "correct_vs_rest_log_odds" else target_letter
        if expected is None or target_letters != (expected,):
            raise ManifestError(
                f"{hint}: {objective_id} requires exactly its designated target letter"
            )
    else:
        wrong = tuple(letter for letter in letters if letter != correct_letter)
        if target_letter is not None or set(target_letters) != set(wrong):
            raise ManifestError(
                f"{hint}: wrong_mass_vs_correct requires all and only wrong labels, with no target_letter"
            )
    messages = _normalize_messages(_first(record, "messages"), path_hint=hint)
    prompt_sha = str(record.get("prompt_sha256", "")).lower()
    if prompt_sha != _canonical_json_sha256(messages):
        raise ManifestError(f"{hint}: context prompt hash does not match messages")
    pair_id = str(record.get("causal_pair_id", "")).strip()
    if not pair_id:
        raise ManifestError(f"{hint}: causal_pair_id is required")
    context_record = dict(record)
    context_record["messages"] = messages
    context_choices = tuple(
        encode_completion(
            context_record,
            tokenizer,
            score_format,
            target_override=_choice_target(record, letter, target_letter or correct_letter),
            max_length=max_length,
            tokenization_mode=tokenization_mode,
        )
        for letter in letters
    )
    reference_choices: tuple[EncodedCompletion, ...] = ()
    if objective_scope == "contextual_uptake":
        reference = record.get("reference_messages")
        reference_messages = _normalize_messages(reference, path_hint=f"{hint} reference")
        reference_hash = str(record.get("reference_prompt_sha256", "")).lower()
        if reference_hash != _canonical_json_sha256(reference_messages):
            raise ManifestError(f"{hint}: reference prompt hash does not match reference_messages")
        if record.get("reference_condition_id") != "neutral":
            raise ManifestError(f"{hint}: contextual uptake reference must be the neutral condition")
        if str(record.get("reference_pair_id", "")) != pair_id:
            raise ManifestError(f"{hint}: context/reference pair identity mismatch")
        reference_record = dict(record)
        reference_record["messages"] = reference_messages
        reference_record["prompt_messages"] = reference_messages
        reference_choices = tuple(
            encode_completion(
                reference_record,
                tokenizer,
                score_format,
                target_override=_choice_target(record, letter, target_letter or correct_letter),
                max_length=max_length,
                tokenization_mode=tokenization_mode,
            )
            for letter in letters
        )
    elif any(
        record.get(field) not in (None, "", [])
        for field in ("reference_messages", "reference_prompt_sha256", "reference_pair_id")
    ):
        raise ManifestError(f"{hint}: raw_margin objective cannot carry a reference prompt")
    return PreparedExample(
        record=record,
        choices=context_choices,
        reference_choices=reference_choices,
        choice_letters=tuple(letters),
        target_choice_index=(
            letters.index(target_letter)
            if target_letter is not None
            else None
        ),
        target_choice_indices=tuple(letters.index(value) for value in target_letters),
        correct_choice_index=letters.index(correct_letter),
        objective_id=objective_id,
        objective_scope=objective_scope,
    )


def prepare_examples(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    score_format: str,
    loss_mode: str,
    max_length: Optional[int],
    tokenization_mode: str = "full_string_offsets",
) -> list[PreparedExample]:
    prepared: list[PreparedExample] = []
    for record in rows:
        letters_value = _first(record, "choice_letters")
        has_choice_letters = (
            isinstance(letters_value, Sequence)
            and not isinstance(letters_value, (str, bytes))
            and len(letters_value) > 0
        ) or (isinstance(letters_value, Mapping) and len(letters_value) > 0)
        choices_value = record.get("choices")
        has_choice_mapping = isinstance(choices_value, Mapping) and len(choices_value) > 0
        is_multiple_choice = has_choice_letters or has_choice_mapping
        if loss_mode == "causal_choice_margin":
            if not is_multiple_choice:
                raise ManifestError(
                    f"manifest line {record.get('_manifest_line', '?')}: "
                    "causal_choice_margin requires explicit answer choices"
                )
            prepared.append(
                _causal_choice_example(
                    record,
                    tokenizer,
                    score_format=score_format,
                    max_length=max_length,
                    tokenization_mode=tokenization_mode,
                )
            )
        elif loss_mode == "choice_token" and is_multiple_choice:
            letters = _choice_letters(record)
            target_letter = _first(record, "target_letter")
            if target_letter is None:
                target_text = _first(record, "target_text")
                target_letter = target_text.strip() if isinstance(target_text, str) else None
            target_letter = str(target_letter)
            if target_letter not in letters:
                raise ManifestError(
                    f"manifest line {record.get('_manifest_line', '?')}: target_letter "
                    f"{target_letter!r} is not in choice_letters"
                )
            choices = tuple(
                encode_completion(
                    record,
                    tokenizer,
                    score_format,
                    target_override=_choice_target(record, letter, target_letter),
                    max_length=max_length,
                    tokenization_mode=tokenization_mode,
                )
                for letter in letters
            )
            prepared.append(
                PreparedExample(
                    record=record,
                    choices=choices,
                    target_choice_index=letters.index(target_letter),
                )
            )
        else:
            prepared.append(
                PreparedExample(
                    record=record,
                    completion=encode_completion(
                        record,
                        tokenizer,
                        score_format,
                        max_length=max_length,
                        tokenization_mode=tokenization_mode,
                    ),
                )
            )
    return prepared


def completion_nll_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """Mean causal NLL over response tokens only."""

    response_logits = logits[:, response_start - 1 : -1, :]
    response_ids = input_ids[:, response_start:]
    if response_logits.shape[1] != response_ids.shape[1] or response_ids.numel() == 0:
        raise RuntimeError("invalid response span for causal loss")
    return F.cross_entropy(
        response_logits.reshape(-1, response_logits.shape[-1]),
        response_ids.reshape(-1),
        reduction="mean",
    )


def sequence_log_probability_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    response_logits = logits[:, response_start - 1 : -1, :]
    response_ids = input_ids[:, response_start:]
    log_probs = F.log_softmax(response_logits.float(), dim=-1)
    return log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1).sum()


def _input_device(model: nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def _forward_logits(model: nn.Module, encoded: EncodedCompletion) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = encoded.input_ids.unsqueeze(0).to(_input_device(model))
    output = model(input_ids=input_ids, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]
    return logits, input_ids


def causal_choice_margin_coefficients(
    logps: torch.Tensor,
    *,
    objective_id: str,
    target_indices: Sequence[int],
    correct_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact scalar margin and analytic dM/d(choice log-probability).

    ``logps`` are full option-sequence log probabilities. Candidate
    renormalization cancels in each registered log-odds objective, so these
    formulas are exact for both raw sequence scores and normalized option
    probabilities.
    """

    values = logps.float()
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise RuntimeError("causal choice margins require finite one-dimensional option logps")
    targets = tuple(int(value) for value in target_indices)
    correct = int(correct_index)
    if correct < 0 or correct >= len(values) or not targets or any(
        value < 0 or value >= len(values) for value in targets
    ):
        raise RuntimeError("invalid causal choice-margin indices")
    coefficients = torch.zeros_like(values)
    if objective_id in {"target_vs_rest_log_odds", "correct_vs_rest_log_odds"}:
        target = targets[0]
        if len(targets) != 1 or (
            objective_id == "correct_vs_rest_log_odds" and target != correct
        ):
            raise RuntimeError(f"invalid targets for {objective_id}")
        rest = [index for index in range(len(values)) if index != target]
        rest_tensor = torch.as_tensor(rest, dtype=torch.long, device=values.device)
        rest_logps = values.index_select(0, rest_tensor)
        rest_weights = torch.softmax(rest_logps, dim=0)
        coefficients[target] = 1.0
        coefficients[rest_tensor] = -rest_weights
        margin = values[target] - torch.logsumexp(rest_logps, dim=0)
    elif objective_id == "wrong_mass_vs_correct_log_odds":
        wrong = tuple(index for index in range(len(values)) if index != correct)
        if set(targets) != set(wrong):
            raise RuntimeError("wrong-mass target set must contain all and only wrong choices")
        wrong_tensor = torch.as_tensor(wrong, dtype=torch.long, device=values.device)
        wrong_logps = values.index_select(0, wrong_tensor)
        coefficients[wrong_tensor] = torch.softmax(wrong_logps, dim=0)
        coefficients[correct] = -1.0
        margin = torch.logsumexp(wrong_logps, dim=0) - values[correct]
    else:
        raise RuntimeError(f"unsupported causal objective {objective_id!r}")
    if not torch.isfinite(margin) or not torch.isfinite(coefficients).all():
        raise RuntimeError("non-finite causal choice margin or coefficient")
    return margin, coefficients


def _choice_logps_no_grad(model: nn.Module, choices: Sequence[EncodedCompletion]) -> torch.Tensor:
    with torch.no_grad():
        values = []
        for candidate in choices:
            logits, ids = _forward_logits(model, candidate)
            values.append(
                sequence_log_probability_from_logits(logits, ids, candidate.response_start)
            )
    return torch.stack(values).float()


def _batched_choice_logps(
    model: nn.Module,
    choices: Sequence[EncodedCompletion],
) -> torch.Tensor:
    """Score variable-length candidate sequences in one exact padded forward."""

    if not choices:
        raise RuntimeError("candidate batch is empty")
    device = _input_device(model)
    lengths = [int(choice.input_ids.numel()) for choice in choices]
    maximum = max(lengths)
    input_ids = torch.zeros((len(choices), maximum), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, choice in enumerate(choices):
        length = lengths[row]
        input_ids[row, :length] = choice.input_ids.to(device)
        attention_mask[row, :length] = 1
    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except TypeError:
        # Tiny test doubles may omit attention_mask. Right padding cannot affect
        # earlier causal positions; production Hugging Face models use the
        # explicit mask above.
        output = model(input_ids=input_ids, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]
    result = []
    for row, choice in enumerate(choices):
        length = lengths[row]
        response_logits = logits[row, choice.response_start - 1 : length - 1, :]
        response_ids = input_ids[row, choice.response_start:length]
        if response_logits.shape[0] != response_ids.numel() or response_ids.numel() == 0:
            raise RuntimeError("invalid batched response span")
        token_logps = F.log_softmax(response_logits.float(), dim=-1)
        result.append(
            token_logps.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1).sum()
        )
    return torch.stack(result)


def _backward_choice_linear_combination(
    model: nn.Module,
    choices: Sequence[EncodedCompletion],
    coefficients: torch.Tensor,
    *,
    sign: float,
) -> None:
    if len(choices) != len(coefficients):
        raise RuntimeError("choice/coefficient length mismatch")
    for index, candidate in enumerate(choices):
        coefficient = float(sign) * coefficients[index]
        if float(coefficient) == 0.0:
            continue
        logits, ids = _forward_logits(model, candidate)
        logp = sequence_log_probability_from_logits(logits, ids, candidate.response_start)
        (coefficient.to(logp.device) * logp).backward()


def _causal_example_value_and_coefficients(
    model: nn.Module,
    example: PreparedExample,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    context_logps = _choice_logps_no_grad(model, example.choices)
    context_value, context_coefficients = causal_choice_margin_coefficients(
        context_logps,
        objective_id=str(example.objective_id),
        target_indices=example.target_choice_indices,
        correct_index=int(example.correct_choice_index),
    )
    if example.objective_scope == "raw_margin":
        if example.reference_choices:
            raise RuntimeError("raw causal margin unexpectedly has reference choices")
        return context_value, context_coefficients, None
    if example.objective_scope != "contextual_uptake" or not example.reference_choices:
        raise RuntimeError("contextual uptake requires matched reference choices")
    reference_logps = _choice_logps_no_grad(model, example.reference_choices)
    reference_value, reference_coefficients = causal_choice_margin_coefficients(
        reference_logps,
        objective_id=str(example.objective_id),
        target_indices=example.target_choice_indices,
        correct_index=int(example.correct_choice_index),
    )
    return context_value - reference_value, context_coefficients, reference_coefficients


def _causal_example_value_batched(
    model: nn.Module,
    example: PreparedExample,
) -> torch.Tensor:
    combined = tuple(example.choices) + tuple(example.reference_choices)
    logps = _batched_choice_logps(model, combined)
    context_logps = logps[: len(example.choices)]
    context_value, _ = causal_choice_margin_coefficients(
        context_logps,
        objective_id=str(example.objective_id),
        target_indices=example.target_choice_indices,
        correct_index=int(example.correct_choice_index),
    )
    if example.objective_scope == "raw_margin":
        if example.reference_choices:
            raise RuntimeError("raw causal margin unexpectedly has reference choices")
        return context_value
    if example.objective_scope != "contextual_uptake" or not example.reference_choices:
        raise RuntimeError("contextual uptake requires matched reference choices")
    reference_logps = logps[len(example.choices) :]
    reference_value, _ = causal_choice_margin_coefficients(
        reference_logps,
        objective_id=str(example.objective_id),
        target_indices=example.target_choice_indices,
        correct_index=int(example.correct_choice_index),
    )
    return context_value - reference_value


def backward_example(
    model: nn.Module,
    example: PreparedExample,
    loss_mode: str,
    *,
    causal_execution_audit: Optional[dict[str, int]] = None,
    allow_causal_candidate_oom_fallback: bool = True,
) -> float:
    """Backpropagate one equal-weight example and return its detached scalar loss."""

    if loss_mode == "causal_choice_margin" and example.choices:
        try:
            value = _causal_example_value_batched(model, example)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or not torch.cuda.is_available():
                raise
            if causal_execution_audit is not None:
                causal_execution_audit["oom_fallback_examples"] = (
                    int(causal_execution_audit.get("oom_fallback_examples", 0)) + 1
                )
            if not allow_causal_candidate_oom_fallback:
                raise RuntimeError(
                    "causal candidate batching exhausted CUDA memory and production "
                    "requires zero sequential fallbacks"
                ) from exc
            torch.cuda.empty_cache()
            value, context_coefficients, reference_coefficients = (
                _causal_example_value_and_coefficients(model, example)
            )
            _backward_choice_linear_combination(
                model, example.choices, context_coefficients, sign=1.0
            )
            if reference_coefficients is not None:
                _backward_choice_linear_combination(
                    model, example.reference_choices, reference_coefficients, sign=-1.0
                )
            return float(value.item())
        if causal_execution_audit is not None:
            causal_execution_audit["batched_examples"] = (
                int(causal_execution_audit.get("batched_examples", 0)) + 1
            )
        value.backward()
        return float(value.item())

    if loss_mode == "choice_token" and example.choices:
        # First pass obtains the exact candidate-renormalization coefficients.
        with torch.no_grad():
            values = []
            for candidate in example.choices:
                logits, ids = _forward_logits(model, candidate)
                values.append(
                    sequence_log_probability_from_logits(logits, ids, candidate.response_start)
                )
        logps = torch.stack(values).float()
        probabilities = torch.softmax(logps, dim=0)
        target = int(example.target_choice_index)
        loss_value = -logps[target] + torch.logsumexp(logps, dim=0)

        # d[-log p_t + logsumexp(log p_j)]/d log p_j = softmax_j - 1[j=t].
        for index, candidate in enumerate(example.choices):
            logits, ids = _forward_logits(model, candidate)
            logp = sequence_log_probability_from_logits(logits, ids, candidate.response_start)
            coefficient = probabilities[index] - float(index == target)
            (coefficient.to(logp.device) * logp).backward()
        return float(loss_value.item())

    if example.completion is None:
        raise RuntimeError("completion_nll example has no encoded completion")
    logits, ids = _forward_logits(model, example.completion)
    loss = completion_nll_from_logits(logits, ids, example.completion.response_start)
    loss.backward()
    return float(loss.detach().item())


@torch.no_grad()
def evaluate_example_loss(model: nn.Module, example: PreparedExample, loss_mode: str) -> float:
    """Evaluate the same scalar example loss used by attribution scoring."""

    if loss_mode == "causal_choice_margin" and example.choices:
        value = _causal_example_value_batched(model, example)
        return float(value.item())

    if loss_mode == "choice_token" and example.choices:
        values = []
        for candidate in example.choices:
            logits, ids = _forward_logits(model, candidate)
            values.append(
                sequence_log_probability_from_logits(logits, ids, candidate.response_start)
            )
        logps = torch.stack(values).float()
        target = int(example.target_choice_index)
        loss = -logps[target] + torch.logsumexp(logps, dim=0)
        return float(loss.item())

    if example.completion is None:
        raise RuntimeError("completion_nll example has no encoded completion")
    logits, ids = _forward_logits(model, example.completion)
    loss = completion_nll_from_logits(logits, ids, example.completion.response_start)
    return float(loss.item())


def evaluate_manifest_mean_loss(
    model: nn.Module,
    examples: Sequence[PreparedExample],
    loss_mode: str,
) -> float:
    """No-grad mean over equal-weight manifest example losses."""

    if not examples:
        raise ValueError("cannot evaluate an empty manifest")
    was_training = model.training
    original_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if original_use_cache is not None:
        model.config.use_cache = False
    try:
        total = math.fsum(
            evaluate_example_loss(model, example, loss_mode) for example in examples
        )
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        model.train(was_training)
    return total / len(examples)


def evaluate_preservation_manifest(args: Any, model: nn.Module, tokenizer: Any) -> float:
    """Load and score the configured preservation manifest after masking."""

    rows = load_manifest(
        args.preserve_manifest,
        nsamples=args.nsamples_preserve or args.nsamples,
        expected_model=args.model,
        expected_revision=args.revision,
        expected_tokenizer_revision=args.tokenizer_revision or args.revision,
        expected_calibration_seed=args.seed,
    )
    examples = prepare_examples(
        rows,
        tokenizer,
        score_format=args.score_format,
        loss_mode=args.loss_mode,
        max_length=args.max_score_length,
    )
    return evaluate_manifest_mean_loss(model, examples, args.loss_mode)


_BLOCK_PATTERN = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")


def eligible_linear_weights(
    model: nn.Module, layers: Optional[Sequence[int]] = None
) -> list[tuple[str, nn.Linear, int]]:
    """Return Linear weights inside transformer blocks, excluding heads/embeddings/norms."""

    allowed = set(layers) if layers is not None else None
    result: list[tuple[str, nn.Linear, int]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        match = _BLOCK_PATTERN.search(name)
        if match is None:
            continue
        block = int(match.group(1))
        if allowed is not None and block not in allowed:
            continue
        result.append((name, module, block))
    if not result:
        raise RuntimeError("no nn.Linear weights were found inside transformer blocks")
    return result


def _safe_tensor_name(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"{digest}.pt"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    """Convert scalar/array objects commonly produced by NumPy and pandas."""

    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
            )
            handle.write("\n")
    os.replace(temporary, path)


def _reset_cuda_peak_memory_stats_all() -> None:
    """Reset allocator peaks for every visible CUDA device, not just device 0."""

    if not torch.cuda.is_available():
        return
    for device_index in range(int(torch.cuda.device_count())):
        torch.cuda.reset_peak_memory_stats(device_index)


def _cuda_memory_audit_all() -> list[dict[str, Any]]:
    """Return a per-visible-device allocator and hardware audit."""

    if not torch.cuda.is_available():
        return []
    result = []
    for device_index in range(int(torch.cuda.device_count())):
        properties = torch.cuda.get_device_properties(device_index)
        result.append(
            {
                "device_index": device_index,
                "device_name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
                "current_allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
                "current_reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
            }
        )
    return result


def score_manifest(
    *,
    model: nn.Module,
    examples: Sequence[PreparedExample],
    output_dir: Union[str, os.PathLike[str]],
    role: str,
    loss_mode: str,
    no_abs: bool,
    role_abs: bool,
    attribution_variant: str,
    layers: Optional[Sequence[int]],
    causal_identity_bindings: Optional[Mapping[str, Any]] = None,
    blocks_per_replay: int = 1,
    require_causal_candidate_batching: bool = False,
    causal_score_sink: Optional[Any] = None,
) -> dict[str, Any]:
    """Compute and persist one score tensor at a time.

    Native scoring replays once per transformer block.  The explicitly causal
    path may accumulate a memory-audited group of blocks per replay, preserving
    exact gradients while reducing full-model forward/backward passes.
    """

    if role not in {"prune", "preserve"}:
        raise ValueError(role)
    if attribution_variant not in {"paper", "released_abs"}:
        raise ValueError(attribution_variant)
    if loss_mode == "causal_choice_margin":
        if attribution_variant != "paper":
            raise ValueError("causal_choice_margin requires paper dataset-mean attribution")
        if not no_abs or role_abs:
            raise ValueError(
                "causal_choice_margin requires signed scores (--no_abs, without role abs)"
            )
    replay_width = int(blocks_per_replay)
    if replay_width <= 0:
        raise ValueError("blocks_per_replay must be positive")
    streaming_signed_completion = (
        causal_score_sink is not None and loss_mode == "completion_nll"
    )
    if loss_mode != "causal_choice_margin" and replay_width != 1 and not streaming_signed_completion:
        raise ValueError(
            "multi-block replay is opt-in for causal_choice_margin or an external "
            "signed completion-NLL sink"
        )
    if causal_score_sink is not None and loss_mode not in {
        "causal_choice_margin",
        "completion_nll",
    }:
        raise ValueError(
            "external score sinks require causal_choice_margin or completion_nll"
        )
    if streaming_signed_completion and (
        attribution_variant != "paper" or not no_abs or role_abs
    ):
        raise ValueError(
            "streamed completion_nll requires signed paper attribution without abs"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    eligible = eligible_linear_weights(model, layers)
    groups: dict[int, list[tuple[str, nn.Linear]]] = {}
    for name, module, block in eligible:
        groups.setdefault(block, []).append((name, module))

    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    model.requires_grad_(False)
    tensor_meta: dict[str, Any] = {}
    losses_by_block: dict[str, float] = {}
    replay_groups: list[dict[str, Any]] = []
    causal_execution_audit = {"batched_examples": 0, "oom_fallback_examples": 0}

    try:
        ordered_blocks = sorted(groups)
        for group_index, offset in enumerate(range(0, len(ordered_blocks), replay_width)):
            group_blocks = ordered_blocks[offset : offset + replay_width]
            modules = [item for block in group_blocks for item in groups[block]]
            group_batching_before = dict(causal_execution_audit)
            accumulators: dict[str, torch.Tensor] = {}
            for name, module in modules:
                module.weight.requires_grad_(True)
                accumulators[name] = torch.zeros_like(module.weight, dtype=torch.float32)

            _reset_cuda_peak_memory_stats_all()

            loss_sum = 0.0
            for example_index, example in enumerate(examples):
                model.zero_grad(set_to_none=True)
                example_loss = backward_example(
                    model,
                    example,
                    loss_mode,
                    causal_execution_audit=(
                        causal_execution_audit
                        if loss_mode == "causal_choice_margin"
                        else None
                    ),
                    allow_causal_candidate_oom_fallback=(
                        not bool(require_causal_candidate_batching)
                    ),
                )
                if not math.isfinite(example_loss):
                    raise RuntimeError(
                        f"non-finite {role} loss in transformer blocks {group_blocks}"
                    )
                loss_sum += example_loss
                for name, module in modules:
                    if module.weight.grad is None:
                        raise RuntimeError(f"no gradient produced for eligible weight {name}")
                    gradient = module.weight.grad.detach().float()
                    if not torch.isfinite(gradient).all():
                        raise RuntimeError(f"non-finite gradient for eligible weight {name}")
                    if attribution_variant == "released_abs":
                        accumulators[name].add_(gradient.abs())
                    else:
                        accumulators[name].add_(gradient)
                    if causal_score_sink is not None and bool(
                        getattr(causal_score_sink, "requires_per_example_scores", False)
                    ):
                        causal_score_sink.consume_example_score(
                            name,
                            example_index,
                            module.weight,
                            gradient,
                        )

            group_mean_loss = loss_sum / len(examples)
            for block in group_blocks:
                losses_by_block[str(block)] = group_mean_loss
            for name, module in modules:
                aggregate = accumulators.pop(name)
                if attribution_variant == "paper":
                    aggregate.div_(len(examples))
                    score = module.weight.detach().float() * aggregate
                    if role_abs or not no_abs:
                        score = score.abs()
                    aggregation = "abs_after_dataset_mean" if role_abs or not no_abs else "signed_dataset_mean"
                else:
                    score = module.weight.detach().float().abs() * aggregate
                    aggregation = "released_abs_weight_times_sum_abs_example_gradients"

                if not torch.isfinite(score).all():
                    raise RuntimeError(f"non-finite attribution score for eligible weight {name}")

                base_tensor_meta = {
                    "shape": list(score.shape),
                    "numel": score.numel(),
                    "block": int(_BLOCK_PATTERN.search(name).group(1)),
                    "aggregation": aggregation,
                }
                if causal_score_sink is None:
                    filename = _safe_tensor_name(name)
                    tensor_path = output / filename
                    torch.save(score.cpu(), tensor_path)
                    tensor_meta[name] = {
                        **base_tensor_meta,
                        "storage": "dense_tensor_file",
                        "file": filename,
                    }
                    if loss_mode == "causal_choice_margin":
                        tensor_meta[name]["sha256"] = sha256_file(tensor_path)
                else:
                    sink_record = causal_score_sink.consume_aggregate(
                        name,
                        score,
                        block=int(_BLOCK_PATTERN.search(name).group(1)),
                        weight=module.weight.detach(),
                    )
                    if not isinstance(sink_record, Mapping) or not sink_record.get(
                        "record_sha256"
                    ):
                        raise RuntimeError("causal score sink returned unauthenticated metadata")
                    tensor_meta[name] = {
                        **base_tensor_meta,
                        "storage": "external_streaming_sink",
                        "sink_record_sha256": str(sink_record["record_sha256"]),
                    }
                module.weight.requires_grad_(False)
                module.weight.grad = None
                del score, aggregate
            device_memory = _cuda_memory_audit_all()
            replay_groups.append(
                {
                    "group_index": group_index,
                    "blocks": list(group_blocks),
                    "module_count": len(modules),
                    "causal_candidate_batching": {
                        "batched_examples": int(
                            causal_execution_audit["batched_examples"]
                            - group_batching_before["batched_examples"]
                        ),
                        "oom_fallback_examples": int(
                            causal_execution_audit["oom_fallback_examples"]
                            - group_batching_before["oom_fallback_examples"]
                        ),
                        "required": bool(require_causal_candidate_batching),
                    },
                    "cuda_devices": device_memory,
                    "peak_cuda_allocated_bytes": max(
                        (int(row["peak_allocated_bytes"]) for row in device_memory),
                        default=0,
                    ),
                }
            )
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        model.requires_grad_(False)
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache

    mean_losses = list(losses_by_block.values())
    if mean_losses and max(mean_losses) - min(mean_losses) > 1e-5 * max(1.0, abs(mean_losses[0])):
        raise RuntimeError("dataset loss changed across block-wise scoring passes")
    sink_audit = causal_score_sink.finalize() if causal_score_sink is not None else None
    metadata = {
        "role": role,
        "loss_mode": loss_mode,
        "attribution_variant": attribution_variant,
        "num_examples": len(examples),
        "mean_dataset_loss": mean_losses[0] if mean_losses else None,
        "eligible_numel": sum(item["numel"] for item in tensor_meta.values()),
        "tensors": tensor_meta,
    }
    if loss_mode == "causal_choice_margin" or causal_score_sink is not None:
        if not causal_identity_bindings:
            raise ValueError("streaming score caches require model/tokenizer/runtime bindings")
        metadata["identity_bindings"] = dict(causal_identity_bindings)
    if loss_mode == "causal_choice_margin":
        metadata["causal_objective_audit"] = causal_manifest_objective_audit(
            [example.record for example in examples]
        )
        metadata["score_sign_convention"] = (
            "theta*dObjective; positive score means setting the weight to zero is "
            "predicted to reduce the registered objective"
        )
        metadata["candidate_execution"] = (
            "single padded context+reference candidate batch; exact full-option-sequence "
            "log probabilities; one backward; CUDA-OOM sequential analytic fallback"
        )
        metadata["blocks_per_replay"] = replay_width
        metadata["replay_groups"] = replay_groups
        metadata["causal_candidate_batching"] = {
            **causal_execution_audit,
            "required": bool(require_causal_candidate_batching),
            "fallback_allowed": not bool(require_causal_candidate_batching),
            "approved_for_production_throughput": (
                int(causal_execution_audit["oom_fallback_examples"]) == 0
            ),
        }
        if require_causal_candidate_batching and int(
            causal_execution_audit["oom_fallback_examples"]
        ) != 0:
            raise RuntimeError("production causal score cache contains candidate-batching fallbacks")
    elif streaming_signed_completion:
        # Completion-NLL production uses the same multi-block replay engine.
        # Persist its exact replay/device telemetry as well so preflight can
        # measure and gate the native/Alpaca objective independently from the
        # much cheaper multiple-choice causal margin.
        metadata["blocks_per_replay"] = replay_width
        metadata["replay_groups"] = replay_groups
        metadata["candidate_batching_not_applicable"] = True
    if sink_audit is not None:
        if not isinstance(sink_audit, Mapping) or not sink_audit.get("artifact_sha256"):
            raise RuntimeError("score sink did not publish an authenticated artifact")
        metadata["external_streaming_sink"] = dict(sink_audit)
        metadata["streamed_signed_objective"] = {
            "loss_mode": loss_mode,
            "aggregation": "theta_times_dataset_mean_gradient",
            "directional_transform": getattr(causal_score_sink, "score_multiplier", 1.0),
        }
    metadata_path = output / "metadata.json"
    _atomic_json(metadata_path, metadata)
    if loss_mode == "causal_choice_margin" or causal_score_sink is not None:
        complete_payload = {
            "schema_version": 1,
            "metadata_sha256": sha256_file(metadata_path),
            "tensor_manifest_sha256": _canonical_json_sha256(tensor_meta),
            "verify_tensor_hashes": causal_score_sink is None,
            "verify_streaming_sink_hash": causal_score_sink is not None,
        }
        complete_payload["complete_sha256"] = _canonical_json_sha256(complete_payload)
        _atomic_json(output / "COMPLETE", complete_payload)
    else:
        (output / "COMPLETE").touch()
    return metadata


def _slug(value: Optional[str]) -> str:
    value = value or "unversioned"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:100]


def score_identity(args: Any) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "tokenizer": args.tokenizer or args.model,
        "tokenizer_revision": getattr(args, "tokenizer_revision", None) or args.revision,
        "prune_manifest": str(Path(args.prune_manifest).resolve()),
        "prune_manifest_sha256": sha256_file(args.prune_manifest),
        "preserve_manifest": str(Path(args.preserve_manifest).resolve()),
        "preserve_manifest_sha256": sha256_file(args.preserve_manifest),
        "nsamples": args.nsamples,
        "nsamples_preserve": args.nsamples_preserve or args.nsamples,
        "seed": args.seed,
        "score_format": args.score_format,
        "loss_mode": args.loss_mode,
        "attribution_variant": args.attribution_variant,
        "no_abs": args.no_abs,
        "abs_prune": args.abs_prune,
        "abs_preserve": args.abs_preserve,
        "layers": args.layers,
        "max_score_length": args.max_score_length,
    }
    if args.loss_mode == "causal_choice_margin":
        causal_manifests: dict[str, Any] = {}
        for role, path, count in (
            ("prune", args.prune_manifest, args.nsamples),
            (
                "preserve",
                args.preserve_manifest,
                args.nsamples_preserve or args.nsamples,
            ),
        ):
            rows = load_manifest(path, nsamples=count)
            causal_manifests[role] = causal_manifest_objective_audit(rows)
        identity["causal_choice_margin_version"] = CAUSAL_CHOICE_MARGIN_VERSION
        identity["causal_manifest_objectives"] = causal_manifests
        runtime_identity = Path(args.runtime_identity)
        identity["runtime_identity"] = str(runtime_identity.resolve())
        identity["runtime_identity_sha256"] = sha256_file(runtime_identity)
        identity["score_blocks_per_replay"] = int(args.score_blocks_per_replay)
        identity["require_causal_candidate_batching"] = bool(
            getattr(args, "require_causal_candidate_batching", False)
        )
        stream_mode = str(getattr(args, "causal_stream_mode", "dense"))
        identity["causal_stream_mode"] = stream_mode
        if stream_mode != "dense":
            identity["causal_stream_component_id"] = str(
                args.causal_stream_component_id
            )
            identity["causal_stream_source_shard_id"] = getattr(
                args, "causal_stream_source_shard_id", None
            )
            identity["causal_stream_top_m"] = getattr(args, "causal_stream_top_m", None)
            if stream_mode == "pass2_union":
                identity["causal_stream_union_sha256"] = sha256_file(
                    args.causal_stream_union
                )
    return identity


def identity_hash(identity: Mapping[str, Any]) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def expected_score_dir(args: Any) -> Path:
    identity = score_identity(args)
    if getattr(args, "score_cache", None):
        return Path(args.score_cache)
    prune_hash = identity["prune_manifest_sha256"][:12]
    preserve_hash = identity["preserve_manifest_sha256"][:12]
    sign_label = (
        f"noabs_{int(args.no_abs)}_absprune_{int(args.abs_prune)}_"
        f"abspreserve_{int(args.abs_preserve)}"
    )
    return (
        Path(args.artifact_root)
        / "scores"
        / _slug(args.model)
        / f"revision_{_slug(args.revision)}"
        / f"format_{args.score_format}"
        / f"loss_{args.loss_mode}"
        / f"attribution_{args.attribution_variant}"
        / f"seed_{args.seed}"
        / f"n_{args.nsamples}_{args.nsamples_preserve or args.nsamples}"
        / f"manifests_{prune_hash}_{preserve_hash}"
        / sign_label
        / identity_hash(identity)
    )


def initialize_score_dir(args: Any) -> tuple[Path, dict[str, Any]]:
    identity = score_identity(args)
    directory = expected_score_dir(args)
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / "identity.json"
    if identity_path.exists():
        with open(identity_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != identity:
            raise RuntimeError(
                f"score cache identity mismatch at {directory}; choose a different --score_cache"
            )
    else:
        _atomic_json(identity_path, identity)
    return directory, identity


def validate_score_cache(args: Any) -> tuple[Path, dict[str, Any]]:
    directory = expected_score_dir(args)
    identity = score_identity(args)
    try:
        with open(directory / "identity.json", "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"no score cache found at {directory}") from exc
    if existing != identity:
        raise RuntimeError(f"score cache identity mismatch at {directory}")
    for role in ("prune", "preserve"):
        complete_path = directory / role / "COMPLETE"
        if not complete_path.exists():
            raise RuntimeError(f"score cache is incomplete: missing {role}/COMPLETE in {directory}")
        if args.loss_mode == "causal_choice_margin":
            metadata_path = directory / role / "metadata.json"
            with open(complete_path, "r", encoding="utf-8") as handle:
                complete = json.load(handle)
            dense_cache = complete.get("verify_tensor_hashes") is True
            streaming_cache = complete.get("verify_streaming_sink_hash") is True
            if dense_cache == streaming_cache:
                raise RuntimeError(
                    f"causal cache {role} must authenticate exactly one storage mode"
                )
            expected_complete_hash = complete.pop("complete_sha256", None)
            if expected_complete_hash != _canonical_json_sha256(complete):
                raise RuntimeError(f"causal cache {role} COMPLETE authentication failed")
            if complete.get("metadata_sha256") != sha256_file(metadata_path):
                raise RuntimeError(f"causal cache {role} metadata hash mismatch")
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            observed_tensor_manifest = _canonical_json_sha256(
                metadata.get("tensors", {})
            )
            if complete.get("tensor_manifest_sha256") != observed_tensor_manifest:
                raise RuntimeError(f"causal cache {role} tensor manifest hash mismatch")
            if dense_cache:
                for name, item in metadata.get("tensors", {}).items():
                    tensor_path = directory / role / str(item.get("file", ""))
                    if not tensor_path.is_file() or item.get("sha256") != sha256_file(
                        tensor_path
                    ):
                        raise RuntimeError(
                            f"causal cache tensor hash mismatch for {role}/{name}"
                        )
            else:
                sink = metadata.get("external_streaming_sink")
                if not isinstance(sink, Mapping) or sink.get("dense_tensors_persisted") is not False:
                    raise RuntimeError(f"causal streaming cache {role} lacks sink audit")
                sink_path = Path(str(sink.get("artifact_path", "")))
                if not sink_path.is_file() or sink.get("artifact_sha256") != sha256_file(
                    sink_path
                ):
                    raise RuntimeError(f"causal streaming cache {role} sink hash mismatch")
    return directory, identity


def load_score_metadata(score_dir: Path, role: str) -> dict[str, Any]:
    with open(score_dir / role / "metadata.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch<2.0 compatibility
        return torch.load(path, map_location="cpu")


def exact_global_topk(
    score_dir: Path,
    metadata: Mapping[str, Any],
    k: int,
    *,
    largest: bool,
    rank_start: int = 0,
) -> dict[str, torch.Tensor]:
    """Memory-bounded exact global rank slice.

    At every merge we retain the best ``k + rank_start`` candidates seen so
    far. By induction, discarded elements can never enter the final global
    top-k. Peak candidate memory is bounded by twice that rank count, not by
    the total model size.
    """

    end = k + rank_start
    if k < 0 or rank_start < 0:
        raise ValueError("k and rank_start must be non-negative")
    if k == 0:
        return {}
    tensors = metadata["tensors"]
    values = torch.empty(0, dtype=torch.float32)
    module_ids = torch.empty(0, dtype=torch.int32)
    local_indices = torch.empty(0, dtype=torch.int64)
    names = list(tensors)

    for module_id, name in enumerate(names):
        item = tensors[name]
        flat = _load_tensor(score_dir / item["file"]).reshape(-1).float()
        if not torch.isfinite(flat).all():
            raise RuntimeError(f"non-finite score values in {score_dir / item['file']}")
        local_k = min(end, flat.numel())
        local_values, local_ids = torch.topk(flat, local_k, largest=largest, sorted=False)
        merged_values = torch.cat((values, local_values))
        merged_modules = torch.cat(
            (module_ids, torch.full((local_k,), module_id, dtype=torch.int32))
        )
        merged_indices = torch.cat((local_indices, local_ids.long()))
        keep_k = min(end, merged_values.numel())
        values, keep = torch.topk(merged_values, keep_k, largest=largest, sorted=False)
        module_ids = merged_modules[keep]
        local_indices = merged_indices[keep]
        del flat, local_values, local_ids, merged_values, merged_modules, merged_indices, keep

    if rank_start:
        # The retained set is exact but unsorted; sort only the small final candidate set.
        order = torch.argsort(values, descending=largest, stable=True)
        order = order[rank_start:end]
        module_ids = module_ids[order]
        local_indices = local_indices[order]

    result: dict[str, torch.Tensor] = {}
    for module_id, name in enumerate(names):
        selected = local_indices[module_ids == module_id]
        if selected.numel():
            result[name] = selected.sort().values
    return result


def set_difference(
    selected: Mapping[str, torch.Tensor], excluded: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, indices in selected.items():
        excluded_indices = excluded.get(name)
        if excluded_indices is None or excluded_indices.numel() == 0:
            keep = indices
        else:
            keep = indices[~torch.isin(indices, excluded_indices)]
        if keep.numel():
            result[name] = keep
    return result


def count_indices(indices: Mapping[str, torch.Tensor]) -> int:
    return sum(value.numel() for value in indices.values())


def _magnitude_matched_random(
    model: nn.Module,
    selected: Mapping[str, torch.Tensor],
    *,
    bins: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Draw a disjoint random mask with exact per-module magnitude-bin counts."""

    if bins <= 0:
        raise ValueError("--match_bins must be positive")
    modules = dict(model.named_modules())
    result: dict[str, torch.Tensor] = {}
    audit: dict[str, Any] = {}
    devices = {module.weight.device for name, module in modules.items() if name in selected}
    if len(devices) > 1:
        # Each CUDA generator is independently seeded by manual_seed_all below.
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    for name, target in selected.items():
        module = modules[name]
        device = module.weight.device
        magnitudes = module.weight.detach().abs().reshape(-1)
        target_device = target.to(device)
        if target_device.numel() == 0:
            continue
        sample_size = min(1_000_000, magnitudes.numel())
        if sample_size < magnitudes.numel():
            sample_indices = torch.randint(
                0, magnitudes.numel(), (sample_size,), device=device
            )
            edge_sample = magnitudes[sample_indices]
        else:
            edge_sample = magnitudes
        quantiles = torch.linspace(0, 1, bins + 1, device=device)
        edges = torch.quantile(edge_sample.float(), quantiles)
        assignments = torch.bucketize(magnitudes, edges[1:-1].contiguous())
        target_counts = torch.bincount(
            assignments[target_device], minlength=bins
        )
        is_target = torch.zeros(magnitudes.numel(), dtype=torch.bool, device=device)
        is_target[target_device] = True
        chosen: list[torch.Tensor] = []
        for bin_index in range(bins):
            need = int(target_counts[bin_index].item())
            if need == 0:
                continue
            pool = ((assignments == bin_index) & ~is_target).nonzero(
                as_tuple=False
            ).squeeze(1)
            if pool.numel() < need:
                raise RuntimeError(
                    "exact magnitude-bin matching is impossible for "
                    f"{name} bin={bin_index}: need {need} disjoint candidates, "
                    f"found {pool.numel()}"
                )
            order = torch.randperm(pool.numel(), device=device)[:need]
            chosen.append(pool[order])
        random_indices = torch.cat(chosen) if chosen else torch.empty(
            0, dtype=torch.long, device=device
        )
        random_counts = torch.bincount(
            assignments[random_indices], minlength=bins
        )
        if random_indices.numel() != target.numel() or not torch.equal(
            target_counts, random_counts
        ):
            raise RuntimeError(f"failed exact magnitude-bin matching for {name}")
        result[name] = random_indices.cpu().sort().values
        audit[name] = {
            "numel": int(target.numel()),
            "bin_edges": [float(value) for value in edges.cpu().tolist()],
            "target_bin_counts": [int(value) for value in target_counts.cpu().tolist()],
            "random_bin_counts": [int(value) for value in random_counts.cpu().tolist()],
            "exact_bin_match": True,
            "disjoint": True,
        }
    return result, audit


def select_global_mask(args: Any, model: nn.Module, score_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    prune_meta = load_score_metadata(score_dir, "prune")
    preserve_meta = load_score_metadata(score_dir, "preserve")
    if prune_meta["tensors"].keys() != preserve_meta["tensors"].keys():
        raise RuntimeError("prune and preserve scores use different parameter universes")
    total = int(prune_meta["eligible_numel"])
    if total != int(preserve_meta["eligible_numel"]):
        raise RuntimeError("prune and preserve eligible weight counts differ")
    if not 0 <= args.p <= 1 or not 0 <= args.q <= 1:
        raise ValueError("--p and --q must lie in [0, 1]")
    q_count = math.floor(args.q * total)
    p_count = math.floor(args.p * total)
    if args.freeze_first_top_q and 2 * q_count > total:
        raise ValueError("--freeze_first_top_q requires 2 * floor(q*N) <= N")
    rank_start = q_count if args.freeze_first_top_q else 0
    prune_selected = exact_global_topk(
        score_dir / "prune",
        prune_meta,
        q_count,
        largest=not args.neg_prune,
        rank_start=rank_start,
    )
    preserve_selected = exact_global_topk(
        score_dir / "preserve",
        preserve_meta,
        p_count,
        largest=True,
    )
    selected = set_difference(prune_selected, preserve_selected)
    before_control = count_indices(selected)
    random_magnitude_audit: Optional[dict[str, Any]] = None
    if args.control == "random_magnitude":
        selected, random_magnitude_audit = _magnitude_matched_random(
            model,
            selected,
            bins=args.match_bins,
            seed=args.seed,
        )
    metadata = {
        "eligible_numel": total,
        "p": args.p,
        "q": args.q,
        "nominal_preserve_count": p_count,
        "nominal_prune_count": q_count,
        "prune_rank_start": rank_start,
        "surviving_count": count_indices(selected),
        "surviving_before_control": before_control,
        "neg_prune": args.neg_prune,
        "freeze_first_top_q": args.freeze_first_top_q,
        "control": args.control,
        "match_bins": int(args.match_bins) if args.control == "random_magnitude" else None,
        "random_magnitude_match": random_magnitude_audit,
        "counts_by_module": {name: value.numel() for name, value in selected.items()},
        "parameter_universe": {
            name: {"shape": item["shape"], "numel": item["numel"], "block": item["block"]}
            for name, item in prune_meta["tensors"].items()
        },
    }
    return selected, metadata


def mask_output_dir(args: Any, score_dir: Path) -> Path:
    control_label = str(args.control)
    if args.control == "random_magnitude":
        control_label += f"_bins_{int(args.match_bins)}"
    label = (
        f"p_{args.p:.12g}_q_{args.q:.12g}_neg_{int(args.neg_prune)}_"
        f"slice2_{int(args.freeze_first_top_q)}_{control_label}"
    )
    scores_root = Path(args.artifact_root) / "scores"
    try:
        relative_score_dir = score_dir.relative_to(scores_root)
    except ValueError:  # explicit --score_cache outside artifact_root
        cache_identity = identity_hash(score_identity(args))
        relative_score_dir = (
            Path(_slug(args.model))
            / f"revision_{_slug(args.revision)}"
            / f"external_{cache_identity}_{_slug(score_dir.name)}"
        )
    return Path(args.artifact_root) / "masks" / relative_score_dir / label


def apply_sparse_mask(
    model: nn.Module,
    indices: Mapping[str, torch.Tensor],
    *,
    alpha: Union[float, str],
) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, flat_indices in indices.items():
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"mask references missing/nonlinear module: {name}")
            flat = module.weight.data.reshape(-1)
            selected = flat_indices.to(flat.device)
            if alpha == "mean":
                flat[selected] = module.weight.data.mean()
            elif alpha:
                flat[selected] *= float(alpha)
            else:
                flat[selected] = 0


def save_sparse_mask(
    output_dir: Path,
    indices: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.cpu() for name, value in indices.items()}, output_dir / "indices.pt")
    payload = dict(metadata)
    payload["score_identity"] = dict(identity)
    _atomic_json(output_dir / "metadata.json", payload)


def dump_scores(args: Any, model: nn.Module, tokenizer: Any) -> Path:
    score_dir, identity = initialize_score_dir(args)
    roles = ("prune", "preserve") if args.score_role == "both" else (args.score_role,)
    for role in roles:
        manifest = args.prune_manifest if role == "prune" else args.preserve_manifest
        count = args.nsamples if role == "prune" else (args.nsamples_preserve or args.nsamples)
        rows = load_manifest(
            manifest,
            nsamples=count,
            expected_model=args.model,
            expected_revision=args.revision,
            expected_tokenizer_revision=args.tokenizer_revision or args.revision,
            expected_calibration_seed=args.seed,
        )
        examples = prepare_examples(
            rows,
            tokenizer,
            score_format=args.score_format,
            loss_mode=args.loss_mode,
            max_length=args.max_score_length,
        )
        role_abs = args.abs_prune if role == "prune" else args.abs_preserve
        causal_identity_bindings = None
        causal_score_sink = None
        if args.loss_mode == "causal_choice_margin":
            chat_template = getattr(tokenizer, "chat_template", None)
            if not isinstance(chat_template, str) or not chat_template:
                raise RuntimeError("causal scoring requires an explicit tokenizer chat_template")
            causal_identity_bindings = {
                "model": args.model,
                "revision": args.revision,
                "tokenizer": args.tokenizer or args.model,
                "tokenizer_revision": args.tokenizer_revision or args.revision,
                "chat_template_sha256": hashlib.sha256(
                    chat_template.encode("utf-8")
                ).hexdigest(),
                "runtime_identity_sha256": sha256_file(args.runtime_identity),
                "score_identity_sha256": _canonical_json_sha256(identity),
                "eligible_tensor_universe": (
                    "all registered nn.Linear weights inside requested transformer blocks"
                ),
            }
            stream_mode = str(getattr(args, "causal_stream_mode", "dense"))
            if stream_mode != "dense":
                from llmssycoph.pruning.streaming_causal_selection import (
                    Pass1TopMSink,
                    Pass2UnionSink,
                )

                question_ids = [
                    str(
                        example.record.get("question_id")
                        or example.record.get("example_id")
                        or example.record.get("pair_id")
                        or f"row_{index}"
                    )
                    for index, example in enumerate(examples)
                ]
                base_output = Path(args.causal_stream_output)
                sink_output = base_output.with_name(
                    f"{base_output.stem}.{role}{base_output.suffix or '.json'}"
                )
                if stream_mode == "pass1_topm":
                    causal_score_sink = Pass1TopMSink(
                        output_path=sink_output,
                        component_id=str(args.causal_stream_component_id),
                        source_shard_id=str(args.causal_stream_source_shard_id),
                        top_m=int(args.causal_stream_top_m),
                        question_ids=question_ids,
                    )
                elif stream_mode == "pass2_union":
                    with open(args.causal_stream_union, "r", encoding="utf-8") as handle:
                        union_artifact = json.load(handle)
                    causal_score_sink = Pass2UnionSink(
                        output_path=sink_output,
                        component_id=str(args.causal_stream_component_id),
                        question_ids=question_ids,
                        union_artifact=union_artifact,
                    )
                else:
                    raise RuntimeError(f"unknown causal stream mode {stream_mode!r}")
        score_manifest(
            model=model,
            examples=examples,
            output_dir=score_dir / role,
            role=role,
            loss_mode=args.loss_mode,
            no_abs=args.no_abs,
            role_abs=role_abs,
            attribution_variant=args.attribution_variant,
            layers=args.layers,
            causal_identity_bindings=causal_identity_bindings,
            blocks_per_replay=int(getattr(args, "score_blocks_per_replay", 1)),
            require_causal_candidate_batching=bool(
                getattr(args, "require_causal_candidate_batching", False)
            ),
            causal_score_sink=causal_score_sink,
        )
    _atomic_json(score_dir / "identity.json", identity)
    return score_dir


def run_manifest_global_pruning(args: Any, model: nn.Module, tokenizer: Any) -> dict[str, Any]:
    if args.dump_score:
        score_dir = dump_scores(args, model, tokenizer)
        return {"dump_only": True, "score_dir": str(score_dir)}

    # q=0 is the unmodified checkpoint baseline. It deliberately avoids score
    # loading/computation so baseline evaluation cannot accidentally depend on
    # calibration artifacts.
    if args.q == 0:
        identity = score_identity(args)
        score_dir = expected_score_dir(args)
        universe = eligible_linear_weights(model, args.layers)
        total = sum(module.weight.numel() for _, module, _ in universe)
        output_dir = mask_output_dir(args, score_dir)
        metadata = {
            "dump_only": False,
            "baseline": True,
            "score_dir": None,
            "mask_dir": str(output_dir),
            "eligible_numel": total,
            "p": args.p,
            "q": 0,
            "nominal_preserve_count": 0,
            "nominal_prune_count": 0,
            "surviving_count": 0,
            "neg_prune": args.neg_prune,
            "freeze_first_top_q": args.freeze_first_top_q,
            "control": args.control,
            "match_bins": (
                int(args.match_bins) if args.control == "random_magnitude" else None
            ),
            "parameter_universe": {
                name: {"shape": list(module.weight.shape), "numel": module.weight.numel(), "block": block}
                for name, module, block in universe
            },
            "score_identity": identity,
        }
        _atomic_json(output_dir / "metadata.json", metadata)
        return metadata

    score_dir, identity = validate_score_cache(args) if args.use_saved_scores else initialize_score_dir(args)
    if not args.use_saved_scores:
        # A convenience path for small smoke tests. Full runs should shard score
        # generation with --dump_score and reuse it for the p/q grid.
        dump_scores(args, model, tokenizer)
        score_dir, identity = validate_score_cache(args)
    indices, metadata = select_global_mask(args, model, score_dir)
    output_dir = mask_output_dir(args, score_dir)
    metadata.update(
        {
            "score_dir": str(score_dir),
            "mask_dir": str(output_dir),
            "attribution_variant": args.attribution_variant,
            "score_format": args.score_format,
            "loss_mode": args.loss_mode,
        }
    )
    if args.dump_mask or args.dump_indices:
        save_sparse_mask(output_dir, indices, metadata, identity)
    else:
        _atomic_json(output_dir / "metadata.json", {**metadata, "score_identity": identity})
    apply_sparse_mask(model, indices, alpha=args.alpha)
    return {"dump_only": False, **metadata}


def _evaluation_path(pruning_result: Mapping[str, Any]) -> Path:
    mask_dir_value = pruning_result.get("mask_dir")
    if not mask_dir_value:
        raise RuntimeError("manifest pruning result is missing mask_dir")
    return Path(str(mask_dir_value)) / "evaluation.json"


def update_evaluation_metadata(
    pruning_result: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> Path:
    """Atomically merge additional manifest-evaluation sections."""

    output_path = _evaluation_path(pruning_result)
    existing: dict[str, Any] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise RuntimeError(f"existing evaluation metadata is not an object: {output_path}")
        existing.update(loaded)
    existing.update(dict(updates))
    _atomic_json(output_path, existing)
    return output_path


def write_evaluation_artifact(
    pruning_result: Mapping[str, Any],
    filename: str,
    payload: Mapping[str, Any],
) -> Path:
    """Write a JSON utility-evaluation artifact next to the selected mask."""

    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("evaluation artifact filename must be a local .json basename")
    output_path = _evaluation_path(pruning_result).parent / filename
    _atomic_json(output_path, dict(payload))
    return output_path


def write_evaluation_items(
    pruning_result: Mapping[str, Any],
    filename: str,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write per-example utility outputs next to the selected mask as JSONL."""

    if Path(filename).name != filename or not filename.endswith(".jsonl"):
        raise ValueError("evaluation items filename must be a local .jsonl basename")
    output_path = _evaluation_path(pruning_result).parent / filename
    _atomic_jsonl(output_path, rows)
    return output_path


def copy_mask_artifacts(
    mask_dir: Union[str, os.PathLike[str]],
    checkpoint_dir: Union[str, os.PathLike[str]],
) -> list[str]:
    """Copy available sparse-mask files into a saved model checkpoint."""

    source = Path(mask_dir)
    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for filename in ("metadata.json", "indices.pt"):
        source_path = source / filename
        if source_path.is_file():
            shutil.copy2(source_path, destination / filename)
            copied.append(filename)
    return copied


def summarize_alpaca_scores(scores: Sequence[Any]) -> dict[str, Any]:
    """Return JSON-safe Alpaca judge mean and sample counts."""

    valid_scores: list[float] = []
    for score in scores:
        if score is None:
            continue
        numeric = float(score)
        if math.isfinite(numeric):
            valid_scores.append(numeric)
    return {
        "mean_score": (
            sum(valid_scores) / len(valid_scores) if valid_scores else None
        ),
        "count": len(scores),
        "valid_count": len(valid_scores),
    }


def summarize_zero_shot_results(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract the stable accuracy fields used by the released utility loop."""

    tasks: dict[str, dict[str, float]] = {}
    accuracies: list[float] = []
    standard_errors: list[float] = []
    for task, result in results.items():
        accuracy = float(result["acc,none"])
        standard_error = float(result["acc_stderr,none"])
        tasks[str(task)] = {
            "accuracy": accuracy,
            "accuracy_stderr": standard_error,
        }
        accuracies.append(accuracy)
        standard_errors.append(standard_error)
    if not accuracies:
        raise ValueError("zero-shot evaluation returned no task results")
    return {
        "tasks": tasks,
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "mean_accuracy_stderr": math.sqrt(
            sum(value * value for value in standard_errors)
            / (len(standard_errors) ** 2)
        ),
        "task_count": len(tasks),
    }


def write_evaluation_metadata(
    args: Any,
    pruning_result: Mapping[str, Any],
    *,
    preservation_loss: float,
    wikitext_perplexity: float,
    sparsity: float,
) -> Path:
    """Atomically create/update the evaluation record beside mask metadata."""

    return update_evaluation_metadata(
        pruning_result,
        {
            "preservation_loss": float(preservation_loss),
            "wikitext_perplexity": float(wikitext_perplexity),
            "sparsity": float(sparsity),
            "model": args.model,
            "revision": args.revision,
            "score_format": args.score_format,
            "loss_mode": args.loss_mode,
            "p": float(args.p),
            "q": float(args.q),
            "seed": int(args.seed),
            "control": args.control,
        },
    )

