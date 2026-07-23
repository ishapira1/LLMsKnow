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
from typing import Any, Iterable, Mapping, Sequence

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
    completion: EncodedCompletion | None = None
    choices: tuple[EncodedCompletion, ...] = ()
    target_choice_index: int | None = None


def _first(record: Mapping[str, Any], canonical: str, default: Any = None) -> Any:
    for key in MANIFEST_ALIASES[canonical]:
        value = record.get(key)
        if value is not None:
            return value
    return default


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(
    path: str | os.PathLike[str],
    *,
    nsamples: int | None,
    expected_model: str | None = None,
    expected_revision: str | None = None,
    expected_tokenizer_revision: str | None = None,
    expected_calibration_seed: int | None = None,
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
    target_override: str | None = None,
    max_length: int | None = None,
) -> EncodedCompletion:
    """Tokenize complete prompt+target text and locate the target by character offsets.

    Fast-tokenizer offsets let us reject boundary-merging ambiguity instead of
    guessing that the last token is the answer.
    """

    prompt = render_prompt(record, tokenizer, score_format)
    target = target_override if target_override is not None else _first(record, "target_text")
    if target is None:
        target = _first(record, "target_letter")
    if not isinstance(target, str) or not target:
        raise ManifestError(
            f"manifest line {record.get('_manifest_line', '?')}: missing non-empty target_text"
        )
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


def prepare_examples(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    score_format: str,
    loss_mode: str,
    max_length: int | None,
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
        if loss_mode == "choice_token" and is_multiple_choice:
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


def backward_example(model: nn.Module, example: PreparedExample, loss_mode: str) -> float:
    """Backpropagate one equal-weight example and return its detached scalar loss."""

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
    model: nn.Module, layers: Sequence[int] | None = None
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


def score_manifest(
    *,
    model: nn.Module,
    examples: Sequence[PreparedExample],
    output_dir: str | os.PathLike[str],
    role: str,
    loss_mode: str,
    no_abs: bool,
    role_abs: bool,
    attribution_variant: str,
    layers: Sequence[int] | None,
) -> dict[str, Any]:
    """Compute and persist one score tensor at a time.

    The model is replayed once per transformer block. This is slower than holding
    all gradients, but bounds live FP32 accumulation to one block and makes 7B/8B
    runs practical on a single accelerator.
    """

    if role not in {"prune", "preserve"}:
        raise ValueError(role)
    if attribution_variant not in {"paper", "released_abs"}:
        raise ValueError(attribution_variant)
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

    try:
        for block, modules in sorted(groups.items()):
            accumulators: dict[str, torch.Tensor] = {}
            for name, module in modules:
                module.weight.requires_grad_(True)
                accumulators[name] = torch.zeros_like(module.weight, dtype=torch.float32)

            loss_sum = 0.0
            for example in examples:
                model.zero_grad(set_to_none=True)
                example_loss = backward_example(model, example, loss_mode)
                if not math.isfinite(example_loss):
                    raise RuntimeError(
                        f"non-finite {role} loss in transformer block {block}"
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

            losses_by_block[str(block)] = loss_sum / len(examples)
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

                filename = _safe_tensor_name(name)
                torch.save(score.cpu(), output / filename)
                tensor_meta[name] = {
                    "file": filename,
                    "shape": list(score.shape),
                    "numel": score.numel(),
                    "block": block,
                    "aggregation": aggregation,
                }
                module.weight.requires_grad_(False)
                module.weight.grad = None
                del score, aggregate
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
    metadata = {
        "role": role,
        "loss_mode": loss_mode,
        "attribution_variant": attribution_variant,
        "num_examples": len(examples),
        "mean_dataset_loss": mean_losses[0] if mean_losses else None,
        "eligible_numel": sum(item["numel"] for item in tensor_meta.values()),
        "tensors": tensor_meta,
    }
    _atomic_json(output / "metadata.json", metadata)
    (output / "COMPLETE").touch()
    return metadata


def _slug(value: str | None) -> str:
    value = value or "unversioned"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:100]


def score_identity(args: Any) -> dict[str, Any]:
    return {
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
        if not (directory / role / "COMPLETE").exists():
            raise RuntimeError(f"score cache is incomplete: missing {role}/COMPLETE in {directory}")
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
    random_magnitude_audit: dict[str, Any] | None = None
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
    alpha: float | str,
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
    mask_dir: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
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
