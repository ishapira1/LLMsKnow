"""Pure adapters for the general-utility and model-damage evaluations.

This module deliberately contains no model loading, network access, or code
execution.  It defines the public evaluation names, deterministic parsers, and
aggregation helpers used by the shared evaluation runner.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence


NEUTRAL_ACCURACY_SAME_DATASETS = "Neutral accuracy (same datasets)"
NEUTRAL_ACCURACY_HELD_OUT_DATASETS = "Neutral accuracy (held out datasets)"
MMLU = "MMLU"
MMLU_PRO = "MMLU-Pro"
WIKITEXT_PERPLEXITY = "WikiText perplexity"
ALPACA_RESPONSE_LOSS = "Alpaca response loss/NLL"
SYMBOLIC_IN_CONTEXT_LEARNING = "Symbolic in-context learning"
EVALPLUS = "EvalPlus"

# Preregistered, tokenizer-output length strata for the frozen Alpaca reference
# responses.  Fixed absolute boundaries keep the estimand independent of any
# model/state losses and therefore prevent post-hoc quantile selection.  The
# same model tokenizer is used for Base and every intervention state, so a
# paired example must remain in the same stratum across states.
ALPACA_RESPONSE_LENGTH_STRATA_VERSION = "fixed_response_token_bins_v1"
ALPACA_RESPONSE_LENGTH_STRATA = (
    ("short_1_64", 1, 64),
    ("medium_65_256", 65, 256),
    ("long_257_plus", 257, None),
)

UTILITY_DISPLAY_NAMES = (
    NEUTRAL_ACCURACY_SAME_DATASETS,
    NEUTRAL_ACCURACY_HELD_OUT_DATASETS,
    MMLU,
    MMLU_PRO,
    WIKITEXT_PERPLEXITY,
    ALPACA_RESPONSE_LOSS,
    SYMBOLIC_IN_CONTEXT_LEARNING,
    EVALPLUS,
)


class UtilityAdapterError(ValueError):
    """Raised when a utility-evaluation artifact violates its contract."""


def _safe_rate(numerator: int | float, denominator: int) -> float:
    if denominator <= 0:
        raise UtilityAdapterError("A rate denominator must be positive")
    return float(numerator) / float(denominator)


def _allowed_labels(values: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(str(value).strip().upper() for value in values)
    if len(labels) < 2 or any(not label for label in labels):
        raise UtilityAdapterError("At least two non-empty answer labels are required")
    if len(labels) != len(set(labels)):
        raise UtilityAdapterError("Answer labels must be unique")
    return labels


def parse_mmlu_pro_answer(raw_response: Any, allowed_labels: Sequence[str]) -> str | None:
    """Parse the final MMLU-Pro answer without mining arbitrary reasoning letters."""

    labels = _allowed_labels(allowed_labels)
    raw = str(raw_response or "")
    escaped = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    pattern = re.compile(
        rf"(?:final\s+answer|answer)\s*(?:is|:)?\s*[\(\[]?({escaped})[\)\]]?",
        flags=re.IGNORECASE,
    )
    matches = pattern.findall(raw)
    if matches:
        candidate = str(matches[-1]).upper()
        return candidate if candidate in labels else None
    stripped = raw.strip().upper()
    strict = re.fullmatch(rf"[\(\[]?({escaped})[\)\]]?[\s.]*", stripped)
    if strict:
        return str(strict.group(1)).upper()
    return None


def summarize_multiple_choice_accuracy(
    records: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
) -> dict[str, Any]:
    """Summarize micro accuracy and an equal-group macro for MMLU-like records."""

    if not records:
        raise UtilityAdapterError("Cannot summarize an empty multiple-choice result")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        if group_field not in row:
            raise UtilityAdapterError(f"Record {index} is missing {group_field!r}")
        group = str(row[group_field]).strip()
        if not group:
            raise UtilityAdapterError(f"Record {index} has an empty {group_field!r}")
        valid = bool(row.get("valid", row.get("prediction") is not None))
        correct = bool(row.get("correct", False)) and valid
        item = {"valid": valid, "correct": correct, "prediction": row.get("prediction")}
        normalized.append(item)
        grouped[group].append(item)

    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(items),
            "accuracy": _safe_rate(sum(bool(item["correct"]) for item in items), len(items)),
            "invalid_rate": _safe_rate(sum(not bool(item["valid"]) for item in items), len(items)),
            "prediction_distribution": dict(
                sorted(
                    Counter(
                        str(item["prediction"])
                        for item in items
                        if item["valid"] and item["prediction"] is not None
                    ).items()
                )
            ),
        }

    by_group = {name: summarize(grouped[name]) for name in sorted(grouped)}
    return {
        **summarize(normalized),
        "group_field": group_field,
        "macro_accuracy": sum(value["accuracy"] for value in by_group.values()) / len(by_group),
        "by_group": by_group,
    }


@dataclass(frozen=True)
class CausalWindow:
    """One causal-LM window with a target interval scored exactly once."""

    input_start: int
    input_end: int
    score_start: int
    score_end: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_start": self.input_start,
            "input_end": self.input_end,
            "score_start": self.score_start,
            "score_end": self.score_end,
        }


def build_causal_windows(
    token_count: int,
    *,
    context_size: int,
    stride: int,
) -> tuple[CausalWindow, ...]:
    """Create overlapping windows whose predicted-token ranges partition ``[1, n)``."""

    if token_count < 0:
        raise UtilityAdapterError("token_count must be non-negative")
    if context_size < 2:
        raise UtilityAdapterError("context_size must be at least two")
    if stride < 1 or stride > context_size - 1:
        raise UtilityAdapterError("stride must lie in [1, context_size - 1]")
    if token_count < 2:
        return ()

    windows: list[CausalWindow] = []
    score_start = 1
    while score_start < token_count:
        score_end = min(token_count, score_start + stride)
        input_end = score_end
        input_start = max(0, input_end - context_size)
        if input_start > score_start - 1:
            raise AssertionError("A scoring window lacks left context")
        windows.append(
            CausalWindow(
                input_start=input_start,
                input_end=input_end,
                score_start=score_start,
                score_end=score_end,
            )
        )
        score_start = score_end

    covered = [
        index
        for window in windows
        for index in range(window.score_start, window.score_end)
    ]
    if covered != list(range(1, token_count)):
        raise AssertionError("Causal windows do not score every predictable token once")
    return tuple(windows)


def token_weighted_perplexity(total_nll: float, scored_token_count: int) -> float:
    if scored_token_count <= 0:
        raise UtilityAdapterError("scored_token_count must be positive")
    total = float(total_nll)
    if not math.isfinite(total) or total < 0.0:
        raise UtilityAdapterError("total_nll must be finite and non-negative")
    return math.exp(total / scored_token_count)


def alpaca_response_length_stratum(response_token_count: int) -> str:
    """Return the fixed preregistered Alpaca response-token length stratum."""

    tokens = int(response_token_count)
    if tokens <= 0:
        raise UtilityAdapterError("response_token_count must be positive")
    for stratum_id, minimum, maximum in ALPACA_RESPONSE_LENGTH_STRATA:
        if tokens >= minimum and (maximum is None or tokens <= maximum):
            return stratum_id
    raise AssertionError("Alpaca response length strata do not cover positive integers")


def _summarize_alpaca_loss_pairs(
    rows: Sequence[tuple[int, float]],
) -> dict[str, Any]:
    if not rows:
        raise UtilityAdapterError("Cannot summarize an empty Alpaca loss stratum")
    total_tokens = sum(tokens for tokens, _nll in rows)
    total_nll = sum(nll for _tokens, nll in rows)
    token_nll = total_nll / total_tokens
    return {
        "n": len(rows),
        "total_response_tokens": total_tokens,
        "total_response_nll": total_nll,
        "token_weighted_response_nll": token_nll,
        "example_macro_response_nll": sum(nll / tokens for tokens, nll in rows)
        / len(rows),
        "response_perplexity": math.exp(token_nll),
    }


def summarize_alpaca_response_losses(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate response-only NLL globally and by fixed response-length bin."""

    if not records:
        raise UtilityAdapterError("Cannot summarize an empty Alpaca result")
    normalized: list[tuple[int, float]] = []
    by_stratum: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, row in enumerate(records):
        tokens = int(row.get("response_token_count", 0))
        nll = float(row.get("response_nll", math.nan))
        if tokens <= 0 or not math.isfinite(nll) or nll < 0.0:
            raise UtilityAdapterError(f"Invalid Alpaca loss record at index {index}")
        pair = (tokens, nll)
        normalized.append(pair)
        by_stratum[alpaca_response_length_stratum(tokens)].append(pair)

    definition = [
        {
            "stratum_id": stratum_id,
            "minimum_response_tokens": minimum,
            "maximum_response_tokens": maximum,
        }
        for stratum_id, minimum, maximum in ALPACA_RESPONSE_LENGTH_STRATA
    ]
    return {
        **_summarize_alpaca_loss_pairs(normalized),
        "response_length_strata_version": ALPACA_RESPONSE_LENGTH_STRATA_VERSION,
        "response_length_strata_unit": "reference_response_tokens",
        "response_length_strata_definition": definition,
        "by_response_length_stratum": {
            stratum_id: _summarize_alpaca_loss_pairs(by_stratum[stratum_id])
            for stratum_id, _minimum, _maximum in ALPACA_RESPONSE_LENGTH_STRATA
            if by_stratum[stratum_id]
        },
    }


def summarize_symbolic_icl(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise UtilityAdapterError("Cannot summarize an empty symbolic-ICL result")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(records):
        dataset = str(row.get("dataset", "")).strip()
        if not dataset:
            raise UtilityAdapterError(f"Symbolic-ICL record {index} has no dataset")
        grouped[dataset].append(row)

    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted(grouped):
        rows = grouped[dataset]
        by_dataset[dataset] = {
            "n": len(rows),
            "accuracy": _safe_rate(
                sum(bool(row.get("valid")) and bool(row.get("correct")) for row in rows),
                len(rows),
            ),
            "invalid_rate": _safe_rate(
                sum(not bool(row.get("valid")) for row in rows), len(rows)
            ),
            "prediction_distribution": dict(
                sorted(
                    Counter(
                        str(row.get("prediction"))
                        for row in rows
                        if bool(row.get("valid"))
                    ).items()
                )
            ),
        }
    return {
        "n": len(records),
        "macro_accuracy": sum(value["accuracy"] for value in by_dataset.values())
        / len(by_dataset),
        "by_dataset": by_dataset,
    }


_FENCED_CODE_RE = re.compile(
    r"```(?:python|py)?\s*\n?(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_python_completion(raw_response: Any) -> str:
    """Extract code for later sandboxed EvalPlus execution; never execute it here."""

    raw = str(raw_response or "")
    matches = _FENCED_CODE_RE.findall(raw)
    code = matches[0] if matches else raw
    return code.strip()


def summarize_evalplus(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise UtilityAdapterError("Cannot summarize an empty EvalPlus result")
    aliases = {
        "humaneval+": "HumanEval+",
        "humaneval": "HumanEval+",
        "mbpp+": "MBPP+",
        "mbpp": "MBPP+",
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(records):
        raw_name = str(row.get("benchmark", "")).strip().lower()
        benchmark = aliases.get(raw_name)
        if benchmark is None:
            raise UtilityAdapterError(
                f"EvalPlus record {index} has unsupported benchmark {raw_name!r}"
            )
        grouped[benchmark].append(row)
    if set(grouped) != {"HumanEval+", "MBPP+"}:
        raise UtilityAdapterError("EvalPlus requires both HumanEval+ and MBPP+")

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        statuses = Counter(str(row.get("status", "unknown")) for row in rows)
        return {
            "n": len(rows),
            "pass_at_1": _safe_rate(sum(bool(row.get("passed")) for row in rows), len(rows)),
            "status_counts": dict(sorted(statuses.items())),
        }

    by_benchmark = {name: summarize(grouped[name]) for name in ("HumanEval+", "MBPP+")}
    return {
        "n": len(records),
        "macro_pass_at_1": sum(value["pass_at_1"] for value in by_benchmark.values()) / 2,
        "by_benchmark": by_benchmark,
    }


__all__ = [
    "ALPACA_RESPONSE_LOSS",
    "ALPACA_RESPONSE_LENGTH_STRATA",
    "ALPACA_RESPONSE_LENGTH_STRATA_VERSION",
    "CausalWindow",
    "EVALPLUS",
    "MMLU",
    "MMLU_PRO",
    "NEUTRAL_ACCURACY_HELD_OUT_DATASETS",
    "NEUTRAL_ACCURACY_SAME_DATASETS",
    "SYMBOLIC_IN_CONTEXT_LEARNING",
    "UTILITY_DISPLAY_NAMES",
    "UtilityAdapterError",
    "WIKITEXT_PERPLEXITY",
    "alpaca_response_length_stratum",
    "build_causal_windows",
    "extract_python_completion",
    "parse_mmlu_pro_answer",
    "summarize_alpaca_response_losses",
    "summarize_evalplus",
    "summarize_multiple_choice_accuracy",
    "summarize_symbolic_icl",
    "token_weighted_perplexity",
]

