from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence

from .constants import MC_MODE_STRICT
from .grading import record_is_usable_for_metrics
from .logging_utils import log_status, ok_status, warn_status


SAMPLING_INTEGRITY_VERSION = 1

_GENERATION_BUCKET_ORDER = [
    "exact_compliance",
    "minor_format_deviation_still_scoreable",
    "format_failure",
]
_GENERATION_BUCKET_LABELS = {
    "exact_compliance": "Exact compliance",
    "minor_format_deviation_still_scoreable": "Minor format deviation, still scoreable",
    "format_failure": "Format failure",
}

_CHOICE_BUCKET_ORDER = [
    "exact_compliance",
    "integrity_failure",
]
_CHOICE_BUCKET_LABELS = {
    "exact_compliance": "Exact compliance",
    "integrity_failure": "Integrity failure",
}


def _strict_mc_generation_contract(record: Dict[str, Any]) -> bool:
    return (
        str(record.get("task_format", "") or "") == "multiple_choice"
        and str(record.get("mc_mode", "") or "") == MC_MODE_STRICT
    )


def _classify_generation_record(record: Dict[str, Any]) -> tuple[str, str]:
    if not record_is_usable_for_metrics(record):
        return (
            "format_failure",
            str(record.get("grading_reason", "") or "unscoreable_generation"),
        )
    if _strict_mc_generation_contract(record) and not bool(record.get("strict_format_exact", False)):
        return (
            "minor_format_deviation_still_scoreable",
            "non_exact_but_scoreable",
        )
    return ("exact_compliance", "exact_or_no_strict_format_contract")


def _choice_labels(record: Dict[str, Any]) -> List[str]:
    letters = str(record.get("letters", "") or "").strip().upper()
    return [letter for letter in letters if letter.strip()]


def _finite_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _is_close(a: Any, b: Any, *, tol: float = 1e-6) -> bool:
    a_num = _finite_number(a)
    b_num = _finite_number(b)
    if a_num is None or b_num is None:
        return False
    return math.isclose(a_num, b_num, rel_tol=tol, abs_tol=tol)


def _choice_probability_issues(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    probabilities_raw = record.get("choice_probabilities", {})
    if not isinstance(probabilities_raw, dict) or not probabilities_raw:
        return ["missing_choice_probabilities"]

    probabilities: Dict[str, float] = {}
    for raw_choice, raw_value in probabilities_raw.items():
        choice = str(raw_choice or "").strip().upper()
        if not choice:
            issues.append("blank_choice_label")
            continue
        value = _finite_number(raw_value)
        if value is None:
            issues.append("non_finite_probability")
            continue
        if value < 0.0:
            issues.append("negative_probability")
            continue
        probabilities[choice] = value

    if not probabilities:
        return sorted(set(issues or ["invalid_choice_probabilities"]))

    total_mass = sum(probabilities.values())
    if not math.isclose(total_mass, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        issues.append("probabilities_not_normalized")

    allowed_choices = _choice_labels(record)
    if allowed_choices:
        missing_choices = [choice for choice in allowed_choices if choice not in probabilities]
        unexpected_choices = [choice for choice in probabilities if choice not in set(allowed_choices)]
        if missing_choices:
            issues.append("missing_allowed_choices")
        if unexpected_choices:
            issues.append("unexpected_choice_labels")
        choice_order = {choice: idx for idx, choice in enumerate(allowed_choices)}
        ranked_choices = sorted(
            probabilities,
            key=lambda choice: (-probabilities[choice], choice_order.get(choice, len(choice_order)), choice),
        )
    else:
        ranked_choices = sorted(probabilities, key=lambda choice: (-probabilities[choice], choice))

    selected_choice = str(record.get("response_raw", "") or "").strip().upper()
    if not selected_choice:
        issues.append("missing_selected_choice")
    elif selected_choice not in probabilities:
        issues.append("selected_choice_missing_from_distribution")
    else:
        if ranked_choices and selected_choice != ranked_choices[0]:
            issues.append("selected_choice_not_argmax")
        if not _is_close(
            record.get("choice_probability_selected", float("nan")),
            probabilities[selected_choice],
        ):
            issues.append("selected_probability_mismatch")

    correct_letter = str(record.get("correct_letter", "") or "").strip().upper()
    if correct_letter:
        if correct_letter not in probabilities:
            issues.append("missing_correct_choice_probability")
        elif not _is_close(
            record.get("choice_probability_correct", float("nan")),
            probabilities[correct_letter],
        ):
            issues.append("correct_probability_mismatch")

    if str(record.get("finish_reason", "") or "") != "choice_probabilities":
        issues.append("unexpected_finish_reason")
    if int(record.get("completion_token_count", 0) or 0) != 1:
        issues.append("unexpected_completion_token_count")
    if bool(record.get("hit_max_new_tokens", False)):
        issues.append("unexpected_cap_hit")
    if bool(record.get("stopped_on_eos", False)):
        issues.append("unexpected_eos_stop")
    if not record_is_usable_for_metrics(record):
        issues.append("not_scoreable")

    return sorted(set(issues))


def _classify_choice_probability_record(record: Dict[str, Any]) -> tuple[str, str]:
    issues = _choice_probability_issues(record)
    if issues:
        return ("integrity_failure", ",".join(issues))
    return ("exact_compliance", "distribution_is_consistent")


def _bucket_counts(total: int, counts: Counter[str], bucket_order: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {
        bucket: {
            "count": int(counts.get(bucket, 0)),
            "rate": 0.0 if total <= 0 else float(counts.get(bucket, 0)) / float(total),
        }
        for bucket in bucket_order
    }


def _mode_human_summary(bucket_summary: Dict[str, Dict[str, Any]], bucket_labels: Dict[str, str], bucket_order: Sequence[str]) -> List[str]:
    lines: List[str] = []
    for bucket in bucket_order:
        stats = bucket_summary[bucket]
        lines.append(f"{bucket_labels[bucket]}: {100.0 * float(stats['rate']):.2f}%")
    return lines


def _top_reasons(reason_counts: Dict[str, Any], limit: int = 3) -> str:
    if not isinstance(reason_counts, dict):
        return ""
    ranked: List[tuple[str, int]] = []
    for reason, count in reason_counts.items():
        try:
            numeric_count = int(count)
        except Exception:
            continue
        ranked.append((str(reason), numeric_count))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{reason}={count}" for reason, count in ranked[:limit])


def _selected_choice_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    for record in records:
        choice = str(record.get("response_raw", "") or "").strip().upper()
        if choice:
            counts[choice] += 1

    total = int(sum(counts.values()))
    dominant_choice = ""
    dominant_count = 0
    if counts:
        dominant_choice, dominant_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]

    return {
        "selected_choice_counts": dict(sorted(counts.items())),
        "selected_choice_total": total,
        "dominant_selected_choice": dominant_choice,
        "dominant_selected_choice_count": int(dominant_count),
        "dominant_selected_choice_rate": 0.0 if total <= 0 else float(dominant_count) / float(total),
    }


def _has_single_selected_choice(summary: Dict[str, Any]) -> bool:
    selected_choice_counts = dict(summary.get("selected_choice_counts", {}) or {})
    selected_choice_total = int(summary.get("selected_choice_total", 0) or 0)
    dominant_selected_choice = str(summary.get("dominant_selected_choice", "") or "").strip().upper()
    return selected_choice_total > 1 and len(selected_choice_counts) == 1 and bool(dominant_selected_choice)


def _integrity_summary_is_clean(sampling_mode: str, summary: Dict[str, Any]) -> bool:
    if sampling_mode == "generation":
        format_failure = int(summary.get("buckets", {}).get("format_failure", {}).get("count", 0) or 0)
        format_drift = int(
            summary.get("buckets", {}).get("minor_format_deviation_still_scoreable", {}).get("count", 0) or 0
        )
        return format_failure == 0 and format_drift == 0
    if sampling_mode == "choice_probabilities":
        integrity_failure = int(summary.get("buckets", {}).get("integrity_failure", {}).get("count", 0) or 0)
        return integrity_failure == 0 and not _has_single_selected_choice(summary)
    return False


def _summarize_generation_records(
    records: Sequence[Dict[str, Any]],
    *,
    include_by_template: bool = True,
) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    by_template: Dict[str, List[Dict[str, Any]]] = {}

    for record in records:
        bucket, reason = _classify_generation_record(record)
        counts[bucket] += 1
        reason_counts[reason] += 1
        by_template.setdefault(str(record.get("template_type", "") or ""), []).append(record)

    total = len(records)
    bucket_summary = _bucket_counts(total, counts, _GENERATION_BUCKET_ORDER)
    summary = {
        "description": "Rows produced by text generation and then graded from the emitted completion.",
        "checks": {
            "exact_compliance": (
                "Row is scoreable. If a strict output contract exists, the emitted answer also matches the exact "
                "required format."
            ),
            "minor_format_deviation_still_scoreable": (
                "Row is still scoreable, but it violates the exact strict format contract."
            ),
            "format_failure": (
                "Row is not scoreable because parsing or format compliance failed."
            ),
        },
        "total": total,
        "buckets": bucket_summary,
        "reason_counts": dict(sorted(reason_counts.items())),
        "human_summary": _mode_human_summary(
            bucket_summary,
            _GENERATION_BUCKET_LABELS,
            _GENERATION_BUCKET_ORDER,
        ),
    }
    if include_by_template:
        summary["by_template"] = {
            template_type: _summarize_generation_records(template_records, include_by_template=False)
            for template_type, template_records in sorted(by_template.items())
        }
    return summary


def _summarize_choice_probability_records(
    records: Sequence[Dict[str, Any]],
    *,
    include_by_template: bool = True,
) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    by_template: Dict[str, List[Dict[str, Any]]] = {}

    for record in records:
        bucket, reason = _classify_choice_probability_record(record)
        counts[bucket] += 1
        reason_counts[reason] += 1
        by_template.setdefault(str(record.get("template_type", "") or ""), []).append(record)

    total = len(records)
    bucket_summary = _bucket_counts(total, counts, _CHOICE_BUCKET_ORDER)
    summary = {
        "description": "Rows produced from first-token choice probabilities instead of sampled text generation.",
        "checks": {
            "exact_compliance": (
                "Stored choice probabilities are normalized and internally consistent with the selected choice, "
                "gold choice, and grading result."
            ),
            "integrity_failure": (
                "At least one probability or bookkeeping consistency check failed."
            ),
        },
        "total": total,
        "buckets": bucket_summary,
        "reason_counts": dict(sorted(reason_counts.items())),
        "human_summary": _mode_human_summary(
            bucket_summary,
            _CHOICE_BUCKET_LABELS,
            _CHOICE_BUCKET_ORDER,
        ),
    }
    summary.update(_selected_choice_summary(records))
    if include_by_template:
        summary["by_template"] = {
            template_type: _summarize_choice_probability_records(template_records, include_by_template=False)
            for template_type, template_records in sorted(by_template.items())
        }
    return summary


def build_sampling_integrity_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    generation_records = [
        record for record in records if str(record.get("sampling_mode", "generation") or "generation") == "generation"
    ]
    choice_probability_records = [
        record
        for record in records
        if str(record.get("sampling_mode", "") or "") == "choice_probabilities"
    ]

    summary: Dict[str, Any] = {
        "sampling_integrity_version": int(SAMPLING_INTEGRITY_VERSION),
        "total_records": int(len(records)),
        "sampling_modes_present": sorted(
            {
                str(record.get("sampling_mode", "generation") or "generation")
                for record in records
            }
        ),
        "by_sampling_mode": {},
    }
    if generation_records:
        summary["by_sampling_mode"]["generation"] = _summarize_generation_records(generation_records)
    if choice_probability_records:
        summary["by_sampling_mode"]["choice_probabilities"] = _summarize_choice_probability_records(
            choice_probability_records
        )
    return summary


def log_sampling_integrity_summary(summary: Dict[str, Any]) -> None:
    if not summary:
        return
    for sampling_mode, mode_summary in sorted(summary.get("by_sampling_mode", {}).items()):
        lines = list(mode_summary.get("human_summary", []))
        if not lines:
            continue
        summary_logger = ok_status if _integrity_summary_is_clean(sampling_mode, mode_summary) else log_status
        summary_logger(
            "sampling_integrity.py",
            f"sampling integrity mode={sampling_mode}: " + " | ".join(lines),
        )
        total = int(mode_summary.get("total", 0) or 0)
        if sampling_mode == "generation":
            format_failure = int(mode_summary.get("buckets", {}).get("format_failure", {}).get("count", 0) or 0)
            format_drift = int(
                mode_summary.get("buckets", {}).get("minor_format_deviation_still_scoreable", {}).get("count", 0) or 0
            )
            if format_failure > 0:
                details = _top_reasons(mode_summary.get("reason_counts", {}))
                warn_status(
                    "sampling_integrity.py",
                    "generation_format_failures",
                    f"{format_failure}/{total} generation rows were not scoreable due to format or parsing failures."
                    + (f" Top reasons: {details}" if details else ""),
                )
            if format_drift > 0:
                warn_status(
                    "sampling_integrity.py",
                    "generation_format_drift",
                    f"{format_drift}/{total} generation rows were scoreable but violated the exact strict-MC format contract.",
                )
        elif sampling_mode == "choice_probabilities":
            integrity_failure = int(mode_summary.get("buckets", {}).get("integrity_failure", {}).get("count", 0) or 0)
            if integrity_failure > 0:
                details = _top_reasons(mode_summary.get("reason_counts", {}))
                warn_status(
                    "sampling_integrity.py",
                    "choice_probability_integrity_failures",
                    f"{integrity_failure}/{total} choice-probability rows failed bookkeeping or integrity checks."
                    + (f" Top reasons: {details}" if details else ""),
                )
            selected_choice_total = int(mode_summary.get("selected_choice_total", 0) or 0)
            dominant_selected_choice = str(mode_summary.get("dominant_selected_choice", "") or "").strip().upper()
            if _has_single_selected_choice(mode_summary):
                warn_status(
                    "sampling_integrity.py",
                    "choice_probability_single_selected_choice",
                    f"all {selected_choice_total}/{total} choice-probability rows selected the same "
                    f"highest-probability option ({dominant_selected_choice}). "
                    "Check answer ordering, prompt construction, and whether the model collapsed to one label.",
                )
        for template_type, template_summary in sorted(mode_summary.get("by_template", {}).items()):
            template_lines = list(template_summary.get("human_summary", []))
            if not template_lines:
                continue
            template_logger = ok_status if _integrity_summary_is_clean(sampling_mode, template_summary) else log_status
            template_logger(
                "sampling_integrity.py",
                f"sampling integrity mode={sampling_mode} template={template_type}: "
                + " | ".join(template_lines),
            )


__all__ = [
    "SAMPLING_INTEGRITY_VERSION",
    "build_sampling_integrity_summary",
    "log_sampling_integrity_summary",
]
