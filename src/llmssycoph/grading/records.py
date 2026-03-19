from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..data import as_prompt_text, dataset_name as _dataset_name
from .grade import (
    extract_gold_answers_from_base as _extract_gold_answers_from_base,
    grade_response_from_base as _grade_response_from_base,
)


def _sort_sample_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(records),
        key=lambda record: (
            str(record.get("split", "")),
            str(record.get("question_id", "")),
            str(record.get("template_type", "")),
            int(record.get("draw_idx", 0)),
        ),
    )


def record_is_usable_for_metrics(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False

    correctness = record.get("correctness")
    usable = record.get("usable_for_metrics")
    has_binary_correctness = isinstance(correctness, Integral) and int(correctness) in {0, 1}

    if usable is None:
        return has_binary_correctness
    return bool(usable) and has_binary_correctness


def refresh_sample_records_for_groups(
    records: Sequence[Dict[str, Any]],
    groups: Sequence[Dict[str, Any]],
    split_name: str,
) -> List[Dict[str, Any]]:
    group_by_question_id = {str(group.get("question_id", "")): group for group in groups}
    refreshed: List[Dict[str, Any]] = []

    for record in records:
        refreshed_record = dict(record)
        group = group_by_question_id.get(str(record.get("question_id", "")))
        template_type = str(record.get("template_type", ""))
        row = None if group is None else group.get("rows_by_type", {}).get(template_type)
        if group is None or row is None:
            refreshed_record["split"] = split_name
            refreshed.append(refreshed_record)
            continue

        prompt_messages = row.get("prompt", [])
        prompt_text = as_prompt_text(prompt_messages)
        prompt_template = (row.get("metadata", {}) or {}).get("prompt_template", "")
        base = row.get("base", {}) or {}
        gold_answers = _extract_gold_answers_from_base(base)
        generation_info = {
            "completion_token_count": record.get("completion_token_count"),
            "hit_max_new_tokens": record.get("hit_max_new_tokens"),
            "stopped_on_eos": record.get("stopped_on_eos"),
            "finish_reason": record.get("finish_reason"),
        }
        grading = _grade_response_from_base(
            str(record.get("response_raw", "")),
            base,
            generation_info=generation_info,
        )

        refreshed_record.update(
            {
                "split": split_name,
                "question_id": group["question_id"],
                "dataset": str(group.get("dataset", "") or _dataset_name(row)),
                "template_type": template_type,
                "task_format": str(base.get("task_format", "") or ""),
                "mc_mode": str(base.get("mc_mode", "") or ""),
                "answer_channel": str(base.get("answer_channel", "") or ""),
                "prompt_spec_version": base.get("prompt_spec_version"),
                "grading_spec_version": grading.get("grading_spec_version", base.get("grading_spec_version")),
                "correct_letter": str(base.get("correct_letter", "") or ""),
                "incorrect_letter": str(base.get("incorrect_letter", "") or ""),
                "letters": str(base.get("letters", "") or ""),
                "answer_options": str(base.get("answers", "") or ""),
                "answers_list": list(base.get("answers_list", []) or []),
                "prompt_messages": prompt_messages,
                "prompt_text": prompt_text,
                "prompt_template": prompt_template,
                "question": group["question"],
                "correct_answer": group["correct_answer"],
                "incorrect_answer": group["incorrect_answer"],
                "incorrect_answer_source": str(base.get("incorrect_answer_source", "") or ""),
                "gold_answers": gold_answers,
                "response": grading["parsed_answer"],
                "committed_answer": grading.get("committed_answer", ""),
                "commitment_kind": grading.get("commitment_kind", ""),
                "commitment_source": grading.get("commitment_source", ""),
                "starts_with_answer_prefix": bool(grading.get("starts_with_answer_prefix", False)),
                "strict_format_exact": bool(grading.get("strict_format_exact", False)),
                "commitment_line": grading.get("commitment_line", ""),
                "answer_marker_count": int(grading.get("answer_marker_count", 0) or 0),
                "multiple_answer_markers": bool(grading.get("multiple_answer_markers", False)),
                "correctness": grading["correctness"],
                "grading_status": grading["status"],
                "grading_reason": grading["reason"],
                "usable_for_metrics": grading["usable_for_metrics"],
                "completion_token_count": record.get("completion_token_count"),
                "hit_max_new_tokens": bool(record.get("hit_max_new_tokens", False)),
                "stopped_on_eos": bool(record.get("stopped_on_eos", False)),
                "finish_reason": str(record.get("finish_reason", "") or ""),
            }
        )
        refreshed.append(refreshed_record)

    return _sort_sample_records(refreshed)


def add_empirical_t(records: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str, str], List[int]] = {}
    for record in records:
        if not record_is_usable_for_metrics(record):
            continue
        key = (record["split"], record["question_id"], record["template_type"])
        grouped.setdefault(key, []).append(int(record["correctness"]))
    tvals = {key: float(np.mean(values)) for key, values in grouped.items()}
    for record in records:
        record["T_prompt"] = tvals.get(
            (record["split"], record["question_id"], record["template_type"]),
            float("nan"),
        )


__all__ = [
    "add_empirical_t",
    "record_is_usable_for_metrics",
    "refresh_sample_records_for_groups",
]
