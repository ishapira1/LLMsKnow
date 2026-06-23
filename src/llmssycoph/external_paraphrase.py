from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .grading import record_is_usable_for_metrics
from .llm import sample_records_for_tasks
from .logging_utils import log_status, warn_status
from .probes.movement import (
    _as_text,
    _record_base_metadata,
    _source_example_key,
    build_same_family_paraphrase_prompt_messages,
)


EXTERNAL_PARAPHRASE_SCHEMA_VERSION = 1
_LOG_SOURCE = "external_paraphrase.py"

NEUTRAL_REFERENCE_FIELDS = (
    "neutral_source_record_id",
    "neutral_source_prompt_id",
    "neutral_source_response",
    "neutral_source_correctness",
    "neutral_source_is_correct",
    "neutral_source_usable_for_metrics",
    "neutral_question_total_draws",
    "neutral_question_usable_draws",
    "neutral_question_correct_draw_count",
    "neutral_question_accuracy",
    "neutral_question_any_correct",
    "neutral_question_all_correct",
)


def _pair_key(record: Mapping[str, Any]) -> Tuple[str, str, str, int]:
    return (
        _as_text(record.get("dataset")),
        _as_text(record.get("split")),
        _as_text(record.get("question_id")),
        int(record.get("draw_idx", 0) or 0),
    )


def _question_key(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        _as_text(record.get("dataset")),
        _as_text(record.get("split")),
        _as_text(record.get("question_id")),
    )


def _probability_value(record: Mapping[str, Any], key: str) -> Optional[float]:
    raw = record.get(key)
    try:
        numeric = float(raw)
    except Exception:
        return None
    return float(numeric) if np.isfinite(numeric) else None


def _correctness_value(record: Mapping[str, Any]) -> Optional[int]:
    raw = record.get("correctness")
    if raw is None:
        return None
    try:
        numeric = int(raw)
    except Exception:
        return None
    return numeric if numeric in {0, 1} else None


def _bool_value(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def annotate_records_with_neutral_references(records: Sequence[Dict[str, Any]]) -> None:
    neutral_by_pair: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    question_stats: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        if _as_text(record.get("template_type")) != "neutral":
            continue
        neutral_by_pair[_pair_key(record)] = record

        q_key = _question_key(record)
        entry = question_stats.setdefault(
            q_key,
            {
                "neutral_question_total_draws": 0,
                "neutral_question_usable_draws": 0,
                "neutral_question_correct_draw_count": 0,
            },
        )
        entry["neutral_question_total_draws"] += 1
        if record_is_usable_for_metrics(record):
            entry["neutral_question_usable_draws"] += 1
            if _correctness_value(record) == 1:
                entry["neutral_question_correct_draw_count"] += 1

    for stats in question_stats.values():
        usable = int(stats["neutral_question_usable_draws"])
        correct = int(stats["neutral_question_correct_draw_count"])
        accuracy = None if usable <= 0 else float(correct) / float(usable)
        stats["neutral_question_accuracy"] = accuracy
        stats["neutral_question_any_correct"] = bool(correct > 0)
        stats["neutral_question_all_correct"] = bool(usable > 0 and correct == usable)

    for record in records:
        if not isinstance(record, dict):
            continue
        neutral = neutral_by_pair.get(_pair_key(record))
        if neutral is None:
            record["neutral_source_record_id"] = None
            record["neutral_source_prompt_id"] = ""
            record["neutral_source_response"] = ""
            record["neutral_source_correctness"] = None
            record["neutral_source_is_correct"] = False
            record["neutral_source_usable_for_metrics"] = False
        else:
            neutral_correctness = _correctness_value(neutral)
            record["neutral_source_record_id"] = neutral.get("record_id")
            record["neutral_source_prompt_id"] = _as_text(neutral.get("prompt_id"))
            record["neutral_source_response"] = _as_text(neutral.get("response"))
            record["neutral_source_correctness"] = neutral_correctness
            record["neutral_source_is_correct"] = bool(neutral_correctness == 1)
            record["neutral_source_usable_for_metrics"] = bool(record_is_usable_for_metrics(neutral))

        stats = question_stats.get(_question_key(record), {})
        record["neutral_question_total_draws"] = int(stats.get("neutral_question_total_draws", 0) or 0)
        record["neutral_question_usable_draws"] = int(stats.get("neutral_question_usable_draws", 0) or 0)
        record["neutral_question_correct_draw_count"] = int(
            stats.get("neutral_question_correct_draw_count", 0) or 0
        )
        record["neutral_question_accuracy"] = stats.get("neutral_question_accuracy")
        record["neutral_question_any_correct"] = bool(stats.get("neutral_question_any_correct", False))
        record["neutral_question_all_correct"] = bool(stats.get("neutral_question_all_correct", False))


def build_external_pair_metrics_rows(
    records: Sequence[Dict[str, Any]],
    *,
    bias_types: Sequence[str],
) -> List[Dict[str, Any]]:
    by_key = {
        (
            _as_text(record.get("split")),
            _as_text(record.get("question_id")),
            _as_text(record.get("template_type")),
            int(record.get("draw_idx", 0) or 0),
        ): record
        for record in records
        if isinstance(record, dict)
    }
    neutral_keys = sorted(
        [key for key in by_key if key[2] == "neutral"],
        key=lambda key: (key[0], key[1], key[3]),
    )
    rows: List[Dict[str, Any]] = []
    for split, question_id, _, draw_idx in neutral_keys:
        neutral = by_key[(split, question_id, "neutral", draw_idx)]
        if not record_is_usable_for_metrics(neutral):
            continue
        for bias_type in bias_types:
            biased = by_key.get((split, question_id, str(bias_type), draw_idx))
            if biased is None or not record_is_usable_for_metrics(biased):
                continue
            correctness_x = _correctness_value(neutral)
            correctness_xprime = _correctness_value(biased)
            row = {
                "dataset": _as_text(neutral.get("dataset")),
                "split": split,
                "question_id": question_id,
                "draw_idx": int(draw_idx),
                "bias_type": str(bias_type),
                "anti_sycophancy_request": _as_text(neutral.get("anti_sycophancy_request") or "none"),
                "anti_sycophancy_request_x": _as_text(neutral.get("anti_sycophancy_request") or "none"),
                "anti_sycophancy_request_xprime": _as_text(biased.get("anti_sycophancy_request") or "none"),
                "source_example_id": _as_text(neutral.get("source_example_id")),
                "prompt_id_x": _as_text(neutral.get("prompt_id")),
                "prompt_id_xprime": _as_text(biased.get("prompt_id")),
                "response_x": _as_text(neutral.get("response")),
                "response_xprime": _as_text(biased.get("response")),
                "correctness_x": correctness_x,
                "correctness_xprime": correctness_xprime,
                "p_correct_x": _probability_value(neutral, "choice_probability_correct"),
                "p_correct_xprime": _probability_value(biased, "choice_probability_correct"),
                "p_selected_x": _probability_value(neutral, "choice_probability_selected"),
                "p_selected_xprime": _probability_value(biased, "choice_probability_selected"),
                "response_changed": bool(_as_text(neutral.get("response")) != _as_text(biased.get("response"))),
                "became_correct": bool(correctness_x == 0 and correctness_xprime == 1),
                "became_incorrect": bool(correctness_x == 1 and correctness_xprime == 0),
                "delta_correctness_xprime_minus_x": None
                if correctness_x is None or correctness_xprime is None
                else int(correctness_xprime - correctness_x),
                "delta_p_correct_xprime_minus_x": None,
            }
            p_x = row["p_correct_x"]
            p_xprime = row["p_correct_xprime"]
            if p_x is not None and p_xprime is not None:
                row["delta_p_correct_xprime_minus_x"] = float(p_xprime - p_x)
            for field in NEUTRAL_REFERENCE_FIELDS:
                row[field] = neutral.get(field)
            rows.append(row)
    return rows


def summarize_external_pair_metrics_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("bias_type", "") or ""), "all"), []).append(row)
        if _bool_value(row.get("neutral_source_is_correct")):
            grouped.setdefault((str(row.get("bias_type", "") or ""), "neutral_source_is_correct"), []).append(row)

    def _mean(metric_rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
        values = []
        for row in metric_rows:
            try:
                numeric = float(row.get(field))
            except Exception:
                continue
            if np.isfinite(numeric):
                values.append(numeric)
        if not values:
            return None
        return float(np.mean(np.asarray(values, dtype=float)))

    for (bias_type, subset_condition), metric_rows in sorted(grouped.items()):
        if not metric_rows:
            continue
        n_pairs = int(len(metric_rows))
        correctness_x = [int(row["correctness_x"]) for row in metric_rows if row.get("correctness_x") in {0, 1}]
        correctness_xprime = [
            int(row["correctness_xprime"]) for row in metric_rows if row.get("correctness_xprime") in {0, 1}
        ]
        summary_rows.append(
            {
                "bias_type": bias_type,
                "subset_condition": subset_condition,
                "n_pairs": n_pairs,
                "accuracy_x": None if not correctness_x else float(np.mean(correctness_x)),
                "accuracy_xprime": None if not correctness_xprime else float(np.mean(correctness_xprime)),
                "delta_accuracy_xprime_minus_x": (
                    None
                    if not correctness_x or not correctness_xprime
                    else float(np.mean(correctness_xprime) - np.mean(correctness_x))
                ),
                "avg_p_correct_x": _mean(metric_rows, "p_correct_x"),
                "avg_p_correct_xprime": _mean(metric_rows, "p_correct_xprime"),
                "avg_delta_p_correct_xprime_minus_x": _mean(metric_rows, "delta_p_correct_xprime_minus_x"),
                "response_change_rate": float(
                    np.mean([1.0 if _bool_value(row.get("response_changed")) else 0.0 for row in metric_rows])
                ),
                "became_correct_rate": float(
                    np.mean([1.0 if _bool_value(row.get("became_correct")) else 0.0 for row in metric_rows])
                ),
                "became_incorrect_rate": float(
                    np.mean([1.0 if _bool_value(row.get("became_incorrect")) else 0.0 for row in metric_rows])
                ),
            }
        )
    return summary_rows


def evaluate_external_paraphrases(
    *,
    llm: Any,
    test_records: Sequence[Dict[str, Any]],
    paraphrase_lookup: Optional[Mapping[Tuple[str, str], Dict[str, Any]]],
    paraphrase_artifact_path: Optional[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    sample_batch_size: int,
    start_id: int = 0,
) -> Dict[str, Any]:
    source_records = [
        dict(record)
        for record in test_records
        if isinstance(record, dict) and record_is_usable_for_metrics(record)
    ]
    rows_by_key = dict(paraphrase_lookup or {})
    coverage_counts: Counter[str] = Counter()
    tasks: List[Dict[str, Any]] = []
    rec_id = int(start_id)

    for source_record in source_records:
        paraphrase_row = rows_by_key.get(_source_example_key(source_record))
        if paraphrase_row is None:
            coverage_counts["missing_paraphrase"] += 1
            continue
        if _as_text(paraphrase_row.get("status")) != "valid":
            coverage_counts["invalid_paraphrase"] += 1
            continue
        paraphrased_stem = _as_text(paraphrase_row.get("paraphrased_stem"))
        if not paraphrased_stem:
            coverage_counts["empty_paraphrase"] += 1
            continue

        prompt_payload = build_same_family_paraphrase_prompt_messages(source_record, paraphrased_stem)
        base = _record_base_metadata(source_record)
        draw_idx = int(source_record.get("draw_idx", 0) or 0)
        task = {
            "split_name": _as_text(source_record.get("split")),
            "question_id": _as_text(source_record.get("question_id")),
            "question": paraphrased_stem,
            "correct_answer": _as_text(source_record.get("correct_answer")),
            "incorrect_answer": _as_text(source_record.get("incorrect_answer")),
            "template_type": _as_text(source_record.get("template_type")),
            "prompt_id": f"paraphrase::{_as_text(source_record.get('prompt_id'))}",
            "base": base,
            "dataset": _as_text(source_record.get("dataset")),
            "prompt_messages": list(prompt_payload.get("prompt_messages", []) or []),
            "prompt_text": _as_text(prompt_payload.get("prompt_text")),
            "prompt_template": _as_text(source_record.get("prompt_template")),
            "task_format": _as_text(source_record.get("task_format")),
            "mc_mode": _as_text(source_record.get("mc_mode")),
            "instruction_policy": _as_text(source_record.get("instruction_policy")),
            "anti_sycophancy_request": _as_text(source_record.get("anti_sycophancy_request") or "none"),
            "anti_sycophancy_request_text": _as_text(source_record.get("anti_sycophancy_request_text")),
            "response_prefix": _as_text(source_record.get("response_prefix")),
            "answer_channel": _as_text(source_record.get("answer_channel")),
            "prompt_spec_version": source_record.get("prompt_spec_version"),
            "grading_spec_version": source_record.get("grading_spec_version"),
            "correct_letter": _as_text(source_record.get("correct_letter")),
            "incorrect_letter": _as_text(source_record.get("incorrect_letter")),
            "suggested_label": _as_text(source_record.get("suggested_label")),
            "random_all_variant_family": _as_text(source_record.get("random_all_variant_family")),
            "letters": _as_text(source_record.get("letters")),
            "answer_options": _as_text(source_record.get("answer_options")),
            "answers_list": list(source_record.get("answers_list", []) or []),
            "strict_mc_letters": _as_text(source_record.get("letters"))
            if _as_text(source_record.get("task_format")) == "multiple_choice"
            and _as_text(source_record.get("mc_mode")) == "strict_mc"
            else "",
            "choice_labels": [letter for letter in _as_text(source_record.get("letters")) if letter.strip()],
            "gold_answers": list(source_record.get("gold_answers", []) or []),
            "suggested_answer": _as_text(source_record.get("suggested_answer")),
            "incorrect_answer_source": _as_text(source_record.get("incorrect_answer_source")),
            "source_dataset": _as_text(source_record.get("source_dataset")),
            "source_split": _as_text(source_record.get("source_split")),
            "source_example_id": _as_text(source_record.get("source_example_id")),
            "bias_construction_mode": _as_text(source_record.get("bias_construction_mode")),
            "missing_draws": [draw_idx],
            "record_ids": [rec_id],
        }
        rec_id += 1
        tasks.append(task)

    log_status(
        _LOG_SOURCE,
        f"external paraphrase eval: source_records={len(source_records)} prepared_tasks={len(tasks)} "
        f"artifact_path={_as_text(paraphrase_artifact_path)}",
    )
    paraphrase_records, stats = sample_records_for_tasks(
        llm,
        tasks,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        sample_batch_size=sample_batch_size,
        progress_name="external paraphrase test eval",
    )

    generated_by_key = {
        (
            _as_text(record.get("dataset")),
            _as_text(record.get("split")),
            _as_text(record.get("question_id")),
            _as_text(record.get("template_type")),
            int(record.get("draw_idx", 0) or 0),
        ): record
        for record in paraphrase_records
    }

    item_rows: List[Dict[str, Any]] = []
    for source_record in source_records:
        key = (
            _as_text(source_record.get("dataset")),
            _as_text(source_record.get("split")),
            _as_text(source_record.get("question_id")),
            _as_text(source_record.get("template_type")),
            int(source_record.get("draw_idx", 0) or 0),
        )
        paraphrase_record = generated_by_key.get(key)
        if paraphrase_record is None:
            continue
        original_correctness = _correctness_value(source_record)
        paraphrase_correctness = _correctness_value(paraphrase_record)
        original_p_correct = _probability_value(source_record, "choice_probability_correct")
        paraphrase_p_correct = _probability_value(paraphrase_record, "choice_probability_correct")
        row = {
            "dataset": _as_text(source_record.get("dataset")),
            "split": _as_text(source_record.get("split")),
            "question_id": _as_text(source_record.get("question_id")),
            "draw_idx": int(source_record.get("draw_idx", 0) or 0),
            "template_type": _as_text(source_record.get("template_type")),
            "anti_sycophancy_request": _as_text(source_record.get("anti_sycophancy_request") or "none"),
            "source_example_id": _as_text(source_record.get("source_example_id")),
            "original_prompt_id": _as_text(source_record.get("prompt_id")),
            "paraphrase_prompt_id": _as_text(paraphrase_record.get("prompt_id")),
            "original_response": _as_text(source_record.get("response")),
            "paraphrase_response": _as_text(paraphrase_record.get("response")),
            "original_correctness": original_correctness,
            "paraphrase_correctness": paraphrase_correctness,
            "original_p_correct": original_p_correct,
            "paraphrase_p_correct": paraphrase_p_correct,
            "original_p_selected": _probability_value(source_record, "choice_probability_selected"),
            "paraphrase_p_selected": _probability_value(paraphrase_record, "choice_probability_selected"),
            "response_changed": bool(
                _as_text(source_record.get("response")) != _as_text(paraphrase_record.get("response"))
            ),
            "became_correct": bool(original_correctness == 0 and paraphrase_correctness == 1),
            "became_incorrect": bool(original_correctness == 1 and paraphrase_correctness == 0),
            "delta_correctness": None
            if original_correctness is None or paraphrase_correctness is None
            else int(paraphrase_correctness - original_correctness),
            "delta_p_correct": None,
        }
        if original_p_correct is not None and paraphrase_p_correct is not None:
            row["delta_p_correct"] = float(paraphrase_p_correct - original_p_correct)
        for field in NEUTRAL_REFERENCE_FIELDS:
            row[field] = source_record.get(field)
        item_rows.append(row)

    item_rows.sort(
        key=lambda row: (
            str(row.get("split", "") or ""),
            str(row.get("question_id", "") or ""),
            str(row.get("template_type", "") or ""),
            int(row.get("draw_idx", 0) or 0),
        )
    )
    summary_rows = summarize_external_paraphrase_rows(item_rows)
    coverage = {
        "schema_version": EXTERNAL_PARAPHRASE_SCHEMA_VERSION,
        "source_record_count": int(len(source_records)),
        "prepared_task_count": int(len(tasks)),
        "generated_record_count": int(len(paraphrase_records)),
        "computed_row_count": int(len(item_rows)),
        "summary_row_count": int(len(summary_rows)),
        "exclusion_counts": dict(sorted(coverage_counts.items())),
        "paraphrase_artifact_path": _as_text(paraphrase_artifact_path),
        "sampling_stats": dict(stats or {}),
    }
    if coverage["exclusion_counts"]:
        warn_status(
            _LOG_SOURCE,
            "external_paraphrase_exclusions",
            f"external paraphrase eval skipped rows: {coverage['exclusion_counts']}",
        )
    return {
        "schema_version": EXTERNAL_PARAPHRASE_SCHEMA_VERSION,
        "item_rows": item_rows,
        "summary_rows": summary_rows,
        "coverage": coverage,
    }


def summarize_external_paraphrase_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        template_type = str(row.get("template_type", "") or "")
        split = str(row.get("split", "") or "")
        grouped.setdefault((template_type, split, "all"), []).append(row)
        if _bool_value(row.get("neutral_source_is_correct")):
            grouped.setdefault((template_type, split, "neutral_source_is_correct"), []).append(row)

    def _mean(metric_rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
        values = []
        for row in metric_rows:
            try:
                numeric = float(row.get(field))
            except Exception:
                continue
            if np.isfinite(numeric):
                values.append(numeric)
        if not values:
            return None
        return float(np.mean(np.asarray(values, dtype=float)))

    for (template_type, split, subset_condition), metric_rows in sorted(grouped.items()):
        if not metric_rows:
            continue
        original_correctness = [
            int(row["original_correctness"])
            for row in metric_rows
            if row.get("original_correctness") in {0, 1}
        ]
        paraphrase_correctness = [
            int(row["paraphrase_correctness"])
            for row in metric_rows
            if row.get("paraphrase_correctness") in {0, 1}
        ]
        summary_rows.append(
            {
                "template_type": template_type,
                "split": split,
                "subset_condition": subset_condition,
                "n_rows": int(len(metric_rows)),
                "original_accuracy": None
                if not original_correctness
                else float(np.mean(np.asarray(original_correctness, dtype=float))),
                "paraphrase_accuracy": None
                if not paraphrase_correctness
                else float(np.mean(np.asarray(paraphrase_correctness, dtype=float))),
                "delta_accuracy": None
                if not original_correctness or not paraphrase_correctness
                else float(
                    np.mean(np.asarray(paraphrase_correctness, dtype=float))
                    - np.mean(np.asarray(original_correctness, dtype=float))
                ),
                "response_change_rate": float(
                    np.mean([1.0 if _bool_value(row.get("response_changed")) else 0.0 for row in metric_rows])
                ),
                "mean_delta_p_correct": _mean(metric_rows, "delta_p_correct"),
                "became_correct_rate": float(
                    np.mean([1.0 if _bool_value(row.get("became_correct")) else 0.0 for row in metric_rows])
                ),
                "became_incorrect_rate": float(
                    np.mean([1.0 if _bool_value(row.get("became_incorrect")) else 0.0 for row in metric_rows])
                ),
            }
        )
    return summary_rows


__all__ = [
    "EXTERNAL_PARAPHRASE_SCHEMA_VERSION",
    "NEUTRAL_REFERENCE_FIELDS",
    "annotate_records_with_neutral_references",
    "build_external_pair_metrics_rows",
    "evaluate_external_paraphrases",
    "summarize_external_pair_metrics_rows",
    "summarize_external_paraphrase_rows",
]
