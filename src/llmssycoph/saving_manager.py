from __future__ import annotations

from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .data import prompt_id_for
from .grading import record_is_usable_for_metrics as _record_is_usable_for_metrics
from .llm.sampling import normalize_sample_records
from .logging_utils import log_status
from .runtime import (
    model_slug,
    preferred_run_artifact_path,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)


SAMPLED_RESPONSE_COLUMNS = [
    "model_name",
    "record_id",
    "split",
    "question_id",
    "prompt_id",
    "dataset",
    "template_type",
    "draw_idx",
    "task_format",
    "mc_mode",
    "answer_channel",
    "question",
    "correct_answer",
    "incorrect_answer",
    "correct_letter",
    "incorrect_letter",
    "prompt_template",
    "prompt_text",
    "response_raw",
    "response",
    "correctness",
    "grading_status",
    "grading_reason",
    "usable_for_metrics",
    "starts_with_answer_prefix",
    "strict_format_exact",
    "completion_token_count",
    "hit_max_new_tokens",
    "stopped_on_eos",
    "finish_reason",
    "sampling_mode",
    "P(correct)",
    "P(selected)",
    "T_prompt",
    "probe_x",
    "probe_xprime",
]

P_CORRECT_COLUMN = "P(correct)"
P_SELECTED_COLUMN = "P(selected)"
_LEGACY_P_CORRECT_COLUMN = "choice_probability_correct"
_LEGACY_P_SELECTED_COLUMN = "choice_probability_selected"

SUMMARY_COLUMNS = [
    "model_name",
    "split",
    "question_id",
    "prompt_id_x",
    "prompt_id_xprime",
    "dataset",
    "bias_type",
    "question",
    "correct_answer",
    "incorrect_answer",
    "prompt_template_x",
    "prompt_template_xprime",
    "prompt_x",
    "prompt_with_bias",
    "T_x",
    "T_xprime",
    "mean_C_x",
    "mean_C_xprime",
    "mean_probe_x",
    "mean_probe_xprime",
    "n_draws",
]

PROBE_CANDIDATE_SCORE_COLUMNS = [
    "model_name",
    "probe_name",
    "split",
    "question_id",
    "prompt_id",
    "template_type",
    "draw_idx",
    "source_record_id",
    "candidate_record_id",
    "correct_letter",
    "selected_choice",
    "candidate_choice",
    "candidate_rank",
    "candidate_probability",
    "probe_sample_weight",
    "candidate_correctness",
    "candidate_is_selected",
    "probe_score",
]

RUN_SUMMARY_SCHEMA_VERSION = 2
MODEL_SUMMARY_SCHEMA_VERSION = 1
PROBE_SUMMARY_SCHEMA_VERSION = 1
REPORTS_SUMMARY_SCHEMA_VERSION = 5

_MC_CONFUSION_STANDALONE_LETTER_RE = re.compile(r"^\(?([A-Za-z])\)?[\]\).,:;\-]?$")
_MC_CONFUSION_AMBIGUOUS_LETTER_RE = re.compile(
    r"^\(?[A-Za-z]\)?(?:\s*(?:or|/|,)\s*\(?[A-Za-z]\)?)+$",
    flags=re.IGNORECASE,
)
_MC_CONFUSION_EXPLICIT_LETTER_RE = re.compile(
    r"\b(?:final\s+answer|correct\s+answer|answer|option|choice)\s*(?:is|:)?\s*\(?([A-Za-z])\)?\b",
    flags=re.IGNORECASE,
)

MODEL_SUMMARY_BY_TEMPLATE_COLUMNS = [
    "template_type",
    "n_rows",
    "n_questions",
    "n_usable_rows",
    "usable_rate",
    "ambiguous_rate",
    "accuracy",
    "avg_p_correct",
    "avg_p_selected",
    "avg_selected_minus_correct_probability_gap",
    "avg_probe_score_selected_prompt",
    "exact_format_rate",
    "starts_with_answer_prefix_rate",
    "cap_hit_rate",
    "stopped_on_eos_rate",
    "avg_completion_token_count",
]

MODEL_SUMMARY_BY_BIAS_COLUMNS = [
    "bias_type",
    "n_pairs",
    "n_questions",
    "accuracy_x",
    "accuracy_xprime",
    "delta_accuracy_x_minus_xprime",
    "avg_p_x",
    "avg_p_xprime",
    "avg_delta_p_x_minus_xprime",
    "avg_probe_x",
    "avg_probe_xprime",
    "avg_delta_probe_x_minus_xprime",
    "harmful_flip_rate",
    "helpful_flip_rate",
    "unchanged_correctness_rate",
    "answer_change_rate",
]

REPORTS_SUMMARY_COLUMNS = [
    "bias_type",
    "n_prompt_rows",
    "n_questions",
    "n_usable_prompt_rows",
    "usable_rate",
    "ambiguous_rate",
    "accuracy",
    "avg_p_correct",
    "avg_p_selected",
    "avg_selected_minus_correct_probability_gap",
    "avg_probe_score_selected_prompt",
    "exact_format_rate",
    "starts_with_answer_prefix_rate",
    "cap_hit_rate",
    "stopped_on_eos_rate",
    "avg_completion_token_count",
    "n_pairs",
    "neutral_accuracy",
    "biased_accuracy",
    "delta_accuracy_biased_minus_neutral",
    "neutral_avg_p_correct",
    "biased_avg_p_correct",
    "avg_delta_p_biased_minus_neutral",
    "neutral_avg_probe",
    "biased_avg_probe",
    "avg_delta_probe_biased_minus_neutral",
    "harmful_flip_rate",
    "helpful_flip_rate",
    "unchanged_correctness_rate",
    "answer_change_rate",
]

PROBE_SUMMARY_COLUMNS = [
    "probe_name",
    "template_type",
    "probe_construction",
    "probe_example_weighting",
    "best_layer",
    "best_dev_auc",
    "train_auc",
    "val_auc",
    "test_auc",
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
    "train_balanced_accuracy",
    "val_balanced_accuracy",
    "test_balanced_accuracy",
    "train_n_total",
    "val_n_total",
    "test_n_total",
    "train_minus_val_auc",
    "val_minus_test_auc",
    "train_minus_test_auc",
    "probe_prefers_correct_rate",
    "probe_prefers_selected_rate",
    "mean_probe_score_correct_candidate",
    "mean_probe_score_incorrect_candidate",
    "mean_probe_score_selected_candidate",
    "mean_probe_score_non_selected_candidate",
    "mean_probe_score_correct_choice",
    "mean_probe_score_selected_choice",
    "mean_correct_minus_selected_probe_gap",
]

MC_PROBE_SCORE_BY_PROMPT_BASE_COLUMNS = [
    "model_name",
    "probe_name",
    "split",
    "question_id",
    "prompt_id",
    "dataset",
    "template_type",
    "draw_idx",
    "source_record_id",
    "correct_letter",
    "selected_choice",
    "selected_choice_is_correct",
    "probe_score_correct_choice",
    "probe_score_selected_choice",
    "correct_choice_probability",
    "selected_choice_probability",
    "probe_argmax_choice",
    "probe_argmax_score",
    "probe_prefers_correct",
    "probe_prefers_selected",
    "probe_score_gap_correct_minus_selected",
]


def build_tuple_rows(
    records: List[Dict[str, Any]],
    model_name: str,
    bias_types: Sequence[str],
) -> List[Dict[str, Any]]:
    by_key = {
        (record["split"], record["question_id"], record["template_type"], record["draw_idx"]): record
        for record in records
    }
    neutral_keys = sorted(
        [key for key in by_key if key[2] == "neutral"],
        key=lambda key: (key[0], key[1], key[3]),
    )

    rows: List[Dict[str, Any]] = []
    for split, question_id, _, draw_idx in neutral_keys:
        neutral_record = by_key[(split, question_id, "neutral", draw_idx)]
        if (
            str(neutral_record.get("task_format", "") or "") == "multiple_choice"
            and str(neutral_record.get("mc_mode", "") or "")
            and str(neutral_record.get("mc_mode", "") or "") != "strict_mc"
        ):
            continue
        if not _record_is_usable_for_metrics(neutral_record):
            continue
        for bias_type in bias_types:
            bias_record = by_key.get((split, question_id, bias_type, draw_idx))
            if bias_record is None or not _record_is_usable_for_metrics(bias_record):
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "question_id": question_id,
                    "prompt_id_x": neutral_record.get("prompt_id", prompt_id_for(question_id, "neutral")),
                    "prompt_id_xprime": bias_record.get("prompt_id", prompt_id_for(question_id, bias_type)),
                    "dataset": neutral_record.get("dataset", ""),
                    "bias_type": bias_type,
                    "draw_idx": draw_idx,
                    "question": neutral_record["question"],
                    "correct_answer": neutral_record["correct_answer"],
                    "incorrect_answer": neutral_record["incorrect_answer"],
                    "gold_answers": json.dumps(neutral_record["gold_answers"], ensure_ascii=False),
                    "prompt_x": neutral_record["prompt_text"],
                    "prompt_with_bias": bias_record["prompt_text"],
                    "prompt_template_x": neutral_record["prompt_template"],
                    "prompt_template_xprime": bias_record["prompt_template"],
                    "y_x": neutral_record["response"],
                    "y_xprime": bias_record["response"],
                    "C_x_y": int(neutral_record["correctness"]),
                    "C_xprime_yprime": int(bias_record["correctness"]),
                    "T_x": float(neutral_record["T_prompt"]),
                    "T_xprime": float(bias_record["T_prompt"]),
                    "probe_x_name": "probe_no_bias",
                    "probe_xprime_name": f"probe_bias_{bias_type}",
                    "probe_x": neutral_record.get("probe_x", np.nan),
                    "probe_xprime": bias_record.get("probe_xprime", np.nan),
                }
            )
    return rows


def to_samples_df(records: List[Dict[str, Any]], model_name: str) -> pd.DataFrame:
    choice_probability_columns = _sample_choice_probability_columns(records)
    rows = []
    for record in records:
        correctness = record.get("correctness")
        row = {
            "model_name": model_name,
            "record_id": record["record_id"],
            "split": record["split"],
            "question_id": record["question_id"],
            "prompt_id": record.get(
                "prompt_id",
                prompt_id_for(record.get("question_id", ""), record.get("template_type", "")),
            ),
            "dataset": record.get("dataset", ""),
            "template_type": record["template_type"],
            "draw_idx": record["draw_idx"],
            "task_format": record.get("task_format", ""),
            "mc_mode": record.get("mc_mode", ""),
            "answer_channel": record.get("answer_channel", ""),
            "question": record["question"],
            "correct_answer": record["correct_answer"],
            "incorrect_answer": record["incorrect_answer"],
            "correct_letter": record.get("correct_letter", ""),
            "incorrect_letter": record.get("incorrect_letter", ""),
            "prompt_template": record["prompt_template"],
            "prompt_text": record["prompt_text"],
            "response_raw": record["response_raw"],
            "response": record["response"],
            "starts_with_answer_prefix": bool(record.get("starts_with_answer_prefix", False)),
            "strict_format_exact": bool(record.get("strict_format_exact", False)),
            "correctness": np.nan if correctness is None else int(correctness),
            "grading_status": record.get(
                "grading_status",
                "graded" if _record_is_usable_for_metrics(record) else "ambiguous",
            ),
            "grading_reason": record.get("grading_reason", ""),
            "usable_for_metrics": bool(
                record.get("usable_for_metrics", _record_is_usable_for_metrics(record))
            ),
            "completion_token_count": record.get("completion_token_count", np.nan),
            "hit_max_new_tokens": bool(record.get("hit_max_new_tokens", False)),
            "stopped_on_eos": bool(record.get("stopped_on_eos", False)),
            "finish_reason": record.get("finish_reason", ""),
            "sampling_mode": record.get("sampling_mode", "generation"),
            P_CORRECT_COLUMN: record.get(P_CORRECT_COLUMN, record.get(_LEGACY_P_CORRECT_COLUMN, np.nan)),
            P_SELECTED_COLUMN: record.get(P_SELECTED_COLUMN, record.get(_LEGACY_P_SELECTED_COLUMN, np.nan)),
            "T_prompt": float(record["T_prompt"]),
            "probe_x": record.get("probe_x", np.nan),
            "probe_xprime": record.get("probe_xprime", np.nan),
        }
        probabilities_raw = record.get("choice_probabilities", {})
        if isinstance(probabilities_raw, dict):
            for raw_choice, probability in probabilities_raw.items():
                choice = str(raw_choice or "").strip().upper()
                if choice:
                    row[_choice_probability_column(choice)] = probability
        rows.append(row)
    return pd.DataFrame(rows, columns=SAMPLED_RESPONSE_COLUMNS + choice_probability_columns)


def to_tuples_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def to_probe_candidate_scores_df(rows: List[Dict[str, Any]], model_name: str) -> pd.DataFrame:
    payload = []
    for row in rows:
        candidate_correctness = row.get("candidate_correctness", row.get("correctness"))
        try:
            candidate_correctness_value = np.nan if candidate_correctness is None else int(candidate_correctness)
        except Exception:
            candidate_correctness_value = np.nan
        payload.append(
            {
                "model_name": model_name,
                "probe_name": str(row.get("probe_name", "") or ""),
                "split": str(row.get("split", "") or ""),
                "question_id": str(row.get("question_id", "") or ""),
                "prompt_id": str(
                    row.get(
                        "prompt_id",
                        prompt_id_for(row.get("question_id", ""), row.get("template_type", "")),
                    )
                ),
                "template_type": str(row.get("template_type", "") or ""),
                "draw_idx": int(row.get("draw_idx", 0) or 0),
                "source_record_id": row.get("source_record_id", np.nan),
                "candidate_record_id": row.get("record_id", np.nan),
                "correct_letter": str(row.get("correct_letter", "") or ""),
                "selected_choice": str(row.get("source_selected_choice", row.get("response_raw", "")) or ""),
                "candidate_choice": str(row.get("candidate_choice", row.get("response_raw", "")) or ""),
                "candidate_rank": int(row.get("candidate_rank", 0) or 0),
                "candidate_probability": row.get("candidate_probability", np.nan),
                "probe_sample_weight": row.get("probe_sample_weight", np.nan),
                "candidate_correctness": candidate_correctness_value,
                "candidate_is_selected": bool(row.get("candidate_is_selected", False)),
                "probe_score": row.get("probe_score", np.nan),
            }
        )
    return pd.DataFrame(payload, columns=PROBE_CANDIDATE_SCORE_COLUMNS)


def build_summary_df(tuples_df: pd.DataFrame) -> pd.DataFrame:
    if len(tuples_df) == 0:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    working_df = tuples_df.copy()
    for column_name in ("dataset", "prompt_id_x", "prompt_id_xprime", "prompt_template_x", "prompt_template_xprime"):
        if column_name not in working_df.columns:
            working_df[column_name] = ""

    summary_df = (
        working_df.groupby(["model_name", "split", "question_id", "dataset", "bias_type"], as_index=False)
        .agg(
            prompt_id_x=("prompt_id_x", "first"),
            prompt_id_xprime=("prompt_id_xprime", "first"),
            question=("question", "first"),
            correct_answer=("correct_answer", "first"),
            incorrect_answer=("incorrect_answer", "first"),
            prompt_template_x=("prompt_template_x", "first"),
            prompt_template_xprime=("prompt_template_xprime", "first"),
            prompt_x=("prompt_x", "first"),
            prompt_with_bias=("prompt_with_bias", "first"),
            T_x=("T_x", "first"),
            T_xprime=("T_xprime", "first"),
            mean_C_x=("C_x_y", "mean"),
            mean_C_xprime=("C_xprime_yprime", "mean"),
            mean_probe_x=("probe_x", "mean"),
            mean_probe_xprime=("probe_xprime", "mean"),
            n_draws=("draw_idx", "nunique"),
        )
    )
    return summary_df[SUMMARY_COLUMNS]


def _float_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _list_like_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    return value


def _mean_or_none(series: pd.Series) -> Optional[float]:
    if not isinstance(series, pd.Series):
        series = pd.Series([series])
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _count_true(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return int((numeric > 0).sum())


def _counts_as_dict(series: pd.Series) -> Dict[str, int]:
    values = [str(value) for value in series.fillna("").astype(str) if str(value).strip()]
    return {key: int(count) for key, count in sorted(Counter(values).items())}


def _series_from_df(df: pd.DataFrame, column_name: str, default: Any = np.nan) -> pd.Series:
    if column_name in df.columns:
        return df[column_name]
    return pd.Series([default] * len(df), index=df.index)


def _choice_probability_column(choice_label: str) -> str:
    return f"P({choice_label})"


def _choice_labels_for_record(record: Dict[str, Any]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    letters = str(record.get("letters", "") or "").strip().upper()
    for choice in letters:
        if choice and choice not in seen:
            ordered.append(choice)
            seen.add(choice)

    probabilities_raw = record.get("choice_probabilities", {})
    if isinstance(probabilities_raw, dict):
        for raw_choice in probabilities_raw:
            choice = str(raw_choice or "").strip().upper()
            if choice and choice not in seen:
                ordered.append(choice)
                seen.add(choice)

    return ordered


def _mc_option_count_from_record(record: Dict[str, Any]) -> Optional[int]:
    if str(record.get("task_format", "") or "") != "multiple_choice":
        return None

    candidate_counts: List[int] = []

    choice_labels = _choice_labels_for_record(record)
    if choice_labels:
        candidate_counts.append(len(choice_labels))

    answers_list = record.get("answers_list")
    if isinstance(answers_list, list):
        non_empty_answers = [str(answer).strip() for answer in answers_list if str(answer).strip()]
        if non_empty_answers:
            candidate_counts.append(len(non_empty_answers))

    if not candidate_counts:
        return None
    return max(candidate_counts)


def _sample_choice_probability_columns(records: Sequence[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    seen: Set[str] = set()
    for record in records:
        for choice in _choice_labels_for_record(record):
            column_name = _choice_probability_column(choice)
            if column_name not in seen:
                columns.append(column_name)
                seen.add(column_name)
    return columns


def _sample_probability_series(
    df: pd.DataFrame,
    current_column_name: str,
    legacy_column_name: str,
    default: Any = np.nan,
) -> pd.Series:
    if current_column_name in df.columns:
        return df[current_column_name]
    if legacy_column_name in df.columns:
        return df[legacy_column_name]
    return pd.Series([default] * len(df), index=df.index)


def _choice_probability_labels_from_df(samples_df: pd.DataFrame) -> List[str]:
    labels: List[str] = []
    for column_name in samples_df.columns:
        if not (column_name.startswith("P(") and column_name.endswith(")")):
            continue
        if column_name in {P_CORRECT_COLUMN, P_SELECTED_COLUMN}:
            continue
        labels.append(column_name[2:-1])
    return sorted(set(labels), key=lambda label: (len(str(label)), str(label)))


def _choice_probability_map_from_row(row: pd.Series, choice_labels: Sequence[str]) -> Dict[str, float]:
    probabilities: Dict[str, float] = {}
    for label in choice_labels:
        numeric = _float_or_none(row.get(_choice_probability_column(label)))
        if numeric is None or numeric < 0.0:
            continue
        probabilities[str(label)] = float(numeric)
    return probabilities


def _selected_choice_from_probability_map(probabilities: Dict[str, float], choice_labels: Sequence[str]) -> str:
    selected_choice = ""
    selected_probability = float("-inf")
    for label in choice_labels:
        numeric = probabilities.get(str(label))
        if numeric is None:
            continue
        if numeric > selected_probability:
            selected_choice = str(label)
            selected_probability = float(numeric)
    return selected_choice


def _effective_option_count_from_probability_map(probabilities: Dict[str, float]) -> Optional[float]:
    if not probabilities:
        return None
    total_mass = float(sum(probabilities.values()))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        return None

    entropy = 0.0
    for probability in probabilities.values():
        normalized = float(probability) / total_mass
        if normalized > 0.0:
            entropy -= normalized * math.log(normalized)
    return float(math.exp(entropy))


def _mc_option_count_values_from_sample_records(
    sample_records: Optional[Sequence[Dict[str, Any]]],
) -> List[int]:
    if not sample_records:
        return []

    option_counts: List[int] = []
    for record in sample_records:
        if not isinstance(record, dict):
            continue
        option_count = _mc_option_count_from_record(record)
        if option_count is not None and option_count > 0:
            option_counts.append(int(option_count))
    return option_counts


def _mc_option_count_values_from_samples_df(samples_df: pd.DataFrame) -> List[int]:
    if samples_df.empty:
        return []

    choice_labels = _choice_probability_labels_from_df(samples_df)
    if not choice_labels:
        return []

    eligible_mask = pd.Series([True] * len(samples_df), index=samples_df.index)
    if "task_format" in samples_df.columns:
        eligible_mask = eligible_mask & (
            _series_from_df(samples_df, "task_format", "").astype(str) == "multiple_choice"
        )

    eligible_df = samples_df[eligible_mask].copy()
    if eligible_df.empty:
        return []

    option_counts: List[int] = []
    for _, row in eligible_df.iterrows():
        option_count = 0
        for label in choice_labels:
            if _float_or_none(row.get(_choice_probability_column(label))) is not None:
                option_count += 1
        if option_count > 0:
            option_counts.append(option_count)
    return option_counts


def _format_mc_option_count_display(option_counts: Sequence[int]) -> Optional[str]:
    normalized_counts = sorted({int(count) for count in option_counts if int(count) > 0})
    if not normalized_counts:
        return None
    if len(normalized_counts) == 1:
        return str(normalized_counts[0])

    contiguous_counts = list(range(normalized_counts[0], normalized_counts[-1] + 1))
    if normalized_counts == contiguous_counts:
        count_text = f"{normalized_counts[0]}-{normalized_counts[-1]}"
    else:
        count_text = "/".join(str(count) for count in normalized_counts)
    return f"{count_text} (varies by question)"


def _build_mc_option_count_summary(
    samples_df: pd.DataFrame,
    *,
    sample_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    option_counts = _mc_option_count_values_from_sample_records(sample_records)
    if not option_counts:
        option_counts = _mc_option_count_values_from_samples_df(samples_df)

    observed_counts = sorted({int(count) for count in option_counts if int(count) > 0})
    return _json_ready(
        {
            "observed_counts": observed_counts,
            "n_multiple_choice_rows": int(len(option_counts)),
            "display": _format_mc_option_count_display(observed_counts),
        }
    )


def _build_mc_option_selection_row(
    samples_df: pd.DataFrame,
    *,
    template_type: str,
    choice_labels: Sequence[str],
) -> Dict[str, Any]:
    selected_counts: Counter[str] = Counter()
    effective_option_counts: List[float] = []
    n_probability_rows = 0

    for _, row in samples_df.iterrows():
        probabilities = _choice_probability_map_from_row(row, choice_labels)
        if not probabilities:
            continue
        n_probability_rows += 1
        selected_choice = _selected_choice_from_probability_map(probabilities, choice_labels)
        if selected_choice:
            selected_counts[selected_choice] += 1
        effective_options = _effective_option_count_from_probability_map(probabilities)
        if effective_options is not None:
            effective_option_counts.append(float(effective_options))

    summary_row: Dict[str, Any] = {
        "template_type": str(template_type),
        "n_rows": int(n_probability_rows),
        "avg_effective_options": (
            None
            if not effective_option_counts
            else float(np.mean(np.asarray(effective_option_counts, dtype=float)))
        ),
    }
    for label in choice_labels:
        summary_row[f"pick_{label}_rate"] = (
            None
            if n_probability_rows <= 0
            else float(selected_counts.get(str(label), 0)) / float(n_probability_rows)
        )
    return _json_ready(summary_row)


def _build_mc_option_selection_summary(
    samples_df: pd.DataFrame,
    *,
    bias_types: Sequence[str],
) -> Dict[str, Any]:
    choice_labels = _choice_probability_labels_from_df(samples_df)
    if not choice_labels:
        return {"choice_labels": [], "summary_rows": [], "n_probability_rows": 0}

    eligible_mask = pd.Series([True] * len(samples_df), index=samples_df.index)
    if "task_format" in samples_df.columns:
        eligible_mask = eligible_mask & (
            _series_from_df(samples_df, "task_format", "").astype(str) == "multiple_choice"
        )

    probability_present_mask = pd.Series([False] * len(samples_df), index=samples_df.index)
    for label in choice_labels:
        numeric = pd.to_numeric(_series_from_df(samples_df, _choice_probability_column(label)), errors="coerce")
        probability_present_mask = probability_present_mask | numeric.notna()
    eligible_df = samples_df[eligible_mask & probability_present_mask].copy()
    if eligible_df.empty:
        return {"choice_labels": list(choice_labels), "summary_rows": [], "n_probability_rows": 0}

    configured_bias_types = [bias_type for bias_type in _list_like_strings(bias_types) if bias_type != "neutral"]
    observed_template_types = sorted(
        {
            str(value).strip()
            for value in _series_from_df(eligible_df, "template_type", "").dropna().astype(str)
            if str(value).strip()
        }
    )
    ordered_template_types = ["neutral"] + configured_bias_types + [
        template_type
        for template_type in observed_template_types
        if template_type not in {"neutral", *set(configured_bias_types)}
    ]

    rows = [
        _build_mc_option_selection_row(
            eligible_df,
            template_type="overall",
            choice_labels=choice_labels,
        )
    ]
    template_series = _series_from_df(eligible_df, "template_type", "").astype(str)
    for template_type in ordered_template_types:
        template_df = eligible_df[template_series == template_type].copy()
        if template_df.empty:
            continue
        rows.append(
            _build_mc_option_selection_row(
                template_df,
                template_type=template_type,
                choice_labels=choice_labels,
            )
        )

    return _json_ready(
        {
            "choice_labels": list(choice_labels),
            "summary_rows": rows,
            "n_probability_rows": int(len(eligible_df)),
        }
    )


def _bool_like(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return bool(value)


def _mc_confusion_choice_labels_from_record(record: Dict[str, Any]) -> List[str]:
    labels: Set[str] = set()

    letters = str(record.get("letters", "") or "").strip().upper()
    for letter in letters:
        if len(letter) == 1 and letter.isalpha():
            labels.add(letter)

    probabilities_raw = record.get("choice_probabilities", {})
    if isinstance(probabilities_raw, dict):
        for raw_choice in probabilities_raw:
            label = str(raw_choice or "").strip().upper()
            if len(label) == 1 and label.isalpha():
                labels.add(label)

    for column_name in record:
        if not (column_name.startswith("P(") and column_name.endswith(")")):
            continue
        if column_name in {P_CORRECT_COLUMN, P_SELECTED_COLUMN}:
            continue
        label = str(column_name[2:-1]).strip().upper()
        if len(label) == 1 and label.isalpha():
            labels.add(label)

    for field_name in ("correct_letter", "incorrect_letter"):
        label = str(record.get(field_name, "") or "").strip().upper()
        if len(label) == 1 and label.isalpha():
            labels.add(label)

    return sorted(labels, key=lambda label: (len(label), label))


def _mc_confusion_exact_letter(value: Any, allowed_labels: Set[str]) -> str:
    if not allowed_labels:
        return ""

    text = str(value or "").strip().strip(" \"'“”‘’\t")
    if not text:
        return ""
    if _MC_CONFUSION_AMBIGUOUS_LETTER_RE.fullmatch(text):
        return ""

    match = _MC_CONFUSION_STANDALONE_LETTER_RE.fullmatch(text)
    if not match:
        return ""

    letter = match.group(1).upper()
    return letter if letter in allowed_labels else ""


def _mc_confusion_letter_from_text(value: Any, allowed_labels: Set[str]) -> str:
    if not allowed_labels:
        return ""

    text = str(value or "").strip()
    if not text:
        return ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        exact = _mc_confusion_exact_letter(line, allowed_labels)
        if exact:
            return exact
        if _MC_CONFUSION_AMBIGUOUS_LETTER_RE.fullmatch(line):
            return ""
        explicit_match = _MC_CONFUSION_EXPLICIT_LETTER_RE.search(line)
        if not explicit_match:
            continue
        letter = explicit_match.group(1).upper()
        if letter in allowed_labels:
            return letter

    exact = _mc_confusion_exact_letter(text, allowed_labels)
    if exact:
        return exact

    explicit_match = _MC_CONFUSION_EXPLICIT_LETTER_RE.search(text)
    if not explicit_match:
        return ""
    letter = explicit_match.group(1).upper()
    return letter if letter in allowed_labels else ""


def _mc_confusion_probability_map(
    record: Dict[str, Any],
    choice_labels: Sequence[str],
) -> Dict[str, float]:
    probabilities: Dict[str, float] = {}

    probabilities_raw = record.get("choice_probabilities", {})
    if isinstance(probabilities_raw, dict):
        for raw_choice, raw_value in probabilities_raw.items():
            label = str(raw_choice or "").strip().upper()
            if label not in choice_labels:
                continue
            numeric = _float_or_none(raw_value)
            if numeric is None or numeric < 0.0:
                continue
            probabilities[label] = float(numeric)

    for label in choice_labels:
        if label in probabilities:
            continue
        numeric = _float_or_none(record.get(_choice_probability_column(label)))
        if numeric is None or numeric < 0.0:
            continue
        probabilities[label] = float(numeric)

    return probabilities


def _mc_confusion_predicted_letter(
    record: Dict[str, Any],
    allowed_labels: Set[str],
) -> str:
    committed_letter = _mc_confusion_exact_letter(record.get("committed_answer"), allowed_labels)
    if committed_letter:
        return committed_letter

    ordered_labels = sorted(allowed_labels, key=lambda label: (len(label), label))
    probability_map = _mc_confusion_probability_map(record, ordered_labels)
    selected_choice = _selected_choice_from_probability_map(probability_map, ordered_labels)
    if selected_choice:
        return selected_choice

    for field_name in ("response", "response_raw"):
        parsed_letter = _mc_confusion_letter_from_text(record.get(field_name), allowed_labels)
        if parsed_letter:
            return parsed_letter
    return ""


def _build_mc_confusion_matrix_summary(
    samples_df: pd.DataFrame,
    *,
    sample_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    source_rows = sample_records if sample_records is not None else samples_df.to_dict(orient="records")
    mc_rows = [
        dict(record)
        for record in source_rows
        if isinstance(record, dict) and str(record.get("task_format", "") or "") == "multiple_choice"
    ]
    if not mc_rows:
        return None

    labels: Set[str] = set()
    mc_row_labels: List[Set[str]] = []
    for record in mc_rows:
        row_labels = set(_mc_confusion_choice_labels_from_record(record))
        mc_row_labels.append(row_labels)
        labels.update(row_labels)

    parsed_pairs: List[tuple[str, str]] = []
    usable_mc_rows = 0
    for record, row_labels in zip(mc_rows, mc_row_labels):
        if not _bool_like(record.get("usable_for_metrics", False)):
            continue
        usable_mc_rows += 1
        allowed_labels = set(row_labels) or set(labels)
        if not allowed_labels:
            continue

        true_letter = _mc_confusion_exact_letter(record.get("correct_letter"), allowed_labels)
        predicted_letter = _mc_confusion_predicted_letter(record, allowed_labels)
        if not true_letter or not predicted_letter:
            continue
        labels.update({true_letter, predicted_letter})
        parsed_pairs.append((predicted_letter, true_letter))

    ordered_labels = sorted(labels, key=lambda label: (len(label), label))
    counts = {
        predicted_label: {true_label: 0 for true_label in ordered_labels}
        for predicted_label in ordered_labels
    }
    for predicted_letter, true_letter in parsed_pairs:
        counts[predicted_letter][true_letter] += 1

    summary_rows = [
        {
            "predicted_letter": predicted_label,
            **{true_label: int(counts[predicted_label][true_label]) for true_label in ordered_labels},
        }
        for predicted_label in ordered_labels
    ]
    return _json_ready(
        {
            "choice_labels": ordered_labels,
            "n_mc_rows": int(len(mc_rows)),
            "n_usable_mc_rows": int(usable_mc_rows),
            "n_confusion_rows": int(len(parsed_pairs)),
            "summary_rows": summary_rows,
        }
    )


def _mc_confusion_matrix_df_from_summary(summary: Any) -> pd.DataFrame:
    if not isinstance(summary, dict):
        return pd.DataFrame()

    choice_labels = [str(label) for label in summary.get("choice_labels", []) if str(label)]
    columns = ["predicted_letter", *choice_labels] if choice_labels else ["predicted_letter"]
    rows = summary.get("summary_rows", [])
    if not isinstance(rows, list):
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _format_duration_human(seconds: Any) -> str:
    numeric = _float_or_none(seconds)
    if numeric is None:
        return "n/a"
    total_seconds = max(0, int(round(float(numeric))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _selected_probe_score_series(samples_df: pd.DataFrame) -> pd.Series:
    if samples_df.empty:
        return pd.Series(dtype=float)

    probe_x = pd.to_numeric(_series_from_df(samples_df, "probe_x"), errors="coerce")
    probe_xprime = pd.to_numeric(_series_from_df(samples_df, "probe_xprime"), errors="coerce")
    template_type = _series_from_df(samples_df, "template_type", "").astype(str)
    selected = probe_x.where(template_type == "neutral", probe_xprime)
    return selected.where(~selected.isna(), probe_x.combine_first(probe_xprime))


def _prompt_metrics_from_df(samples_df: pd.DataFrame) -> Dict[str, Any]:
    if samples_df.empty:
        return {
            "n_rows": 0,
            "n_questions": 0,
            "n_usable_rows": 0,
            "usable_rate": None,
            "ambiguous_rate": None,
            "accuracy": None,
            "avg_p_correct": None,
            "avg_p_selected": None,
            "avg_selected_minus_correct_probability_gap": None,
            "avg_probe_score_selected_prompt": None,
            "exact_format_rate": None,
            "starts_with_answer_prefix_rate": None,
            "cap_hit_rate": None,
            "stopped_on_eos_rate": None,
            "avg_completion_token_count": None,
            "grading_status_counts": {},
            "finish_reason_counts": {},
            "sampling_mode_counts": {},
        }

    usable_series = pd.to_numeric(_series_from_df(samples_df, "usable_for_metrics", 0), errors="coerce").fillna(0.0)
    # T_prompt is the prompt-level soft correctness field:
    # strict-MC uses exact P(correct), while generation-based paths use
    # empirical mean correctness over repeated usable draws.
    p_correct = pd.to_numeric(_series_from_df(samples_df, "T_prompt"), errors="coerce")
    p_selected = pd.to_numeric(
        _sample_probability_series(samples_df, P_SELECTED_COLUMN, _LEGACY_P_SELECTED_COLUMN),
        errors="coerce",
    )
    selected_probe_score = _selected_probe_score_series(samples_df)
    metrics = {
        "n_rows": int(len(samples_df)),
        "n_questions": int(samples_df["question_id"].nunique()) if "question_id" in samples_df.columns else 0,
        "n_usable_rows": _count_true(usable_series),
        "usable_rate": _mean_or_none(usable_series),
        "ambiguous_rate": _mean_or_none(1.0 - usable_series),
        "accuracy": _mean_or_none(_series_from_df(samples_df, "correctness")),
        "avg_p_correct": _mean_or_none(p_correct),
        "avg_p_selected": _mean_or_none(p_selected),
        "avg_selected_minus_correct_probability_gap": _mean_or_none(p_selected - p_correct),
        "avg_probe_score_selected_prompt": _mean_or_none(selected_probe_score),
        "exact_format_rate": _mean_or_none(_series_from_df(samples_df, "strict_format_exact")),
        "starts_with_answer_prefix_rate": _mean_or_none(_series_from_df(samples_df, "starts_with_answer_prefix")),
        "cap_hit_rate": _mean_or_none(_series_from_df(samples_df, "hit_max_new_tokens")),
        "stopped_on_eos_rate": _mean_or_none(_series_from_df(samples_df, "stopped_on_eos")),
        "avg_completion_token_count": _mean_or_none(_series_from_df(samples_df, "completion_token_count")),
        "grading_status_counts": _counts_as_dict(_series_from_df(samples_df, "grading_status", "")),
        "finish_reason_counts": _counts_as_dict(_series_from_df(samples_df, "finish_reason", "")),
        "sampling_mode_counts": _counts_as_dict(_series_from_df(samples_df, "sampling_mode", "")),
    }
    return _json_ready(metrics)


def _pair_metrics_from_df(tuples_df: pd.DataFrame) -> Dict[str, Any]:
    if tuples_df.empty:
        return {
            "n_pairs": 0,
            "n_questions": 0,
            "accuracy_x": None,
            "accuracy_xprime": None,
            "delta_accuracy_x_minus_xprime": None,
            "avg_p_x": None,
            "avg_p_xprime": None,
            "avg_delta_p_x_minus_xprime": None,
            "avg_probe_x": None,
            "avg_probe_xprime": None,
            "avg_delta_probe_x_minus_xprime": None,
            "harmful_flip_rate": None,
            "helpful_flip_rate": None,
            "unchanged_correctness_rate": None,
            "answer_change_rate": None,
        }

    accuracy_x = pd.to_numeric(tuples_df.get("C_x_y", np.nan), errors="coerce")
    accuracy_xprime = pd.to_numeric(tuples_df.get("C_xprime_yprime", np.nan), errors="coerce")
    p_x = pd.to_numeric(tuples_df.get("T_x", np.nan), errors="coerce")
    p_xprime = pd.to_numeric(tuples_df.get("T_xprime", np.nan), errors="coerce")
    probe_x = pd.to_numeric(tuples_df.get("probe_x", np.nan), errors="coerce")
    probe_xprime = pd.to_numeric(tuples_df.get("probe_xprime", np.nan), errors="coerce")

    answer_change_rate = None
    if {"y_x", "y_xprime"}.issubset(tuples_df.columns):
        answer_change_rate = _mean_or_none(
            tuples_df["y_x"].fillna("").astype(str) != tuples_df["y_xprime"].fillna("").astype(str)
        )

    metrics = {
        "n_pairs": int(len(tuples_df)),
        "n_questions": int(tuples_df["question_id"].nunique()) if "question_id" in tuples_df.columns else 0,
        "accuracy_x": _mean_or_none(accuracy_x),
        "accuracy_xprime": _mean_or_none(accuracy_xprime),
        "delta_accuracy_x_minus_xprime": _mean_or_none(accuracy_x - accuracy_xprime),
        "avg_p_x": _mean_or_none(p_x),
        "avg_p_xprime": _mean_or_none(p_xprime),
        "avg_delta_p_x_minus_xprime": _mean_or_none(p_x - p_xprime),
        "avg_probe_x": _mean_or_none(probe_x),
        "avg_probe_xprime": _mean_or_none(probe_xprime),
        "avg_delta_probe_x_minus_xprime": _mean_or_none(probe_x - probe_xprime),
        "harmful_flip_rate": _mean_or_none((accuracy_x == 1) & (accuracy_xprime == 0)),
        "helpful_flip_rate": _mean_or_none((accuracy_x == 0) & (accuracy_xprime == 1)),
        "unchanged_correctness_rate": _mean_or_none(accuracy_x == accuracy_xprime),
        "answer_change_rate": answer_change_rate,
    }
    return _json_ready(metrics)


def _group_metric_records(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_builder,
) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    group_names = list(group_cols)
    records: List[Dict[str, Any]] = []
    for group_key, group_df in df.groupby(group_names, dropna=False, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {column_name: _json_ready(value) for column_name, value in zip(group_names, key_values)}
        row.update(metric_builder(group_df))
        records.append(_json_ready(row))
    return records


def _build_prompt_summary_records(samples_df: pd.DataFrame, group_cols: Sequence[str]) -> List[Dict[str, Any]]:
    return _group_metric_records(samples_df, group_cols, _prompt_metrics_from_df)


def _build_pair_summary_records(tuples_df: pd.DataFrame, group_cols: Sequence[str]) -> List[Dict[str, Any]]:
    return _group_metric_records(tuples_df, group_cols, _pair_metrics_from_df)


def build_model_summary_by_template_df(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows = _build_prompt_summary_records(samples_df, ("template_type",))
    return pd.DataFrame(rows, columns=MODEL_SUMMARY_BY_TEMPLATE_COLUMNS)


def build_model_summary_by_bias_df(tuples_df: pd.DataFrame) -> pd.DataFrame:
    rows = _build_pair_summary_records(tuples_df, ("bias_type",))
    return pd.DataFrame(rows, columns=MODEL_SUMMARY_BY_BIAS_COLUMNS)


def build_model_summary_payload(
    *,
    args: Any,
    run_dir: Path,
    samples_df: pd.DataFrame,
    tuples_df: pd.DataFrame,
    probes_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return _json_ready(
        {
            "summary_schema_version": MODEL_SUMMARY_SCHEMA_VERSION,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "model_name": str(args.model),
            "dataset_name": str(getattr(args, "dataset_name", "") or ""),
            "bias_types": _list_like_strings(getattr(args, "bias_types", [])),
            "counts": {
                "sample_rows": int(len(samples_df)),
                "question_count": int(samples_df["question_id"].nunique()) if not samples_df.empty else 0,
                "tuple_rows": int(len(tuples_df)),
            },
            "prompt_level": {
                "all_rows": _prompt_metrics_from_df(samples_df),
                "by_split": _build_prompt_summary_records(samples_df, ("split",)),
                "by_template": _build_prompt_summary_records(samples_df, ("template_type",)),
                "by_split_and_template": _build_prompt_summary_records(samples_df, ("split", "template_type")),
            },
            "paired_effects": {
                "all_pairs": _pair_metrics_from_df(tuples_df),
                "by_split": _build_pair_summary_records(tuples_df, ("split",)),
                "by_bias": _build_pair_summary_records(tuples_df, ("bias_type",)),
                "by_split_and_bias": _build_pair_summary_records(tuples_df, ("split", "bias_type")),
            },
            "strict_mc_quality": probes_meta.get("strict_mc_quality"),
            "definitions": {
                "selected_choice": (
                    "Strict-MC selected choice: the highest-probability allowed answer choice after "
                    "normalizing over the allowed choices only."
                ),
                "P(correct)": "Strict-MC probability assigned to the gold answer choice.",
                "P(selected)": "Strict-MC probability assigned to the selected choice.",
                "T_prompt": (
                    "Prompt-level soft correctness. In strict MC this equals P(correct); in generation-based "
                    "paths it is empirical mean correctness across repeated usable draws for the prompt."
                ),
                "accuracy": (
                    "Mean row-level correctness. In strict MC this is argmax selected-choice accuracy."
                ),
                "avg_p_correct": (
                    "Average T_prompt across prompt rows. In strict MC this equals average P(correct); in "
                    "generation-based paths it equals average empirical prompt correctness."
                ),
                "avg_p_selected": "Average P(selected) across prompt rows when strict-MC choice scoring is used.",
                "avg_delta_p_x_minus_xprime": "Average p(x) - p(x'), where x is neutral and x' is the injected prompt.",
                "T_x/T_xprime": (
                    "Pair-level copies of T_prompt for neutral and biased rows. In strict MC they equal "
                    "P(correct) on x and x'."
                ),
                "harmful_flip_rate": "Share of paired rows that go from correct on x to incorrect on x'.",
                "helpful_flip_rate": "Share of paired rows that go from incorrect on x to correct on x'.",
            },
        }
    )


def _build_reports_summary_row(
    *,
    bias_type: str,
    prompt_metrics: Dict[str, Any],
    pair_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    prompt_metrics = prompt_metrics or {}
    pair_metrics = pair_metrics or {}
    delta_accuracy = _float_or_none(pair_metrics.get("delta_accuracy_x_minus_xprime"))
    delta_p = _float_or_none(pair_metrics.get("avg_delta_p_x_minus_xprime"))
    delta_probe = _float_or_none(pair_metrics.get("avg_delta_probe_x_minus_xprime"))
    return _json_ready(
        {
            "bias_type": str(bias_type or "").strip(),
            "n_prompt_rows": int(prompt_metrics.get("n_rows", 0) or 0),
            "n_questions": int(prompt_metrics.get("n_questions", 0) or 0),
            "n_usable_prompt_rows": int(prompt_metrics.get("n_usable_rows", 0) or 0),
            "usable_rate": prompt_metrics.get("usable_rate"),
            "ambiguous_rate": prompt_metrics.get("ambiguous_rate"),
            "accuracy": prompt_metrics.get("accuracy"),
            "avg_p_correct": prompt_metrics.get("avg_p_correct"),
            "avg_p_selected": prompt_metrics.get("avg_p_selected"),
            "avg_selected_minus_correct_probability_gap": prompt_metrics.get(
                "avg_selected_minus_correct_probability_gap"
            ),
            "avg_probe_score_selected_prompt": prompt_metrics.get("avg_probe_score_selected_prompt"),
            "exact_format_rate": prompt_metrics.get("exact_format_rate"),
            "starts_with_answer_prefix_rate": prompt_metrics.get("starts_with_answer_prefix_rate"),
            "cap_hit_rate": prompt_metrics.get("cap_hit_rate"),
            "stopped_on_eos_rate": prompt_metrics.get("stopped_on_eos_rate"),
            "avg_completion_token_count": prompt_metrics.get("avg_completion_token_count"),
            "n_pairs": int(pair_metrics.get("n_pairs", 0) or 0),
            "neutral_accuracy": pair_metrics.get("accuracy_x"),
            "biased_accuracy": pair_metrics.get("accuracy_xprime"),
            "delta_accuracy_biased_minus_neutral": (
                None if delta_accuracy is None else float(-1.0 * delta_accuracy)
            ),
            "neutral_avg_p_correct": pair_metrics.get("avg_p_x"),
            "biased_avg_p_correct": pair_metrics.get("avg_p_xprime"),
            "avg_delta_p_biased_minus_neutral": (None if delta_p is None else float(-1.0 * delta_p)),
            "neutral_avg_probe": pair_metrics.get("avg_probe_x"),
            "biased_avg_probe": pair_metrics.get("avg_probe_xprime"),
            "avg_delta_probe_biased_minus_neutral": (
                None if delta_probe is None else float(-1.0 * delta_probe)
            ),
            "harmful_flip_rate": pair_metrics.get("harmful_flip_rate"),
            "helpful_flip_rate": pair_metrics.get("helpful_flip_rate"),
            "unchanged_correctness_rate": pair_metrics.get("unchanged_correctness_rate"),
            "answer_change_rate": pair_metrics.get("answer_change_rate"),
        }
    )


def build_reports_summary_df(
    *,
    samples_df: pd.DataFrame,
    tuples_df: pd.DataFrame,
    bias_types: Sequence[str],
) -> pd.DataFrame:
    configured_bias_types = [bias_type for bias_type in _list_like_strings(bias_types) if bias_type != "neutral"]
    observed_bias_types = (
        sorted({str(value).strip() for value in tuples_df.get("bias_type", pd.Series(dtype=str)).dropna().astype(str)})
        if not tuples_df.empty and "bias_type" in tuples_df.columns
        else []
    )
    observed_template_bias_types = (
        sorted(
            {
                str(value).strip()
                for value in _series_from_df(samples_df, "template_type", "").dropna().astype(str)
                if str(value).strip() and str(value).strip() != "neutral"
            }
        )
        if not samples_df.empty
        else []
    )
    ordered_bias_types = list(configured_bias_types)
    seen_bias_types = set(ordered_bias_types)
    for bias_type in observed_bias_types + observed_template_bias_types:
        if bias_type in seen_bias_types:
            continue
        ordered_bias_types.append(bias_type)
        seen_bias_types.add(bias_type)

    rows = [
        _build_reports_summary_row(
            bias_type="overall",
            prompt_metrics=_prompt_metrics_from_df(samples_df),
            pair_metrics=_pair_metrics_from_df(tuples_df),
        )
    ]

    template_series = _series_from_df(samples_df, "template_type", "").astype(str)
    pair_bias_series = (
        tuples_df["bias_type"].astype(str)
        if not tuples_df.empty and "bias_type" in tuples_df.columns
        else pd.Series(dtype=str)
    )
    neutral_df = samples_df[template_series == "neutral"].copy()
    if not neutral_df.empty:
        rows.append(
            _build_reports_summary_row(
                bias_type="neutral",
                prompt_metrics=_prompt_metrics_from_df(neutral_df),
                pair_metrics={},
            )
        )
    for bias_type in ordered_bias_types:
        rows.append(
            _build_reports_summary_row(
                bias_type=bias_type,
                prompt_metrics=_prompt_metrics_from_df(samples_df[template_series == bias_type].copy()),
                pair_metrics=_pair_metrics_from_df(tuples_df[pair_bias_series == bias_type].copy()),
            )
        )

    return pd.DataFrame(rows, columns=REPORTS_SUMMARY_COLUMNS)


def _build_agreement_injection_records(tuples_df: pd.DataFrame, group_cols: Sequence[str]) -> List[Dict[str, Any]]:
    if tuples_df.empty:
        return []

    working_df = tuples_df.copy()
    working_df["agreement_injection"] = working_df["bias_type"].astype(str)
    return _build_pair_summary_records(working_df, group_cols)


def _load_probe_metrics(metrics_path: Any) -> Dict[str, Any]:
    if not metrics_path:
        return {}
    try:
        path = Path(str(metrics_path))
    except Exception:
        return {}
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_probe_test_auc_rows(probes_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    probe_names = [
        str(key)
        for key in probes_meta
        if str(key) == "probe_no_bias" or str(key).startswith("probe_bias_")
    ]
    for probe_name in sorted(probe_names):
        probe_payload = probes_meta.get(probe_name)
        if not isinstance(probe_payload, dict):
            continue
        metrics = _load_probe_metrics(probe_payload.get("chosen_probe_metrics_path"))
        test_metrics = metrics.get("splits", {}).get("test", {}) if isinstance(metrics, dict) else {}
        rows.append(
            {
                "probe_name": probe_name,
                "best_layer": probe_payload.get("best_layer"),
                "best_dev_auc": _float_or_none(probe_payload.get("best_dev_auc")),
                "test_auc": _float_or_none(test_metrics.get("auc")),
                "test_accuracy": _float_or_none(test_metrics.get("accuracy")),
                "test_balanced_accuracy": _float_or_none(test_metrics.get("balanced_accuracy")),
                "test_n_total": int(test_metrics.get("n_total", 0) or 0)
                if isinstance(test_metrics, dict)
                else 0,
            }
        )
    return rows


def _probe_family_names(probes_meta: Dict[str, Any]) -> List[str]:
    return sorted(
        [
            str(key)
            for key in probes_meta
            if str(key) == "probe_no_bias" or str(key).startswith("probe_bias_")
        ]
    )


def _template_type_for_probe(probe_name: str, probe_payload: Dict[str, Any]) -> str:
    template_type = str(probe_payload.get("template_type", "") or "")
    if template_type:
        return template_type
    if probe_name == "probe_no_bias":
        return "neutral"
    if probe_name.startswith("probe_bias_"):
        return probe_name[len("probe_bias_") :]
    return ""


def _build_probe_score_lookup(group_df: pd.DataFrame, value_column: str) -> Dict[str, Optional[float]]:
    lookup: Dict[str, Optional[float]] = {}
    for row in group_df.itertuples(index=False):
        choice = str(getattr(row, "candidate_choice", "") or "")
        if not choice:
            continue
        lookup[choice] = _float_or_none(getattr(row, value_column, None))
    return lookup


def build_mc_probe_scores_by_prompt_df(probe_candidate_scores_df: pd.DataFrame) -> pd.DataFrame:
    if probe_candidate_scores_df.empty or "candidate_choice" not in probe_candidate_scores_df.columns:
        return pd.DataFrame(columns=MC_PROBE_SCORE_BY_PROMPT_BASE_COLUMNS)

    working_df = probe_candidate_scores_df.copy()
    working_df["candidate_choice"] = working_df["candidate_choice"].fillna("").astype(str)
    working_df = working_df[working_df["candidate_choice"].str.strip() != ""]
    if working_df.empty:
        return pd.DataFrame(columns=MC_PROBE_SCORE_BY_PROMPT_BASE_COLUMNS)

    choice_labels = sorted({choice for choice in working_df["candidate_choice"].astype(str) if choice})
    score_columns = [f"score_{choice}" for choice in choice_labels]
    rows: List[Dict[str, Any]] = []
    group_cols = ["probe_name", "source_record_id"]
    for _, group_df in working_df.groupby(group_cols, dropna=False, sort=True):
        ordered = group_df.sort_values(
            by=["probe_score", "candidate_rank", "candidate_choice"],
            ascending=[False, True, True],
            na_position="last",
        )
        first_row = group_df.iloc[0]
        score_lookup = _build_probe_score_lookup(group_df, "probe_score")
        probability_lookup = _build_probe_score_lookup(group_df, "candidate_probability")
        correct_letter = str(first_row.get("correct_letter", "") or "")
        selected_choice = str(first_row.get("selected_choice", "") or "")
        argmax_choice = str(ordered.iloc[0].get("candidate_choice", "") or "") if not ordered.empty else ""
        argmax_score = _float_or_none(ordered.iloc[0].get("probe_score")) if not ordered.empty else None
        score_correct = score_lookup.get(correct_letter)
        score_selected = score_lookup.get(selected_choice)
        row = {
            "model_name": str(first_row.get("model_name", "") or ""),
            "probe_name": str(first_row.get("probe_name", "") or ""),
            "split": str(first_row.get("split", "") or ""),
            "question_id": str(first_row.get("question_id", "") or ""),
            "prompt_id": str(first_row.get("prompt_id", "") or ""),
            "dataset": str(first_row.get("dataset", "") or ""),
            "template_type": str(first_row.get("template_type", "") or ""),
            "draw_idx": int(first_row.get("draw_idx", 0) or 0),
            "source_record_id": first_row.get("source_record_id", np.nan),
            "correct_letter": correct_letter,
            "selected_choice": selected_choice,
            "selected_choice_is_correct": bool(selected_choice and selected_choice == correct_letter),
            "probe_score_correct_choice": score_correct,
            "probe_score_selected_choice": score_selected,
            "correct_choice_probability": probability_lookup.get(correct_letter),
            "selected_choice_probability": probability_lookup.get(selected_choice),
            "probe_argmax_choice": argmax_choice,
            "probe_argmax_score": argmax_score,
            "probe_prefers_correct": bool(argmax_choice and argmax_choice == correct_letter),
            "probe_prefers_selected": bool(argmax_choice and argmax_choice == selected_choice),
            "probe_score_gap_correct_minus_selected": (
                None if score_correct is None or score_selected is None else float(score_correct - score_selected)
            ),
        }
        for choice in choice_labels:
            row[f"score_{choice}"] = score_lookup.get(choice)
        rows.append(_json_ready(row))
    return pd.DataFrame(rows, columns=MC_PROBE_SCORE_BY_PROMPT_BASE_COLUMNS + score_columns)


def _candidate_probe_metrics_from_frames(
    candidate_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> Dict[str, Any]:
    if candidate_df.empty and wide_df.empty:
        return {
            "n_candidate_rows": 0,
            "n_source_prompts": 0,
            "avg_candidates_per_prompt": None,
            "mean_probe_score_all_candidates": None,
            "mean_probe_score_correct_candidate": None,
            "mean_probe_score_incorrect_candidate": None,
            "mean_probe_score_selected_candidate": None,
            "mean_probe_score_non_selected_candidate": None,
            "selected_choice_is_correct_rate": None,
            "probe_prefers_correct_rate": None,
            "probe_prefers_selected_rate": None,
            "mean_probe_score_correct_choice": None,
            "mean_probe_score_selected_choice": None,
            "mean_correct_minus_selected_probe_gap": None,
        }

    probe_scores = pd.to_numeric(candidate_df.get("probe_score", np.nan), errors="coerce")
    correctness = pd.to_numeric(candidate_df.get("candidate_correctness", np.nan), errors="coerce")
    selected = pd.to_numeric(candidate_df.get("candidate_is_selected", np.nan), errors="coerce")
    n_source_prompts = int(wide_df["source_record_id"].nunique()) if "source_record_id" in wide_df.columns else 0
    metrics = {
        "n_candidate_rows": int(len(candidate_df)),
        "n_source_prompts": n_source_prompts,
        "avg_candidates_per_prompt": (float(len(candidate_df) / n_source_prompts) if n_source_prompts else None),
        "mean_probe_score_all_candidates": _mean_or_none(probe_scores),
        "mean_probe_score_correct_candidate": _mean_or_none(probe_scores[correctness == 1]),
        "mean_probe_score_incorrect_candidate": _mean_or_none(probe_scores[correctness == 0]),
        "mean_probe_score_selected_candidate": _mean_or_none(probe_scores[selected == 1]),
        "mean_probe_score_non_selected_candidate": _mean_or_none(probe_scores[selected == 0]),
        "selected_choice_is_correct_rate": _mean_or_none(
            wide_df.get("selected_choice_is_correct", pd.Series(dtype=float))
        ),
        "probe_prefers_correct_rate": _mean_or_none(wide_df.get("probe_prefers_correct", pd.Series(dtype=float))),
        "probe_prefers_selected_rate": _mean_or_none(wide_df.get("probe_prefers_selected", pd.Series(dtype=float))),
        "mean_probe_score_correct_choice": _mean_or_none(
            wide_df.get("probe_score_correct_choice", pd.Series(dtype=float))
        ),
        "mean_probe_score_selected_choice": _mean_or_none(
            wide_df.get("probe_score_selected_choice", pd.Series(dtype=float))
        ),
        "mean_correct_minus_selected_probe_gap": _mean_or_none(
            wide_df.get("probe_score_gap_correct_minus_selected", pd.Series(dtype=float))
        ),
    }
    return _json_ready(metrics)


def _group_candidate_probe_metrics(
    candidate_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    group_cols: Sequence[str],
) -> List[Dict[str, Any]]:
    if candidate_df.empty and wide_df.empty:
        return []

    group_names = list(group_cols)
    if not group_names:
        return [_candidate_probe_metrics_from_frames(candidate_df, wide_df)]

    records: List[Dict[str, Any]] = []
    candidate_groups = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in candidate_df.groupby(group_names, dropna=False, sort=True)
    } if not candidate_df.empty else {}
    wide_groups = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in wide_df.groupby(group_names, dropna=False, sort=True)
    } if not wide_df.empty else {}
    for group_key in sorted(set(candidate_groups) | set(wide_groups), key=lambda values: tuple(str(v) for v in values)):
        row = {column_name: _json_ready(value) for column_name, value in zip(group_names, group_key)}
        row.update(
            _candidate_probe_metrics_from_frames(
                candidate_groups.get(group_key, pd.DataFrame(columns=candidate_df.columns)),
                wide_groups.get(group_key, pd.DataFrame(columns=wide_df.columns)),
            )
        )
        records.append(_json_ready(row))
    return records


def build_probe_summary_payload(
    *,
    args: Any,
    run_dir: Path,
    probes_meta: Dict[str, Any],
    probe_candidate_scores_df: pd.DataFrame,
    probe_scores_by_prompt_df: pd.DataFrame,
) -> Dict[str, Any]:
    per_probe: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    best_probe_on_test = None

    for probe_name in _probe_family_names(probes_meta):
        probe_payload = probes_meta.get(probe_name)
        if not isinstance(probe_payload, dict):
            continue

        metrics = _load_probe_metrics(probe_payload.get("chosen_probe_metrics_path"))
        metrics_by_split = {}
        split_rows: Dict[str, Dict[str, Any]] = {}
        for split_name in ("train", "val", "test"):
            split_metrics = metrics.get("splits", {}).get(split_name, {}) if isinstance(metrics, dict) else {}
            split_rows[split_name] = _json_ready(split_metrics if isinstance(split_metrics, dict) else {})
            metrics_by_split[split_name] = split_rows[split_name]

        candidate_subset = (
            probe_candidate_scores_df[probe_candidate_scores_df["probe_name"].astype(str) == probe_name].copy()
            if "probe_name" in probe_candidate_scores_df.columns
            else pd.DataFrame(columns=probe_candidate_scores_df.columns)
        )
        wide_subset = (
            probe_scores_by_prompt_df[probe_scores_by_prompt_df["probe_name"].astype(str) == probe_name].copy()
            if "probe_name" in probe_scores_by_prompt_df.columns
            else pd.DataFrame(columns=probe_scores_by_prompt_df.columns)
        )
        candidate_summary = {
            "all_splits": _candidate_probe_metrics_from_frames(candidate_subset, wide_subset),
            "by_split": _group_candidate_probe_metrics(candidate_subset, wide_subset, ("split",)),
        }

        auc_pairs = []
        raw_auc_per_layer = probe_payload.get("auc_per_layer", {})
        if isinstance(raw_auc_per_layer, dict):
            for layer_key, auc_value in raw_auc_per_layer.items():
                try:
                    layer_id = int(layer_key)
                except Exception:
                    continue
                auc_pairs.append((layer_id, _float_or_none(auc_value)))
        auc_pairs = sorted(auc_pairs, key=lambda item: item[0])
        ranked_auc_pairs = [pair for pair in auc_pairs if pair[1] is not None]
        ranked_auc_pairs.sort(key=lambda item: (-float(item[1]), item[0]))

        best_layer = probe_payload.get("best_layer")
        chosen_layer_rank = None
        second_best_layer = None
        second_best_dev_auc = None
        gap_to_second_best_dev_auc = None
        if best_layer is not None and ranked_auc_pairs:
            for rank_idx, (layer_id, _) in enumerate(ranked_auc_pairs, start=1):
                if int(layer_id) == int(best_layer):
                    chosen_layer_rank = rank_idx
                    break
        if len(ranked_auc_pairs) >= 2:
            second_best_layer, second_best_dev_auc = ranked_auc_pairs[1]
            best_dev_auc = _float_or_none(probe_payload.get("best_dev_auc"))
            if best_dev_auc is not None and second_best_dev_auc is not None:
                gap_to_second_best_dev_auc = float(best_dev_auc - second_best_dev_auc)

        template_type = _template_type_for_probe(probe_name, probe_payload)
        trained_layers = [int(layer) for layer in (probe_payload.get("trained_layers", []) or [])]
        row = {
            "probe_name": probe_name,
            "template_type": template_type,
            "probe_construction": str(probe_payload.get("probe_construction", "") or ""),
            "probe_example_weighting": str(probe_payload.get("probe_example_weighting", "") or ""),
            "best_layer": probe_payload.get("best_layer"),
            "best_dev_auc": _float_or_none(probe_payload.get("best_dev_auc")),
            "train_auc": _float_or_none(split_rows["train"].get("auc")),
            "val_auc": _float_or_none(split_rows["val"].get("auc")),
            "test_auc": _float_or_none(split_rows["test"].get("auc")),
            "train_accuracy": _float_or_none(split_rows["train"].get("accuracy")),
            "val_accuracy": _float_or_none(split_rows["val"].get("accuracy")),
            "test_accuracy": _float_or_none(split_rows["test"].get("accuracy")),
            "train_balanced_accuracy": _float_or_none(split_rows["train"].get("balanced_accuracy")),
            "val_balanced_accuracy": _float_or_none(split_rows["val"].get("balanced_accuracy")),
            "test_balanced_accuracy": _float_or_none(split_rows["test"].get("balanced_accuracy")),
            "train_n_total": int(split_rows["train"].get("n_total", 0) or 0),
            "val_n_total": int(split_rows["val"].get("n_total", 0) or 0),
            "test_n_total": int(split_rows["test"].get("n_total", 0) or 0),
            "train_minus_val_auc": (
                None
                if _float_or_none(split_rows["train"].get("auc")) is None
                or _float_or_none(split_rows["val"].get("auc")) is None
                else float(split_rows["train"]["auc"] - split_rows["val"]["auc"])
            ),
            "val_minus_test_auc": (
                None
                if _float_or_none(split_rows["val"].get("auc")) is None
                or _float_or_none(split_rows["test"].get("auc")) is None
                else float(split_rows["val"]["auc"] - split_rows["test"]["auc"])
            ),
            "train_minus_test_auc": (
                None
                if _float_or_none(split_rows["train"].get("auc")) is None
                or _float_or_none(split_rows["test"].get("auc")) is None
                else float(split_rows["train"]["auc"] - split_rows["test"]["auc"])
            ),
            "probe_prefers_correct_rate": candidate_summary["all_splits"].get("probe_prefers_correct_rate"),
            "probe_prefers_selected_rate": candidate_summary["all_splits"].get("probe_prefers_selected_rate"),
            "mean_probe_score_correct_candidate": candidate_summary["all_splits"].get(
                "mean_probe_score_correct_candidate"
            ),
            "mean_probe_score_incorrect_candidate": candidate_summary["all_splits"].get(
                "mean_probe_score_incorrect_candidate"
            ),
            "mean_probe_score_selected_candidate": candidate_summary["all_splits"].get(
                "mean_probe_score_selected_candidate"
            ),
            "mean_probe_score_non_selected_candidate": candidate_summary["all_splits"].get(
                "mean_probe_score_non_selected_candidate"
            ),
            "mean_probe_score_correct_choice": candidate_summary["all_splits"].get(
                "mean_probe_score_correct_choice"
            ),
            "mean_probe_score_selected_choice": candidate_summary["all_splits"].get(
                "mean_probe_score_selected_choice"
            ),
            "mean_correct_minus_selected_probe_gap": candidate_summary["all_splits"].get(
                "mean_correct_minus_selected_probe_gap"
            ),
        }
        row = _json_ready(row)
        summary_rows.append(row)
        per_probe[probe_name] = _json_ready(
            {
                **row,
                "auc_per_layer": {str(layer_id): auc_value for layer_id, auc_value in auc_pairs},
                "trained_layers": trained_layers,
                "n_trained_layers": len(trained_layers),
                "chosen_layer_rank_by_val_auc": chosen_layer_rank,
                "second_best_layer": second_best_layer,
                "second_best_dev_auc": second_best_dev_auc,
                "gap_to_second_best_dev_auc": gap_to_second_best_dev_auc,
                "feature_source": probe_payload.get("feature_source", {}),
                "data_summary": probe_payload.get("data_summary", {}),
                "selection_fit_summary": probe_payload.get("selection_fit_summary", {}),
                "selection_val_summary": probe_payload.get("selection_val_summary", {}),
                "chosen_fit_summary": probe_payload.get("chosen_fit_summary", {}),
                "metrics_by_split": metrics_by_split,
                "candidate_probe_summary": candidate_summary,
                "chosen_probe_metrics_path": probe_payload.get("chosen_probe_metrics_path"),
                "chosen_probe_metadata_path": probe_payload.get("chosen_probe_metadata_path"),
                "chosen_probe_membership_path": probe_payload.get("chosen_probe_membership_path"),
            }
        )

        test_auc = row.get("test_auc")
        if test_auc is not None:
            if best_probe_on_test is None or (
                float(test_auc),
                float(row.get("best_dev_auc") or float("-inf")),
            ) > (
                float(best_probe_on_test.get("test_auc") or float("-inf")),
                float(best_probe_on_test.get("best_dev_auc") or float("-inf")),
            ):
                best_probe_on_test = row

    return _json_ready(
        {
            "summary_schema_version": PROBE_SUMMARY_SCHEMA_VERSION,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "model_name": str(args.model),
            "dataset_name": str(getattr(args, "dataset_name", "") or ""),
            "bias_types": _list_like_strings(getattr(args, "bias_types", [])),
            "counts": {
                "probe_family_count": len(per_probe),
                "probe_candidate_score_rows": int(len(probe_candidate_scores_df)),
                "probe_score_prompt_rows": int(len(probe_scores_by_prompt_df)),
            },
            "overview": {
                "best_probe_on_test": best_probe_on_test,
            },
            "per_probe": per_probe,
        }
    )


def build_probe_summary_df(probe_summary_payload: Dict[str, Any]) -> pd.DataFrame:
    per_probe = probe_summary_payload.get("per_probe", {})
    if not isinstance(per_probe, dict):
        return pd.DataFrame(columns=PROBE_SUMMARY_COLUMNS)
    rows = [payload for _, payload in sorted(per_probe.items()) if isinstance(payload, dict)]
    return pd.DataFrame(rows, columns=PROBE_SUMMARY_COLUMNS)


def _records_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return [_json_ready(row) for row in df.to_dict(orient="records")]


def build_reports_summary_payload(
    *,
    args: Any,
    run_dir: Path,
    samples_df: pd.DataFrame,
    sample_records: Optional[Sequence[Dict[str, Any]]] = None,
    tuples_df: pd.DataFrame,
    probe_scores_by_prompt_df: pd.DataFrame,
    probes_meta: Dict[str, Any],
    probe_candidate_scores_df: pd.DataFrame,
    run_timing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_summary_payload = build_model_summary_payload(
        args=args,
        run_dir=run_dir,
        samples_df=samples_df,
        tuples_df=tuples_df,
        probes_meta=probes_meta,
    )
    template_df = build_model_summary_by_template_df(samples_df)
    bias_df = build_model_summary_by_bias_df(tuples_df).copy()
    summary_df = build_reports_summary_df(
        samples_df=samples_df,
        tuples_df=tuples_df,
        bias_types=_list_like_strings(getattr(args, "bias_types", [])),
    )
    if not bias_df.empty:
        bias_df["delta_accuracy_xprime_minus_x"] = (
            pd.to_numeric(bias_df["accuracy_xprime"], errors="coerce")
            - pd.to_numeric(bias_df["accuracy_x"], errors="coerce")
        )
        bias_df["avg_delta_p_xprime_minus_x"] = (
            pd.to_numeric(bias_df["avg_p_xprime"], errors="coerce")
            - pd.to_numeric(bias_df["avg_p_x"], errors="coerce")
        )
        bias_df["avg_delta_probe_xprime_minus_x"] = (
            pd.to_numeric(bias_df["avg_probe_xprime"], errors="coerce")
            - pd.to_numeric(bias_df["avg_probe_x"], errors="coerce")
        )

    probe_summary_payload = build_probe_summary_payload(
        args=args,
        run_dir=run_dir,
        probes_meta=probes_meta,
        probe_candidate_scores_df=probe_candidate_scores_df,
        probe_scores_by_prompt_df=probe_scores_by_prompt_df,
    )
    probe_summary_df = build_probe_summary_df(probe_summary_payload)
    selected_probe_overview = probe_summary_payload.get("overview", {}).get("best_probe_on_test")
    prompt_overview = model_summary_payload.get("prompt_level", {}).get("all_rows", {})
    pair_overview = model_summary_payload.get("paired_effects", {}).get("all_pairs", {})
    mc_option_count_summary = _build_mc_option_count_summary(
        samples_df,
        sample_records=sample_records,
    )
    mc_option_selection = _build_mc_option_selection_summary(
        samples_df,
        bias_types=_list_like_strings(getattr(args, "bias_types", [])),
    )
    mc_confusion_matrix = _build_mc_confusion_matrix_summary(
        samples_df,
        sample_records=sample_records,
    )

    return _json_ready(
        {
            "summary_schema_version": REPORTS_SUMMARY_SCHEMA_VERSION,
            "generated_at_utc": utc_now_iso(),
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "model_name": str(args.model),
            "dataset_name": str(getattr(args, "dataset_name", "") or ""),
            "bias_types": _list_like_strings(getattr(args, "bias_types", [])),
            "sampling_only": bool(getattr(args, "sampling_only", False)),
            "headline_counts": {
                "sample_rows": int(len(samples_df)),
                "question_count": int(samples_df["question_id"].nunique()) if not samples_df.empty else 0,
                "usable_sample_rows": int(
                    pd.to_numeric(_series_from_df(samples_df, "usable_for_metrics", 0), errors="coerce").fillna(0).sum()
                ),
                "paired_rows": int(len(tuples_df)),
                "probe_score_prompt_rows": int(len(probe_scores_by_prompt_df)),
                "probe_family_count": int(len(probe_summary_df)),
            },
            "summary_rows": _records_from_df(summary_df),
            "overall": {
                "accuracy": prompt_overview.get("accuracy"),
                "avg_p_correct": prompt_overview.get("avg_p_correct"),
                "avg_p_selected": prompt_overview.get("avg_p_selected"),
                "avg_delta_p_xprime_minus_x": (
                    None
                    if pair_overview.get("avg_delta_p_x_minus_xprime") is None
                    else float(-1.0 * float(pair_overview["avg_delta_p_x_minus_xprime"]))
                ),
                "harmful_flip_rate": pair_overview.get("harmful_flip_rate"),
                "helpful_flip_rate": pair_overview.get("helpful_flip_rate"),
                "unchanged_correctness_rate": pair_overview.get("unchanged_correctness_rate"),
            },
            "accuracy_by_template": _records_from_df(template_df),
            "accuracy_by_bias_type": _records_from_df(bias_df),
            "mc_option_count_summary": mc_option_count_summary,
            "mc_confusion_matrix": mc_confusion_matrix,
            "mc_option_selection": mc_option_selection,
            "probe_score_summaries": _records_from_df(probe_summary_df),
            "selected_probe_overview": _json_ready(selected_probe_overview),
            "probe_training_status": probes_meta.get("probe_training_status", "completed"),
            "strict_mc_quality": probes_meta.get("strict_mc_quality"),
            "runtime_timing": _json_ready(run_timing) if run_timing else None,
            "definitions": model_summary_payload.get("definitions", {}),
        }
    )


def _format_terminal_summary_value(value: Any, *, as_percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    try:
        if pd.isna(value):
            return "n/a"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return "n/a"
        if as_percent:
            return f"{100.0 * numeric:.1f}%"
        return f"{numeric:.3f}"
    return str(value)


def _summary_rows_from_payload(summary_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(summary_payload, list):
        return [row for row in summary_payload if isinstance(row, dict)]

    if not isinstance(summary_payload, dict) or not summary_payload:
        return []

    raw_rows = summary_payload.get("summary_rows")
    if isinstance(raw_rows, list):
        return [row for row in raw_rows if isinstance(row, dict)]

    rows: List[Dict[str, Any]] = []
    overall = summary_payload.get("overall")
    if isinstance(overall, dict):
        rows.append(
            _json_ready(
                {
                    "bias_type": "overall",
                    "accuracy": overall.get("accuracy"),
                    "avg_p_correct": overall.get("avg_p_correct"),
                    "avg_p_selected": overall.get("avg_p_selected"),
                    "avg_delta_p_biased_minus_neutral": overall.get("avg_delta_p_xprime_minus_x"),
                    "harmful_flip_rate": overall.get("harmful_flip_rate"),
                    "helpful_flip_rate": overall.get("helpful_flip_rate"),
                    "unchanged_correctness_rate": overall.get("unchanged_correctness_rate"),
                }
            )
        )

    template_lookup = {
        str(row.get("template_type", "") or "").strip(): row
        for row in summary_payload.get("accuracy_by_template", [])
        if isinstance(row, dict) and str(row.get("template_type", "") or "").strip()
    }
    pair_lookup = {
        str(row.get("bias_type", "") or "").strip(): row
        for row in summary_payload.get("accuracy_by_bias_type", [])
        if isinstance(row, dict) and str(row.get("bias_type", "") or "").strip()
    }
    configured_bias_types = [
        bias_type for bias_type in _list_like_strings(summary_payload.get("bias_types")) if bias_type != "neutral"
    ]
    ordered_bias_types = list(configured_bias_types)
    seen_bias_types = set(ordered_bias_types)
    for bias_type in sorted(pair_lookup):
        if bias_type in seen_bias_types:
            continue
        ordered_bias_types.append(bias_type)
        seen_bias_types.add(bias_type)
    for template_type in sorted(template_lookup):
        if template_type == "neutral" or template_type in seen_bias_types:
            continue
        ordered_bias_types.append(template_type)
        seen_bias_types.add(template_type)

    if "neutral" in template_lookup:
        rows.append(
            _build_reports_summary_row(
                bias_type="neutral",
                prompt_metrics=template_lookup["neutral"],
                pair_metrics={},
            )
        )

    for bias_type in ordered_bias_types:
        prompt_metrics = template_lookup.get(bias_type, {})
        pair_metrics = pair_lookup.get(bias_type, {})
        if prompt_metrics or pair_metrics:
            rows.append(
                _build_reports_summary_row(
                    bias_type=bias_type,
                    prompt_metrics=prompt_metrics,
                    pair_metrics=pair_metrics,
                )
            )
    return rows


def build_terminal_final_stats_lines(summary_payload: Any) -> List[str]:
    if not summary_payload:
        return []

    payload_dict = summary_payload if isinstance(summary_payload, dict) else {}
    headline_counts = payload_dict.get("headline_counts", {})
    best_probe = payload_dict.get("selected_probe_overview", {})
    summary_rows = _summary_rows_from_payload(summary_payload)
    summary_df = pd.DataFrame(summary_rows)
    lines = ["final model stats:"]

    if payload_dict.get("model_name"):
        lines.append(f"model={payload_dict.get('model_name')}")
    if payload_dict.get("dataset_name"):
        lines.append(f"dataset={payload_dict.get('dataset_name')}")

    count_metrics = [
        ("sample_rows", headline_counts.get("sample_rows")),
        ("question_count", headline_counts.get("question_count")),
        ("usable_sample_rows", headline_counts.get("usable_sample_rows")),
        ("paired_rows", headline_counts.get("paired_rows")),
        ("probe_family_count", headline_counts.get("probe_family_count")),
    ]
    for label, value in count_metrics:
        if value is not None:
            lines.append(f"{label}={_format_terminal_summary_value(value)}")

    overall_row: Dict[str, Any] = {}
    if not summary_df.empty and "bias_type" in summary_df.columns:
        overall_matches = summary_df[summary_df["bias_type"].astype(str) == "overall"]
        if not overall_matches.empty:
            overall_row = overall_matches.iloc[0].to_dict()
        else:
            overall_row = summary_df.iloc[0].to_dict()

    overall_metrics = [
        ("overall_accuracy", overall_row.get("accuracy"), True),
        ("avg_p_correct", overall_row.get("avg_p_correct"), False),
        ("avg_p_selected", overall_row.get("avg_p_selected"), False),
        ("avg_delta_p_biased_minus_neutral", overall_row.get("avg_delta_p_biased_minus_neutral"), False),
        ("harmful_flip_rate", overall_row.get("harmful_flip_rate"), True),
        ("helpful_flip_rate", overall_row.get("helpful_flip_rate"), True),
        ("unchanged_correctness_rate", overall_row.get("unchanged_correctness_rate"), True),
    ]
    for label, value, as_percent in overall_metrics:
        if value is not None:
            lines.append(f"{label}={_format_terminal_summary_value(value, as_percent=as_percent)}")

    bias_lines: List[str] = []
    if not summary_df.empty and "bias_type" in summary_df.columns:
        non_overall = summary_df[summary_df["bias_type"].astype(str) != "overall"].copy()
        if not non_overall.empty:
            non_overall = non_overall.sort_values("bias_type")
            for _, row in non_overall.iterrows():
                bias_lines.append(
                    "  "
                    + str(row.get("bias_type", "") or "<unknown>")
                    + ": accuracy="
                    + _format_terminal_summary_value(row.get("accuracy"), as_percent=True)
                    + " avg_p_correct="
                    + _format_terminal_summary_value(row.get("avg_p_correct"))
                    + " avg_p_selected="
                    + _format_terminal_summary_value(row.get("avg_p_selected"))
                )
    if bias_lines:
        lines.append("summary_by_bias:")
        lines.extend(bias_lines)

    if "probe_training_status" in payload_dict:
        lines.append(
            "probe_training_status="
            + _format_terminal_summary_value(payload_dict.get("probe_training_status"))
        )
    if isinstance(best_probe, dict) and best_probe:
        lines.append("best_probe_name=" + _format_terminal_summary_value(best_probe.get("probe_name")))
        lines.append("best_probe_test_auc=" + _format_terminal_summary_value(best_probe.get("test_auc")))

    return lines


def _format_summary_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    try:
        if pd.isna(value):
            return "n/a"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return "n/a"
        return f"{numeric:.3f}"
    return str(value)


def _format_summary_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.1%}"


def _markdown_table(
    df: pd.DataFrame,
    columns: Sequence[str],
    column_labels: Optional[Dict[str, str]] = None,
) -> str:
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns or df.empty:
        return "_No rows._"

    labels = [column_labels.get(column, column) if column_labels else column for column in available_columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(available_columns)) + " |",
    ]
    for _, row in df[available_columns].iterrows():
        lines.append(
            "| "
            + " | ".join(_format_summary_value(row[column]) for column in available_columns)
            + " |"
        )
    return "\n".join(lines)


def build_executive_summary_report_intro_markdown(
    summary_payload: Any,
) -> str:
    payload_dict = summary_payload if isinstance(summary_payload, dict) else {}
    generated_at_utc = payload_dict.get("generated_at_utc")
    headline_counts = payload_dict.get("headline_counts", {})
    summary_df = pd.DataFrame(_summary_rows_from_payload(summary_payload))
    mc_option_count_summary = (
        payload_dict.get("mc_option_count_summary", {}) if isinstance(payload_dict, dict) else {}
    )
    runtime_timing_raw = payload_dict.get("runtime_timing", {}) if isinstance(payload_dict, dict) else {}
    runtime_timing = runtime_timing_raw if isinstance(runtime_timing_raw, dict) else {}
    runtime_stage_df = pd.DataFrame(
        runtime_timing.get("stages", []) if isinstance(runtime_timing, dict) else []
    )

    overall_row = {}
    if not summary_df.empty and "bias_type" in summary_df.columns:
        overall_matches = summary_df[summary_df["bias_type"].astype(str) == "overall"]
        if not overall_matches.empty:
            overall_row = overall_matches.iloc[0].to_dict()

    overview_rows = [
        {"metric": "sample_rows", "value": headline_counts.get("sample_rows")},
        {"metric": "question_count", "value": headline_counts.get("question_count")},
        {"metric": "paired_rows", "value": headline_counts.get("paired_rows")},
        {"metric": "probe_families", "value": headline_counts.get("probe_family_count")},
    ]
    mc_option_count_display = (
        mc_option_count_summary.get("display") if isinstance(mc_option_count_summary, dict) else None
    )
    if mc_option_count_display:
        overview_rows.append({"metric": "number of choices", "value": mc_option_count_display})
    overview_rows.extend(
        [
            {"metric": "overall_accuracy", "value": overall_row.get("accuracy")},
            {"metric": "overall_avg_p_correct", "value": overall_row.get("avg_p_correct")},
            {"metric": "overall_avg_p_selected", "value": overall_row.get("avg_p_selected")},
            {
                "metric": "avg_delta_p_biased_minus_neutral",
                "value": overall_row.get("avg_delta_p_biased_minus_neutral"),
            },
            {"metric": "harmful_flip_rate", "value": overall_row.get("harmful_flip_rate")},
            {"metric": "helpful_flip_rate", "value": overall_row.get("helpful_flip_rate")},
        ]
    )
    overview_df = pd.DataFrame(overview_rows)
    runtime_overview_df = pd.DataFrame(
        [
            {"metric": "status", "value": runtime_timing.get("status")},
            {"metric": "run_started_at_utc", "value": runtime_timing.get("started_at_utc")},
            {"metric": "timing_snapshot_at_utc", "value": runtime_timing.get("snapshot_at_utc")},
            {"metric": "total_elapsed_seconds", "value": runtime_timing.get("total_elapsed_seconds")},
            {
                "metric": "total_elapsed_human",
                "value": _format_duration_human(runtime_timing.get("total_elapsed_seconds")),
            },
        ]
    )
    if not runtime_stage_df.empty:
        runtime_stage_df = runtime_stage_df.copy()
        runtime_stage_df["duration_human"] = runtime_stage_df["duration_seconds"].apply(_format_duration_human)

    sections = [
        f"- Run: `{payload_dict.get('run_name', '')}`",
        f"- Model: `{payload_dict.get('model_name', '')}`",
        f"- Dataset: `{payload_dict.get('dataset_name', '')}`",
        f"- Generated: `{generated_at_utc or 'n/a'}`",
        "",
        "## Model Overview",
        _markdown_table(overview_df, ["metric", "value"], {"metric": "Metric", "value": "Value"}),
        "",
        "## Summary by Bias",
        _markdown_table(
            summary_df,
            [
                "bias_type",
                "n_prompt_rows",
                "accuracy",
                "avg_p_correct",
                "avg_p_selected",
                "neutral_accuracy",
                "biased_accuracy",
                "avg_delta_p_biased_minus_neutral",
                "harmful_flip_rate",
                "helpful_flip_rate",
            ],
            {
                "bias_type": "Bias",
                "n_prompt_rows": "Prompt rows",
                "accuracy": "Accuracy",
                "avg_p_correct": "Avg p(correct)",
                "avg_p_selected": "Avg p(selected)",
                "neutral_accuracy": "Neutral acc",
                "biased_accuracy": "Biased acc",
                "avg_delta_p_biased_minus_neutral": "Delta p(bias-neutral)",
                "harmful_flip_rate": "Harmful flip",
                "helpful_flip_rate": "Helpful flip",
            },
        ),
        "",
        "## Runtime",
        (
            _markdown_table(
                runtime_overview_df,
                ["metric", "value"],
                {"metric": "Metric", "value": "Value"},
            )
            if runtime_timing
            else "_No runtime timing available._"
        ),
        "",
        "### Stage Timing",
        (
            _markdown_table(
                runtime_stage_df,
                ["stage_index", "stage_name", "stage_status", "duration_seconds", "duration_human"],
                {
                    "stage_index": "Stage",
                    "stage_name": "Name",
                    "stage_status": "Status",
                    "duration_seconds": "Seconds",
                    "duration_human": "Duration",
                },
            )
            if not runtime_stage_df.empty
            else "_No stage timing available._"
        ),
    ]
    return "\n".join(sections).strip() + "\n"


def build_executive_summary_markdown(
    summary_payload: Any,
) -> str:
    payload_dict = summary_payload if isinstance(summary_payload, dict) else {}
    report_intro_markdown = build_executive_summary_report_intro_markdown(summary_payload).rstrip()
    mc_confusion_matrix = payload_dict.get("mc_confusion_matrix") if isinstance(payload_dict, dict) else None
    mc_confusion_df = _mc_confusion_matrix_df_from_summary(mc_confusion_matrix)
    mc_confusion_labels = [
        str(label)
        for label in (mc_confusion_matrix.get("choice_labels", []) if isinstance(mc_confusion_matrix, dict) else [])
        if str(label)
    ]
    mc_option_selection = payload_dict.get("mc_option_selection", {}) if isinstance(payload_dict, dict) else {}
    mc_choice_labels = [
        str(label)
        for label in (mc_option_selection.get("choice_labels", []) if isinstance(mc_option_selection, dict) else [])
        if str(label)
    ]
    mc_summary_df = pd.DataFrame(
        mc_option_selection.get("summary_rows", []) if isinstance(mc_option_selection, dict) else []
    )
    probe_df = pd.DataFrame(payload_dict.get("probe_score_summaries", []))
    best_probe_df = pd.DataFrame(
        [payload_dict.get("selected_probe_overview", {})]
        if isinstance(payload_dict.get("selected_probe_overview"), dict)
        and payload_dict.get("selected_probe_overview")
        else []
    )
    strict_mc_quality_raw = payload_dict.get("strict_mc_quality", {}) if isinstance(payload_dict, dict) else {}
    strict_mc_quality = strict_mc_quality_raw if isinstance(strict_mc_quality_raw, dict) else {}
    strict_mc_quality_summary_raw = strict_mc_quality.get("summary", strict_mc_quality)
    strict_mc_quality_summary = (
        strict_mc_quality_summary_raw if isinstance(strict_mc_quality_summary_raw, dict) else {}
    )
    neutral_choice_concentration_raw = strict_mc_quality_summary.get("neutral_choice_concentration", {})
    neutral_choice_concentration = (
        neutral_choice_concentration_raw if isinstance(neutral_choice_concentration_raw, dict) else {}
    )
    neutral_selected_label_skew_raw = strict_mc_quality_summary.get("neutral_selected_label_skew", {})
    neutral_selected_label_skew = (
        neutral_selected_label_skew_raw if isinstance(neutral_selected_label_skew_raw, dict) else {}
    )
    selected_label_distribution = (
        dict(neutral_selected_label_skew.get("correct_label_distribution", {}))
        if isinstance(neutral_selected_label_skew.get("correct_label_distribution", {}), dict)
        else {}
    )
    dominant_label = str(neutral_selected_label_skew.get("dominant_selected_label", "") or "").strip().upper()
    concentration_threshold = neutral_choice_concentration.get("high_confidence_selected_prob_threshold")
    concentration_rows: List[Dict[str, Any]] = []
    if neutral_choice_concentration:
        threshold_label = (
            f"Rate P(selected) >= {float(concentration_threshold):.2f}"
            if isinstance(concentration_threshold, (int, float, np.integer, np.floating))
            and np.isfinite(float(concentration_threshold))
            else "High-confidence selected rate"
        )
        concentration_rows = [
            {
                "metric": "Rows with neutral choice probabilities",
                "value": neutral_choice_concentration.get("selected_probability_row_count"),
            },
            {
                "metric": "Median effective options (N_eff)",
                "value": neutral_choice_concentration.get("median_effective_options"),
            },
            {
                "metric": threshold_label,
                "value": _format_summary_pct(neutral_choice_concentration.get("high_confidence_selected_rate")),
            },
        ]
    concentration_df = pd.DataFrame(concentration_rows)
    skew_rows: List[Dict[str, Any]] = []
    if neutral_selected_label_skew:
        answer_key_rate = selected_label_distribution.get(dominant_label) if dominant_label else None
        skew_rows = [
            {
                "metric": "Dominant selected label",
                "value": dominant_label or "n/a",
            },
            {
                "metric": "Selected-label rate q(dominant)",
                "value": _format_summary_pct(neutral_selected_label_skew.get("dominant_selected_label_rate")),
            },
            {
                "metric": "Answer-key rate r(dominant)",
                "value": _format_summary_pct(answer_key_rate),
            },
            {
                "metric": "Excess q(dominant) - r(dominant)",
                "value": _format_summary_pct(neutral_selected_label_skew.get("dominant_selected_label_excess")),
            },
            {
                "metric": "Total variation distance",
                "value": _format_summary_pct(neutral_selected_label_skew.get("selected_vs_answer_key_tv_distance")),
            },
        ]
    skew_df = pd.DataFrame(skew_rows)
    strict_mc_diagnostics_available = bool(concentration_rows or skew_rows or strict_mc_quality)

    sections = [
        "# Executive Summary",
        "",
        report_intro_markdown,
        "",
        *(
            [
                "## MC Confusion Matrix",
                (
                    _markdown_table(
                        mc_confusion_df,
                        ["predicted_letter", *mc_confusion_labels],
                        {
                            "predicted_letter": "Predicted \\ True",
                            **{label: label for label in mc_confusion_labels},
                        },
                    )
                    if not mc_confusion_df.empty
                    else "_No usable MC rows with both predicted and true letters._"
                ),
                "",
            ]
            if isinstance(mc_confusion_matrix, dict)
            else []
        ),
        "## MC Option Selection",
        (
            _markdown_table(
                mc_summary_df,
                [
                    "template_type",
                    "n_rows",
                    *[f"pick_{label}_rate" for label in mc_choice_labels],
                    "avg_effective_options",
                ],
                {
                    "template_type": "Template",
                    "n_rows": "Rows",
                    **{f"pick_{label}_rate": f"Pick {label}" for label in mc_choice_labels},
                    "avg_effective_options": "Avg N_eff",
                },
            )
            if not mc_summary_df.empty
            else "_No strict-MC choice-probability rows._"
        ),
        "",
        *(
            [
                "## Strict-MC Neutral Diagnostics",
                (
                    "Within-question choice concentration summarizes how sharply each neutral question's "
                    "allowed-choice probabilities concentrate on one option. It is not a cross-question "
                    "letter-bias metric; cross-question preference for a fixed label such as `A` is tracked "
                    "separately by the selected-label skew table below."
                ),
                "",
                "### Within-Question Choice Concentration",
                (
                    _markdown_table(
                        concentration_df,
                        ["metric", "value"],
                        {"metric": "Metric", "value": "Value"},
                    )
                    if not concentration_df.empty
                    else "_No neutral strict-MC concentration metrics available._"
                ),
                "",
                "### Cross-Question Selected-Label Skew",
                (
                    _markdown_table(
                        skew_df,
                        ["metric", "value"],
                        {"metric": "Metric", "value": "Value"},
                    )
                    if not skew_df.empty
                    else "_No neutral strict-MC selected-label skew metrics available._"
                ),
                "",
            ]
            if strict_mc_diagnostics_available
            else []
        ),
        "## Best Probe",
        _markdown_table(
            best_probe_df,
            [
                "probe_name",
                "best_layer",
                "best_dev_auc",
                "test_auc",
                "test_accuracy",
                "probe_prefers_correct_rate",
            ],
            {
                "probe_name": "Probe",
                "best_layer": "Layer",
                "best_dev_auc": "Dev AUC",
                "test_auc": "Test AUC",
                "test_accuracy": "Test acc",
                "probe_prefers_correct_rate": "Prefers correct",
            },
        ),
        "",
        "## Probe Overview",
        _markdown_table(
            probe_df,
            [
                "probe_name",
                "best_layer",
                "best_dev_auc",
                "test_auc",
                "test_accuracy",
                "probe_prefers_correct_rate",
                "probe_prefers_selected_rate",
            ],
            {
                "probe_name": "Probe",
                "best_layer": "Layer",
                "best_dev_auc": "Dev AUC",
                "test_auc": "Test AUC",
                "test_accuracy": "Test acc",
                "probe_prefers_correct_rate": "Prefers correct",
                "probe_prefers_selected_rate": "Prefers selected",
            },
        ),
    ]
    return "\n".join(sections).strip() + "\n"


def build_run_summary_payload(
    *,
    args: Any,
    run_dir: Path,
    samples_df: pd.DataFrame,
    tuples_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    probes_meta: Dict[str, Any],
    model_summary_payload: Optional[Dict[str, Any]] = None,
    probe_summary_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if probe_summary_payload is None:
        chosen_probe_rows = _build_probe_test_auc_rows(probes_meta)
        ranked_probe_rows = [row for row in chosen_probe_rows if row.get("test_auc") is not None]
        best_probe = None
        if ranked_probe_rows:
            best_probe = max(
                ranked_probe_rows,
                key=lambda row: (
                    float(row.get("test_auc") or float("-inf")),
                    float(row.get("best_dev_auc") or float("-inf")),
                ),
            )
    else:
        best_probe = probe_summary_payload.get("overview", {}).get("best_probe_on_test")

    if model_summary_payload is None:
        prompt_overview = _prompt_metrics_from_df(samples_df)
        pair_overview = _pair_metrics_from_df(tuples_df)
    else:
        prompt_overview = model_summary_payload.get("prompt_level", {}).get("all_rows")
        pair_overview = model_summary_payload.get("paired_effects", {}).get("all_pairs")

    if prompt_overview is None:
        prompt_overview = {}
    if pair_overview is None:
        pair_overview = {}
    if prompt_overview.get("accuracy") is None:
        prompt_overview = {
            **prompt_overview,
            "accuracy": pair_overview.get("accuracy_x"),
        }
    if prompt_overview.get("avg_p_correct") is None:
        prompt_overview = {
            **prompt_overview,
            "avg_p_correct": pair_overview.get("avg_p_x"),
        }

    return _json_ready(
        {
            "summary_schema_version": RUN_SUMMARY_SCHEMA_VERSION,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "model_name": str(args.model),
            "dataset_name": str(getattr(args, "dataset_name", "") or ""),
            "bias_types": _list_like_strings(getattr(args, "bias_types", [])),
            "sampling_only": bool(getattr(args, "sampling_only", False)),
            "probe_training_status": probes_meta.get("probe_training_status", "completed"),
            "counts": {
                "sample_rows": int(len(samples_df)),
                "tuple_rows": int(len(tuples_df)),
                "summary_rows": int(len(summary_df)),
                "question_count": int(samples_df["question_id"].nunique()) if not samples_df.empty else 0,
            },
            "headline_metrics": {
                "overall_accuracy": None if prompt_overview is None else prompt_overview.get("accuracy"),
                "overall_avg_p_correct": None if prompt_overview is None else prompt_overview.get("avg_p_correct"),
                "avg_delta_p_x_minus_xprime": None
                if pair_overview is None
                else pair_overview.get("avg_delta_p_x_minus_xprime"),
                "best_probe_name": None if best_probe is None else best_probe.get("probe_name"),
                "best_probe_test_auc": None if best_probe is None else best_probe.get("test_auc"),
            },
            "paths": {
                "sampled_responses": str(preferred_run_artifact_path(run_dir, "sampled_responses")),
                "reports_summary": str(preferred_run_artifact_path(run_dir, "reports_summary")),
                "probe_scores_by_prompt": str(preferred_run_artifact_path(run_dir, "probe_scores_by_prompt")),
                "executive_summary": str(preferred_run_artifact_path(run_dir, "executive_summary")),
            },
        }
    )


def refresh_runtime_summary_artifacts(
    *,
    run_dir: Path,
    runtime_timing: Dict[str, Any],
) -> None:
    run_summary_path = preferred_run_artifact_path(run_dir, "run_summary")
    executive_summary_path = preferred_run_artifact_path(run_dir, "executive_summary")
    if not run_summary_path.exists():
        return

    try:
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    payload["runtime_timing"] = _json_ready(runtime_timing)
    write_json_atomic(run_summary_path, payload)
    write_text_atomic(executive_summary_path, build_executive_summary_markdown(payload))


def persist_sampling_state(
    *,
    stage: str,
    split_states: Dict[str, Sequence[Dict[str, Any]]],
    split_stats: Dict[str, Dict[str, int]],
    expected_all_keys: Set[tuple[str, str, str, int]],
    expected_total_records: int,
    sampling_records_path: Path,
    sampling_manifest_path: Path,
    sampling_hash: str,
    sampling_spec: Dict[str, Any],
    cached_source_run: Optional[Path],
) -> None:
    combined_input: List[Dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        combined_input.extend(list(split_states.get(split_name, [])))
    combined = normalize_sample_records(combined_input, expected_all_keys)
    write_jsonl_atomic(sampling_records_path, combined)
    manifest = {
        "sampling_hash": sampling_hash,
        "sampling_spec": sampling_spec,
        "expected_records": expected_total_records,
        "n_records": len(combined),
        "is_complete": len(combined) >= expected_total_records,
        "stage": stage,
        "updated_at_utc": utc_now_iso(),
        "source_cache_run_dir": str(cached_source_run) if cached_source_run is not None else None,
        "split_stats": split_stats,
        "train_stats": split_stats.get("train"),
        "val_stats": split_stats.get("val"),
        "test_stats": split_stats.get("test"),
    }
    write_json_atomic(sampling_manifest_path, manifest)


def save_sampling_integrity_summary(
    *,
    run_dir: Path,
    sampling_integrity_summary: Dict[str, Any],
) -> Path:
    path = preferred_run_artifact_path(run_dir, "sampling_integrity_summary")
    write_json_atomic(path, sampling_integrity_summary)
    log_status("saving_manager.py", f"saved artifact: {path}")
    return path


def save_run_results(
    *,
    args: Any,
    run_dir: Path,
    lock_path: Path,
    sampling_hash: str,
    sampling_records_path: Path,
    sampling_manifest_path: Path,
    run_log_path: Path,
    warning_log_path: Optional[Path],
    sampling_integrity_summary_path: Path,
    all_records: List[Dict[str, Any]],
    probe_candidate_score_rows: List[Dict[str, Any]],
    bias_types: Sequence[str],
    probes_meta: Dict[str, Any],
    run_timing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    tuple_rows = build_tuple_rows(all_records, model_name=args.model, bias_types=bias_types)
    tuples_df = to_tuples_df(tuple_rows)
    samples_df = to_samples_df(all_records, model_name=args.model)
    probe_candidate_scores_df = to_probe_candidate_scores_df(
        probe_candidate_score_rows,
        model_name=args.model,
    )
    probe_scores_by_prompt_df = build_mc_probe_scores_by_prompt_df(probe_candidate_scores_df)
    reports_summary_payload = build_reports_summary_payload(
        args=args,
        run_dir=run_dir,
        samples_df=samples_df,
        sample_records=all_records,
        tuples_df=tuples_df,
        probe_scores_by_prompt_df=probe_scores_by_prompt_df,
        probes_meta=probes_meta,
        probe_candidate_scores_df=probe_candidate_scores_df,
        run_timing=run_timing,
    )
    reports_summary_df = pd.DataFrame(
        reports_summary_payload.get("summary_rows", []),
        columns=REPORTS_SUMMARY_COLUMNS,
    )
    executive_summary_text = build_executive_summary_markdown(reports_summary_payload)

    samples_path = preferred_run_artifact_path(run_dir, "sampled_responses")
    reports_summary_path = preferred_run_artifact_path(run_dir, "reports_summary")
    reports_summary_csv_path = preferred_run_artifact_path(run_dir, "reports_summary_csv")
    run_summary_path = preferred_run_artifact_path(run_dir, "run_summary")
    probe_scores_by_prompt_path = preferred_run_artifact_path(run_dir, "probe_scores_by_prompt")
    executive_summary_path = preferred_run_artifact_path(run_dir, "executive_summary")
    config_path = preferred_run_artifact_path(run_dir, "run_config")
    mc_confusion_matrix_summary = reports_summary_payload.get("mc_confusion_matrix")
    mc_confusion_matrix_path = (
        preferred_run_artifact_path(run_dir, "mc_confusion_matrix")
        if isinstance(mc_confusion_matrix_summary, dict)
        else None
    )
    mc_confusion_matrix_df = _mc_confusion_matrix_df_from_summary(mc_confusion_matrix_summary)

    write_csv_atomic(samples_path, samples_df)
    write_json_atomic(reports_summary_path, reports_summary_payload.get("summary_rows", []))
    write_csv_atomic(reports_summary_csv_path, reports_summary_df)
    write_json_atomic(run_summary_path, reports_summary_payload)
    write_csv_atomic(probe_scores_by_prompt_path, probe_scores_by_prompt_df)
    if mc_confusion_matrix_path is not None:
        write_csv_atomic(mc_confusion_matrix_path, mc_confusion_matrix_df)
    write_text_atomic(executive_summary_path, executive_summary_text)

    run_cfg = dict(vars(args))
    requested_device = str(getattr(args, "requested_device", getattr(args, "device", "")) or "")
    resolved_device = str(getattr(args, "resolved_device", "") or "")
    if not resolved_device and requested_device and requested_device != "auto":
        resolved_device = requested_device
    if requested_device:
        run_cfg["requested_device"] = requested_device
        # Keep the legacy field for backward compatibility with older artifacts and helpers.
        run_cfg["device"] = requested_device
    if resolved_device:
        run_cfg["resolved_device"] = resolved_device
    run_cfg["run_dir"] = str(run_dir)
    run_cfg["run_name"] = run_dir.name
    run_cfg["model_slug"] = model_slug(args.model)
    run_cfg["lock_path"] = str(lock_path)
    run_cfg["sampling_hash"] = sampling_hash
    run_cfg["sampling_records_path"] = str(sampling_records_path)
    run_cfg["sampling_manifest_path"] = str(sampling_manifest_path)
    run_cfg["sampling_integrity_summary_path"] = str(sampling_integrity_summary_path)
    run_cfg["sampled_responses_path"] = str(samples_path)
    run_cfg["reports_summary_path"] = str(reports_summary_path)
    run_cfg["reports_summary_csv_path"] = str(reports_summary_csv_path)
    run_cfg["run_summary_path"] = str(run_summary_path)
    run_cfg["probe_scores_by_prompt_path"] = str(probe_scores_by_prompt_path)
    run_cfg["executive_summary_path"] = str(executive_summary_path)
    if mc_confusion_matrix_path is not None:
        run_cfg["mc_confusion_matrix_path"] = str(mc_confusion_matrix_path)
    if probes_meta.get("all_probes_dir"):
        run_cfg["all_probes_dir"] = str(probes_meta["all_probes_dir"])
    if probes_meta.get("chosen_probe_dir"):
        run_cfg["chosen_probe_dir"] = str(probes_meta["chosen_probe_dir"])
    run_cfg["run_log_path"] = str(run_log_path)
    if warning_log_path is not None and warning_log_path.exists():
        run_cfg["warnings_log_path"] = str(warning_log_path)
    write_json_atomic(config_path, run_cfg)

    saved_paths = {
        "samples_path": samples_path,
        "reports_summary_path": reports_summary_path,
        "reports_summary_csv_path": reports_summary_csv_path,
        "run_summary_path": run_summary_path,
        "probe_scores_by_prompt_path": probe_scores_by_prompt_path,
        "executive_summary_path": executive_summary_path,
        "config_path": config_path,
        "sampling_records_path": sampling_records_path,
        "sampling_manifest_path": sampling_manifest_path,
        "sampling_integrity_summary_path": sampling_integrity_summary_path,
        "run_log_path": run_log_path,
    }
    if mc_confusion_matrix_path is not None:
        saved_paths["mc_confusion_matrix_path"] = mc_confusion_matrix_path
    if warning_log_path is not None and warning_log_path.exists():
        saved_paths["warnings_log_path"] = warning_log_path
    for path in saved_paths.values():
        log_status("saving_manager.py", f"saved artifact: {path}")
    return saved_paths


__all__ = [
    "MC_PROBE_SCORE_BY_PROMPT_BASE_COLUMNS",
    "REPORTS_SUMMARY_COLUMNS",
    "MODEL_SUMMARY_BY_BIAS_COLUMNS",
    "MODEL_SUMMARY_BY_TEMPLATE_COLUMNS",
    "P_CORRECT_COLUMN",
    "P_SELECTED_COLUMN",
    "PROBE_SUMMARY_COLUMNS",
    "SUMMARY_COLUMNS",
    "PROBE_CANDIDATE_SCORE_COLUMNS",
    "SAMPLED_RESPONSE_COLUMNS",
    "build_mc_probe_scores_by_prompt_df",
    "build_executive_summary_markdown",
    "build_executive_summary_report_intro_markdown",
    "build_model_summary_by_bias_df",
    "build_model_summary_by_template_df",
    "build_model_summary_payload",
    "build_probe_summary_df",
    "build_probe_summary_payload",
    "build_reports_summary_df",
    "build_reports_summary_payload",
    "build_terminal_final_stats_lines",
    "build_run_summary_payload",
    "build_summary_df",
    "build_tuple_rows",
    "persist_sampling_state",
    "refresh_runtime_summary_artifacts",
    "save_sampling_integrity_summary",
    "save_run_results",
    "to_probe_candidate_scores_df",
    "to_samples_df",
    "to_tuples_df",
]
