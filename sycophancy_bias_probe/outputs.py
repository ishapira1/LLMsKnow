from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .correctness import record_is_usable_for_metrics as _record_is_usable_for_metrics


SUMMARY_COLUMNS = [
    "model_name",
    "split",
    "question_id",
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
    rows = []
    for record in records:
        correctness = record.get("correctness")
        rows.append(
            {
                "model_name": model_name,
                "record_id": record["record_id"],
                "split": record["split"],
                "question_id": record["question_id"],
                "dataset": record.get("dataset", ""),
                "template_type": record["template_type"],
                "draw_idx": record["draw_idx"],
                "question": record["question"],
                "correct_answer": record["correct_answer"],
                "incorrect_answer": record["incorrect_answer"],
                "incorrect_answer_source": record.get("incorrect_answer_source", ""),
                "task_format": record.get("task_format", ""),
                "mc_mode": record.get("mc_mode", ""),
                "answer_channel": record.get("answer_channel", ""),
                "prompt_spec_version": record.get("prompt_spec_version", ""),
                "grading_spec_version": record.get("grading_spec_version", ""),
                "correct_letter": record.get("correct_letter", ""),
                "incorrect_letter": record.get("incorrect_letter", ""),
                "letters": record.get("letters", ""),
                "answer_options": record.get("answer_options", ""),
                "answers_list": json.dumps(record.get("answers_list", []), ensure_ascii=False),
                "gold_answers": json.dumps(record["gold_answers"], ensure_ascii=False),
                "prompt_template": record["prompt_template"],
                "prompt_text": record["prompt_text"],
                "response_raw": record["response_raw"],
                "response": record["response"],
                "committed_answer": record.get("committed_answer", ""),
                "commitment_kind": record.get("commitment_kind", ""),
                "commitment_source": record.get("commitment_source", ""),
                "starts_with_answer_prefix": bool(record.get("starts_with_answer_prefix", False)),
                "strict_format_exact": bool(record.get("strict_format_exact", False)),
                "commitment_line": record.get("commitment_line", ""),
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
                "T_prompt": float(record["T_prompt"]),
                "probe_x": record.get("probe_x", np.nan),
                "probe_xprime": record.get("probe_xprime", np.nan),
            }
        )
    return pd.DataFrame(rows)


def to_tuples_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def build_summary_df(tuples_df: pd.DataFrame) -> pd.DataFrame:
    if len(tuples_df) == 0:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    working_df = tuples_df.copy()
    for column_name in ("dataset", "prompt_template_x", "prompt_template_xprime"):
        if column_name not in working_df.columns:
            working_df[column_name] = ""

    return (
        working_df.groupby(["model_name", "split", "question_id", "dataset", "bias_type"], as_index=False)
        .agg(
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
