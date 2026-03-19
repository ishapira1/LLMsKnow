from __future__ import annotations

import json
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
    "choice_probability_correct",
    "choice_probability_selected",
    "T_prompt",
    "probe_x",
    "probe_xprime",
]

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

RUN_SUMMARY_SCHEMA_VERSION = 1


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
    rows = []
    for record in records:
        correctness = record.get("correctness")
        rows.append(
            {
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
                "choice_probabilities": json.dumps(
                    record.get("choice_probabilities", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "choice_probability_correct": record.get("choice_probability_correct", np.nan),
                "choice_probability_selected": record.get("choice_probability_selected", np.nan),
                "T_prompt": float(record["T_prompt"]),
                "probe_x": record.get("probe_x", np.nan),
                "probe_xprime": record.get("probe_xprime", np.nan),
            }
        )
    return pd.DataFrame(rows, columns=SAMPLED_RESPONSE_COLUMNS)


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


def _build_agreement_injection_records(tuples_df: pd.DataFrame, group_cols: Sequence[str]) -> List[Dict[str, Any]]:
    if tuples_df.empty:
        return []

    working_df = tuples_df.copy()
    working_df["agreement_injection"] = working_df["bias_type"].astype(str)
    working_df["delta_p_x_minus_xprime"] = (
        pd.to_numeric(working_df["T_x"], errors="coerce")
        - pd.to_numeric(working_df["T_xprime"], errors="coerce")
    )
    grouped = (
        working_df.groupby(list(group_cols), as_index=False)
        .agg(
            n_pairs=("draw_idx", "size"),
            n_questions=("question_id", "nunique"),
            accuracy_x=("C_x_y", "mean"),
            accuracy_xprime=("C_xprime_yprime", "mean"),
            avg_p_x=("T_x", "mean"),
            avg_p_xprime=("T_xprime", "mean"),
            avg_delta_p_x_minus_xprime=("delta_p_x_minus_xprime", "mean"),
        )
        .sort_values(list(group_cols))
    )
    return grouped.to_dict(orient="records")


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


def build_run_summary_payload(
    *,
    args: Any,
    run_dir: Path,
    samples_df: pd.DataFrame,
    tuples_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    probes_meta: Dict[str, Any],
) -> Dict[str, Any]:
    chosen_probe_rows = _build_probe_test_auc_rows(probes_meta)
    ranked_probe_rows = [row for row in chosen_probe_rows if row.get("test_auc") is not None]
    best_probe_on_test = None
    if ranked_probe_rows:
        best_probe_on_test = max(
            ranked_probe_rows,
            key=lambda row: (
                float(row.get("test_auc") or float("-inf")),
                float(row.get("best_dev_auc") or float("-inf")),
            ),
        )

    return {
        "summary_schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": str(args.model),
        "dataset_name": str(getattr(args, "dataset_name", "") or ""),
        "bias_types": _list_like_strings(getattr(args, "bias_types", [])),
        "definitions": {
            "agreement_injection": "The prompt-side bias_type applied to the neutral prompt x to form x'.",
            "accuracy_x": "Accuracy before agreement injection, averaged over neutral rows.",
            "accuracy_xprime": "Accuracy after agreement injection, averaged over biased rows.",
            "avg_p_x": (
                "Average p(x), where p(x) is the probability of the correct response. "
                "For strict_mc this is the model probability on the gold choice; otherwise it is the empirical "
                "correctness rate across draws."
            ),
            "avg_p_xprime": "Average p(x') after agreement injection, using the same definition as p(x).",
            "avg_delta_p_x_minus_xprime": "Average of p(x) - p(x'). Positive values mean the injected prompt reduced p(correct).",
            "best_probe_on_test": "The chosen probe family with the highest test AUC among the saved chosen probes.",
        },
        "counts": {
            "sample_rows": int(len(samples_df)),
            "tuple_rows": int(len(tuples_df)),
            "summary_rows": int(len(summary_df)),
            "question_count": int(samples_df["question_id"].nunique()) if not samples_df.empty else 0,
        },
        "agreement_injection_summary": {
            "all_splits": _build_agreement_injection_records(
                tuples_df=tuples_df,
                group_cols=("agreement_injection",),
            ),
            "by_split": _build_agreement_injection_records(
                tuples_df=tuples_df,
                group_cols=("split", "agreement_injection"),
            ),
        },
        "chosen_probe_test_metrics": chosen_probe_rows,
        "best_probe_on_test": best_probe_on_test,
    }


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
    sampling_integrity_summary_path: Path,
    all_records: List[Dict[str, Any]],
    probe_candidate_score_rows: List[Dict[str, Any]],
    bias_types: Sequence[str],
    probes_meta: Dict[str, Any],
) -> Dict[str, Path]:
    tuple_rows = build_tuple_rows(all_records, model_name=args.model, bias_types=bias_types)
    tuples_df = to_tuples_df(tuple_rows)
    summary_df = build_summary_df(tuples_df)
    samples_df = to_samples_df(all_records, model_name=args.model)
    probe_candidate_scores_df = to_probe_candidate_scores_df(
        probe_candidate_score_rows,
        model_name=args.model,
    )
    run_summary_payload = build_run_summary_payload(
        args=args,
        run_dir=run_dir,
        samples_df=samples_df,
        tuples_df=tuples_df,
        summary_df=summary_df,
        probes_meta=probes_meta,
    )

    samples_path = preferred_run_artifact_path(run_dir, "sampled_responses")
    tuples_path = preferred_run_artifact_path(run_dir, "final_tuples")
    summary_path = preferred_run_artifact_path(run_dir, "summary_by_question")
    run_summary_path = preferred_run_artifact_path(run_dir, "run_summary")
    probe_candidate_scores_path = preferred_run_artifact_path(run_dir, "probe_candidate_scores")
    meta_path = preferred_run_artifact_path(run_dir, "probe_metadata")
    config_path = preferred_run_artifact_path(run_dir, "run_config")

    write_csv_atomic(samples_path, samples_df)
    write_csv_atomic(tuples_path, tuples_df)
    write_csv_atomic(summary_path, summary_df)
    write_json_atomic(run_summary_path, run_summary_payload)
    write_csv_atomic(probe_candidate_scores_path, probe_candidate_scores_df)
    write_json_atomic(meta_path, probes_meta)

    run_cfg = dict(vars(args))
    run_cfg["run_dir"] = str(run_dir)
    run_cfg["run_name"] = run_dir.name
    run_cfg["model_slug"] = model_slug(args.model)
    run_cfg["lock_path"] = str(lock_path)
    run_cfg["sampling_hash"] = sampling_hash
    run_cfg["sampling_records_path"] = str(sampling_records_path)
    run_cfg["sampling_manifest_path"] = str(sampling_manifest_path)
    run_cfg["sampling_integrity_summary_path"] = str(sampling_integrity_summary_path)
    run_cfg["run_summary_path"] = str(run_summary_path)
    run_cfg["probe_candidate_scores_path"] = str(probe_candidate_scores_path)
    run_cfg["run_log_path"] = str(run_log_path)
    write_json_atomic(config_path, run_cfg)

    saved_paths = {
        "samples_path": samples_path,
        "tuples_path": tuples_path,
        "summary_path": summary_path,
        "run_summary_path": run_summary_path,
        "probe_candidate_scores_path": probe_candidate_scores_path,
        "meta_path": meta_path,
        "config_path": config_path,
        "sampling_records_path": sampling_records_path,
        "sampling_manifest_path": sampling_manifest_path,
        "sampling_integrity_summary_path": sampling_integrity_summary_path,
        "run_log_path": run_log_path,
    }
    for path in saved_paths.values():
        log_status("saving_manager.py", f"saved artifact: {path}")
    return saved_paths


__all__ = [
    "SUMMARY_COLUMNS",
    "PROBE_CANDIDATE_SCORE_COLUMNS",
    "SAMPLED_RESPONSE_COLUMNS",
    "build_run_summary_payload",
    "build_summary_df",
    "build_tuple_rows",
    "persist_sampling_state",
    "save_sampling_integrity_summary",
    "save_run_results",
    "to_probe_candidate_scores_df",
    "to_samples_df",
    "to_tuples_df",
]
