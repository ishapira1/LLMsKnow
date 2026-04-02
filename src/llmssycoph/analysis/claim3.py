from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_ENDORSED_OPTION_GRID_TEMPLATE_TYPE = "endorsed_option_grid_supportive_weak"


def default_claim3_sampling_subdir() -> str:
    return f"sampling_backfills/{DEFAULT_ENDORSED_OPTION_GRID_TEMPLATE_TYPE}"


def default_claim3_probe_subdir(probe_name: str) -> str:
    safe_probe_name = str(probe_name or "").strip() or "probe"
    return f"probes/backfills/{safe_probe_name}_on_{DEFAULT_ENDORSED_OPTION_GRID_TEMPLATE_TYPE}"


def default_claim3_output_subdir(probe_name: str) -> str:
    safe_probe_name = str(probe_name or "").strip() or "probe"
    return f"analysis/claim3/{safe_probe_name}__{DEFAULT_ENDORSED_OPTION_GRID_TEMPLATE_TYPE}"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _summary_label(value: Any) -> str:
    normalized = _normalize_bool(value)
    if normalized is True:
        return "true"
    if normalized is False:
        return "false"
    return "all"


def _coerce_string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


def _coerce_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def load_claim3_backfill_frames(
    run_dir: Path | str,
    *,
    probe_name: str = "probe_no_bias",
    sampling_subdir: Optional[str] = None,
    probe_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_run_dir = Path(run_dir).resolve()
    sampling_dir = resolved_run_dir / str(sampling_subdir or default_claim3_sampling_subdir()).strip()
    probe_dir = resolved_run_dir / str(probe_subdir or default_claim3_probe_subdir(probe_name)).strip()

    sampling_records_path = sampling_dir / "sampling_records.jsonl"
    candidate_scores_path = probe_dir / "probe_candidate_scores.csv"
    prompt_scores_path = probe_dir / "probe_scores_by_prompt.csv"

    if not sampling_records_path.exists():
        raise FileNotFoundError(f"Missing claim-3 sampling records: {sampling_records_path}")
    if not candidate_scores_path.exists():
        raise FileNotFoundError(f"Missing claim-3 candidate scores: {candidate_scores_path}")

    sampling_records = _load_jsonl(sampling_records_path)
    sampling_df = pd.DataFrame(sampling_records)
    candidate_df = pd.read_csv(candidate_scores_path)
    prompt_df = pd.read_csv(prompt_scores_path) if prompt_scores_path.exists() else pd.DataFrame()

    return {
        "run_dir": resolved_run_dir,
        "sampling_dir": sampling_dir,
        "probe_dir": probe_dir,
        "sampling_records_path": sampling_records_path,
        "candidate_scores_path": candidate_scores_path,
        "prompt_scores_path": prompt_scores_path,
        "sampling_df": sampling_df,
        "candidate_df": candidate_df,
        "prompt_df": prompt_df,
    }


def enrich_claim3_candidate_scores(
    candidate_df: pd.DataFrame,
    sampling_df: pd.DataFrame,
    *,
    requested_splits: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()
    if sampling_df.empty:
        raise ValueError("Claim-3 sampling dataframe is empty.")

    working_candidates = candidate_df.copy()
    working_candidates["source_record_id"] = _coerce_numeric_column(working_candidates, "source_record_id").astype("Int64")
    working_candidates["candidate_choice"] = _coerce_string_column(working_candidates, "candidate_choice").str.upper()
    working_candidates["correct_letter"] = _coerce_string_column(working_candidates, "correct_letter").str.upper()
    working_candidates["candidate_correctness"] = _coerce_numeric_column(
        working_candidates, "candidate_correctness"
    ).astype("Int64")
    working_candidates["probe_score"] = _coerce_numeric_column(working_candidates, "probe_score")
    working_candidates["candidate_rank"] = _coerce_numeric_column(working_candidates, "candidate_rank")

    working_sampling = sampling_df.copy()
    working_sampling["record_id"] = _coerce_numeric_column(working_sampling, "record_id").astype("Int64")
    working_sampling["prompt_id"] = _coerce_string_column(working_sampling, "prompt_id")
    working_sampling["split"] = _coerce_string_column(working_sampling, "split")
    working_sampling["question_id"] = _coerce_string_column(working_sampling, "question_id")
    working_sampling["template_type"] = _coerce_string_column(working_sampling, "template_type")
    working_sampling["backfill_mode"] = _coerce_string_column(working_sampling, "backfill_mode")
    working_sampling["framing_family"] = _coerce_string_column(working_sampling, "framing_family")
    working_sampling["tone"] = _coerce_string_column(working_sampling, "tone")
    working_sampling["endorsed_letter"] = _coerce_string_column(working_sampling, "endorsed_letter").str.upper()
    working_sampling["endorsed_text"] = _coerce_string_column(working_sampling, "endorsed_text")
    working_sampling["neutral_source_record_id"] = _coerce_numeric_column(
        working_sampling, "neutral_source_record_id"
    ).astype("Int64")
    working_sampling["neutral_source_prompt_id"] = _coerce_string_column(
        working_sampling, "neutral_source_prompt_id"
    )
    working_sampling["neutral_source_selected_choice"] = _coerce_string_column(
        working_sampling, "neutral_source_selected_choice"
    ).str.upper()
    working_sampling["neutral_source_selected_answer"] = _coerce_string_column(
        working_sampling, "neutral_source_selected_answer"
    )
    working_sampling["endorsed_is_correct"] = working_sampling.get("endorsed_is_correct", False).map(_normalize_bool)
    working_sampling["neutral_source_selected_is_correct"] = working_sampling.get(
        "neutral_source_selected_is_correct", False
    ).map(_normalize_bool)

    merge_columns = [
        "record_id",
        "prompt_id",
        "split",
        "question_id",
        "template_type",
        "backfill_mode",
        "framing_family",
        "tone",
        "endorsed_letter",
        "endorsed_text",
        "endorsed_is_correct",
        "neutral_source_record_id",
        "neutral_source_prompt_id",
        "neutral_source_selected_choice",
        "neutral_source_selected_answer",
        "neutral_source_selected_is_correct",
    ]
    enriched = working_candidates.merge(
        working_sampling[merge_columns],
        how="left",
        left_on="source_record_id",
        right_on="record_id",
        suffixes=("", "_sampling"),
    )
    enriched["candidate_is_endorsed"] = enriched["candidate_choice"].eq(enriched["endorsed_letter"])
    enriched["candidate_matches_neutral_selected"] = enriched["candidate_choice"].eq(
        enriched["neutral_source_selected_choice"]
    )

    if requested_splits:
        wanted = {str(split).strip() for split in requested_splits if str(split).strip()}
        enriched = enriched.loc[enriched["split"].astype(str).isin(wanted)].copy()
    return enriched


def build_claim3_prompt_metrics(enriched_df: pd.DataFrame) -> pd.DataFrame:
    if enriched_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for source_record_id, group in enriched_df.groupby("source_record_id", dropna=False, sort=True):
        valid = group.loc[
            group["probe_score"].notna() & group["candidate_correctness"].isin([0, 1])
        ].copy()
        correct = valid.loc[valid["candidate_correctness"].eq(1)]
        wrong = valid.loc[valid["candidate_correctness"].eq(0)]
        if correct.empty or wrong.empty:
            continue

        correct_score = float(correct["probe_score"].mean())
        wrong_scores = pd.to_numeric(wrong["probe_score"], errors="coerce").dropna()
        if wrong_scores.empty:
            continue

        ordered = valid.sort_values(
            by=["probe_score", "candidate_rank", "candidate_choice"],
            ascending=[False, True, True],
            na_position="last",
        )
        top_row = ordered.iloc[0]
        reference = valid.iloc[0]
        rows.append(
            {
                "source_record_id": source_record_id,
                "split": str(reference.get("split", "") or ""),
                "question_id": str(reference.get("question_id", "") or ""),
                "prompt_id": str(reference.get("prompt_id", "") or ""),
                "template_type": str(reference.get("template_type", "") or ""),
                "framing_family": str(reference.get("framing_family", "") or ""),
                "tone": str(reference.get("tone", "") or ""),
                "endorsed_letter": str(reference.get("endorsed_letter", "") or ""),
                "endorsed_is_correct": _normalize_bool(reference.get("endorsed_is_correct")),
                "neutral_source_record_id": reference.get("neutral_source_record_id", pd.NA),
                "neutral_source_prompt_id": str(reference.get("neutral_source_prompt_id", "") or ""),
                "neutral_source_selected_choice": str(reference.get("neutral_source_selected_choice", "") or ""),
                "correct_letter": str(reference.get("correct_letter", "") or ""),
                "correct_score": correct_score,
                "wrong_mean_score": float(wrong_scores.mean()),
                "truth_gap": float(correct_score - wrong_scores.mean()),
                "pairwise_k": float(np.mean(correct_score > wrong_scores.to_numpy(dtype=float))),
                "probe_top1_choice": str(top_row.get("candidate_choice", "") or ""),
                "probe_top1_correct": bool(int(top_row.get("candidate_correctness", 0) or 0) == 1),
            }
        )
    return pd.DataFrame(rows)


def build_claim3_candidate_leakage(enriched_df: pd.DataFrame) -> pd.DataFrame:
    if enriched_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    group_columns = ["split", "neutral_source_record_id", "question_id", "candidate_choice"]
    for group_key, group in enriched_df.groupby(group_columns, dropna=False, sort=True):
        valid = group.loc[group["probe_score"].notna()].copy()
        endorsed_scores = pd.to_numeric(
            valid.loc[valid["candidate_is_endorsed"], "probe_score"],
            errors="coerce",
        ).dropna()
        not_endorsed_scores = pd.to_numeric(
            valid.loc[~valid["candidate_is_endorsed"], "probe_score"],
            errors="coerce",
        ).dropna()
        if endorsed_scores.empty or not_endorsed_scores.empty:
            continue

        reference = valid.iloc[0]
        split, neutral_source_record_id, question_id, candidate_choice = group_key
        rows.append(
            {
                "split": str(split or ""),
                "neutral_source_record_id": neutral_source_record_id,
                "question_id": str(question_id or ""),
                "candidate_choice": str(candidate_choice or ""),
                "candidate_correctness": int(reference.get("candidate_correctness", 0) or 0),
                "mean_endorsed_score": float(endorsed_scores.mean()),
                "mean_not_endorsed_score": float(not_endorsed_scores.mean()),
                "endorsement_leakage": float(endorsed_scores.mean() - not_endorsed_scores.mean()),
                "abs_endorsement_leakage": float(
                    abs(float(endorsed_scores.mean() - not_endorsed_scores.mean()))
                ),
                "n_endorsed_rows": int(len(endorsed_scores)),
                "n_not_endorsed_rows": int(len(not_endorsed_scores)),
            }
        )
    return pd.DataFrame(rows)


def _build_prompt_summary_rows(prompt_metrics_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if prompt_metrics_df.empty:
        return []

    rows: List[Dict[str, Any]] = []

    def append_summary(subset: pd.DataFrame, *, split: str, endorsed_is_correct: str) -> None:
        if subset.empty:
            return
        rows.append(
            {
                "split": split,
                "endorsed_is_correct": endorsed_is_correct,
                "n_prompts": int(len(subset)),
                "mean_truth_gap": float(pd.to_numeric(subset["truth_gap"], errors="coerce").mean()),
                "mean_pairwise_k": float(pd.to_numeric(subset["pairwise_k"], errors="coerce").mean()),
                "probe_top1_correct_rate": float(
                    pd.to_numeric(subset["probe_top1_correct"].astype(int), errors="coerce").mean()
                ),
                "mean_correct_score": float(pd.to_numeric(subset["correct_score"], errors="coerce").mean()),
                "mean_wrong_score": float(pd.to_numeric(subset["wrong_mean_score"], errors="coerce").mean()),
            }
        )

    append_summary(prompt_metrics_df, split="all", endorsed_is_correct="all")
    for split_value, split_subset in prompt_metrics_df.groupby("split", dropna=False, sort=True):
        append_summary(split_subset, split=str(split_value or ""), endorsed_is_correct="all")
    for endorsed_value, endorsed_subset in prompt_metrics_df.groupby("endorsed_is_correct", dropna=False, sort=True):
        append_summary(prompt_metrics_df.loc[prompt_metrics_df["endorsed_is_correct"].eq(endorsed_value)], split="all", endorsed_is_correct=_summary_label(endorsed_value))
    for (split_value, endorsed_value), subset in prompt_metrics_df.groupby(
        ["split", "endorsed_is_correct"],
        dropna=False,
        sort=True,
    ):
        append_summary(subset, split=str(split_value or ""), endorsed_is_correct=_summary_label(endorsed_value))
    return rows


def summarize_claim3_truth_separation(prompt_metrics_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(_build_prompt_summary_rows(prompt_metrics_df))


def summarize_claim3_endorsement_leakage(candidate_leakage_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_leakage_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    def append_summary(subset: pd.DataFrame, *, split: str, candidate_correctness: str) -> None:
        if subset.empty:
            return
        leakage = pd.to_numeric(subset["endorsement_leakage"], errors="coerce")
        abs_leakage = pd.to_numeric(subset["abs_endorsement_leakage"], errors="coerce")
        rows.append(
            {
                "split": split,
                "candidate_correctness": candidate_correctness,
                "n_candidates": int(len(subset)),
                "mean_endorsement_leakage": float(leakage.mean()),
                "mean_abs_endorsement_leakage": float(abs_leakage.mean()),
            }
        )

    append_summary(candidate_leakage_df, split="all", candidate_correctness="all")
    for split_value, split_subset in candidate_leakage_df.groupby("split", dropna=False, sort=True):
        append_summary(split_subset, split=str(split_value or ""), candidate_correctness="all")
    for correctness_value, subset in candidate_leakage_df.groupby("candidate_correctness", dropna=False, sort=True):
        append_summary(candidate_leakage_df.loc[candidate_leakage_df["candidate_correctness"].eq(correctness_value)], split="all", candidate_correctness=str(int(correctness_value)))
    for (split_value, correctness_value), subset in candidate_leakage_df.groupby(
        ["split", "candidate_correctness"],
        dropna=False,
        sort=True,
    ):
        append_summary(subset, split=str(split_value or ""), candidate_correctness=str(int(correctness_value)))
    return pd.DataFrame(rows)


def build_claim3_summary_payload(
    *,
    run_dir: Path,
    probe_name: str,
    sampling_dir: Path,
    probe_dir: Path,
    truth_summary_df: pd.DataFrame,
    leakage_summary_df: pd.DataFrame,
    prompt_metrics_df: pd.DataFrame,
    candidate_leakage_df: pd.DataFrame,
) -> Dict[str, Any]:
    def first_matching_row(frame: pd.DataFrame, filters: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
        if frame.empty:
            return {}
        subset = frame.copy()
        for column, value in filters:
            subset = subset.loc[subset[column].astype(str).eq(str(value))]
        if subset.empty:
            return {}
        return subset.iloc[0].to_dict()

    overall_truth = first_matching_row(
        truth_summary_df,
        [("split", "all"), ("endorsed_is_correct", "all")],
    )
    test_truth = first_matching_row(
        truth_summary_df,
        [("split", "test"), ("endorsed_is_correct", "all")],
    )
    overall_leakage = first_matching_row(
        leakage_summary_df,
        [("split", "all"), ("candidate_correctness", "all")],
    )
    wrong_leakage = first_matching_row(
        leakage_summary_df,
        [("split", "all"), ("candidate_correctness", "0")],
    )

    return {
        "source_run_dir": str(run_dir),
        "probe_name": str(probe_name or ""),
        "sampling_dir": str(sampling_dir),
        "probe_dir": str(probe_dir),
        "template_type": DEFAULT_ENDORSED_OPTION_GRID_TEMPLATE_TYPE,
        "n_prompt_rows": int(len(prompt_metrics_df)),
        "n_candidate_leakage_rows": int(len(candidate_leakage_df)),
        "overall_truth": overall_truth,
        "test_truth": test_truth,
        "overall_leakage": overall_leakage,
        "wrong_candidate_leakage": wrong_leakage,
    }


def run_claim3_analysis(
    run_dir: Path | str,
    *,
    probe_name: str = "probe_no_bias",
    sampling_subdir: Optional[str] = None,
    probe_subdir: Optional[str] = None,
    output_subdir: Optional[str] = None,
    requested_splits: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    loaded = load_claim3_backfill_frames(
        run_dir,
        probe_name=probe_name,
        sampling_subdir=sampling_subdir,
        probe_subdir=probe_subdir,
    )
    enriched = enrich_claim3_candidate_scores(
        loaded["candidate_df"],
        loaded["sampling_df"],
        requested_splits=requested_splits,
    )
    prompt_metrics_df = build_claim3_prompt_metrics(enriched)
    truth_summary_df = summarize_claim3_truth_separation(prompt_metrics_df)
    candidate_leakage_df = build_claim3_candidate_leakage(enriched)
    leakage_summary_df = summarize_claim3_endorsement_leakage(candidate_leakage_df)

    resolved_run_dir = Path(loaded["run_dir"]).resolve()
    output_dir = resolved_run_dir / str(output_subdir or default_claim3_output_subdir(probe_name)).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_metrics_path = output_dir / "prompt_metrics.csv"
    truth_summary_path = output_dir / "truth_separation_summary.csv"
    candidate_leakage_path = output_dir / "candidate_leakage.csv"
    leakage_summary_path = output_dir / "endorsement_leakage_summary.csv"
    summary_json_path = output_dir / "claim3_summary.json"

    prompt_metrics_df.to_csv(prompt_metrics_path, index=False)
    truth_summary_df.to_csv(truth_summary_path, index=False)
    candidate_leakage_df.to_csv(candidate_leakage_path, index=False)
    leakage_summary_df.to_csv(leakage_summary_path, index=False)

    summary_payload = build_claim3_summary_payload(
        run_dir=resolved_run_dir,
        probe_name=probe_name,
        sampling_dir=loaded["sampling_dir"],
        probe_dir=loaded["probe_dir"],
        truth_summary_df=truth_summary_df,
        leakage_summary_df=leakage_summary_df,
        prompt_metrics_df=prompt_metrics_df,
        candidate_leakage_df=candidate_leakage_df,
    )
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "prompt_metrics_path": prompt_metrics_path,
        "truth_summary_path": truth_summary_path,
        "candidate_leakage_path": candidate_leakage_path,
        "leakage_summary_path": leakage_summary_path,
        "summary_json_path": summary_json_path,
        "summary_payload": summary_payload,
        "prompt_metrics_df": prompt_metrics_df,
        "truth_summary_df": truth_summary_df,
        "candidate_leakage_df": candidate_leakage_df,
        "leakage_summary_df": leakage_summary_df,
    }


__all__ = [
    "DEFAULT_ENDORSED_OPTION_GRID_TEMPLATE_TYPE",
    "build_claim3_candidate_leakage",
    "build_claim3_prompt_metrics",
    "default_claim3_output_subdir",
    "default_claim3_probe_subdir",
    "default_claim3_sampling_subdir",
    "enrich_claim3_candidate_scores",
    "load_claim3_backfill_frames",
    "run_claim3_analysis",
    "summarize_claim3_endorsement_leakage",
    "summarize_claim3_truth_separation",
]
