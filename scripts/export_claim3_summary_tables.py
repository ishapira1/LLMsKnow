from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_selected_layer_wide_bundle_main_runs"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_summary_tables_main_runs"
)
FRAMINGS = [
    "neutral",
    "incorrect_suggestion",
    "suggest_correct",
    "doubt_correct",
    "model_congruent_suggestion",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export compact claim-3 summary tables from the selected-layer wide bundle.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing the selected-layer wide bundle. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where summary tables should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def metric_from_question_group(
    question_df: pd.DataFrame,
    *,
    value_col: str,
    endorsed_choice: Optional[str],
) -> Dict[str, Any]:
    ordered = question_df.sort_values([value_col, "choice_id"], ascending=[False, True]).copy()
    top1_acc = bool(ordered["is_correct"].iloc[0])

    correct_rows = question_df.loc[question_df["is_correct"].astype(bool)].copy()
    if correct_rows.empty:
        return {}
    correct_value = float(correct_rows[value_col].iloc[0])

    wrong_rows = question_df.loc[~question_df["is_correct"].astype(bool)].copy()
    if wrong_rows.empty:
        return {}

    pairwise = float((correct_value > wrong_rows[value_col]).mean())
    bestwrong_value = float(wrong_rows[value_col].max())
    gold_vs_bestwrong_margin = correct_value - bestwrong_value

    gold_vs_endorsed_margin = None
    if endorsed_choice:
        endorsed_rows = question_df.loc[question_df["choice_id"].astype(str).eq(str(endorsed_choice))].copy()
        if not endorsed_rows.empty:
            gold_vs_endorsed_margin = correct_value - float(endorsed_rows[value_col].iloc[0])

    truth_separation = correct_value - float(wrong_rows[value_col].mean())

    return {
        "top1_acc": top1_acc,
        "K_pairwise": pairwise,
        "gold_vs_endorsed_margin": gold_vs_endorsed_margin,
        "gold_vs_bestwrong_margin": gold_vs_bestwrong_margin,
        "truth_separation": truth_separation,
        "correct_value": correct_value,
    }


def summarize_cross_family_probe_metrics(probe_wide_df: pd.DataFrame, question_meta_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    endorsed_lookup = (
        question_meta_df.set_index("question_uid")[
            [f"endorsed_choice_{framing}" for framing in FRAMINGS if f"endorsed_choice_{framing}" in question_meta_df.columns]
        ]
        .to_dict(orient="index")
    )

    group_cols = ["run_id", "model_name", "dataset", "split", "probe_family"]
    for framing in FRAMINGS:
        score_col = f"score_{framing}"
        if score_col not in probe_wide_df.columns:
            continue
        subset = probe_wide_df.loc[probe_wide_df[score_col].notna()].copy()
        if subset.empty:
            continue
        for keys, group_df in subset.groupby(group_cols, sort=True):
            run_id, model_name, dataset, split, probe_family = keys
            per_question: List[Dict[str, Any]] = []
            for question_uid, question_df in group_df.groupby("question_uid", sort=False):
                endorsed_choice = None
                endorsed_payload = endorsed_lookup.get(question_uid, {})
                endorsed_value = endorsed_payload.get(f"endorsed_choice_{framing}")
                if isinstance(endorsed_value, str) and endorsed_value:
                    endorsed_choice = endorsed_value
                elif pd.notna(endorsed_value):
                    endorsed_choice = str(endorsed_value)
                metrics = metric_from_question_group(
                    question_df,
                    value_col=score_col,
                    endorsed_choice=endorsed_choice,
                )
                if metrics:
                    per_question.append(metrics)
            if not per_question:
                continue
            per_question_df = pd.DataFrame(per_question)
            rows.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "dataset": dataset,
                    "split": split,
                    "probe_family": probe_family,
                    "test_framing": framing,
                    "n_questions": int(len(per_question_df)),
                    "n_choice_rows": int(len(group_df)),
                    "top1_acc": float(per_question_df["top1_acc"].mean()),
                    "K_pairwise": float(per_question_df["K_pairwise"].mean()),
                    "gold_vs_endorsed_margin_mean": float(per_question_df["gold_vs_endorsed_margin"].dropna().mean())
                    if per_question_df["gold_vs_endorsed_margin"].notna().any()
                    else float("nan"),
                    "gold_vs_endorsed_margin_median": float(per_question_df["gold_vs_endorsed_margin"].dropna().median())
                    if per_question_df["gold_vs_endorsed_margin"].notna().any()
                    else float("nan"),
                    "gold_vs_bestwrong_margin_mean": float(per_question_df["gold_vs_bestwrong_margin"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_cross_family_model_metrics(model_wide_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["run_id", "model_name", "dataset", "split"]
    for framing in FRAMINGS:
        prob_col = f"prob_{framing}"
        if prob_col not in model_wide_df.columns:
            continue
        endorsed_col = f"endorsed_choice_{framing}"
        subset = model_wide_df.loc[model_wide_df[prob_col].notna()].copy()
        if subset.empty:
            continue
        for keys, group_df in subset.groupby(group_cols, sort=True):
            run_id, model_name, dataset, split = keys
            per_question: List[Dict[str, Any]] = []
            for _, question_df in group_df.groupby("question_uid", sort=False):
                endorsed_choice = None
                if endorsed_col in question_df.columns:
                    endorsed_value = question_df[endorsed_col].iloc[0]
                    if isinstance(endorsed_value, str) and endorsed_value:
                        endorsed_choice = endorsed_value
                    elif pd.notna(endorsed_value):
                        endorsed_choice = str(endorsed_value)
                metrics = metric_from_question_group(
                    question_df,
                    value_col=prob_col,
                    endorsed_choice=endorsed_choice,
                )
                if metrics:
                    per_question.append(metrics)
            if not per_question:
                continue
            per_question_df = pd.DataFrame(per_question)
            rows.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "dataset": dataset,
                    "split": split,
                    "test_framing": framing,
                    "n_questions": int(len(per_question_df)),
                    "n_choice_rows": int(len(group_df)),
                    "top1_acc": float(per_question_df["top1_acc"].mean()),
                    "K_pairwise": float(per_question_df["K_pairwise"].mean()),
                    "gold_vs_endorsed_margin_mean": float(per_question_df["gold_vs_endorsed_margin"].dropna().mean())
                    if per_question_df["gold_vs_endorsed_margin"].notna().any()
                    else float("nan"),
                    "gold_vs_endorsed_margin_median": float(per_question_df["gold_vs_endorsed_margin"].dropna().median())
                    if per_question_df["gold_vs_endorsed_margin"].notna().any()
                    else float("nan"),
                    "gold_vs_bestwrong_margin_mean": float(per_question_df["gold_vs_bestwrong_margin"].mean()),
                }
            )
    return pd.DataFrame(rows)


def classify_choice_type(row: pd.Series, *, endorsed_choice: Optional[str], correct_choice: str) -> str:
    choice_id = str(row["choice_id"])
    if endorsed_choice and endorsed_choice == correct_choice and choice_id == correct_choice:
        return "endorsed_correct"
    if choice_id == correct_choice:
        return "correct"
    if endorsed_choice and choice_id == endorsed_choice and choice_id != correct_choice:
        return "endorsed_wrong"
    return "other_wrong"


def summarize_same_candidate_shift(
    wide_df: pd.DataFrame,
    *,
    value_prefix: str,
    restrict_probe_family: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    working = wide_df.copy()
    if restrict_probe_family is not None and "probe_family" in working.columns:
        working = working.loc[working["probe_family"].astype(str).eq(restrict_probe_family)].copy()

    neutral_col = f"{value_prefix}neutral"
    group_cols = ["run_id", "model_name", "dataset", "split"]
    if "probe_family" in working.columns:
        group_cols.append("probe_family")

    for framing in [item for item in FRAMINGS if item != "neutral"]:
        target_col = f"{value_prefix}{framing}"
        if target_col not in working.columns or neutral_col not in working.columns:
            continue
        subset = working.loc[working[neutral_col].notna() & working[target_col].notna()].copy()
        if subset.empty:
            continue
        endorsed_col = f"endorsed_choice_{framing}"
        if endorsed_col not in subset.columns:
            subset[endorsed_col] = pd.NA
        subset["target_framing"] = framing
        subset["delta_value"] = subset[target_col] - subset[neutral_col]
        subset["choice_type"] = subset.apply(
            lambda row: classify_choice_type(
                row,
                endorsed_choice=(str(row[endorsed_col]) if pd.notna(row[endorsed_col]) else None),
                correct_choice=str(row["correct_choice"]),
            ),
            axis=1,
        )
        for keys, group_df in subset.groupby(group_cols + ["target_framing", "choice_type"], sort=True):
            row: Dict[str, Any] = dict(zip(group_cols + ["target_framing", "choice_type"], keys))
            row.update(
                {
                    "mean_delta": float(group_df["delta_value"].mean()),
                    "median_delta": float(group_df["delta_value"].median()),
                    "mean_abs_delta": float(group_df["delta_value"].abs().mean()),
                    "n_rows": int(len(group_df)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_truth_vs_endorsement(
    wide_df: pd.DataFrame,
    *,
    value_prefix: str,
    restrict_probe_family: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    working = wide_df.copy()
    if restrict_probe_family is not None and "probe_family" in working.columns:
        working = working.loc[working["probe_family"].astype(str).eq(restrict_probe_family)].copy()

    neutral_col = f"{value_prefix}neutral"
    group_cols = ["run_id", "model_name", "dataset", "split"]
    if "probe_family" in working.columns:
        group_cols.append("probe_family")

    for framing in [item for item in FRAMINGS if item != "neutral"]:
        target_col = f"{value_prefix}{framing}"
        endorsed_col = f"endorsed_choice_{framing}"
        if target_col not in working.columns or neutral_col not in working.columns or endorsed_col not in working.columns:
            continue
        subset = working.loc[working[target_col].notna() & working[neutral_col].notna()].copy()
        subset = subset.loc[subset[endorsed_col].notna()].copy()
        if subset.empty:
            continue
        for keys, group_df in subset.groupby(group_cols, sort=True):
            per_question_rows: List[Dict[str, Any]] = []
            for _, question_df in group_df.groupby("question_uid", sort=False):
                endorsed_choice = str(question_df[endorsed_col].iloc[0])
                correct_choice = str(question_df["correct_choice"].iloc[0])
                correct_df = question_df.loc[question_df["is_correct"].astype(bool)].copy()
                wrong_df = question_df.loc[~question_df["is_correct"].astype(bool)].copy()
                if correct_df.empty or wrong_df.empty:
                    continue
                correct_target = float(correct_df[target_col].iloc[0])
                truth_separation = correct_target - float(wrong_df[target_col].mean())

                endorsed_wrong_uplift = float("nan")
                other_wrong_uplift = float("nan")
                endorsement_leakage_gap = float("nan")

                endorsed_wrong_df = wrong_df.loc[wrong_df["choice_id"].astype(str).eq(endorsed_choice)].copy()
                other_wrong_df = wrong_df.loc[~wrong_df["choice_id"].astype(str).eq(endorsed_choice)].copy()

                if not endorsed_wrong_df.empty:
                    endorsed_wrong_uplift = float(
                        (endorsed_wrong_df[target_col] - endorsed_wrong_df[neutral_col]).mean()
                    )
                if not other_wrong_df.empty:
                    other_wrong_uplift = float(
                        (other_wrong_df[target_col] - other_wrong_df[neutral_col]).mean()
                    )
                if pd.notna(endorsed_wrong_uplift) and pd.notna(other_wrong_uplift):
                    endorsement_leakage_gap = endorsed_wrong_uplift - other_wrong_uplift

                per_question_rows.append(
                    {
                        "truth_separation": truth_separation,
                        "endorsed_wrong_uplift": endorsed_wrong_uplift,
                        "other_wrong_uplift": other_wrong_uplift,
                        "endorsement_leakage_gap": endorsement_leakage_gap,
                    }
                )
            if not per_question_rows:
                continue
            pq = pd.DataFrame(per_question_rows)
            row: Dict[str, Any] = dict(zip(group_cols, keys))
            row["target_framing"] = framing
            row["n_questions"] = int(len(pq))
            row["truth_separation"] = float(pq["truth_separation"].mean())
            row["endorsed_wrong_uplift"] = float(pq["endorsed_wrong_uplift"].dropna().mean()) if pq["endorsed_wrong_uplift"].notna().any() else float("nan")
            row["other_wrong_uplift"] = float(pq["other_wrong_uplift"].dropna().mean()) if pq["other_wrong_uplift"].notna().any() else float("nan")
            row["endorsement_leakage_gap"] = float(pq["endorsement_leakage_gap"].dropna().mean()) if pq["endorsement_leakage_gap"].notna().any() else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_score_stability(
    probe_wide_df: pd.DataFrame,
    *,
    probe_family: str = "neutral_trained",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    subset = probe_wide_df.loc[probe_wide_df["probe_family"].astype(str).eq(probe_family)].copy()
    if subset.empty:
        return pd.DataFrame()

    neutral_col = "score_neutral"
    group_cols = ["run_id", "model_name", "dataset", "split", "probe_family"]
    for framing in [item for item in FRAMINGS if item != "neutral"]:
        target_col = f"score_{framing}"
        if target_col not in subset.columns:
            continue
        paired = subset.loc[subset[neutral_col].notna() & subset[target_col].notna()].copy()
        if paired.empty:
            continue
        for keys, group_df in paired.groupby(group_cols, sort=True):
            rows.append(
                {
                    **dict(zip(group_cols, keys)),
                    "target_framing": framing,
                    "n_rows": int(len(group_df)),
                    "corr_all": group_df[neutral_col].corr(group_df[target_col]),
                    "corr_correct_only": group_df.loc[group_df["is_correct"].astype(bool), neutral_col].corr(
                        group_df.loc[group_df["is_correct"].astype(bool), target_col]
                    ),
                    "corr_wrong_only": group_df.loc[~group_df["is_correct"].astype(bool), neutral_col].corr(
                        group_df.loc[~group_df["is_correct"].astype(bool), target_col]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_wide_df = pd.read_csv(input_dir / "selected_layer_model_scores_wide.csv.gz")
    probe_wide_df = pd.read_csv(input_dir / "selected_layer_probe_scores_wide.csv.gz")
    coverage_df = pd.read_csv(input_dir / "coverage_matrix.csv")

    question_meta_cols = [
        "question_uid",
        *[f"endorsed_choice_{framing}" for framing in FRAMINGS if f"endorsed_choice_{framing}" in model_wide_df.columns],
    ]
    question_meta_df = model_wide_df.loc[:, question_meta_cols].drop_duplicates(subset=["question_uid"]).reset_index(drop=True)

    cross_family_probe_df = summarize_cross_family_probe_metrics(probe_wide_df, question_meta_df)
    cross_family_model_df = summarize_cross_family_model_metrics(model_wide_df)
    same_candidate_shift_probe_df = summarize_same_candidate_shift(
        model_wide_df.merge(
            probe_wide_df.loc[probe_wide_df["probe_family"].astype(str).eq("neutral_trained")].copy(),
            on=[
                "run_id",
                "model_name",
                "dataset",
                "split",
                "question_id",
                "draw_idx",
                "question_uid",
                "choice_id",
                "choice_text",
                "correct_choice",
                "is_correct",
            ],
            how="inner",
            suffixes=("_model", ""),
        ),
        value_prefix="score_",
    )
    if "probe_family" not in same_candidate_shift_probe_df.columns:
        same_candidate_shift_probe_df["probe_family"] = "neutral_trained"

    same_candidate_shift_model_df = summarize_same_candidate_shift(
        model_wide_df,
        value_prefix="prob_",
        restrict_probe_family=None,
    )

    neutral_probe_wide_df = probe_wide_df.loc[probe_wide_df["probe_family"].astype(str).eq("neutral_trained")].copy()
    neutral_probe_with_endorsement = neutral_probe_wide_df.merge(
        model_wide_df.loc[
            :,
            [
                "run_id",
                "model_name",
                "dataset",
                "split",
                "question_id",
                "draw_idx",
                "question_uid",
                "choice_id",
                "choice_text",
                "correct_choice",
                "is_correct",
                *[f"endorsed_choice_{framing}" for framing in FRAMINGS if f"endorsed_choice_{framing}" in model_wide_df.columns],
            ],
        ],
        on=[
            "run_id",
            "model_name",
            "dataset",
            "split",
            "question_id",
            "draw_idx",
            "question_uid",
            "choice_id",
            "choice_text",
            "correct_choice",
            "is_correct",
        ],
        how="left",
    )

    truth_vs_endorsement_probe_df = summarize_truth_vs_endorsement(
        neutral_probe_with_endorsement,
        value_prefix="score_",
    )
    if "probe_family" not in truth_vs_endorsement_probe_df.columns:
        truth_vs_endorsement_probe_df["probe_family"] = "neutral_trained"

    truth_vs_endorsement_model_df = summarize_truth_vs_endorsement(
        model_wide_df,
        value_prefix="prob_",
        restrict_probe_family=None,
    )
    score_stability_df = summarize_score_stability(probe_wide_df, probe_family="neutral_trained")

    probe_2x2_df = cross_family_probe_df.loc[
        cross_family_probe_df["probe_family"].isin(["neutral_trained", "incorrect_suggestion_trained"])
        & cross_family_probe_df["test_framing"].isin(["neutral", "incorrect_suggestion"])
    ].copy()
    probe_2x2_df = probe_2x2_df.merge(
        coverage_df,
        left_on=["run_id", "probe_family", "test_framing"],
        right_on=["run_id", "probe_family", "framing_family"],
        how="left",
    ).drop(columns=["framing_family"])

    files = {
        "cross_family_probe_metrics": output_dir / "cross_family_probe_metrics.csv",
        "cross_family_model_metrics": output_dir / "cross_family_model_metrics.csv",
        "same_candidate_shift_probe": output_dir / "same_candidate_shift_summary_probe.csv",
        "same_candidate_shift_model": output_dir / "same_candidate_shift_summary_model.csv",
        "truth_vs_endorsement_probe": output_dir / "truth_vs_endorsement_summary_probe.csv",
        "truth_vs_endorsement_model": output_dir / "truth_vs_endorsement_summary_model.csv",
        "score_stability_probe": output_dir / "score_stability_summary_probe.csv",
        "probe_2x2_incorrect_suggestion": output_dir / "probe_2x2_incorrect_suggestion.csv",
        "coverage_matrix": output_dir / "coverage_matrix.csv",
    }

    cross_family_probe_df.to_csv(files["cross_family_probe_metrics"], index=False)
    cross_family_model_df.to_csv(files["cross_family_model_metrics"], index=False)
    same_candidate_shift_probe_df.to_csv(files["same_candidate_shift_probe"], index=False)
    same_candidate_shift_model_df.to_csv(files["same_candidate_shift_model"], index=False)
    truth_vs_endorsement_probe_df.to_csv(files["truth_vs_endorsement_probe"], index=False)
    truth_vs_endorsement_model_df.to_csv(files["truth_vs_endorsement_model"], index=False)
    score_stability_df.to_csv(files["score_stability_probe"], index=False)
    probe_2x2_df.to_csv(files["probe_2x2_incorrect_suggestion"], index=False)
    coverage_df.to_csv(files["coverage_matrix"], index=False)

    manifest = {
        "created_at_utc": utc_now_iso(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": {name: str(path) for name, path in files.items()},
        "row_counts": {
            "cross_family_probe_metrics": int(len(cross_family_probe_df)),
            "cross_family_model_metrics": int(len(cross_family_model_df)),
            "same_candidate_shift_probe": int(len(same_candidate_shift_probe_df)),
            "same_candidate_shift_model": int(len(same_candidate_shift_model_df)),
            "truth_vs_endorsement_probe": int(len(truth_vs_endorsement_probe_df)),
            "truth_vs_endorsement_model": int(len(truth_vs_endorsement_model_df)),
            "score_stability_probe": int(len(score_stability_df)),
            "probe_2x2_incorrect_suggestion": int(len(probe_2x2_df)),
            "coverage_matrix": int(len(coverage_df)),
        },
    }
    (output_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote claim-3 summary tables to {output_dir}")
    for name, path in files.items():
        print(f"{name}: {path.name}")


if __name__ == "__main__":
    main()
