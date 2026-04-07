from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_choice_level_package_main_runs"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_selected_layer_wide_bundle_main_runs"
)
FRAMING_ORDER = [
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
        description=(
            "Transform the claim-3 long-format choice-level package into a smaller selected-layer "
            "wide bundle with compressed files and coverage summaries."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing the claim-3 long-format package. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where the selected-layer wide bundle should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--write-run-splits",
        action="store_true",
        help="Also write per-run compressed model/probe wide files in addition to the combined files.",
    )
    return parser


def ordered_columns(df: pd.DataFrame, preferred_prefix: str) -> List[str]:
    ordered: List[str] = []
    for framing in FRAMING_ORDER:
        candidate = f"{preferred_prefix}{framing}"
        if candidate in df.columns:
            ordered.append(candidate)
    remaining = sorted(column for column in df.columns if column.startswith(preferred_prefix) and column not in ordered)
    ordered.extend(remaining)
    return ordered


def ensure_framing_value_columns(
    df: pd.DataFrame,
    *,
    prefix: str,
    default_value: Any,
) -> pd.DataFrame:
    for framing in FRAMING_ORDER:
        column = f"{prefix}{framing}"
        if column not in df.columns:
            df[column] = default_value
    return df


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, compression={"method": "gzip", "compresslevel": 1})


def build_model_wide_df(model_long_df: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
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
    ]

    prob_wide = (
        model_long_df.pivot_table(
            index=index_cols,
            columns="framing_family",
            values="model_prob",
            aggfunc="first",
        )
        .reset_index()
    )
    prob_wide.columns = [
        f"prob_{column}" if column not in index_cols else column
        for column in prob_wide.columns
    ]

    endorsed_wide = (
        model_long_df.pivot_table(
            index=index_cols,
            columns="framing_family",
            values="endorsed_choice",
            aggfunc="first",
        )
        .reset_index()
    )
    endorsed_wide.columns = [
        f"endorsed_choice_{column}" if column not in index_cols else column
        for column in endorsed_wide.columns
    ]

    merged = prob_wide.merge(endorsed_wide, on=index_cols, how="left")
    merged = ensure_framing_value_columns(merged, prefix="prob_", default_value=float("nan"))
    merged = ensure_framing_value_columns(merged, prefix="endorsed_choice_", default_value=pd.NA)
    ordered = index_cols + ordered_columns(merged, "prob_") + ordered_columns(merged, "endorsed_choice_")
    return merged.loc[:, ordered].sort_values(
        ["model_name", "dataset", "run_id", "split", "question_id", "draw_idx", "choice_id"]
    ).reset_index(drop=True)


def build_probe_wide_df(probe_long_df: pd.DataFrame) -> pd.DataFrame:
    selected = probe_long_df.loc[probe_long_df["is_selected_layer"].astype(bool)].copy()
    index_cols = [
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
        "probe_family",
        "layer",
    ]

    wide = (
        selected.pivot_table(
            index=index_cols,
            columns="framing_family",
            values="probe_score",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"layer": "selected_layer"})
    )
    wide.columns = [
        f"score_{column}" if column not in index_cols and column != "selected_layer" else column
        for column in wide.columns
    ]
    wide = ensure_framing_value_columns(wide, prefix="score_", default_value=float("nan"))
    ordered = [
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
        "probe_family",
        "selected_layer",
    ] + ordered_columns(wide, "score_")
    return wide.loc[:, ordered].sort_values(
        ["model_name", "dataset", "run_id", "split", "question_id", "draw_idx", "probe_family", "choice_id"]
    ).reset_index(drop=True)


def build_coverage_df(model_long_df: pd.DataFrame, probe_long_df: pd.DataFrame) -> pd.DataFrame:
    model_totals = (
        model_long_df.groupby(["run_id", "framing_family"], as_index=False)
        .agg(
            n_possible_questions=("question_uid", "nunique"),
            n_possible_choice_rows=("choice_id", "size"),
        )
    )
    probe_counts = (
        probe_long_df.loc[probe_long_df["is_selected_layer"].astype(bool)]
        .groupby(["run_id", "probe_family", "framing_family"], as_index=False)
        .agg(
            n_questions=("question_uid", "nunique"),
            n_choice_rows=("choice_id", "size"),
        )
    )
    coverage = probe_counts.merge(model_totals, on=["run_id", "framing_family"], how="left")
    coverage["coverage_fraction"] = coverage["n_choice_rows"] / coverage["n_possible_choice_rows"]
    return coverage.sort_values(["run_id", "probe_family", "framing_family"]).reset_index(drop=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copy2(src, dst)


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_long_path = input_dir / "choice_level_model_scores.csv"
    probe_long_path = input_dir / "choice_level_probe_scores.csv"
    probe_metadata_path = input_dir / "probe_metadata.csv"
    framing_metadata_path = input_dir / "framing_metadata.csv"
    input_manifest_path = input_dir / "package_manifest.json"

    model_usecols = [
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
        "framing_family",
        "model_prob",
        "endorsed_choice",
    ]
    probe_usecols = [
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
        "probe_family",
        "layer",
        "is_selected_layer",
        "framing_family",
        "probe_score",
    ]

    model_long_df = pd.read_csv(model_long_path, usecols=model_usecols)
    probe_long_df = pd.read_csv(probe_long_path, usecols=probe_usecols)

    model_wide_df = build_model_wide_df(model_long_df)
    probe_wide_df = build_probe_wide_df(probe_long_df)
    coverage_df = build_coverage_df(model_long_df, probe_long_df)

    model_wide_path = output_dir / "selected_layer_model_scores_wide.csv.gz"
    probe_wide_path = output_dir / "selected_layer_probe_scores_wide.csv.gz"
    coverage_path = output_dir / "coverage_matrix.csv"
    output_manifest_path = output_dir / "package_manifest.json"

    write_csv_gz(model_wide_df, model_wide_path)
    write_csv_gz(probe_wide_df, probe_wide_path)
    coverage_df.to_csv(coverage_path, index=False)

    copy_file(probe_metadata_path, output_dir / "probe_metadata.csv")
    copy_file(framing_metadata_path, output_dir / "framing_metadata.csv")

    run_split_files: Dict[str, Dict[str, str]] = {}
    if args.write_run_splits:
        for run_id, run_model_df in model_wide_df.groupby("run_id", sort=True):
            safe_run_id = str(run_id)
            run_model_path = output_dir / f"selected_layer_model_scores_wide_{safe_run_id}.csv.gz"
            write_csv_gz(run_model_df, run_model_path)
            run_split_files.setdefault(safe_run_id, {})["model"] = str(run_model_path)

        for run_id, run_probe_df in probe_wide_df.groupby("run_id", sort=True):
            safe_run_id = str(run_id)
            run_probe_path = output_dir / f"selected_layer_probe_scores_wide_{safe_run_id}.csv.gz"
            write_csv_gz(run_probe_df, run_probe_path)
            run_split_files.setdefault(safe_run_id, {})["probe"] = str(run_probe_path)

    input_manifest: Dict[str, Any] = {}
    if input_manifest_path.exists():
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))

    manifest = {
        "created_at_utc": utc_now_iso(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_manifest": str(input_manifest_path) if input_manifest_path.exists() else "",
        "source_row_counts": input_manifest.get("row_counts", {}),
        "files": {
            "selected_layer_model_scores_wide": str(model_wide_path),
            "selected_layer_probe_scores_wide": str(probe_wide_path),
            "probe_metadata": str(output_dir / "probe_metadata.csv"),
            "framing_metadata": str(output_dir / "framing_metadata.csv"),
            "coverage_matrix": str(coverage_path),
        },
        "row_counts": {
            "selected_layer_model_scores_wide": int(len(model_wide_df)),
            "selected_layer_probe_scores_wide": int(len(probe_wide_df)),
            "coverage_matrix": int(len(coverage_df)),
        },
        "framing_columns": {
            "model_prob_columns": ordered_columns(model_wide_df, "prob_"),
            "model_endorsed_choice_columns": ordered_columns(model_wide_df, "endorsed_choice_"),
            "probe_score_columns": ordered_columns(probe_wide_df, "score_"),
        },
        "run_split_files": run_split_files,
    }
    output_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote selected-layer wide bundle to {output_dir}")
    print(f"Model wide rows: {len(model_wide_df)}")
    print(f"Probe wide rows: {len(probe_wide_df)}")
    print(f"Coverage rows: {len(coverage_df)}")
    print(f"Per-run split files written: {bool(args.write_run_splits)}")


if __name__ == "__main__":
    main()
