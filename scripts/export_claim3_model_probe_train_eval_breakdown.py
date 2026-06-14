from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_choice_level_package_main_runs"
)
DEFAULT_PROBE_INPUT_PATH = DEFAULT_PACKAGE_DIR / "choice_level_probe_scores.csv"
DEFAULT_MODEL_INPUT_PATH = DEFAULT_PACKAGE_DIR / "choice_level_model_scores.csv"
DEFAULT_PROBE_METADATA_PATH = DEFAULT_PACKAGE_DIR / "probe_metadata.csv"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_model_probe_train_eval_breakdown_main_runs"
)

FRAMING_ORDER = [
    "neutral",
    "incorrect_suggestion",
    "suggest_correct",
    "doubt_correct",
    "model_congruent_suggestion",
]
TRAINED_ON_ORDER = [
    "model",
    "neutral",
    "incorrect_suggestion",
    "suggest_correct",
    "doubt_correct",
]
PROBE_FAMILY_ORDER = [
    "neutral_trained",
    "incorrect_suggestion_trained",
    "suggest_correct_trained",
    "doubt_correct_trained",
]
PROBE_LABELS = {
    "neutral_trained": "neutral",
    "incorrect_suggestion_trained": "incorrect_suggestion",
    "suggest_correct_trained": "suggest_correct",
    "doubt_correct_trained": "doubt_correct",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a per-model/per-dataset claim-3 table with probe trained-on vs eval-on "
            "metrics, plus the model's own numbers."
        )
    )
    parser.add_argument(
        "--probe-input-path",
        default=str(DEFAULT_PROBE_INPUT_PATH),
        help=f"Choice-level probe scores CSV. Default: {DEFAULT_PROBE_INPUT_PATH}",
    )
    parser.add_argument(
        "--model-input-path",
        default=str(DEFAULT_MODEL_INPUT_PATH),
        help=f"Choice-level model scores CSV. Default: {DEFAULT_MODEL_INPUT_PATH}",
    )
    parser.add_argument(
        "--probe-metadata-path",
        default=str(DEFAULT_PROBE_METADATA_PATH),
        help=f"Probe metadata CSV. Default: {DEFAULT_PROBE_METADATA_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where outputs should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to summarize. Default: test",
    )
    parser.add_argument(
        "--output-stem",
        default="claim3_model_probe_train_eval_breakdown_main_runs",
        help="Output stem without extension.",
    )
    return parser


def bool_from_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes"})


def auc_from_scores(scores: pd.Series, labels: pd.Series) -> float:
    labels_int = labels.astype(int)
    n_pos = int(labels_int.sum())
    n_neg = int((1 - labels_int).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = scores.rank(method="average")
    sum_pos_ranks = float(ranks[labels_int.eq(1)].sum())
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def iter_prompt_pairwise_scores(df: pd.DataFrame, *, score_col: str) -> Iterable[float]:
    grouped = df.groupby("question_uid", sort=False)
    for _, prompt_df in grouped:
        correct_scores = prompt_df.loc[prompt_df["is_correct"].astype(bool), score_col]
        wrong_scores = prompt_df.loc[~prompt_df["is_correct"].astype(bool), score_col]
        if correct_scores.empty or wrong_scores.empty:
            continue
        correct_score = float(correct_scores.iloc[0])
        yield float((correct_score > wrong_scores.astype(float)).mean())


def summarize_group(df: pd.DataFrame, *, score_col: str, source_col: str) -> pd.Series:
    sorted_df = df.sort_values(
        ["question_uid", score_col, "choice_id"],
        ascending=[True, False, True],
    )
    top1_df = sorted_df.groupby("question_uid", sort=False).head(1)
    pairwise_scores = list(iter_prompt_pairwise_scores(sorted_df, score_col=score_col))
    source_kinds = sorted(df[source_col].dropna().astype(str).unique().tolist()) if source_col in df.columns else []

    return pd.Series(
        {
            "top1": float(top1_df["is_correct"].astype(bool).mean()),
            "pairwise_k": float(sum(pairwise_scores) / len(pairwise_scores)) if pairwise_scores else float("nan"),
            "auc": auc_from_scores(sorted_df[score_col].astype(float), sorted_df["is_correct"].astype(bool)),
            "n_prompts": int(top1_df.shape[0]),
            "n_candidate_rows": int(sorted_df.shape[0]),
            "source_kinds": json.dumps(source_kinds),
        }
    )


def framing_sort_key(name: str) -> tuple[int, str]:
    if name in FRAMING_ORDER:
        return (FRAMING_ORDER.index(name), name)
    return (len(FRAMING_ORDER), name)


def trained_on_sort_key(name: str) -> tuple[int, str]:
    if name in TRAINED_ON_ORDER:
        return (TRAINED_ON_ORDER.index(name), name)
    return (len(TRAINED_ON_ORDER), name)


def probe_family_sort_key(name: str) -> tuple[int, str]:
    if name in PROBE_FAMILY_ORDER:
        return (PROBE_FAMILY_ORDER.index(name), name)
    return (len(PROBE_FAMILY_ORDER), name)


def parse_training_families(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(loaded, list):
        return [str(item).strip() for item in loaded if str(item).strip()]
    if loaded is None:
        return []
    return [str(loaded).strip()]


def normalize_probe_metadata(probe_metadata_df: pd.DataFrame) -> pd.DataFrame:
    metadata = probe_metadata_df.copy()
    metadata["trained_on"] = metadata["training_families"].map(parse_training_families).map(
        lambda items: "|".join(items) if items else ""
    )
    metadata["trained_on"] = metadata["trained_on"].mask(
        metadata["trained_on"].eq(""),
        metadata["probe_family"].map(PROBE_LABELS).fillna(metadata["probe_family"]),
    )
    return (
        metadata.loc[:, ["run_id", "model_name", "dataset", "probe_name", "probe_family", "trained_on"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def load_probe_scores(path: Path, *, split: str) -> pd.DataFrame:
    usecols = [
        "run_id",
        "model_name",
        "dataset",
        "split",
        "question_uid",
        "choice_id",
        "is_correct",
        "probe_name",
        "probe_family",
        "probe_training_families",
        "framing_family",
        "probe_score",
        "prompt_score_source_kind",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df = df.loc[df["split"].astype(str).eq(split)].copy()
    df["is_correct"] = bool_from_series(df["is_correct"])
    df["probe_score"] = pd.to_numeric(df["probe_score"], errors="coerce")
    df = df.loc[df["probe_score"].notna()].copy()
    df["trained_on"] = df["probe_training_families"].map(parse_training_families).map(
        lambda items: "|".join(items) if items else ""
    )
    df["trained_on"] = df["trained_on"].mask(
        df["trained_on"].eq(""),
        df["probe_family"].map(PROBE_LABELS).fillna(df["probe_family"]),
    )
    return df


def load_model_scores(path: Path, *, split: str) -> pd.DataFrame:
    usecols = [
        "run_id",
        "run_name",
        "run_dir",
        "model_name",
        "dataset",
        "split",
        "question_uid",
        "choice_id",
        "is_correct",
        "framing_family",
        "model_prob",
        "record_source_kind",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df = df.loc[df["split"].astype(str).eq(split)].copy()
    df["is_correct"] = bool_from_series(df["is_correct"])
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df = df.loc[df["model_prob"].notna()].copy()
    return df


def summarize_probe_scores(probe_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        probe_df.groupby(
            ["run_id", "model_name", "dataset", "split", "probe_name", "probe_family", "trained_on", "framing_family"],
            sort=False,
        )
        .apply(summarize_group, score_col="probe_score", source_col="prompt_score_source_kind", include_groups=False)
        .reset_index()
        .rename(columns={"framing_family": "eval_on"})
    )
    return grouped


def summarize_model_scores(model_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        model_df.groupby(["run_id", "run_name", "run_dir", "model_name", "dataset", "split", "framing_family"], sort=False)
        .apply(summarize_group, score_col="model_prob", source_col="record_source_kind", include_groups=False)
        .reset_index()
        .rename(columns={"framing_family": "eval_on"})
    )
    return grouped


def build_probe_grid(
    probe_metadata_df: pd.DataFrame,
    model_df: pd.DataFrame,
    probe_summary_df: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    model_framings = (
        model_df.loc[:, ["run_id", "framing_family"]]
        .drop_duplicates()
        .rename(columns={"framing_family": "eval_on"})
        .reset_index(drop=True)
    )
    probe_inventory = (
        probe_metadata_df.loc[:, ["run_id", "model_name", "dataset", "probe_name", "probe_family", "trained_on"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    probe_inventory["merge_key"] = 1
    model_framings["merge_key"] = 1
    probe_grid = probe_inventory.merge(model_framings, on=["run_id", "merge_key"], how="left").drop(columns=["merge_key"])
    probe_grid["split"] = split

    merged = probe_grid.merge(
        probe_summary_df,
        on=["run_id", "model_name", "dataset", "split", "probe_name", "probe_family", "trained_on", "eval_on"],
        how="left",
    )
    merged["row_kind"] = "probe"
    merged["available"] = merged["n_candidate_rows"].fillna(0).astype(int).gt(0)
    merged["run_name"] = pd.NA
    merged["run_dir"] = pd.NA
    return merged


def build_model_rows(model_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = model_summary_df.copy()
    rows["row_kind"] = "model"
    rows["probe_name"] = pd.NA
    rows["probe_family"] = pd.NA
    rows["trained_on"] = "model"
    rows["available"] = rows["n_candidate_rows"].fillna(0).astype(int).gt(0)
    return rows


def build_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
        "model_name",
        "dataset",
        "run_id",
        "run_name",
        "run_dir",
        "split",
        "row_kind",
        "trained_on",
        "probe_family",
        "probe_name",
    ]
    value_cols = ["available", "top1", "pairwise_k", "auc", "n_prompts", "n_candidate_rows"]
    wide = (
        long_df.pivot_table(
            index=index_cols,
            columns="eval_on",
            values=value_cols,
            aggfunc="first",
        )
        .reset_index()
    )
    flattened_columns: list[str] = []
    for column in wide.columns:
        if isinstance(column, tuple):
            left, right = column
            if not right:
                flattened_columns.append(str(left))
            else:
                flattened_columns.append(f"{right}__{left}")
        else:
            flattened_columns.append(str(column))
    wide.columns = flattened_columns
    return wide


def sort_long_df(long_df: pd.DataFrame) -> pd.DataFrame:
    row_kind_order = {"model": 0, "probe": 1}
    sorted_df = long_df.copy()
    sorted_df["_row_kind_order"] = sorted_df["row_kind"].map(lambda value: row_kind_order.get(str(value), 99))
    sorted_df["_trained_on_order"] = sorted_df["trained_on"].map(lambda value: trained_on_sort_key(str(value)))
    sorted_df["_probe_family_order"] = sorted_df["probe_family"].map(
        lambda value: probe_family_sort_key("" if pd.isna(value) else str(value))
    )
    sorted_df["_eval_on_order"] = sorted_df["eval_on"].map(lambda value: framing_sort_key(str(value)))
    sorted_df = sorted_df.sort_values(
        by=[
            "model_name",
            "dataset",
            "_row_kind_order",
            "_trained_on_order",
            "_probe_family_order",
            "_eval_on_order",
            "run_id",
        ]
    ).reset_index(drop=True)
    return sorted_df.drop(
        columns=["_row_kind_order", "_trained_on_order", "_probe_family_order", "_eval_on_order"]
    )


def main() -> None:
    args = build_parser().parse_args()
    probe_input_path = Path(args.probe_input_path).expanduser().resolve()
    model_input_path = Path(args.model_input_path).expanduser().resolve()
    probe_metadata_path = Path(args.probe_metadata_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split = str(args.split).strip()
    output_stem = str(args.output_stem).strip()

    probe_df = load_probe_scores(probe_input_path, split=split)
    model_df = load_model_scores(model_input_path, split=split)
    probe_metadata_df = normalize_probe_metadata(pd.read_csv(probe_metadata_path))

    probe_summary_df = summarize_probe_scores(probe_df)
    model_summary_df = summarize_model_scores(model_df)

    probe_rows_df = build_probe_grid(
        probe_metadata_df=probe_metadata_df,
        model_df=model_df,
        probe_summary_df=probe_summary_df,
        split=split,
    )
    model_rows_df = build_model_rows(model_summary_df)

    long_df = pd.concat([model_rows_df, probe_rows_df], ignore_index=True, sort=False)
    long_df = sort_long_df(long_df)
    wide_df = build_wide_table(long_df)

    missing_probe_df = (
        long_df.loc[long_df["row_kind"].eq("probe") & ~long_df["available"].astype(bool)]
        .loc[:, ["model_name", "dataset", "run_id", "split", "trained_on", "probe_family", "probe_name", "eval_on"]]
        .reset_index(drop=True)
    )

    long_path = output_dir / f"{output_stem}_long.csv"
    wide_path = output_dir / f"{output_stem}_wide.csv"
    missing_path = output_dir / f"{output_stem}_missing_probe_combos.csv"
    metadata_path = output_dir / f"{output_stem}.json"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    missing_probe_df.to_csv(missing_path, index=False)

    metadata = {
        "created_at_utc": utc_now_iso(),
        "split": split,
        "probe_input_path": str(probe_input_path),
        "model_input_path": str(model_input_path),
        "probe_metadata_path": str(probe_metadata_path),
        "output_long_path": str(long_path),
        "output_wide_path": str(wide_path),
        "output_missing_probe_path": str(missing_path),
        "counts": {
            "model_rows": int(model_rows_df.shape[0]),
            "probe_rows": int(probe_rows_df.shape[0]),
            "missing_probe_rows": int(missing_probe_df.shape[0]),
            "runs": int(long_df["run_id"].dropna().nunique()),
            "datasets": sorted(long_df["dataset"].dropna().astype(str).unique().tolist()),
            "models": sorted(long_df["model_name"].dropna().astype(str).unique().tolist()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote long table to {long_path}")
    print(f"Wrote wide table to {wide_path}")
    print(f"Wrote missing-combo table to {missing_path}")
    print(
        long_df.loc[:, ["model_name", "dataset", "row_kind", "trained_on", "eval_on", "top1", "pairwise_k", "auc", "available"]]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
