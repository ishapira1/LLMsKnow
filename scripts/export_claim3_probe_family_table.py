from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_choice_level_package_main_runs"
    / "choice_level_probe_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_probe_family_tables"
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
            "Export a compact claim-3 summary table for a single probe family "
            "using the choice-level probe-score package."
        ),
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help=f"Choice-level probe-score CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where the table should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--probe-family",
        default="neutral_trained",
        help="Probe family to summarize, for example neutral_trained or incorrect_suggestion_trained.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to summarize. Default: test",
    )
    parser.add_argument(
        "--output-stem",
        default="",
        help="Optional explicit output stem without extension.",
    )
    return parser


def auc_from_scores(scores: pd.Series, labels: pd.Series) -> float:
    labels_int = labels.astype(int)
    n_pos = int(labels_int.sum())
    n_neg = int((1 - labels_int).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = scores.rank(method="average")
    sum_pos_ranks = float(ranks[labels_int.eq(1)].sum())
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def iter_prompt_pairwise_scores(df: pd.DataFrame) -> Iterable[float]:
    grouped = df.groupby(["run_id", "question_uid"], sort=False)
    for _, prompt_df in grouped:
        correct_scores = prompt_df.loc[prompt_df["is_correct"].astype(bool), "probe_score"]
        wrong_scores = prompt_df.loc[~prompt_df["is_correct"].astype(bool), "probe_score"]
        if correct_scores.empty or wrong_scores.empty:
            continue
        correct_score = float(correct_scores.iloc[0])
        yield float((correct_score > wrong_scores.astype(float)).mean())


def summarize_framing(df: pd.DataFrame) -> pd.Series:
    sorted_df = df.sort_values(
        ["run_id", "question_uid", "probe_score", "choice_id"],
        ascending=[True, True, False, True],
    )
    top1_df = sorted_df.groupby(["run_id", "question_uid"], sort=False).head(1)
    pairwise_scores = list(iter_prompt_pairwise_scores(sorted_df))

    return pd.Series(
        {
            "Top-1": float(top1_df["is_correct"].astype(bool).mean()),
            "Pairwise K": float(sum(pairwise_scores) / len(pairwise_scores)) if pairwise_scores else float("nan"),
            "AUC": auc_from_scores(sorted_df["probe_score"].astype(float), sorted_df["is_correct"].astype(bool)),
            "n_prompts": int(top1_df.shape[0]),
            "n_candidate_rows": int(sorted_df.shape[0]),
        }
    )


def framing_sort_key(name: str) -> tuple[int, str]:
    if name in FRAMING_ORDER:
        return (FRAMING_ORDER.index(name), name)
    return (len(FRAMING_ORDER), name)


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "split",
        "framing_family",
        "probe_family",
        "probe_score",
        "is_correct",
        "question_uid",
        "run_id",
        "choice_id",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=500_000):
        filtered = chunk.loc[
            chunk["split"].astype(str).eq(str(args.split))
            & chunk["probe_family"].astype(str).eq(str(args.probe_family))
        ].copy()
        if not filtered.empty:
            frames.append(filtered)

    if not frames:
        raise ValueError(
            f"No rows found for probe_family={args.probe_family!r} and split={args.split!r} in {input_path}."
        )

    all_df = pd.concat(frames, ignore_index=True)
    summary_df = (
        all_df.groupby("framing_family", sort=False)
        .apply(summarize_framing, include_groups=False)
        .reset_index()
        .rename(columns={"framing_family": "Framing family"})
    )
    summary_df = summary_df.sort_values(
        by="Framing family",
        key=lambda series: series.map(lambda name: framing_sort_key(str(name))),
    ).reset_index(drop=True)

    table_df = summary_df.loc[:, ["Framing family", "Top-1", "Pairwise K", "AUC"]].copy()

    output_stem = (
        args.output_stem.strip()
        if args.output_stem.strip()
        else f"claim3_probe_family_table_{args.probe_family}_{args.split}"
    )
    table_path = output_dir / f"{output_stem}.csv"
    metadata_path = output_dir / f"{output_stem}.json"

    table_df.to_csv(table_path, index=False)
    metadata = {
        "created_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "output_table_path": str(table_path),
        "probe_family": str(args.probe_family),
        "split": str(args.split),
        "available_framings": summary_df["Framing family"].tolist(),
        "row_counts": summary_df.loc[:, ["Framing family", "n_prompts", "n_candidate_rows"]].to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote table to {table_path}")
    print(table_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
