from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGED_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_choice_level_package_main_runs"
    / "choice_level_probe_scores.csv"
)
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "sycophancy_bias_probe"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_probe_family_tables"
)
PROBE_FAMILY_ORDER = ["neutral_trained", "incorrect_suggestion_trained"]
PROBE_LABELS = {
    "neutral_trained": "neutral",
    "incorrect_suggestion_trained": "incorrect_suggestion",
}
FRAMING_ORDER = [
    "neutral",
    "incorrect_suggestion",
    "suggest_correct",
    "doubt_correct",
    "model_congruent_suggestion",
]
RAW_BACKFILL_SPECS = [
    {
        "probe_family": "incorrect_suggestion_trained",
        "glob": "**/probe_bias_incorrect_suggestion_all_templates/probe_candidate_scores.csv",
    }
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a pooled claim-3 table with rows for "
            "(probe trained on, eval on) across the main runs."
        ),
    )
    parser.add_argument(
        "--packaged-input-path",
        default=str(DEFAULT_PACKAGED_INPUT_PATH),
        help=f"Packaged choice-level CSV. Default: {DEFAULT_PACKAGED_INPUT_PATH}",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help=f"Raw results root for backfill discovery. Default: {DEFAULT_RESULTS_ROOT}",
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
        default="claim3_probe_train_eval_table_main_runs",
        help="Output stem without extension.",
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


def summarize_group(df: pd.DataFrame) -> pd.Series:
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
            "n_runs": int(top1_df["run_id"].nunique()),
            "sources": sorted(df["source_kind"].astype(str).unique().tolist()),
        }
    )


def bool_from_str(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes"})


def framing_sort_key(name: str) -> tuple[int, str]:
    if name in FRAMING_ORDER:
        return (FRAMING_ORDER.index(name), name)
    return (len(FRAMING_ORDER), name)


def probe_sort_key(name: str) -> tuple[int, str]:
    if name in PROBE_FAMILY_ORDER:
        return (PROBE_FAMILY_ORDER.index(name), name)
    return (len(PROBE_FAMILY_ORDER), name)


def find_run_id(path: Path) -> str:
    run_id = next((parent.name for parent in path.parents if parent.name.startswith("full_")), "")
    if not run_id:
        raise ValueError(f"Could not infer run_id from {path}")
    return run_id


def load_allowed_run_ids(packaged_input_path: Path) -> set[str]:
    run_ids: set[str] = set()
    for chunk in pd.read_csv(packaged_input_path, usecols=["run_id"], chunksize=500_000):
        run_ids.update(chunk["run_id"].dropna().astype(str).unique().tolist())
    return run_ids


def load_raw_backfill_candidate_scores(
    results_root: Path,
    split: str,
    allowed_run_ids: set[str],
) -> tuple[pd.DataFrame, set[tuple[str, str, str, str]]]:
    frames: list[pd.DataFrame] = []
    override_combos: set[tuple[str, str, str, str]] = set()

    for spec in RAW_BACKFILL_SPECS:
        probe_family = str(spec["probe_family"])
        for path in sorted(results_root.glob(str(spec["glob"]))):
            run_id = find_run_id(path)
            if allowed_run_ids and run_id not in allowed_run_ids:
                continue
            usecols = [
                "split",
                "prompt_id",
                "template_type",
                "candidate_choice",
                "candidate_correctness",
                "probe_score",
            ]
            df = pd.read_csv(path, usecols=usecols)
            df = df.loc[df["split"].astype(str).eq(split)].copy()
            if df.empty:
                continue

            df["run_id"] = run_id
            df["probe_family"] = probe_family
            df["framing_family"] = df["template_type"].astype(str)
            df["question_uid"] = (
                df["run_id"].astype(str)
                + "::"
                + df["split"].astype(str)
                + "::"
                + df["prompt_id"].astype(str)
            )
            df["choice_id"] = df["candidate_choice"].astype(str)
            df["is_correct"] = bool_from_str(df["candidate_correctness"])
            df["probe_score"] = df["probe_score"].astype(float)
            df["source_kind"] = "raw_backfill_candidate_scores"

            normalized = df.loc[
                :,
                [
                    "run_id",
                    "split",
                    "probe_family",
                    "framing_family",
                    "question_uid",
                    "choice_id",
                    "is_correct",
                    "probe_score",
                    "source_kind",
                ],
            ].copy()
            frames.append(normalized)

            combo_df = normalized.loc[:, ["run_id", "split", "probe_family", "framing_family"]].drop_duplicates()
            override_combos.update(tuple(row) for row in combo_df.itertuples(index=False, name=None))

    if not frames:
        return pd.DataFrame(), override_combos

    return pd.concat(frames, ignore_index=True), override_combos


def load_packaged_choice_scores(
    packaged_input_path: Path,
    split: str,
    override_combos: set[tuple[str, str, str, str]],
) -> pd.DataFrame:
    usecols = [
        "run_id",
        "split",
        "probe_family",
        "framing_family",
        "probe_score",
        "is_correct",
        "question_uid",
        "choice_id",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(packaged_input_path, usecols=usecols, chunksize=500_000):
        filtered = chunk.loc[
            chunk["split"].astype(str).eq(split)
            & chunk["probe_family"].astype(str).isin(PROBE_FAMILY_ORDER)
        ].copy()
        if filtered.empty:
            continue

        if override_combos:
            combo_index = pd.MultiIndex.from_frame(
                filtered.loc[:, ["run_id", "split", "probe_family", "framing_family"]]
            )
            override_index = pd.MultiIndex.from_tuples(
                sorted(override_combos),
                names=["run_id", "split", "probe_family", "framing_family"],
            )
            filtered = filtered.loc[~combo_index.isin(override_index)].copy()
            if filtered.empty:
                continue

        filtered["source_kind"] = "packaged_choice_level"
        frames.append(filtered)

    if not frames:
        return pd.DataFrame(columns=usecols + ["source_kind"])
    return pd.concat(frames, ignore_index=True)


def build_summary_table(all_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        all_df.groupby(["probe_family", "framing_family"], sort=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )

    full_grid = pd.MultiIndex.from_product(
        [PROBE_FAMILY_ORDER, FRAMING_ORDER],
        names=["probe_family", "framing_family"],
    ).to_frame(index=False)
    summary_df = full_grid.merge(summary_df, on=["probe_family", "framing_family"], how="left")

    summary_df["Probe trained on"] = summary_df["probe_family"].map(PROBE_LABELS).fillna(summary_df["probe_family"])
    summary_df["Eval on"] = summary_df["framing_family"]

    summary_df = summary_df.sort_values(
        by=["probe_family", "framing_family"],
        key=lambda series: series.map(
            lambda name: probe_sort_key(str(name)) if series.name == "probe_family" else framing_sort_key(str(name))
        ),
    ).reset_index(drop=True)
    return summary_df


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def write_latex_table(summary_df: pd.DataFrame, tex_path: Path, split: str) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Probe trained on & Eval on & Top-1 & Pairwise $K$ & AUC \\\\",
        "\\midrule",
    ]

    for _, row in summary_df.iterrows():
        probe_label = str(row["Probe trained on"]).replace("_", "\\_")
        eval_label = str(row["Eval on"]).replace("_", "\\_")
        lines.append(
            f"{probe_label} & {eval_label} & {format_metric(row['Top-1'])} & "
            f"{format_metric(row['Pairwise K'])} & {format_metric(row['AUC'])} \\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            (
                "\\caption{Pooled "
                + split.replace("_", "\\_")
                + "-set probe metrics across the main runs, with newer raw backfills used when available.}"
            ),
            "\\label{tab:claim3-probe-train-eval-main-runs}",
            "\\end{table}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    packaged_input_path = Path(args.packaged_input_path).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_run_ids = load_allowed_run_ids(packaged_input_path)
    raw_df, override_combos = load_raw_backfill_candidate_scores(
        results_root=results_root,
        split=str(args.split),
        allowed_run_ids=allowed_run_ids,
    )
    packaged_df = load_packaged_choice_scores(
        packaged_input_path=packaged_input_path,
        split=str(args.split),
        override_combos=override_combos,
    )

    frames = [df for df in [packaged_df, raw_df] if not df.empty]
    if not frames:
        raise ValueError("No rows found for the requested split and probe families.")

    all_df = pd.concat(frames, ignore_index=True)
    summary_df = build_summary_table(all_df)

    table_df = summary_df.loc[:, ["Probe trained on", "Eval on", "Top-1", "Pairwise K", "AUC"]].copy()
    output_stem = args.output_stem.strip()
    csv_path = output_dir / f"{output_stem}.csv"
    json_path = output_dir / f"{output_stem}.json"
    tex_path = output_dir / f"{output_stem}.tex"

    table_df.to_csv(csv_path, index=False)
    write_latex_table(summary_df, tex_path=tex_path, split=str(args.split))

    metadata = {
        "created_at_utc": utc_now_iso(),
        "packaged_input_path": str(packaged_input_path),
        "results_root": str(results_root),
        "split": str(args.split),
        "output_csv_path": str(csv_path),
        "output_tex_path": str(tex_path),
        "allowed_run_ids": sorted(allowed_run_ids),
        "override_combo_count": len(override_combos),
        "row_counts": summary_df.loc[
            :,
            [
                "Probe trained on",
                "Eval on",
                "n_prompts",
                "n_candidate_rows",
                "n_runs",
                "sources",
            ],
        ].to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote TeX to {tex_path}")
    printable_df = table_df.copy()
    for col in ["Top-1", "Pairwise K", "AUC"]:
        printable_df[col] = printable_df[col].map(format_metric)
    print(printable_df.to_string(index=False))


if __name__ == "__main__":
    main()
