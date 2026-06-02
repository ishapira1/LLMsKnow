from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
    / "claim3_targeted_transport_main_runs"
)

NEUTRAL_COLOR = "#73b3ab"
INCORRECT_COLOR = "#d4651a"
SUPPORT_COLOR = "#4c4c4c"
MARGIN_QUARTILE_PALETTE = {
    "Q1": "#d4651a",
    "Q2": "#e6ad79",
    "Q3": "#9fc9c3",
    "Q4": "#73b3ab",
}
SUBSET_PALETTE = {
    "all": SUPPORT_COLOR,
    "no_flip": NEUTRAL_COLOR,
    "stay_correct": INCORRECT_COLOR,
}

sns.set_style("white")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _bootstrap_src_path() -> None:
    import sys

    src_dir = REPO_ROOT / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_bootstrap_src_path()

from llmssycoph.analysis.transport import (  # noqa: E402
    build_incorrect_suggestion_transport_df,
    summarize_transport_by_margin_quartile,
    summarize_transport_by_subset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export targeted-transport diagnostics for the neutral -> incorrect_suggestion shift "
            "from the claim-3 selected-layer wide bundle."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing selected_layer_model_scores_wide.csv.gz and selected_layer_probe_scores_wide.csv.gz. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where transport tables and plots should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def _friendly_model_name(value: object) -> str:
    text = str(value or "").strip()
    mapping = {
        "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    }
    return mapping.get(text, text)


def _friendly_dataset_name(value: object) -> str:
    text = str(value or "").strip()
    mapping = {
        "commonsense_qa": "CommonsenseQA",
        "arc_challenge": "ARC-Challenge",
    }
    return mapping.get(text, text)


def _with_run_labels(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "model_name" in working.columns:
        working["model_label"] = working["model_name"].map(_friendly_model_name)
    if "dataset" in working.columns:
        working["dataset_label"] = working["dataset"].map(_friendly_dataset_name)
    if {"model_label", "dataset_label"}.issubset(working.columns):
        working["run_label"] = working["model_label"] + " / " + working["dataset_label"]
    return working


def _write_csv(df: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    if gzip:
        df.to_csv(path, index=False, compression={"method": "gzip", "compresslevel": 1})
    else:
        df.to_csv(path, index=False)


def _collapsed_subset_summary(transport_df: pd.DataFrame) -> pd.DataFrame:
    if transport_df.empty:
        return pd.DataFrame()

    collapsed = transport_df.copy()
    collapsed["split"] = "all"
    by_run = summarize_transport_by_subset(collapsed)

    pooled = transport_df.copy()
    pooled["run_id"] = "all_runs"
    pooled["model_name"] = "all_runs"
    pooled["dataset"] = "all_runs"
    pooled["split"] = "all"
    all_runs = summarize_transport_by_subset(pooled)
    combined = pd.concat([by_run, all_runs], ignore_index=True)
    return _with_run_labels(combined)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)


def plot_targeted_ratio_histogram(transport_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = transport_df.loc[transport_df["directional_transport_share"].notna()].copy()
    plot_df = _with_run_labels(plot_df)
    if plot_df.empty:
        return
    run_order = sorted(plot_df["run_label"].dropna().unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, run_label in zip(axes, run_order):
        subset = plot_df.loc[plot_df["run_label"].eq(run_label)].copy()
        sns.histplot(
            data=subset,
            x="directional_transport_share",
            bins=40,
            color=NEUTRAL_COLOR,
            edgecolor="white",
            ax=ax,
        )
        ax.axvline(0.0, color=SUPPORT_COLOR, linestyle="--", linewidth=1.5)
        ax.axvline(1.0, color=INCORRECT_COLOR, linestyle=":", linewidth=1.5)
        _style_axes(
            ax,
            xlabel=r"Directional transport share $\alpha_{cb} / (2 \cdot TV)$",
            ylabel="Question count",
            title=run_label,
        )
    for ax in axes[len(run_order) :]:
        ax.axis("off")
    fig.suptitle("Directional Transport Share for Neutral to Incorrect-Suggestion Shift", fontsize=22)
    _save_figure(fig, output_path)


def plot_alpha_vs_tv_scatter(transport_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = transport_df.loc[
        transport_df["tv"].notna()
        & transport_df["alpha_cb"].notna()
        & transport_df["neutral_margin_quartile_global"].notna()
    ].copy()
    plot_df = _with_run_labels(plot_df)
    if plot_df.empty:
        return
    run_order = sorted(plot_df["run_label"].dropna().unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, run_label in zip(axes, run_order):
        subset = plot_df.loc[plot_df["run_label"].eq(run_label)].copy()
        sns.scatterplot(
            data=subset,
            x="tv",
            y="alpha_cb",
            hue="neutral_margin_quartile_global",
            palette=MARGIN_QUARTILE_PALETTE,
            alpha=0.25,
            s=18,
            linewidth=0,
            ax=ax,
        )
        max_x = float(subset["tv"].max()) if len(subset) else 0.0
        guide_x = np.linspace(0.0, max_x, 50)
        ax.plot(guide_x, 2.0 * guide_x, linestyle="--", color=SUPPORT_COLOR, linewidth=1.5)
        _style_axes(
            ax,
            xlabel=r"Total movement $TV$",
            ylabel=r"Directional movement $\alpha_{cb}$",
            title=run_label,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles=handles,
                labels=labels,
                title="Neutral margin quartile",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                ncol=3,
                frameon=False,
                fontsize=11,
                title_fontsize=12,
            )
    for ax in axes[len(run_order) :]:
        ax.axis("off")
    fig.suptitle("Directional Movement Versus Total Redistribution", fontsize=22)
    _save_figure(fig, output_path)


def plot_targeted_share_by_margin_quartile(transport_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = transport_df.loc[
        transport_df["neutral_margin_quartile_run"].notna()
        & transport_df["directional_transport_share"].notna()
    ].copy()
    if plot_df.empty:
        return
    plot_df = _with_run_labels(plot_df)
    pieces: List[pd.DataFrame] = []
    for subset_name in ["all", "no_flip", "stay_correct"]:
        if subset_name == "all":
            subset = plot_df.copy()
        elif subset_name == "no_flip":
            subset = plot_df.loc[~plot_df["answer_changed"].astype(bool)].copy()
        else:
            subset = plot_df.loc[plot_df["stay_correct"].astype(bool)].copy()
        subset["subset"] = subset_name
        pieces.append(subset)
    combined = pd.concat(pieces, ignore_index=True)
    summary = (
        combined.groupby(["run_label", "subset", "neutral_margin_quartile_run"], as_index=False)
        .agg(mean_share=("directional_transport_share", "mean"))
        .sort_values(["run_label", "subset", "neutral_margin_quartile_run"])
    )
    run_order = sorted(summary["run_label"].dropna().unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    for ax, run_label in zip(axes, run_order):
        subset = summary.loc[summary["run_label"].eq(run_label)].copy()
        sns.pointplot(
            data=subset,
            x="neutral_margin_quartile_run",
            y="mean_share",
            hue="subset",
            palette=SUBSET_PALETTE,
            markers=["o", "s", "^"],
            linestyles=["-", "--", ":"],
            errorbar=None,
            ax=ax,
        )
        ax.axhline(0.0, color=SUPPORT_COLOR, linestyle="--", linewidth=1.2)
        ax.axhline(1.0, color=INCORRECT_COLOR, linestyle=":", linewidth=1.2)
        _style_axes(
            ax,
            xlabel="Neutral margin quartile within run/split",
            ylabel=r"Mean directional transport share",
            title=run_label,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles=handles,
                labels=labels,
                title="Subset",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                ncol=3,
                frameon=False,
                fontsize=11,
                title_fontsize=12,
            )
    for ax in axes[len(run_order) :]:
        ax.axis("off")
    fig.suptitle("Targeted Transport Share by Neutral-Margin Quartile", fontsize=22)
    _save_figure(fig, output_path)


def plot_delta_b_vs_other_wrong(transport_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = transport_df.loc[
        transport_df["delta_b"].notna()
        & transport_df["delta_best_other_wrong_neutral"].notna()
        & transport_df["neutral_margin_quartile_global"].notna()
    ].copy()
    plot_df = _with_run_labels(plot_df)
    if plot_df.empty:
        return
    run_order = sorted(plot_df["run_label"].dropna().unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, run_label in zip(axes, run_order):
        subset = plot_df.loc[plot_df["run_label"].eq(run_label)].copy()
        sns.scatterplot(
            data=subset,
            x="delta_best_other_wrong_neutral",
            y="delta_b",
            hue="neutral_margin_quartile_global",
            palette=MARGIN_QUARTILE_PALETTE,
            alpha=0.25,
            s=18,
            linewidth=0,
            ax=ax,
        )
        limits = [
            np.nanmin([subset["delta_best_other_wrong_neutral"].min(), subset["delta_b"].min()]),
            np.nanmax([subset["delta_best_other_wrong_neutral"].max(), subset["delta_b"].max()]),
        ]
        ax.plot(limits, limits, linestyle="--", color=SUPPORT_COLOR, linewidth=1.5)
        _style_axes(
            ax,
            xlabel=r"$\Delta p(\mathrm{best\ wrong}\neq b)$",
            ylabel=r"$\Delta p(b)$",
            title=run_label,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles=handles,
                labels=labels,
                title="Neutral margin quartile",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                ncol=3,
                frameon=False,
                fontsize=11,
                title_fontsize=12,
            )
    for ax in axes[len(run_order) :]:
        ax.axis("off")
    fig.suptitle("Mass to the Endorsed Wrong Answer Versus the Best Other Wrong Answer", fontsize=22)
    _save_figure(fig, output_path)


def plot_output_vs_probe_gap_closure(transport_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = transport_df.loc[
        transport_df["output_gap_closure"].notna() & transport_df["probe_closed_gap_closure"].notna()
    ].copy()
    plot_df = _with_run_labels(plot_df)
    if plot_df.empty:
        return
    subset_frames: List[pd.DataFrame] = []
    subset_specs = {
        "flip": plot_df["answer_changed"].astype(bool),
        "no_flip": ~plot_df["answer_changed"].astype(bool),
        "stay_correct": plot_df["stay_correct"].astype(bool),
    }
    for subset_name, mask in subset_specs.items():
        subset = plot_df.loc[mask].copy()
        if subset.empty:
            continue
        subset["subset"] = subset_name
        subset_frames.append(subset)
    if not subset_frames:
        return
    combined = pd.concat(subset_frames, ignore_index=True)
    run_order = sorted(combined["run_label"].dropna().unique().tolist())
    subset_order = ["flip", "no_flip", "stay_correct"]
    fig, axes = plt.subplots(len(subset_order), len(run_order), figsize=(4.8 * len(run_order), 4.1 * len(subset_order)), sharex=True, sharey=True)
    if len(subset_order) == 1 and len(run_order) == 1:
        axes = np.array([[axes]])
    elif len(subset_order) == 1:
        axes = np.array([axes])
    elif len(run_order) == 1:
        axes = np.array([[ax] for ax in axes])
    for row_idx, subset_name in enumerate(subset_order):
        for col_idx, run_label in enumerate(run_order):
            ax = axes[row_idx, col_idx]
            subset = combined.loc[
                combined["subset"].eq(subset_name) & combined["run_label"].eq(run_label)
            ].copy()
            if subset.empty:
                ax.axis("off")
                continue
            sns.scatterplot(
                data=subset,
                x="output_gap_closure",
                y="probe_closed_gap_closure",
                color=NEUTRAL_COLOR if subset_name != "stay_correct" else INCORRECT_COLOR,
                alpha=0.18,
                s=14,
                linewidth=0,
                ax=ax,
            )
            limits = [
                np.nanmin([subset["output_gap_closure"].min(), subset["probe_closed_gap_closure"].min()]),
                np.nanmax([subset["output_gap_closure"].max(), subset["probe_closed_gap_closure"].max()]),
            ]
            ax.plot(limits, limits, linestyle="--", color=SUPPORT_COLOR, linewidth=1.2)
            title = f"{run_label}\n{subset_name.replace('_', ' ')}"
            _style_axes(
                ax,
                xlabel="Output gap closure",
                ylabel="Probe gap closure (closed scores)",
                title=title,
            )
    fig.suptitle("Output Gap Closure Versus Probe Gap Closure", fontsize=22)
    _save_figure(fig, output_path)


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_wide_df = pd.read_csv(input_dir / "selected_layer_model_scores_wide.csv.gz", low_memory=False)
    probe_wide_df = pd.read_csv(input_dir / "selected_layer_probe_scores_wide.csv.gz", low_memory=False)

    transport_df = build_incorrect_suggestion_transport_df(
        model_wide_df,
        probe_wide_df=probe_wide_df,
        framing="incorrect_suggestion",
        probe_family="neutral_trained",
    )
    transport_df = _with_run_labels(transport_df)

    subset_summary_df = _with_run_labels(summarize_transport_by_subset(transport_df))
    collapsed_subset_summary_df = _collapsed_subset_summary(transport_df)
    margin_quartile_summary_df = _with_run_labels(
        summarize_transport_by_margin_quartile(transport_df, quartile_column="neutral_margin_quartile_run")
    )

    files: Dict[str, Path] = {
        "per_question_transport_metrics": output_dir / "per_question_transport_metrics.csv.gz",
        "subset_summary_by_split": output_dir / "subset_summary_by_split.csv",
        "subset_summary_collapsed": output_dir / "subset_summary_collapsed.csv",
        "margin_quartile_summary": output_dir / "margin_quartile_summary.csv",
        "plot_targeted_ratio_histogram": plots_dir / "01_targeted_ratio_histogram.png",
        "plot_alpha_vs_tv_scatter": plots_dir / "02_alpha_vs_tv_scatter.png",
        "plot_targeted_share_by_margin_quartile": plots_dir / "03_targeted_share_by_margin_quartile.png",
        "plot_delta_b_vs_other_wrong": plots_dir / "04_delta_b_vs_other_wrong.png",
        "plot_output_vs_probe_gap_closure": plots_dir / "05_output_vs_probe_gap_closure.png",
    }

    _write_csv(transport_df, files["per_question_transport_metrics"], gzip=True)
    _write_csv(subset_summary_df, files["subset_summary_by_split"])
    _write_csv(collapsed_subset_summary_df, files["subset_summary_collapsed"])
    _write_csv(margin_quartile_summary_df, files["margin_quartile_summary"])

    plot_targeted_ratio_histogram(transport_df, files["plot_targeted_ratio_histogram"])
    plot_alpha_vs_tv_scatter(transport_df, files["plot_alpha_vs_tv_scatter"])
    plot_targeted_share_by_margin_quartile(transport_df, files["plot_targeted_share_by_margin_quartile"])
    plot_delta_b_vs_other_wrong(transport_df, files["plot_delta_b_vs_other_wrong"])
    plot_output_vs_probe_gap_closure(transport_df, files["plot_output_vs_probe_gap_closure"])

    manifest = {
        "created_at_utc": _utc_now_iso(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": {name: str(path) for name, path in files.items()},
        "row_counts": {
            "per_question_transport_metrics": int(len(transport_df)),
            "subset_summary_by_split": int(len(subset_summary_df)),
            "subset_summary_collapsed": int(len(collapsed_subset_summary_df)),
            "margin_quartile_summary": int(len(margin_quartile_summary_df)),
        },
    }
    (output_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote transport diagnostics to {output_dir}")
    for name, path in files.items():
        print(f"{name}: {path.name}")


if __name__ == "__main__":
    main()
