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
    / "claim3_self_commitment_main_runs"
)

NEUTRAL_COLOR = "#73b3ab"
INCORRECT_COLOR = "#d4651a"
SUPPORT_COLOR = "#4c4c4c"
C_VS_D_PALETTE = {
    "Neutral top = correct (c)": NEUTRAL_COLOR,
    "Neutral top = other wrong (d)": INCORRECT_COLOR,
}
PROBE_SPLIT_PALETTE = {
    "Probe favors c over neutral top": NEUTRAL_COLOR,
    "Probe does not favor c over neutral top": INCORRECT_COLOR,
}
CONDITION_PALETTE = {
    "Incorrect suggestion -> b": INCORRECT_COLOR,
    "Congruent suggestion -> neutral top": NEUTRAL_COLOR,
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
    build_self_commitment_comparison_df,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export self-commitment diagnostics for the neutral -> incorrect_suggestion shift, "
            "including C-vs-D comparisons and model-congruent controls."
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
        help=f"Directory where the self-commitment diagnostics package should be written. Default: {DEFAULT_OUTPUT_DIR}",
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


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)


def _legend_below(ax: plt.Axes, *, title: str, ncol: int = 2) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles=handles,
        labels=labels,
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol,
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )


def _run_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["run_label"].dropna().unique().tolist())


def _c_vs_d_summary(
    comparison_df: pd.DataFrame,
    *,
    no_flip_only: bool = False,
) -> pd.DataFrame:
    plot_df = comparison_df.loc[
        comparison_df["included_in_c_vs_d"].astype(bool)
        & comparison_df["self_margin_to_b_quartile_run"].notna()
    ].copy()
    if no_flip_only:
        plot_df = plot_df.loc[~plot_df["answer_changed"].astype(bool)].copy()
    if plot_df.empty:
        return pd.DataFrame()
    plot_df = _with_run_labels(plot_df)
    plot_df["group_label"] = plot_df["neutral_top_group"].map(
        {
            "c_top": "Neutral top = correct (c)",
            "d_top": "Neutral top = other wrong (d)",
        }
    )
    summary = (
        plot_df.groupby(["run_label", "group_label", "self_margin_to_b_quartile_run"], as_index=False)
        .agg(
            n_questions=("question_uid", "nunique"),
            mean_delta_b=("delta_b", "mean"),
            mean_self_to_b_gap_closure=("self_to_b_gap_closure", "mean"),
        )
        .sort_values(["run_label", "group_label", "self_margin_to_b_quartile_run"])
    )
    return summary


def _d_group_probe_truth_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = comparison_df.loc[
        comparison_df["neutral_top_group"].eq("d_top")
        & comparison_df["self_margin_to_b_quartile_run"].notna()
        & comparison_df["probe_prefers_correct_to_self_neutral"].notna()
    ].copy()
    if plot_df.empty:
        return pd.DataFrame()
    plot_df = _with_run_labels(plot_df)
    plot_df["probe_truth_group"] = plot_df["probe_prefers_correct_to_self_neutral"].map(
        {
            True: "Probe favors c over neutral top",
            False: "Probe does not favor c over neutral top",
        }
    )
    summary = (
        plot_df.groupby(["run_label", "probe_truth_group", "self_margin_to_b_quartile_run"], as_index=False)
        .agg(
            n_questions=("question_uid", "nunique"),
            mean_delta_b=("delta_b", "mean"),
        )
        .sort_values(["run_label", "probe_truth_group", "self_margin_to_b_quartile_run"])
    )
    return summary


def _d_group_condition_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = comparison_df.loc[
        comparison_df["neutral_top_group"].eq("d_top")
        & comparison_df["self_margin_to_b_quartile_run"].notna()
        & comparison_df["congruent_prompt_available"].astype(bool)
        & comparison_df["congruent_endorses_self_choice"].astype(bool)
    ].copy()
    if plot_df.empty:
        return pd.DataFrame()
    plot_df = _with_run_labels(plot_df)
    incorrect = plot_df.loc[:, ["run_label", "self_margin_to_b_quartile_run", "question_uid", "delta_b"]].copy()
    incorrect["condition_label"] = "Incorrect suggestion -> b"
    incorrect["delta_prompt_endorsed_target"] = incorrect["delta_b"]
    incorrect = incorrect.drop(columns=["delta_b"])

    congruent = plot_df.loc[
        :,
        [
            "run_label",
            "self_margin_to_b_quartile_run",
            "question_uid",
            "delta_prompt_endorsed_target_congruent",
        ],
    ].copy()
    congruent["condition_label"] = "Congruent suggestion -> neutral top"
    congruent = congruent.rename(
        columns={"delta_prompt_endorsed_target_congruent": "delta_prompt_endorsed_target"}
    )

    long_df = pd.concat([incorrect, congruent], ignore_index=True)
    summary = (
        long_df.groupby(["run_label", "condition_label", "self_margin_to_b_quartile_run"], as_index=False)
        .agg(
            n_questions=("question_uid", "nunique"),
            mean_delta_prompt_endorsed_target=("delta_prompt_endorsed_target", "mean"),
        )
        .sort_values(["run_label", "condition_label", "self_margin_to_b_quartile_run"])
    )
    return summary


def plot_delta_b_by_self_margin_group(comparison_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = _c_vs_d_summary(comparison_df, no_flip_only=False)
    if summary.empty:
        return summary
    run_order = _run_order(summary)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    for ax, run_label in zip(axes, run_order):
        subset = summary.loc[summary["run_label"].eq(run_label)].copy()
        sns.pointplot(
            data=subset,
            x="self_margin_to_b_quartile_run",
            y="mean_delta_b",
            hue="group_label",
            palette=C_VS_D_PALETTE,
            markers=["o", "s"],
            linestyles=["-", "--"],
            errorbar=None,
            ax=ax,
        )
        ax.axhline(0.0, color=SUPPORT_COLOR, linestyle="--", linewidth=1.2)
        _style_axes(
            ax,
            xlabel="Neutral self-margin quartile within run/split",
            ylabel=r"Mean $\Delta p(b)$",
            title=run_label,
        )
        _legend_below(ax, title="Neutral top group", ncol=2)
    for ax in axes[len(run_order) :]:
        ax.axis("off")
    fig.suptitle("Movement Toward b by Neutral Self-Commitment", fontsize=22)
    _save_figure(fig, output_path)
    return summary


def plot_self_gap_closure_by_self_margin_group(comparison_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = _c_vs_d_summary(comparison_df, no_flip_only=False)
    if summary.empty:
        return summary
    run_order = _run_order(summary)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    for ax, run_label in zip(axes, run_order):
        subset = summary.loc[summary["run_label"].eq(run_label)].copy()
        sns.pointplot(
            data=subset,
            x="self_margin_to_b_quartile_run",
            y="mean_self_to_b_gap_closure",
            hue="group_label",
            palette=C_VS_D_PALETTE,
            markers=["o", "s"],
            linestyles=["-", "--"],
            errorbar=None,
            ax=ax,
        )
        ax.axhline(0.0, color=SUPPORT_COLOR, linestyle="--", linewidth=1.2)
        _style_axes(
            ax,
            xlabel="Neutral self-margin quartile within run/split",
            ylabel="Mean self-to-b gap closure",
            title=run_label,
        )
        _legend_below(ax, title="Neutral top group", ncol=2)
    for ax in axes[len(run_order) :]:
        ax.axis("off")
    fig.suptitle("Self-to-b Gap Closure by Neutral Self-Commitment", fontsize=22)
    _save_figure(fig, output_path)
    return summary


def plot_no_flip_dual_outcome(comparison_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = _c_vs_d_summary(comparison_df, no_flip_only=True)
    if summary.empty:
        return summary
    run_order = _run_order(summary)
    outcome_specs = [
        ("mean_delta_b", r"Mean $\Delta p(b)$", "No-Flip Movement Toward b"),
        ("mean_self_to_b_gap_closure", "Mean self-to-b gap closure", "No-Flip Self-to-b Gap Closure"),
    ]
    fig, axes = plt.subplots(
        len(outcome_specs),
        len(run_order),
        figsize=(4.8 * len(run_order), 8.0),
        sharex=True,
        sharey="row",
    )
    if len(run_order) == 1:
        axes = np.array([[axes[0]], [axes[1]]]) if len(outcome_specs) == 2 else np.array([axes])
    for row_idx, (metric_col, ylabel, row_title) in enumerate(outcome_specs):
        for col_idx, run_label in enumerate(run_order):
            ax = axes[row_idx, col_idx]
            subset = summary.loc[summary["run_label"].eq(run_label)].copy()
            sns.pointplot(
                data=subset,
                x="self_margin_to_b_quartile_run",
                y=metric_col,
                hue="group_label",
                palette=C_VS_D_PALETTE,
                markers=["o", "s"],
                linestyles=["-", "--"],
                errorbar=None,
                ax=ax,
            )
            ax.axhline(0.0, color=SUPPORT_COLOR, linestyle="--", linewidth=1.2)
            title = run_label if row_idx == 0 else row_title
            _style_axes(
                ax,
                xlabel="Neutral self-margin quartile within run/split",
                ylabel=ylabel,
                title=title,
            )
            _legend_below(ax, title="Neutral top group", ncol=2)
    fig.suptitle("No-Flip C-vs-D Comparison by Neutral Self-Commitment", fontsize=22)
    _save_figure(fig, output_path)
    return summary


def plot_d_group_probe_truth_delta_b(comparison_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = _d_group_probe_truth_summary(comparison_df)
    if summary.empty:
        return summary
    run_order = _run_order(summary)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    for ax, run_label in zip(axes, run_order):
        subset = summary.loc[summary["run_label"].eq(run_label)].copy()
        sns.pointplot(
            data=subset,
            x="self_margin_to_b_quartile_run",
            y="mean_delta_b",
            hue="probe_truth_group",
            palette=PROBE_SPLIT_PALETTE,
            markers=["o", "s"],
            linestyles=["-", "--"],
            errorbar=None,
            ax=ax,
        )
        ax.axhline(0.0, color=SUPPORT_COLOR, linestyle="--", linewidth=1.2)
        _style_axes(
            ax,
            xlabel="Neutral self-margin quartile within run/split",
            ylabel=r"Mean $\Delta p(b)$",
            title=run_label,
        )
        _legend_below(ax, title="Neutral probe signal", ncol=2)
    for ax in axes[len(run_order) :]:
        ax.axis("off")
    fig.suptitle("D-Group Movement Toward b by Neutral Probe Truth Signal", fontsize=22)
    _save_figure(fig, output_path)
    return summary


def plot_d_group_incorrect_vs_congruent(comparison_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = _d_group_condition_summary(comparison_df)
    if summary.empty:
        return summary
    run_order = _run_order(summary)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    for ax, run_label in zip(axes, run_order):
        subset = summary.loc[summary["run_label"].eq(run_label)].copy()
        sns.pointplot(
            data=subset,
            x="self_margin_to_b_quartile_run",
            y="mean_delta_prompt_endorsed_target",
            hue="condition_label",
            palette=CONDITION_PALETTE,
            markers=["o", "s"],
            linestyles=["-", "--"],
            errorbar=None,
            ax=ax,
        )
        ax.axhline(0.0, color=SUPPORT_COLOR, linestyle="--", linewidth=1.2)
        _style_axes(
            ax,
            xlabel="Neutral self-margin quartile within run/split",
            ylabel="Mean movement toward prompt-endorsed target",
            title=run_label,
        )
        _legend_below(ax, title="Prompt condition", ncol=2)
    for ax in axes[len(run_order) :]:
        ax.axis("off")
    fig.suptitle("D-Group: Incorrect Versus Congruent Suggestion", fontsize=22)
    _save_figure(fig, output_path)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_wide_df = pd.read_csv(input_dir / "selected_layer_model_scores_wide.csv.gz", low_memory=False)
    probe_wide_df = pd.read_csv(input_dir / "selected_layer_probe_scores_wide.csv.gz", low_memory=False)

    comparison_df = build_self_commitment_comparison_df(
        model_wide_df,
        probe_wide_df=probe_wide_df,
        framing="incorrect_suggestion",
        congruent_framing="model_congruent_suggestion",
        probe_family="neutral_trained",
    )
    comparison_df = _with_run_labels(comparison_df)

    files: Dict[str, Path] = {
        "per_question_self_commitment_metrics": output_dir / "per_question_self_commitment_metrics.csv.gz",
        "c_vs_d_summary_by_margin": output_dir / "c_vs_d_summary_by_margin.csv",
        "c_vs_d_no_flip_summary_by_margin": output_dir / "c_vs_d_no_flip_summary_by_margin.csv",
        "d_group_probe_truth_summary_by_margin": output_dir / "d_group_probe_truth_summary_by_margin.csv",
        "d_group_incorrect_vs_congruent_summary_by_margin": output_dir / "d_group_incorrect_vs_congruent_summary_by_margin.csv",
        "plot_delta_b_by_self_margin_group": plots_dir / "01_delta_b_by_self_margin_group.png",
        "plot_self_gap_closure_by_self_margin_group": plots_dir / "02_self_gap_closure_by_self_margin_group.png",
        "plot_no_flip_dual_outcome": plots_dir / "03_no_flip_dual_outcome.png",
        "plot_d_group_probe_truth_delta_b": plots_dir / "04_d_group_probe_truth_delta_b.png",
        "plot_d_group_incorrect_vs_congruent": plots_dir / "05_d_group_incorrect_vs_congruent.png",
    }

    _write_csv(comparison_df, files["per_question_self_commitment_metrics"], gzip=True)
    c_vs_d_summary = plot_delta_b_by_self_margin_group(comparison_df, files["plot_delta_b_by_self_margin_group"])
    gap_summary = plot_self_gap_closure_by_self_margin_group(
        comparison_df,
        files["plot_self_gap_closure_by_self_margin_group"],
    )
    no_flip_summary = plot_no_flip_dual_outcome(comparison_df, files["plot_no_flip_dual_outcome"])
    probe_truth_summary = plot_d_group_probe_truth_delta_b(
        comparison_df,
        files["plot_d_group_probe_truth_delta_b"],
    )
    condition_summary = plot_d_group_incorrect_vs_congruent(
        comparison_df,
        files["plot_d_group_incorrect_vs_congruent"],
    )

    summary_to_write = c_vs_d_summary if not c_vs_d_summary.empty else gap_summary
    _write_csv(summary_to_write, files["c_vs_d_summary_by_margin"])
    _write_csv(no_flip_summary, files["c_vs_d_no_flip_summary_by_margin"])
    _write_csv(probe_truth_summary, files["d_group_probe_truth_summary_by_margin"])
    _write_csv(condition_summary, files["d_group_incorrect_vs_congruent_summary_by_margin"])

    manifest = {
        "created_at_utc": _utc_now_iso(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": {name: str(path) for name, path in files.items()},
        "row_counts": {
            "per_question_self_commitment_metrics": int(len(comparison_df)),
            "c_vs_d_summary_by_margin": int(len(summary_to_write)),
            "c_vs_d_no_flip_summary_by_margin": int(len(no_flip_summary)),
            "d_group_probe_truth_summary_by_margin": int(len(probe_truth_summary)),
            "d_group_incorrect_vs_congruent_summary_by_margin": int(len(condition_summary)),
        },
    }
    (output_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote self-commitment diagnostics to {output_dir}")
    for name, path in files.items():
        print(f"{name}: {path.name}")


if __name__ == "__main__":
    main()
