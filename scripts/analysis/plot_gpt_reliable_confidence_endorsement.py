#!/usr/bin/env python3
"""Plot GPT-only pre/post margins using reliable normalized choice scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, trim_mean

from plot_confidence_conditioned_user_endorsement import (
    DATASET_ORDER,
    GPT_INPUT,
    PROMPT_TEXT,
    assign_confidence_bins,
    bootstrap_equal_dataset_mean,
    equal_dataset_mean,
    load_gpt_rows,
    summarize_bins,
)


MODEL = "GPT-5.4 nano"
COLOR = "#756bb1"
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/analysis/gpt_reliable_confidence_endorsement_20260828"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=GPT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-probability", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def select_reliable_gpt_rows(
    rows: pd.DataFrame, *, min_probability: float
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not 0 < min_probability < 1:
        raise ValueError("--min-probability must be strictly between zero and one")

    probability_columns = ["p0_a0", "p0_x", "p1_a0", "p1_x"]
    complete = rows["valid_probabilities"].fillna(False).astype(bool)
    target_not_top = rows["endorsed_answer"].astype(str).ne(
        rows["neutral_top_answer"].astype(str)
    )
    finite = np.isfinite(rows[probability_columns].to_numpy(float)).all(axis=1)
    above_floor = rows[probability_columns].gt(min_probability).all(axis=1)
    included = complete & target_not_top & finite & above_floor

    reliable = rows[included].copy()
    # Recompute without clipping: every retained probability is strictly above
    # the declared reliability threshold.
    reliable["initial_resistance_log_odds"] = np.log(reliable["p0_a0"]) - np.log(
        reliable["p0_x"]
    )
    reliable["post_resistance_log_odds"] = np.log(reliable["p1_a0"]) - np.log(
        reliable["p1_x"]
    )
    reliable["movement_toward_x_log_odds"] = (
        reliable["initial_resistance_log_odds"]
        - reliable["post_resistance_log_odds"]
    )

    audit = {
        "n_available_pairs": int(len(rows)),
        "n_valid_pre_post_distributions": int(complete.sum()),
        "n_target_not_neutral_top": int((complete & target_not_top).sum()),
        "n_finite_pairwise_probabilities": int(
            (complete & target_not_top & finite).sum()
        ),
        "n_above_probability_threshold": int(included.sum()),
        "n_excluded_at_or_below_threshold": int(
            (complete & target_not_top & finite & ~above_floor).sum()
        ),
        "min_probability_exclusive": float(min_probability),
        "n_included_by_dataset": {
            str(dataset): int(count)
            for dataset, count in reliable["dataset"].value_counts().items()
        },
    }
    return reliable, audit


def plot_gpt_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    sns.set_style("white")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
    metric_specs = [
        (
            "post_resistance_log_odds",
            r"Post-endorsement margin $m_1=\log\,\frac{\pi_1(a_0)}{\pi_1(X)}$",
            "A. The same margin before and after endorsement",
        ),
        (
            "movement_toward_x_log_odds",
            r"Margin reduction toward $X$: $m_0-m_1$",
            "B. Endorsement-induced reduction in that margin",
        ),
    ]
    legend_handle = None

    for ax, (metric, ylabel, title) in zip(axes, metric_specs, strict=True):
        metric_summary = summary[summary["metric"].eq(metric)].sort_values(
            "confidence_bin"
        )
        x = metric_summary["initial_resistance_mean"].to_numpy(float)
        legend_handle = sns.lineplot(
            data=metric_summary,
            x="initial_resistance_mean",
            y="estimate",
            color=COLOR,
            marker="o",
            markersize=9,
            linewidth=2.8,
            legend=False,
            ax=ax,
        ).lines[-1]
        ax.fill_between(
            x,
            metric_summary["ci_low"].to_numpy(float),
            metric_summary["ci_high"].to_numpy(float),
            color=COLOR,
            alpha=0.18,
            linewidth=0,
        )
        ax.set_title(title, fontsize=18, pad=13)
        ax.set_xlabel(
            r"Neutral margin $m_0=\log[\pi_0(a_0)/\pi_0(X)]$"
            "\n(point = equal-dataset mean within a margin quintile)",
            fontsize=15,
            labelpad=10,
        )
        ax.set_ylabel(ylabel, fontsize=15, labelpad=9)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.9)
        sns.despine(ax=ax)

    x_values = summary["initial_resistance_mean"].to_numpy(float)
    post = summary[summary["metric"].eq("post_resistance_log_odds")]
    domain_low = float(min(x_values.min(), post["ci_low"].min(), 0.0))
    domain_high = float(max(x_values.max(), post["ci_high"].max(), 0.0))
    padding = 0.05 * max(domain_high - domain_low, 1.0)
    domain_low -= padding
    domain_high += padding

    axes[0].plot(
        [domain_low, domain_high],
        [domain_low, domain_high],
        color="#666666",
        linestyle="--",
        linewidth=1.5,
        zorder=0,
    )
    axes[0].text(
        domain_high - 0.02 * (domain_high - domain_low),
        domain_high - 0.04 * (domain_high - domain_low),
        r"no movement: $m_1=m_0$",
        ha="right",
        va="top",
        fontsize=12,
        color="#555555",
        rotation=38,
    )
    axes[0].axhline(0, color="#999999", linestyle=":", linewidth=1.1, zorder=0)
    axes[0].set_xlim(domain_low, domain_high)
    axes[0].set_ylim(domain_low, domain_high)
    axes[1].axhline(0, color="#777777", linestyle=":", linewidth=1.2, zorder=0)
    axes[1].set_xlim(domain_low, domain_high)

    fig.suptitle(
        "GPT-5.4 nano: unsupported endorsement shifts the original margin",
        fontsize=22,
        y=0.998,
    )
    fig.text(
        0.5,
        0.94,
        r"$\pi_t(x)=\exp z_t(x)\,/\,\sum_{y\in\mathcal{A}}\exp z_t(y)$"
        r"; $a_0=\arg\max_{a\in\mathcal{A}}\pi_0(a)$",
        ha="center",
        va="center",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.90,
        rf'Prompt: “{PROMPT_TEXT}”',
        ha="center",
        va="center",
        fontsize=13,
    )
    fig.legend(
        [legend_handle],
        [MODEL],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=1,
        frameon=True,
        edgecolor="#dddddd",
        fontsize=12,
        handlelength=2.7,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.24,
        top=0.83,
        wspace=0.26,
    )

    for suffix in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"gpt_reliable_pre_post_margin.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def assign_probability_margin_bins(rows: pd.DataFrame, bins: int) -> pd.DataFrame:
    output = rows.copy()
    output["probability_margin_bin"] = -1
    for dataset, index in output.groupby("dataset").groups.items():
        values = output.loc[index, "initial_probability_margin"]
        if len(values) < bins:
            raise ValueError(f"{dataset} has fewer observations than bins")
        ranked = values.rank(method="first")
        output.loc[index, "probability_margin_bin"] = (
            pd.qcut(ranked, q=bins, labels=False).astype(int).to_numpy() + 1
        )
    output["probability_margin_bin"] = output[
        "probability_margin_bin"
    ].astype(int)
    return output


def summarize_probability_bins(
    rows: pd.DataFrame, *, replicates: int, seed: int
) -> pd.DataFrame:
    metrics = [
        "initial_probability_margin",
        "post_probability_margin",
        "probability_margin_reduction",
    ]
    summaries: list[dict[str, object]] = []
    for bin_number, frame in rows.groupby("probability_margin_bin", sort=True):
        for metric_number, metric in enumerate(metrics):
            rng = np.random.default_rng(seed + int(bin_number) * 10 + metric_number)
            low, high = bootstrap_equal_dataset_mean(
                frame,
                metric=metric,
                replicates=replicates,
                rng=rng,
            )
            values = frame[metric].to_numpy(float)
            summaries.append(
                {
                    "probability_margin_bin": int(bin_number),
                    "metric": metric,
                    "estimate": equal_dataset_mean(frame, metric),
                    "ci_low": low,
                    "ci_high": high,
                    "median": float(np.median(values)),
                    "trimmed_mean_10pct": float(trim_mean(values, 0.1)),
                    "n_items": int(len(frame)),
                    "n_arc_challenge": int(
                        frame["dataset"].eq("arc_challenge").sum()
                    ),
                    "n_commonsense_qa": int(
                        frame["dataset"].eq("commonsense_qa").sum()
                    ),
                    "dataset_weighting": "equal ARC-Challenge / CommonsenseQA",
                    "uncertainty": (
                        "stratified item bootstrap percentile 95% CI"
                    ),
                }
            )
    return pd.DataFrame(summaries).sort_values(
        ["metric", "probability_margin_bin"]
    ).reset_index(drop=True)


def plot_gpt_probability_summary(
    summary: pd.DataFrame, output_dir: Path
) -> None:
    sns.set_style("white")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
    condition_specs = [
        ("initial_probability_margin", "Neutral margin $d_0$", "#73b3ab"),
        ("post_probability_margin", "Post-endorsement margin $d_1$", "#d4651a"),
    ]
    handles: list[object] = []
    labels: list[str] = []

    for metric, label, color in condition_specs:
        metric_summary = summary[summary["metric"].eq(metric)].sort_values(
            "probability_margin_bin"
        )
        x = metric_summary["probability_margin_bin"].to_numpy(float)
        line = sns.lineplot(
            data=metric_summary,
            x="probability_margin_bin",
            y="estimate",
            color=color,
            marker="o",
            markersize=9,
            linewidth=2.8,
            legend=False,
            ax=axes[0],
        ).lines[-1]
        axes[0].fill_between(
            x,
            metric_summary["ci_low"].to_numpy(float),
            metric_summary["ci_high"].to_numpy(float),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        handles.append(line)
        labels.append(label)

    movement = summary[
        summary["metric"].eq("probability_margin_reduction")
    ].sort_values("probability_margin_bin")
    movement_x = movement["probability_margin_bin"].to_numpy(float)
    movement_line = sns.lineplot(
        data=movement,
        x="probability_margin_bin",
        y="estimate",
        color=COLOR,
        marker="o",
        markersize=9,
        linewidth=2.8,
        legend=False,
        ax=axes[1],
    ).lines[-1]
    axes[1].fill_between(
        movement_x,
        movement["ci_low"].to_numpy(float),
        movement["ci_high"].to_numpy(float),
        color=COLOR,
        alpha=0.18,
        linewidth=0,
    )
    handles.append(movement_line)
    labels.append(r"Margin reduction $d_0-d_1$")

    initial = summary[
        summary["metric"].eq("initial_probability_margin")
    ].sort_values("probability_margin_bin")
    bin_labels = [
        f"Q{int(row.probability_margin_bin)}\n{row.estimate:.4f}"
        for row in initial.itertuples()
    ]
    for ax in axes:
        ax.set_xticks(range(1, len(bin_labels) + 1))
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel(
            r"Neutral probability-margin quintile, "
            r"$d_0=\pi_0(a_0)-\pi_0(X)$"
            "\n(tick gives the equal-dataset mean)",
            fontsize=15,
            labelpad=10,
        )
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        )
        sns.despine(ax=ax)

    axes[0].set_title(
        "A. Probability margin before and after endorsement",
        fontsize=18,
        pad=13,
    )
    axes[0].set_ylabel(
        r"Probability margin: $\pi(a_0)-\pi(X)$",
        fontsize=15,
        labelpad=9,
    )
    axes[0].axhline(0, color="#777777", linestyle=":", linewidth=1.2, zorder=0)
    axes[0].set_ylim(-0.22, 1.05)

    axes[1].set_title(
        "B. Endorsement-induced reduction in that margin",
        fontsize=18,
        pad=13,
    )
    axes[1].set_ylabel(
        r"Movement toward $X$: $d_0-d_1$",
        fontsize=15,
        labelpad=9,
    )
    axes[1].axhline(0, color="#777777", linestyle=":", linewidth=1.2, zorder=0)
    axes[1].set_ylim(-0.04, 0.95)

    fig.suptitle(
        "GPT-5.4 nano: unsupported endorsement in probability space",
        fontsize=22,
        y=0.998,
    )
    fig.text(
        0.5,
        0.94,
        r"$\pi_t(x)=\exp z_t(x)\,/\,\sum_{y\in\mathcal{A}}\exp z_t(y)$"
        r"; $a_0=\arg\max_{a\in\mathcal{A}}\pi_0(a)$",
        ha="center",
        va="center",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.90,
        rf'Prompt: “{PROMPT_TEXT}”',
        ha="center",
        va="center",
        fontsize=13,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=True,
        edgecolor="#dddddd",
        fontsize=12,
        handlelength=2.7,
        columnspacing=1.6,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.25,
        top=0.83,
        wspace=0.26,
    )

    for suffix in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"gpt_reliable_probability_margin.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def log_margin_robustness(rows: pd.DataFrame, bins: int) -> pd.DataFrame:
    binned = assign_confidence_bins(rows, bins)
    summaries: list[dict[str, object]] = []
    for bin_number, frame in binned.groupby("confidence_bin", sort=True):
        values = frame["movement_toward_x_log_odds"].to_numpy(float)
        summaries.append(
            {
                "confidence_bin": int(bin_number),
                "n_items": int(len(frame)),
                "mean": equal_dataset_mean(
                    frame, "movement_toward_x_log_odds"
                ),
                "median": float(np.median(values)),
                "trimmed_mean_10pct": float(trim_mean(values, 0.1)),
            }
        )
    return pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpt_rows, _ = load_gpt_rows(args.input)
    reliable, audit = select_reliable_gpt_rows(
        gpt_rows, min_probability=args.min_probability
    )
    missing_datasets = set(DATASET_ORDER).difference(reliable["dataset"].unique())
    if missing_datasets:
        raise ValueError(f"Reliable cohort is missing datasets: {sorted(missing_datasets)}")

    binned = assign_confidence_bins(reliable, args.bins)
    summary = summarize_bins(
        binned,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    trend = spearmanr(
        reliable["initial_resistance_log_odds"].to_numpy(float),
        reliable["movement_toward_x_log_odds"].to_numpy(float),
    )
    reliable["initial_probability_margin"] = reliable["p0_a0"] - reliable["p0_x"]
    reliable["post_probability_margin"] = reliable["p1_a0"] - reliable["p1_x"]
    reliable["probability_margin_reduction"] = (
        reliable["initial_probability_margin"]
        - reliable["post_probability_margin"]
    )
    probability_binned = assign_probability_margin_bins(reliable, args.bins)
    probability_summary = summarize_probability_bins(
        probability_binned,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    probability_trend = spearmanr(
        reliable["initial_probability_margin"].to_numpy(float),
        reliable["probability_margin_reduction"].to_numpy(float),
    )
    robustness = log_margin_robustness(reliable, args.bins)

    reliable.to_csv(args.output_dir / "gpt_reliable_item_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "gpt_reliable_bin_summary.csv", index=False)
    probability_summary.to_csv(
        args.output_dir / "gpt_reliable_probability_bin_summary.csv", index=False
    )
    robustness.to_csv(
        args.output_dir / "gpt_log_margin_outlier_robustness.csv", index=False
    )
    metadata = {
        "model": MODEL,
        "prompt": PROMPT_TEXT,
        "normalized_choice_probability": (
            "pi_t(x) = exp(z_t(x)) / sum_{y in A} exp(z_t(y))"
        ),
        "answer_set": "multiple-choice options shown for the item",
        "neutral_anchor": "a0 = argmax_{a in A} pi_0(a)",
        "initial_margin": "m0 = log[pi_0(a0) / pi_0(X)]",
        "post_margin": "m1 = log[pi_1(a0) / pi_1(X)], with a0 fixed from neutral",
        "movement_toward_x": "m0 - m1",
        "reliability_filter": (
            "pi_0(a0), pi_0(X), pi_1(a0), and pi_1(X) are all finite and "
            f"strictly greater than {args.min_probability:g}"
        ),
        "cohort_audit": audit,
        "confidence_binning": (
            f"{args.bins} equal-count rank bins within dataset"
        ),
        "dataset_weighting": "equal ARC-Challenge / CommonsenseQA",
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.seed),
        "continuous_movement_spearman_rho": float(trend.statistic),
        "continuous_movement_two_sided_p_value": float(trend.pvalue),
        "probability_initial_margin": "d0 = pi_0(a0) - pi_0(X)",
        "probability_post_margin": "d1 = pi_1(a0) - pi_1(X)",
        "probability_movement_toward_x": "d0 - d1",
        "continuous_probability_movement_spearman_rho": float(
            probability_trend.statistic
        ),
        "continuous_probability_movement_two_sided_p_value": float(
            probability_trend.pvalue
        ),
        "outlier_robustness": (
            "log-margin bucket means, medians, and 10% trimmed means are saved "
            "in gpt_log_margin_outlier_robustness.csv"
        ),
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    plot_gpt_summary(summary, args.output_dir)
    plot_gpt_probability_summary(probability_summary, args.output_dir)

    print(f"Wrote {args.output_dir}")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        "movement Spearman rho="
        f"{trend.statistic:.4f}, p={trend.pvalue:.4g}"
    )
    print(
        "probability movement Spearman rho="
        f"{probability_trend.statistic:.4f}, p={probability_trend.pvalue:.4g}"
    )


if __name__ == "__main__":
    main()
