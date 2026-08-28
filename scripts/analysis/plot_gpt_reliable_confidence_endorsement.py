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
from scipy.stats import spearmanr

from plot_confidence_conditioned_user_endorsement import (
    DATASET_ORDER,
    GPT_INPUT,
    PROMPT_TEXT,
    assign_confidence_bins,
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

    reliable.to_csv(args.output_dir / "gpt_reliable_item_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "gpt_reliable_bin_summary.csv", index=False)
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
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    plot_gpt_summary(summary, args.output_dir)

    print(f"Wrote {args.output_dir}")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        "movement Spearman rho="
        f"{trend.statistic:.4f}, p={trend.pvalue:.4g}"
    )


if __name__ == "__main__":
    main()
