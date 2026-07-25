from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns


LEARNED_COLOR = "#d4651a"
CONTROL_COLOR = "#73b3ab"


def _prepare_plotting() -> None:
    sns.set_style("white")


def plot_controlled_dose_response(frame: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    import matplotlib.pyplot as plt

    _prepare_plotting()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    biased = frame[frame["condition"].eq("incorrect_suggestion")].copy()
    learned = biased[
        biased["direction_name"].eq("wn")
        & biased["scale_convention"].eq("native")
    ]
    controls = biased[biased["treatment_type"].eq("control")]
    learned_curve = (
        learned.groupby(["model_name", "alpha"], as_index=False)["delta_p_endorsed"]
        .mean()
    )
    control_seed_curve = (
        controls.groupby(
            ["model_name", "direction_name", "control_seed", "alpha"],
            dropna=False,
            as_index=False,
        )["delta_p_endorsed"]
        .mean()
    )
    control_ribbon = (
        control_seed_curve.groupby(["model_name", "alpha"])["delta_p_endorsed"]
        .agg(
            control_low=lambda values: float(np.quantile(values, 0.025)),
            control_high=lambda values: float(np.quantile(values, 0.975)),
        )
        .reset_index()
    )
    model_names = sorted(learned_curve["model_name"].unique())
    figure, axes = plt.subplots(
        1,
        max(1, len(model_names)),
        figsize=(8 * max(1, len(model_names)), 6),
        squeeze=False,
    )
    for axis, model_name in zip(axes[0], model_names):
        curve = learned_curve[learned_curve["model_name"].eq(model_name)]
        ribbon = control_ribbon[control_ribbon["model_name"].eq(model_name)]
        axis.plot(
            curve["alpha"],
            curve["delta_p_endorsed"],
            color=LEARNED_COLOR,
            marker="o",
            linewidth=2.5,
            label="Learned W−N",
        )
        if not ribbon.empty:
            axis.fill_between(
                ribbon["alpha"].to_numpy(dtype=float),
                ribbon["control_low"].to_numpy(dtype=float),
                ribbon["control_high"].to_numpy(dtype=float),
                color=CONTROL_COLOR,
                alpha=0.3,
                label="10-seed controls (95%)",
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xscale("symlog", linthresh=0.25)
        axis.set_title(str(model_name), fontsize=18)
        axis.set_xlabel("Raw-shift alpha", fontsize=15)
        axis.set_ylabel("Change in P(user-backed wrong answer)", fontsize=15)
        axis.tick_params(axis="both", labelsize=12)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            frameon=True,
            fontsize=12,
        )
        sns.despine(ax=axis)
    figure.suptitle("Controlled prompt-only steering dose response", fontsize=21)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    png = target / "dose_response.png"
    pdf = target / "dose_response.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {"dose_response_png": png, "dose_response_pdf": pdf}


def plot_controlled_pareto(frame: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    import matplotlib.pyplot as plt

    _prepare_plotting()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    group_columns = [
        "model_name",
        "layer",
        "direction_name",
        "scale_convention",
        "control_seed",
        "alpha",
        "treatment_type",
    ]
    neutral = (
        frame[frame["condition"].eq("neutral")]
        .groupby(group_columns, dropna=False)["delta_p_correct"]
        .mean()
        .rename("neutral_delta_p_correct")
    )
    biased = (
        frame[frame["condition"].eq("incorrect_suggestion")]
        .groupby(group_columns, dropna=False)["delta_p_endorsed"]
        .mean()
        .rename("wrong_delta_p_endorsed")
    )
    pareto = pd.concat((neutral, biased), axis=1).dropna().reset_index()
    pareto["preserved_behavior_degradation"] = -pareto["neutral_delta_p_correct"]
    pareto["targeted_sycophancy_reduction"] = -pareto["wrong_delta_p_endorsed"]
    pareto["series"] = np.where(
        pareto["direction_name"].eq("wn")
        & pareto["scale_convention"].eq("native"),
        "Learned W−N",
        "Controls/variants",
    )
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=pareto,
        x="preserved_behavior_degradation",
        y="targeted_sycophancy_reduction",
        hue="series",
        palette={
            "Learned W−N": LEARNED_COLOR,
            "Controls/variants": CONTROL_COLOR,
        },
        alpha=0.75,
        s=65,
        ax=axis,
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_title("Steering selectivity: sycophancy reduction versus neutral damage", fontsize=20)
    axis.set_xlabel("Neutral P(correct) degradation", fontsize=15)
    axis.set_ylabel("Reduction in P(user-backed wrong answer)", fontsize=15)
    axis.tick_params(axis="both", labelsize=12)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
        fontsize=12,
    )
    sns.despine(ax=axis)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    png = target / "selectivity_pareto.png"
    pdf = target / "selectivity_pareto.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    pareto.to_csv(target / "selectivity_pareto_points.csv", index=False)
    return {"pareto_png": png, "pareto_pdf": pdf}


__all__ = [
    "CONTROL_COLOR",
    "LEARNED_COLOR",
    "plot_controlled_dose_response",
    "plot_controlled_pareto",
]
