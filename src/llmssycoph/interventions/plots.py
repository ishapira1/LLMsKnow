from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


RESTORATION_COLOR = "#73b3ab"
CONTRAST_COLOR = "#d4651a"
CONTROL_COLOR = "#777777"


def _plot_runtime():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("white")
    return plt, sns


def plot_validation_layer_profile(frame: pd.DataFrame, output_path: Path) -> Optional[Path]:
    working = frame[
        (frame["split"] == "val")
        & frame["neutral_correct"].astype(bool)
        & frame["intervention"].isin(["patch_paired_full", "patch_reverse_full"])
    ].copy()
    if "is_terminal_layer" in working.columns:
        working = working[~working["is_terminal_layer"].astype(bool)]
    if working.empty:
        return None
    working["Expected-direction effect"] = np.where(
        working["intervention"] == "patch_paired_full",
        working["delta_margin"].astype(float),
        -working["delta_margin"].astype(float),
    )
    working["Patch"] = working["intervention"].map(
        {
            "patch_paired_full": "Neutral → biased",
            "patch_reverse_full": "Biased → neutral (sign reversed)",
        }
    )
    summary = (
        working.groupby(["layer", "Patch"], as_index=False)["Expected-direction effect"]
        .mean()
        .sort_values("layer")
    )
    plt, sns = _plot_runtime()
    figure, axis = plt.subplots(figsize=(11, 6.5))
    sns.lineplot(
        data=summary,
        x="layer",
        y="Expected-direction effect",
        hue="Patch",
        marker="o",
        linewidth=2.5,
        palette={
            "Neutral → biased": RESTORATION_COLOR,
            "Biased → neutral (sign reversed)": CONTRAST_COLOR,
        },
        ax=axis,
    )
    axis.axhline(0.0, color="#222222", linewidth=1.0, alpha=0.7)
    axis.set_title("Bidirectional paired-patch localization (validation)", fontsize=20, pad=14)
    axis.set_xlabel("Residual layer", fontsize=16)
    axis.set_ylabel("Mean effect in expected direction\n(log P(correct) − log P(user answer))", fontsize=15)
    axis.tick_params(axis="both", labelsize=12)
    axis.legend(
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
        fontsize=12,
    )
    sns.despine(axis=axis)
    figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_selected_dose_response(
    frame: pd.DataFrame,
    selection: Mapping[str, Any],
    output_path: Path,
    *,
    split: str = "val",
) -> Optional[Path]:
    layer = int(selection["selected_layer"])
    interventions = {
        "steer_restoration_meandiff": "Restoration MeanDiff",
        "steer_rademacher_null": "Label-sign null",
        "steer_random_direction": "Random direction",
    }
    working = frame[
        (frame["split"] == str(split))
        & (frame["layer"].astype(int) == layer)
        & (frame["condition"] == "incorrect_suggestion_strong")
        & frame["neutral_correct"].astype(bool)
        & frame["intervention"].isin(interventions)
    ].copy()
    if working.empty:
        return None
    working["Direction"] = working["intervention"].map(interventions)
    summary = (
        working.groupby(["alpha", "Direction"], as_index=False)["delta_margin"]
        .mean()
        .sort_values("alpha")
    )
    plt, sns = _plot_runtime()
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    sns.lineplot(
        data=summary,
        x="alpha",
        y="delta_margin",
        hue="Direction",
        marker="o",
        linewidth=2.5,
        palette={
            "Restoration MeanDiff": RESTORATION_COLOR,
            "Label-sign null": CONTRAST_COLOR,
            "Random direction": CONTROL_COLOR,
        },
        ax=axis,
    )
    selected_alpha = float(selection["selected_alpha"])
    axis.axvline(selected_alpha, color=RESTORATION_COLOR, linestyle="--", linewidth=1.5)
    axis.axhline(0.0, color="#222222", linewidth=1.0, alpha=0.7)
    axis.set_title(f"Steering dose response at selected layer {layer} ({split})", fontsize=20, pad=14)
    axis.set_xlabel("Dose α (projection standard deviations)", fontsize=16)
    axis.set_ylabel("Mean Δ [log P(correct) − log P(user answer)]", fontsize=15)
    axis.tick_params(axis="both", labelsize=12)
    axis.legend(
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        fontsize=12,
    )
    sns.despine(axis=axis)
    figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


__all__ = [
    "CONTRAST_COLOR",
    "CONTROL_COLOR",
    "RESTORATION_COLOR",
    "plot_selected_dose_response",
    "plot_validation_layer_profile",
]
