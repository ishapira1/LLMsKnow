#!/usr/bin/env python3
"""Export publication-sized vector panels for the source-credibility figure."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "pdf" / "source_credibility_panels"
DEFAULT_SUMMARY_CSV = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "mmlupro_source_credibility_correct81_single_multiturn_20260903"
    / "analysis"
    / "summary.csv"
)

PANEL_SIZE = (3.25, 3.0)
# Match the compact panel's actual paper footprint.  Rendering directly at
# final size keeps its 8--11 pt labels crisp instead of shrinking them again in
# LaTeX.
CONFLICT_PANEL_SIZE = (2.20, 2.66)
FORTY_CONFLICT_PANEL_SIZE = (2.60, 2.66)
FORTY_FIVE_CONFLICT_PANEL_SIZE = (2.925, 2.66)
FIFTY_CONFLICT_PANEL_SIZE = (3.25, 2.66)
HALF_CONFLICT_PANEL_SIZE = (3.18, 3.30)
WIDE_CONFLICT_PANEL_SIZE = (6.50, 2.55)
MODEL_LEGEND_SIZE = (6.5, 0.55)
CONFLICT_LEGEND_SIZE = (3.25, 0.55)

AXIS_TITLE_SIZE = 20
TICK_SIZE = 16
RESPONSE_X_TICK_SIZE = 14
ANNOTATION_SIZE = 15
LEGEND_SIZE = 15
LINE_WIDTH = 2.15
MARKER_SIZE = 6.5
LEGEND_MARKER_SIZE = 8.0
CI_WIDTH = 1.15
CI_CAP_SIZE = 2.7
CI_ALPHA = 0.62

MODEL_ORDER = (
    "GPT-5.6 Terra",
    "GPT-5.6 Luna",
    "Claude Opus 5",
    "Claude Sonnet 5",
)
MODEL_COLORS = {
    "GPT-5.6 Terra": "#264653",
    "GPT-5.6 Luna": "#4F7CAC",
    "Claude Opus 5": "#6D597A",
    "Claude Sonnet 5": "#C05A87",
}
MODEL_MARKERS = {
    "GPT-5.6 Terra": "o",
    "GPT-5.6 Luna": "s",
    "Claude Opus 5": "D",
    "Claude Sonnet 5": "^",
}
SOURCE_COLOR = "#73B3AB"
USER_COLOR = "#D4651A"
CONNECTOR_COLOR = "#B8B8B8"
GRID_COLOR = "#E8E8E8"
TEXT_COLOR = "#2B2B2B"
SPINE_COLOR = "#5A5A5A"
RULE_COLOR = "#dddddd"
OVERLEAF_TEXT_FONT = "Times"

CONFLICT_COLORS = {
    "user": "#B73229",
    "source": "#D7928F",
    "correct": "#E7E7E7",
    "other_incorrect": "#8B9CAD",
}
CONFLICT_CALLOUTS = {
    "user": "User\nassertion",
    "source": "Source\nassertion",
    "correct": "Correct",
    "other_incorrect": "Other\nincorrect",
}
CONFLICT_ORDER = ("user", "source", "correct", "other_incorrect")
CONFLICT_BAR_HEIGHT = 0.86
AXIS_TICK_COLOR = "#A7A7A7"
ROW_LABEL_COLOR = "#6A6A6A"
CORRECT_LABEL_COLOR = "#565656"
BRACKET_COLOR = "#4A4A4A"
REFERENCE_FONT = "Times"


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float


MODEL_KEYS = {
    "GPT-5.6 Terra": "terra",
    "GPT-5.6 Luna": "luna",
    "Claude Opus 5": "opus5",
    "Claude Sonnet 5": "sonnet5",
}


def _load_primary_summary(
    path: Path,
) -> tuple[
    Mapping[str, tuple[Estimate, ...]],
    Mapping[str, tuple[Estimate, ...]],
    tuple[Mapping[str, object], ...],
]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    primary = {
        (str(row["model_key"]), str(row["metric"])): row
        for row in rows
        if row["analysis"] == "primary"
    }

    def estimate(model_key: str, metric: str) -> Estimate:
        try:
            row = primary[(model_key, metric)]
        except KeyError as exc:
            raise ValueError(f"Missing primary estimate for {model_key}/{metric}") from exc
        return Estimate(
            100 * float(row["estimate"]),
            100 * float(row["ci_low"]),
            100 * float(row["ci_high"]),
        )

    identity = {
        model: tuple(
            estimate(model_key, metric)
            for metric in ("p_x_user", "p_x_professor", "p_x_expert")
        )
        for model, model_key in MODEL_KEYS.items()
    }
    reliability = {
        model: tuple(
            estimate(model_key, metric)
            for metric in ("p_x_reliable20", "p_x_reliable50", "p_x_reliable90")
        )
        for model, model_key in MODEL_KEYS.items()
    }
    conflict = tuple(
        {
            "model": model.replace("GPT-5.6 ", "").replace("Claude ", ""),
            "source": estimate(model_key, "conflict_source_rate_credible_average"),
            "user": estimate(model_key, "conflict_preference_rate_credible_average"),
            "correct": estimate(model_key, "conflict_correct_rate_credible_average"),
            "other_incorrect": estimate(
                model_key, "conflict_other_incorrect_rate_credible_average"
            ),
        }
        for model, model_key in MODEL_KEYS.items()
    )
    return identity, reliability, conflict


def _configure_matplotlib() -> None:
    sns.set_style("white")
    mpl.rcParams.update(
        {
            # style/arxiv.sty sets \rmdefault to ptm (a Times-family serif).
            "font.family": "serif",
            "font.serif": [OVERLEAF_TEXT_FONT, "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.grid": False,
            "grid.color": GRID_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
        }
    )


def _save_fixed_pdf(fig: plt.Figure, output: Path, *, transparent: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        transparent=transparent,
        facecolor="none" if transparent else "white",
    )
    plt.close(fig)


def _draw_response_panel(
    data: Mapping[str, tuple[Estimate, ...]],
    tick_labels: Sequence[str],
    output: Path,
    *,
    x_label: str | None,
    show_y_label: bool,
    x_positions: Sequence[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    fig.subplots_adjust(left=0.18, right=0.995, bottom=0.27, top=0.93)

    x = np.asarray(
        x_positions if x_positions is not None else np.arange(len(tick_labels)),
        dtype=float,
    )
    for model in MODEL_ORDER:
        estimates = data[model]
        values = np.asarray([item.value for item in estimates], dtype=float)
        lows = np.asarray([item.low for item in estimates], dtype=float)
        highs = np.asarray([item.high for item in estimates], dtype=float)
        ax.errorbar(
            x,
            values,
            yerr=np.vstack([values - lows, highs - values]),
            fmt="none",
            ecolor=MODEL_COLORS[model],
            elinewidth=CI_WIDTH,
            capsize=CI_CAP_SIZE,
            capthick=CI_WIDTH,
            alpha=CI_ALPHA,
            zorder=2,
        )
        ax.plot(
            x,
            values,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=MARKER_SIZE,
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor=MODEL_COLORS[model],
            linewidth=LINE_WIDTH,
            alpha=1.0,
            zorder=3,
        )

    ax.set_xlim(float(x.min()) - 0.03, float(x.max()) + 0.03)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x, tick_labels)
    if x_label:
        ax.set_xlabel(x_label, fontsize=AXIS_TITLE_SIZE, labelpad=6)
    if show_y_label:
        fig.text(
            0.07,
            0.62,
            "Endorsed incorrect\nanswer selected (%)",
            fontsize=AXIS_TITLE_SIZE,
            rotation=90,
            ha="center",
            va="center",
            linespacing=0.8,
        )
    ax.tick_params(axis="x", labelsize=RESPONSE_X_TICK_SIZE, pad=3)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, pad=3)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.65)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(SPINE_COLOR)
        ax.spines[spine_name].set_linewidth(0.7)
    ax.set_axisbelow(True)
    ax.get_xticklabels()[-1].set_ha("right")
    _save_fixed_pdf(fig, output)


def _draw_direct_conflict(
    data: Sequence[Mapping[str, object]],
    output: Path,
    *,
    wide: bool = False,
    half: bool = False,
    forty: bool = False,
    forty_five: bool = False,
    fifty: bool = False,
) -> None:
    if sum((wide, half, forty, forty_five, fifty)) > 1:
        raise ValueError("direct-conflict layout modes are mutually exclusive")
    layout = (
        "wide"
        if wide
        else "half"
        if half
        else "fifty"
        if fifty
        else "forty_five"
        if forty_five
        else "forty"
        if forty
        else "compact"
    )
    fig, ax = plt.subplots(
        figsize=(
            WIDE_CONFLICT_PANEL_SIZE
            if wide
            else FIFTY_CONFLICT_PANEL_SIZE
            if fifty
            else FORTY_FIVE_CONFLICT_PANEL_SIZE
            if forty_five
            else FORTY_CONFLICT_PANEL_SIZE
            if forty
            else HALF_CONFLICT_PANEL_SIZE
            if half
            else CONFLICT_PANEL_SIZE
        )
    )
    if wide:
        fig.subplots_adjust(left=0.13, right=0.975, bottom=0.24, top=0.92)
    elif fifty:
        fig.subplots_adjust(left=0.20, right=0.97, bottom=0.25, top=0.92)
    elif forty_five:
        fig.subplots_adjust(left=0.22, right=0.97, bottom=0.25, top=0.92)
    elif forty:
        fig.subplots_adjust(left=0.25, right=0.96, bottom=0.25, top=0.92)
    elif half:
        fig.subplots_adjust(left=0.23, right=0.96, bottom=0.24, top=0.92)
    else:
        # Leave a real gutter for the longest model name at compact scale;
        # otherwise the leading "S" in "Sonnet 5" is clipped by the PDF box.
        fig.subplots_adjust(left=0.29, right=0.95, bottom=0.25, top=0.92)

    if layout == "wide":
        large_segment_size = 10.0
        small_segment_size = 8.2
        tiny_callout_size = 8.3
        residual_size = 8.8
        bracket_size = 9.2
        category_size = 8.8
        endpoint_size = 10.0
        row_label_size = 11.0
    elif layout == "half":
        large_segment_size = 9.0
        small_segment_size = 8.0
        tiny_callout_size = 8.3
        residual_size = 8.6
        bracket_size = 9.1
        category_size = 8.6
        endpoint_size = 9.5
        row_label_size = 10.2
    elif layout == "forty":
        large_segment_size = 8.4
        small_segment_size = 7.5
        tiny_callout_size = 7.9
        residual_size = 8.2
        bracket_size = 8.7
        category_size = 8.1
        endpoint_size = 9.0
        row_label_size = 9.8
    elif layout == "forty_five":
        large_segment_size = 8.8
        small_segment_size = 7.8
        tiny_callout_size = 8.0
        residual_size = 8.4
        bracket_size = 9.0
        category_size = 8.4
        endpoint_size = 9.2
        row_label_size = 10.0
    elif layout == "fifty":
        large_segment_size = 8.0
        small_segment_size = 7.0
        tiny_callout_size = 7.2
        residual_size = 7.5
        bracket_size = 8.2
        category_size = 7.6
        endpoint_size = 8.5
        row_label_size = 7.4
    else:
        # The one-row panel is rendered directly at its final physical size.
        # A tighter, internally consistent type hierarchy is cleaner here than
        # shrinking the wide-panel typography wholesale.
        large_segment_size = 7.6
        small_segment_size = 7.0
        tiny_callout_size = 7.5
        residual_size = 7.7
        bracket_size = 8.0
        category_size = 7.3
        endpoint_size = 8.5
        row_label_size = 9.1

    bar_height = 0.78 if layout == "fifty" else CONFLICT_BAR_HEIGHT
    y_positions = np.arange(len(data))[::-1] * 1.80
    endpoint_y = -1.42
    for row_index, (y, row) in enumerate(zip(y_positions, data)):
        widths = []
        for key in CONFLICT_ORDER:
            estimate = row[key]
            assert isinstance(estimate, Estimate)
            widths.append(max(0.0, estimate.value))
        total = sum(widths)
        if total <= 0:
            raise ValueError(f"Conflict outcome percentages sum to {total}")
        widths = [100.0 * value / total for value in widths]
        display_percentages = [int(np.floor(value + 0.5)) for value in widths]
        # Let the large correct-response segment absorb any one-point rounding
        # discrepancy. This keeps the displayed row total at exactly 100 while
        # preserving conventional rounding for the three focal categories.
        display_percentages[2] += 100 - sum(display_percentages)
        accepting_display = sum(display_percentages[:2])

        left = 0.0
        centers = []
        segments = []
        for key, width, displayed in zip(
            CONFLICT_ORDER, widths, display_percentages
        ):
            center = left + width / 2
            centers.append(center)
            segments.append((key, left, width, center, displayed))
            ax.barh(
                y,
                width,
                left=left,
                height=bar_height,
                color=CONFLICT_COLORS[key],
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )

            # Only place labels inside segments that can comfortably contain
            # them. Tiny focal segments and the residual segment are labeled
            # outside the bar below.
            if displayed == 0 or width < 8 or key == "other_incorrect":
                left += width
                continue
            label_size = large_segment_size
            if width < 11:
                label_size = small_segment_size
            elif width < 15:
                label_size = small_segment_size
            ax.text(
                center,
                y,
                f"{displayed}%" if layout != "compact" else f"{displayed}",
                ha="center",
                va="center",
                fontsize=label_size,
                fontweight="regular" if key == "correct" else "medium",
                fontfamily=REFERENCE_FONT,
                color=CORRECT_LABEL_COLOR if key == "correct" else "white",
                clip_on=False,
                zorder=3,
            )
            left += width

        # Put adjacent tiny focal labels in a dedicated callout lane above the
        # row.  Their elbow leaders separate the labels horizontally, avoiding
        # both the neighboring row and the category labels below the chart.
        tiny_focal_segments = [
            segment
            for segment in segments
            if segment[0] in {"user", "source"}
            and 0 < segment[4]
            and segment[2] < 8
        ]
        if len(tiny_focal_segments) == 1:
            target_xs = [tiny_focal_segments[0][3]]
        else:
            target_xs = [-1.5, 13.0]
        for (_, _, _, center, displayed), target_x in zip(
            tiny_focal_segments, target_xs
        ):
            bar_edge = y + bar_height / 2
            leader_y = bar_edge + 0.18
            ax.plot(
                [center, center, target_x],
                [bar_edge + 0.03, leader_y, leader_y],
                color=ROW_LABEL_COLOR,
                linewidth=0.6,
                clip_on=False,
                zorder=3,
            )
            ax.text(
                target_x,
                leader_y + 0.07,
                f"{displayed}%" if layout != "compact" else f"{displayed}",
                ha="center",
                va="bottom",
                fontsize=tiny_callout_size,
                fontweight="regular",
                fontfamily=REFERENCE_FONT,
                color=ROW_LABEL_COLOR,
                clip_on=False,
                zorder=4,
            )

        # Give one-percent residual outcomes a dedicated right-side gutter.
        other_segment = next(
            segment for segment in segments if segment[0] == "other_incorrect"
        )
        if other_segment[4] > 0:
            ax.text(
                101.5,
                y,
                f"{other_segment[4]}%" if layout != "compact" else f"{other_segment[4]}",
                ha="left",
                va="center",
                fontsize=residual_size,
                fontweight="regular",
                fontfamily=REFERENCE_FONT,
                color=ROW_LABEL_COLOR,
                clip_on=False,
                zorder=4,
            )

        if row_index == 0:
            accepting = widths[0] + widths[1]
            bracket_y = y + CONFLICT_BAR_HEIGHT / 2 + 0.24
            bracket_tip = 0.11
            ax.plot(
                [0, 0, accepting, accepting],
                [bracket_y - bracket_tip, bracket_y, bracket_y, bracket_y - bracket_tip],
                color=BRACKET_COLOR,
                linewidth=0.6,
                clip_on=False,
                zorder=4,
            )
            ax.text(
                accepting / 2,
                bracket_y + 0.11,
                f"{accepting_display}% accepting\nfalse premise",
                ha="center",
                va="bottom",
                fontsize=bracket_size,
                fontweight="regular",
                fontfamily=REFERENCE_FONT,
                color=BRACKET_COLOR,
                linespacing=0.90,
                clip_on=False,
                zorder=4,
            )

        if row_index == len(data) - 1:
            bar_bottom = y - bar_height / 2
            tick_bottom = bar_bottom - 0.18
            label_y = tick_bottom - 0.14
            endpoint_y = label_y - (1.05 if layout != "compact" else 1.15)
            for key, center in zip(CONFLICT_ORDER, centers):
                ax.plot(
                    [center, center],
                    [bar_bottom - 0.04, tick_bottom],
                    color=ROW_LABEL_COLOR,
                    linewidth=0.6,
                    clip_on=False,
                    zorder=1,
                )
                horizontal_alignment = "center"
                label_x = center
                if key == "user":
                    horizontal_alignment = "right"
                elif key == "source":
                    horizontal_alignment = "left"
                elif key == "other_incorrect":
                    horizontal_alignment = "right"
                    label_x = 100.0
                ax.text(
                    label_x,
                    label_y,
                    CONFLICT_CALLOUTS[key],
                    ha=horizontal_alignment,
                    va="top",
                    fontsize=category_size,
                    fontfamily=REFERENCE_FONT,
                    color=ROW_LABEL_COLOR,
                    linespacing=0.92,
                    clip_on=False,
                )

    ax.set_xlim(-3, 108)
    ax.set_ylim(-2.42, y_positions[0] + 1.50)
    ax.set_xticks([])
    ax.text(
        0,
        endpoint_y,
        "0%",
        ha="left",
        va="top",
        fontsize=endpoint_size,
        fontfamily=REFERENCE_FONT,
        color=AXIS_TICK_COLOR,
        clip_on=False,
    )
    ax.text(
        100,
        endpoint_y,
        "100%",
        ha="right",
        va="top",
        fontsize=endpoint_size,
        fontfamily=REFERENCE_FONT,
        color=AXIS_TICK_COLOR,
        clip_on=False,
    )
    ax.set_yticks(y_positions, [str(row["model"]) for row in data])
    ax.tick_params(
        axis="y",
        labelsize=row_label_size,
        labelcolor=ROW_LABEL_COLOR,
        length=0,
        pad=8,
    )
    ax.grid(False)
    for spine_name in ("left", "top", "right", "bottom"):
        ax.spines[spine_name].set_visible(False)
    ax.set_axisbelow(True)

    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontfamily(REFERENCE_FONT)

    _save_fixed_pdf(fig, output)


def _model_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [],
            [],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=LINE_WIDTH,
            markersize=LEGEND_MARKER_SIZE,
            label=model,
        )
        for model in MODEL_ORDER
    ]


def _draw_model_legend(output: Path) -> None:
    fig = plt.figure(figsize=MODEL_LEGEND_SIZE)
    fig.legend(
        handles=_model_handles(),
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.05,
        handletextpad=0.25,
        columnspacing=0.45,
        borderaxespad=0,
    )
    _save_fixed_pdf(fig, output, transparent=True)


def _draw_conflict_legend(output: Path) -> None:
    handles = [
        mlines.Line2D(
            [],
            [],
            color=SOURCE_COLOR,
            marker="o",
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            label="Source-endorsed",
        ),
        mlines.Line2D(
            [],
            [],
            color=USER_COLOR,
            marker="o",
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            label="User-preferred",
        ),
    ]
    fig = plt.figure(figsize=CONFLICT_LEGEND_SIZE)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=0.40,
        handletextpad=0.15,
        columnspacing=0.25,
        borderaxespad=0,
    )
    _save_fixed_pdf(fig, output, transparent=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument(
        "--wide-conflict-output",
        type=Path,
        default=None,
        help="Optionally render a full-text-width direct-conflict panel.",
    )
    parser.add_argument(
        "--half-conflict-output",
        type=Path,
        default=None,
        help="Optionally render a direct-conflict panel for an even two-panel row.",
    )
    parser.add_argument(
        "--forty-conflict-output",
        type=Path,
        default=None,
        help="Optionally render a direct-conflict panel for a 40-percent-width slot.",
    )
    parser.add_argument(
        "--forty-five-conflict-output",
        type=Path,
        default=None,
        help="Optionally render a direct-conflict panel for a 45-percent-width slot.",
    )
    parser.add_argument(
        "--fifty-conflict-output",
        type=Path,
        default=None,
        help="Optionally render a direct-conflict panel for a 50-percent-width slot.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _configure_matplotlib()
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    identity, reliability, conflict = _load_primary_summary(args.summary_csv.resolve())
    outputs = {
        "claimed_source_identity": output_dir / "claimed_source_identity.pdf",
        "stated_source_reliability": output_dir / "stated_source_reliability.pdf",
        "direct_conflict": output_dir / "direct_conflict.pdf",
        "model_legend": output_dir / "model_legend.pdf",
        "conflict_legend": output_dir / "conflict_legend.pdf",
    }

    _draw_response_panel(
        identity,
        ("Unsupported\nuser", '"Professor"', "Relevant\nexpert"),
        outputs["claimed_source_identity"],
        x_label=None,
        show_y_label=False,
        x_positions=(0.0, 1.0, 2.0),
    )
    _draw_response_panel(
        reliability,
        ("20%", "50%", "90%"),
        outputs["stated_source_reliability"],
        x_label="Stated reliability",
        show_y_label=False,
    )
    _draw_direct_conflict(conflict, outputs["direct_conflict"])
    if args.wide_conflict_output is not None:
        _draw_direct_conflict(
            conflict,
            args.wide_conflict_output.resolve(),
            wide=True,
        )
    if args.half_conflict_output is not None:
        _draw_direct_conflict(
            conflict,
            args.half_conflict_output.resolve(),
            half=True,
        )
    if args.forty_conflict_output is not None:
        _draw_direct_conflict(
            conflict,
            args.forty_conflict_output.resolve(),
            forty=True,
        )
    if args.forty_five_conflict_output is not None:
        _draw_direct_conflict(
            conflict,
            args.forty_five_conflict_output.resolve(),
            forty_five=True,
        )
    if args.fifty_conflict_output is not None:
        _draw_direct_conflict(
            conflict,
            args.fifty_conflict_output.resolve(),
            fifty=True,
        )
    _draw_model_legend(outputs["model_legend"])
    _draw_conflict_legend(outputs["conflict_legend"])

    for output in outputs.values():
        print(output)
    if args.wide_conflict_output is not None:
        print(args.wide_conflict_output.resolve())
    if args.half_conflict_output is not None:
        print(args.half_conflict_output.resolve())
    if args.forty_conflict_output is not None:
        print(args.forty_conflict_output.resolve())
    if args.forty_five_conflict_output is not None:
        print(args.forty_five_conflict_output.resolve())
    if args.fifty_conflict_output is not None:
        print(args.fifty_conflict_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
