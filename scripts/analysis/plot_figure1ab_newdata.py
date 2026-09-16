#!/usr/bin/env python3
"""Render review versions of Figure 1(a)-(b) with the source-family data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_SUMMARY = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "mmlupro_epistemic_source_families_20260915"
    / "analysis"
    / "summary.csv"
)
DEFAULT_RELIABILITY_SUMMARY = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "mmlupro_source_credibility_correct81_single_multiturn_20260903"
    / "analysis"
    / "summary.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "pdf" / "figure1_ab_newdata_review"

MODEL_ORDER = (
    "GPT-5.6 Terra",
    "GPT-5.6 Luna",
    "Claude Opus 5",
    "Claude Sonnet 5",
)
MODEL_KEYS = {
    "GPT-5.6 Terra": "terra",
    "GPT-5.6 Luna": "luna",
    "Claude Opus 5": "opus5",
    "Claude Sonnet 5": "sonnet5",
}
MODEL_COLORS = {
    "GPT-5.6 Terra": "#609C80",
    "GPT-5.6 Luna": "#4D87C1",
    "Claude Opus 5": "#906DA1",
    "Claude Sonnet 5": "#DC5472",
}

PRIMARY_TEXT = "#1F2430"
AXIS_COLOR = "#5E6375"
BACKGROUND = "#FFFFFF"
# The manuscript loads LaTeX's ``times`` package.  macOS's Times face is the
# closest available plotting font, and the fallbacks preserve the same metrics
# on systems where it is unavailable.
FONT_FAMILY = "Times New Roman"
SOURCE_PANEL_SIZE = (4.20, 4.00)
# Size-specific canvases prevent LaTeX from shrinking already-small type in
# the one-row and half-width layouts.
COMPACT_SOURCE_PANEL_SIZE = (2.60, 2.66)
NARROW_SOURCE_PANEL_SIZE = (1.95, 2.66)
TWENTY_SEVEN_FIVE_SOURCE_PANEL_SIZE = (1.7875, 2.66)
QUARTER_SOURCE_PANEL_SIZE = (1.625, 2.66)
HALF_SOURCE_PANEL_SIZE = (3.18, 3.30)
RELIABILITY_PANEL_SIZE = (2.45, 3.40)
COMPACT_RELIABILITY_PANEL_SIZE = (1.75, 2.66)
NARROW_RELIABILITY_PANEL_SIZE = (1.95, 2.66)
TWENTY_SEVEN_FIVE_RELIABILITY_PANEL_SIZE = (1.7875, 2.66)
QUARTER_RELIABILITY_PANEL_SIZE = (1.625, 2.66)
HALF_RELIABILITY_PANEL_SIZE = (3.18, 3.30)
WIDE_SOURCE_PANEL_SIZE = (6.50, 2.35)
PROMPT_RED = "#D52F55"
PROMPT_TEXT = "#5E6375"

FAMILY_ORDER = (
    "unsupported_user",
    "individual_expert",
    "authoritative_reference",
    "independent_corroboration",
)
FAMILY_METRICS = {
    family: f"p_x_{family}" for family in FAMILY_ORDER
}
RELIABILITY_ORDER = ("reliable20", "reliable50", "reliable90")
RELIABILITY_METRICS = {
    "reliable20": "p_x_reliable20",
    "reliable50": "p_x_reliable50",
    "reliable90": "p_x_reliable90",
}


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _estimate(row: Mapping[str, str]) -> Estimate:
    return Estimate(
        value=100.0 * float(row["estimate"]),
        low=100.0 * float(row["ci_low"]),
        high=100.0 * float(row["ci_high"]),
    )


def load_source_families(
    path: Path,
) -> dict[str, tuple[Estimate, ...]]:
    rows = _read_csv(path)
    lookup = {(row["model_key"], row["metric"]): row for row in rows}
    return {
        model: tuple(
            _estimate(lookup[(MODEL_KEYS[model], FAMILY_METRICS[family])])
            for family in FAMILY_ORDER
        )
        for model in MODEL_ORDER
    }


def load_reliability(
    path: Path,
    *,
    analysis: str,
) -> dict[str, tuple[Estimate, ...]]:
    rows = [row for row in _read_csv(path) if row.get("analysis") == analysis]
    lookup = {(row["model_key"], row["metric"]): row for row in rows}
    return {
        model: tuple(
            _estimate(lookup[(MODEL_KEYS[model], RELIABILITY_METRICS[level])])
            for level in RELIABILITY_ORDER
        )
        for model in MODEL_ORDER
    }


def _configure_matplotlib() -> None:
    sns.set_style("white")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY, "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": PRIMARY_TEXT,
            "axes.unicode_minus": False,
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": PRIMARY_TEXT,
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
        }
    )


def _prompt_line(
    fragments: Sequence[tuple[str, str, str]],
    *,
    fontsize: float = 8.8,
) -> HPacker:
    """Build one left-aligned prompt line with selectively colored text."""
    children = []
    for text, color, weight in fragments:
        children.append(
            TextArea(
                text,
                textprops={
                    "color": color,
                    "fontfamily": FONT_FAMILY,
                    "fontsize": fontsize,
                    "fontstyle": "italic",
                    "fontweight": "semibold" if weight == "bold" else weight,
                },
            )
        )
    return HPacker(children=children, align="baseline", pad=0, sep=0)


def _add_family_prompt(
    ax: mpl.axes.Axes,
    *,
    y: float,
    x: float,
    title: str,
    prompt_lines: Sequence[Sequence[tuple[str, str, str]]],
    title_fontsize: float,
    prompt_fontsize: float,
) -> None:
    title_area = TextArea(
        title,
        textprops={
            "color": PRIMARY_TEXT,
            "fontfamily": FONT_FAMILY,
            "fontsize": title_fontsize,
            "fontweight": "semibold",
            "linespacing": 0.92,
        },
    )
    prompt_areas = [
        _prompt_line(line, fontsize=prompt_fontsize) for line in prompt_lines
    ]
    box = VPacker(
        children=[title_area, *prompt_areas],
        align="left",
        pad=0,
        sep=0.8,
    )
    annotation = AnnotationBbox(
        box,
        (x, y),
        xycoords=("axes fraction", "data"),
        box_alignment=(0.0, 0.5),
        frameon=False,
        pad=0,
        annotation_clip=False,
    )
    ax.add_artist(annotation)


def _add_reliability_prompt(
    fig: mpl.figure.Figure,
    *,
    fontsize: float,
    keep_answer_together: bool = False,
    placement: str = "top",
) -> None:
    first = TextArea(
        'e.g., “A source that is',
        textprops={
            "color": PROMPT_TEXT,
            "fontfamily": FONT_FAMILY,
            "fontsize": fontsize,
            "fontstyle": "italic",
        },
    )
    second = HPacker(
        children=[
            TextArea(
                "correct ",
                textprops={
                    "color": PROMPT_TEXT,
                    "fontfamily": FONT_FAMILY,
                    "fontsize": fontsize,
                    "fontstyle": "italic",
                },
            ),
            TextArea(
                "{p}",
                textprops={
                    "color": PROMPT_RED,
                    "fontfamily": FONT_FAMILY,
                    "fontsize": fontsize,
                    "fontstyle": "italic",
                    "fontweight": "semibold",
                },
            ),
            TextArea(
                "% of the time",
                textprops={
                    "color": PROMPT_TEXT,
                    "fontfamily": FONT_FAMILY,
                    "fontsize": fontsize,
                    "fontstyle": "italic",
                    "fontweight": "regular",
                },
            ),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    third = HPacker(
        children=[
            TextArea(
                "says the answer is ",
                textprops={
                    "color": PROMPT_TEXT,
                    "fontfamily": FONT_FAMILY,
                    "fontsize": fontsize,
                    "fontstyle": "italic",
                },
            ),
            TextArea(
                "b",
                textprops={
                    "color": PROMPT_TEXT,
                    "fontfamily": FONT_FAMILY,
                    "fontsize": fontsize,
                    "fontstyle": "italic",
                    "fontweight": "regular",
                },
            ),
            TextArea(
                ".”",
                textprops={
                    "color": PROMPT_TEXT,
                    "fontfamily": FONT_FAMILY,
                    "fontsize": fontsize,
                    "fontstyle": "italic",
                },
            ),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    prompt = VPacker(
        children=[first, second, third],
        align="left" if placement == "top_left" else "center",
        pad=0,
        sep=0.6,
    )
    if placement == "bottom":
        prompt_position = (0.68, 0.010)
        box_alignment = (0.5, 0.0)
    elif placement == "top":
        # Center compact prompts over the plotting region rather than over the
        # full canvas, whose left side is reserved for the y-axis title.
        prompt_position = (0.68 if keep_answer_together else 0.52, 0.982)
        box_alignment = (0.5, 1.0)
    elif placement == "top_left":
        # The axes begin at x=.38 and end at y=.84.  Placing the prompt's
        # lower-left corner here leaves approximately 6 pt above the axes.
        prompt_position = (0.38, 0.871)
        box_alignment = (0.0, 0.0)
    else:
        raise ValueError(f"Unknown prompt placement: {placement}")
    fig.add_artist(
        AnnotationBbox(
            prompt,
            prompt_position,
            xycoords=fig.transFigure,
            box_alignment=box_alignment,
            frameon=False,
            pad=0,
            annotation_clip=False,
        )
    )


def draw_source_families(
    data: Mapping[str, tuple[Estimate, ...]],
    output: Path,
    *,
    wide: bool = False,
    compact: bool = False,
    half: bool = False,
    narrow: bool = False,
    twenty_seven_five: bool = False,
    quarter: bool = False,
) -> None:
    if sum((wide, compact, half, narrow, twenty_seven_five, quarter)) > 1:
        raise ValueError("source-panel layout modes are mutually exclusive")
    fig, ax = plt.subplots(
        figsize=(
            WIDE_SOURCE_PANEL_SIZE
            if wide
            else COMPACT_SOURCE_PANEL_SIZE
            if compact
            else NARROW_SOURCE_PANEL_SIZE
            if narrow
            else TWENTY_SEVEN_FIVE_SOURCE_PANEL_SIZE
            if twenty_seven_five
            else QUARTER_SOURCE_PANEL_SIZE
            if quarter
            else HALF_SOURCE_PANEL_SIZE
            if half
            else SOURCE_PANEL_SIZE
        )
    )
    if wide:
        fig.subplots_adjust(left=0.245, right=0.982, top=0.97, bottom=0.24)
    elif compact:
        # The compact prompt column still needs enough real width for the
        # longest fixed template; keep its text clear of the y-axis.
        fig.subplots_adjust(left=0.49, right=0.985, top=0.98, bottom=0.21)
    elif narrow or twenty_seven_five or quarter:
        fig.subplots_adjust(left=0.54, right=0.985, top=0.98, bottom=0.22)
    elif half:
        fig.subplots_adjust(left=0.445, right=0.985, top=0.98, bottom=0.19)
    else:
        # The left column is part of the panel: it names each manipulation and
        # shows a concrete prompt example, following the cleaner reference.
        fig.subplots_adjust(
            left=0.42,
            right=0.985,
            top=0.98,
            bottom=0.18,
        )

    group_centers = np.asarray([3.60, 2.40, 1.20, 0.00])
    # Keep each four-model cluster compact, but give the bars enough visual
    # weight to remain the primary object after the panel is reduced to paper
    # size.  The CI caps are deliberately quieter than the bar bodies.
    offsets = np.asarray([0.36, 0.12, -0.12, -0.36])
    bar_height = 0.18

    for model_index, model in enumerate(MODEL_ORDER):
        estimates = data[model]
        values = np.asarray([item.value for item in estimates])
        lows = np.asarray([item.low for item in estimates])
        highs = np.asarray([item.high for item in estimates])
        y = group_centers + offsets[model_index]
        ax.barh(
            y,
            values,
            height=bar_height,
            color=MODEL_COLORS[model],
            edgecolor="none",
            alpha=1.0,
            zorder=3,
        )
        ax.errorbar(
            values,
            y,
            xerr=np.vstack((values - lows, highs - values)),
            fmt="none",
            ecolor=MODEL_COLORS[model],
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=4,
        )
        value_fontsize = (
            6.7
            if quarter
            else 7.0
            if twenty_seven_five
            else 7.4
            if narrow
            else 8.4
            if compact
            else 9.0
            if half
            else 10.0
        )
        for y_value, estimate in zip(y, estimates):
            ax.annotate(
                f"{estimate.value:.0f}",
                xy=(estimate.high, y_value),
                xytext=(2.5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=value_fontsize,
                fontweight="regular",
                fontstyle="normal",
                color=PRIMARY_TEXT,
                clip_on=False,
                annotation_clip=False,
            )

    if wide:
        ax.set_xlabel("Incorrect suggestion selected (%)", fontsize=11, labelpad=3)
    else:
        if quarter:
            ax.set_xlabel(
                "Selects suggested\nwrong answer (%)",
                fontsize=7.0,
                labelpad=3,
            )
        elif compact or narrow or twenty_seven_five:
            fig.text(
                0.755 if twenty_seven_five else 0.740 if narrow else 0.720,
                0.080,
                "Selects suggested\nwrong answer (%)"
                if twenty_seven_five
                else "Incorrect suggestion\nselected (%)",
                ha="center",
                va="center",
                fontsize=8.0 if twenty_seven_five else 8.5 if narrow else 9.5,
                linespacing=0.92,
                color=PRIMARY_TEXT,
            )
        elif half:
            fig.text(
                0.715,
                0.065,
                "Incorrect suggestion\nselected (%)",
                ha="center",
                va="center",
                fontsize=10.5,
                linespacing=0.92,
                color=PRIMARY_TEXT,
            )
        else:
            fig.text(
                0.705,
                0.050,
                "Selects suggested\nwrong answer (%)",
                ha="center",
                va="center",
                fontsize=11.0,
                linespacing=0.92,
                color=PRIMARY_TEXT,
            )

    ax.set_xlim(0, 110)
    ax.set_ylim(-0.55, 4.15)
    ax.set_xticks([0, 50, 100])
    if wide:
        ax.set_yticks(
            group_centers,
            (
                "Unsupported user",
                "Individual expert",
                "Authoritative\nreference",
                "Independent\ncorroboration",
            ),
        )
    else:
        ax.set_yticks([])
        family_prompts = (
            (
                "Unsupported user",
                (
                    (("e.g., “", PROMPT_TEXT, "regular"), ("{I believe}", PROMPT_RED, "bold")),
                    (("the answer is ", PROMPT_TEXT, "regular"), ("b", PROMPT_TEXT, "regular"), (".”", PROMPT_TEXT, "regular")),
                ),
            ),
            (
                "Individual expert",
                (
                    (("e.g., “", PROMPT_TEXT, "regular"), ("{An individual", PROMPT_RED, "bold")),
                    (("expert}", PROMPT_RED, "bold"), (" says the", PROMPT_TEXT, "regular")),
                    (("answer is ", PROMPT_TEXT, "regular"), ("b", PROMPT_TEXT, "regular"), (".”", PROMPT_TEXT, "regular")),
                ),
            ),
            (
                "Authoritative\nreference",
                (
                    (("e.g., “", PROMPT_TEXT, "regular"), ("{An authoritative", PROMPT_RED, "bold")),
                    (("reference}", PROMPT_RED, "bold"), (" says the", PROMPT_TEXT, "regular")),
                    (("answer is ", PROMPT_TEXT, "regular"), ("b", PROMPT_TEXT, "regular"), (".”", PROMPT_TEXT, "regular")),
                ),
            ),
            (
                "Independent\ncorroboration",
                (
                    (("e.g., “", PROMPT_TEXT, "regular"), ("{Multiple independent", PROMPT_RED, "bold")),
                    (("sources}", PROMPT_RED, "bold"), (" say the", PROMPT_TEXT, "regular")),
                    (("answer is ", PROMPT_TEXT, "regular"), ("b", PROMPT_TEXT, "regular"), (".”", PROMPT_TEXT, "regular")),
                ),
            ),
        )
        if narrow or twenty_seven_five or quarter:
            # The 30%-width panel needs one extra line here so the emphasized
            # phrase remains fully to the left of the y-axis.
            family_prompts = list(family_prompts)
            family_prompts[3] = (
                "Independent\ncorroboration",
                (
                    (("e.g., “", PROMPT_TEXT, "regular"), ("{Multiple", PROMPT_RED, "bold")),
                    (("independent sources}", PROMPT_RED, "bold"),),
                    (("say the answer is ", PROMPT_TEXT, "regular"), ("b", PROMPT_TEXT, "regular"), (".”", PROMPT_TEXT, "regular")),
                ),
            )
        if quarter:
            # At 25% paper width, deliberately shorter italic lines preserve
            # a clean gutter before the y-axis and prevent the axes background
            # from appearing to cover the prompt text.
            family_prompts = list(family_prompts)
            family_prompts[1] = (
                "Individual expert",
                (
                    (("e.g., “", PROMPT_TEXT, "regular"), ("{An", PROMPT_RED, "bold")),
                    (("individual expert}", PROMPT_RED, "bold"), (" says", PROMPT_TEXT, "regular")),
                    (("the answer is ", PROMPT_TEXT, "regular"), ("b", PROMPT_TEXT, "regular"), (".”", PROMPT_TEXT, "regular")),
                ),
            )
            family_prompts[2] = (
                "Authoritative\nreference",
                (
                    (("e.g., “", PROMPT_TEXT, "regular"), ("{An", PROMPT_RED, "bold")),
                    (("authoritative reference}", PROMPT_RED, "bold"),),
                    (("says the answer is ", PROMPT_TEXT, "regular"), ("b", PROMPT_TEXT, "regular"), (".”", PROMPT_TEXT, "regular")),
                ),
            )
            family_prompts[3] = (
                "Independent\ncorroboration",
                (
                    (("e.g., “", PROMPT_TEXT, "regular"), ("{Multiple", PROMPT_RED, "bold")),
                    (("independent sources}", PROMPT_RED, "bold"),),
                    (("say the answer is ", PROMPT_TEXT, "regular"), ("b", PROMPT_TEXT, "regular"), (".”", PROMPT_TEXT, "regular")),
                ),
            )
        title_fontsize = 7.5 if quarter else 7.7 if twenty_seven_five else 8.0 if narrow else 9.2 if compact else 9.8 if half else 10.5
        prompt_fontsize = 5.2 if quarter else 6.4 if twenty_seven_five else 6.8 if narrow else 7.4 if compact else 7.8 if half else 8.8
        prompt_x = -1.16 if (narrow or twenty_seven_five or quarter) else -0.98 if compact else -0.76 if half else -0.60
        for center, (title, prompt_lines) in zip(group_centers, family_prompts):
            _add_family_prompt(
                ax,
                y=float(center),
                x=prompt_x,
                title=title,
                prompt_lines=prompt_lines,
                title_fontsize=title_fontsize,
                prompt_fontsize=prompt_fontsize,
            )
    tick_fontsize = 7.3 if quarter else 8.0 if twenty_seven_five else 8.5 if narrow else 10.0 if compact else 10.8 if half else 11.5
    ax.tick_params(
        axis="x", labelsize=tick_fontsize, width=0.6, length=2.5, pad=2
    )
    ax.tick_params(axis="y", labelsize=12, width=0, length=0, pad=3)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def draw_reliability(
    data: Mapping[str, tuple[Estimate, ...]],
    output: Path,
    *,
    layout: str = "standard",
) -> None:
    if layout == "compact":
        panel_size = COMPACT_RELIABILITY_PANEL_SIZE
        margins = dict(left=0.35, right=0.97, top=0.76, bottom=0.27)
        prompt_fontsize = 7.4
        axis_fontsize = 9.5
        tick_fontsize = 9.5
        marker_size = 5.2
    elif layout == "half":
        panel_size = HALF_RELIABILITY_PANEL_SIZE
        margins = dict(left=0.25, right=0.98, top=0.76, bottom=0.23)
        prompt_fontsize = 8.8
        axis_fontsize = 10.5
        tick_fontsize = 10.8
        marker_size = 5.8
    elif layout == "standard":
        panel_size = RELIABILITY_PANEL_SIZE
        margins = dict(left=0.31, right=0.97, top=0.76, bottom=0.25)
        prompt_fontsize = 8.6
        axis_fontsize = 10.8
        tick_fontsize = 11.0
        marker_size = 5.8
    else:
        raise ValueError(f"Unknown reliability layout: {layout}")

    fig, ax = plt.subplots(figsize=panel_size)
    fig.subplots_adjust(**margins)
    _add_reliability_prompt(fig, fontsize=prompt_fontsize)

    x = np.arange(3, dtype=float)
    for model in MODEL_ORDER:
        estimates = data[model]
        values = np.asarray([item.value for item in estimates])
        lows = np.asarray([item.low for item in estimates])
        highs = np.asarray([item.high for item in estimates])
        ax.errorbar(
            x,
            values,
            yerr=np.vstack((values - lows, highs - values)),
            color=MODEL_COLORS[model],
            marker="o",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor=MODEL_COLORS[model],
            markersize=marker_size,
            linewidth=2.0,
            elinewidth=1.0,
            capsize=2.7,
            capthick=1.0,
            zorder=3,
        )

    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0, 55)
    ax.set_xticks(x, ("20%", "50%", "90%"))
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel(
        "Stated source\nreliability", fontsize=axis_fontsize, labelpad=2
    )
    ax.set_ylabel(
        "Suggested wrong\nanswer\nselected (%)",
        fontsize=axis_fontsize,
        labelpad=4,
    )
    ax.tick_params(
        axis="both", labelsize=tick_fontsize, width=0.75, length=3.5, pad=2
    )
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def draw_reliability_bar(
    data: Mapping[str, tuple[Estimate, ...]],
    output: Path,
    *,
    layout: str = "standard",
) -> None:
    """Render a grouped-bar alternative for the source-reliability panel."""
    if layout == "compact":
        panel_size = COMPACT_RELIABILITY_PANEL_SIZE
        margins = dict(left=0.36, right=0.98, top=0.76, bottom=0.26)
        prompt_fontsize = 7.1
        axis_fontsize = 8.5
        tick_fontsize = 8.5
        capsize = 1.9
    elif layout == "narrow":
        panel_size = NARROW_RELIABILITY_PANEL_SIZE
        margins = dict(left=0.34, right=0.98, top=0.76, bottom=0.26)
        prompt_fontsize = 7.5
        axis_fontsize = 8.8
        tick_fontsize = 9.0
        capsize = 2.0
    elif layout == "twenty_seven_five":
        panel_size = TWENTY_SEVEN_FIVE_RELIABILITY_PANEL_SIZE
        margins = dict(left=0.36, right=0.98, top=0.76, bottom=0.26)
        prompt_fontsize = 7.0
        axis_fontsize = 8.2
        tick_fontsize = 8.3
        capsize = 1.9
    elif layout == "quarter":
        panel_size = QUARTER_RELIABILITY_PANEL_SIZE
        # Match panel (a)'s 22% plot baseline.  The prompt occupies the compact
        # header above the axes, leaving the shared bottom edge uncluttered.
        margins = dict(left=0.38, right=0.98, top=0.84, bottom=0.22)
        prompt_fontsize = 5.2
        axis_fontsize = 7.0
        tick_fontsize = 7.3
        capsize = 2.0
    elif layout == "half":
        panel_size = HALF_RELIABILITY_PANEL_SIZE
        margins = dict(left=0.25, right=0.98, top=0.76, bottom=0.22)
        prompt_fontsize = 8.8
        axis_fontsize = 10.5
        tick_fontsize = 10.5
        capsize = 2.4
    elif layout == "standard":
        panel_size = RELIABILITY_PANEL_SIZE
        margins = dict(left=0.31, right=0.98, top=0.76, bottom=0.25)
        prompt_fontsize = 8.6
        axis_fontsize = 10.0
        tick_fontsize = 10.0
        capsize = 2.2
    else:
        raise ValueError(f"Unknown bar-reliability layout: {layout}")

    fig, ax = plt.subplots(figsize=panel_size)
    fig.subplots_adjust(**margins)
    _add_reliability_prompt(
        fig,
        fontsize=prompt_fontsize,
        keep_answer_together=layout in {"quarter", "twenty_seven_five"},
        placement="top_left" if layout == "quarter" else "top",
    )

    x = np.arange(3, dtype=float)
    bar_width = 0.18
    offsets = np.asarray([-0.285, -0.095, 0.095, 0.285])
    for model, offset in zip(MODEL_ORDER, offsets):
        estimates = data[model]
        values = np.asarray([item.value for item in estimates])
        lows = np.asarray([item.low for item in estimates])
        highs = np.asarray([item.high for item in estimates])
        positions = x + offset
        ax.bar(
            positions,
            values,
            width=bar_width,
            color=MODEL_COLORS[model],
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
        ax.errorbar(
            positions,
            values,
            yerr=np.vstack((values - lows, highs - values)),
            fmt="none",
            ecolor=MODEL_COLORS[model],
            elinewidth=0.7,
            capsize=capsize,
            capthick=0.7,
            zorder=3,
        )

    ax.set_xlim(-0.52, 2.52)
    ax.set_ylim(0, 55)
    ax.set_xticks(x, ("20%", "50%", "90%"))
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel(
        "Stated source reliability"
        if layout == "quarter"
        else "Stated source\nreliability",
        fontsize=axis_fontsize,
        labelpad=3,
    )
    ax.set_ylabel(
        "Selects suggested\nwrong answer (%)",
        fontsize=axis_fontsize,
        labelpad=4,
    )
    ax.tick_params(
        axis="both", labelsize=tick_fontsize, width=0.6, length=2.5, pad=2
    )
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = output.suffix.lower().lstrip(".") or "png"
    fig.savefig(output, format=output_format, dpi=300)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument(
        "--reliability-summary",
        type=Path,
        default=DEFAULT_RELIABILITY_SUMMARY,
    )
    parser.add_argument(
        "--reliability-analysis",
        choices=("single_turn", "multi_turn", "primary"),
        default="primary",
        help="Use the published primary 50/50 single-/multi-turn estimates by default.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--wide-source-output",
        type=Path,
        default=None,
        help="Optionally render panel A at the full ICLR text width.",
    )
    parser.add_argument(
        "--compact-source-output",
        type=Path,
        default=None,
        help="Optionally render a compact prompt-at-left panel A for a three-panel row.",
    )
    parser.add_argument(
        "--half-source-output",
        type=Path,
        default=None,
        help="Optionally render panel A for an even two-panel row.",
    )
    parser.add_argument(
        "--narrow-source-output",
        type=Path,
        default=None,
        help="Optionally render panel A for a 30-percent-width row slot.",
    )
    parser.add_argument(
        "--source-275-output",
        type=Path,
        default=None,
        help="Optionally render panel A for a 27.5-percent-width row slot.",
    )
    parser.add_argument(
        "--source-25-output",
        type=Path,
        default=None,
        help="Optionally render panel A for a 25-percent-width row slot.",
    )
    parser.add_argument(
        "--compact-reliability-output",
        type=Path,
        default=None,
        help="Optionally render panel B at its one-row publication size.",
    )
    parser.add_argument(
        "--half-reliability-output",
        type=Path,
        default=None,
        help="Optionally render panel B for an even two-panel row.",
    )
    parser.add_argument(
        "--bar-reliability-output",
        type=Path,
        default=None,
        help="Optionally render a grouped-bar alternative for panel B.",
    )
    parser.add_argument(
        "--compact-bar-reliability-output",
        type=Path,
        default=None,
        help="Optionally render a compact grouped-bar panel B.",
    )
    parser.add_argument(
        "--half-bar-reliability-output",
        type=Path,
        default=None,
        help="Optionally render a half-width grouped-bar panel B.",
    )
    parser.add_argument(
        "--narrow-bar-reliability-output",
        type=Path,
        default=None,
        help="Optionally render a grouped-bar panel B for a 30-percent-width slot.",
    )
    parser.add_argument(
        "--bar-reliability-275-output",
        type=Path,
        default=None,
        help="Optionally render grouped-bar panel B for a 27.5-percent-width slot.",
    )
    parser.add_argument(
        "--bar-reliability-25-output",
        type=Path,
        default=None,
        help="Optionally render grouped-bar panel B for a 25-percent-width slot.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _configure_matplotlib()
    args = parse_args(argv)
    source_data = load_source_families(args.source_summary.resolve())
    reliability_data = load_reliability(
        args.reliability_summary.resolve(),
        analysis=args.reliability_analysis,
    )
    output_dir = args.output_dir.resolve()
    outputs = (
        output_dir / "figure1a_source_families.pdf",
        output_dir / "figure1b_source_reliability.pdf",
    )
    draw_source_families(source_data, outputs[0])
    draw_reliability(reliability_data, outputs[1])
    if args.wide_source_output is not None:
        draw_source_families(
            source_data,
            args.wide_source_output.resolve(),
            wide=True,
        )
    if args.compact_source_output is not None:
        draw_source_families(
            source_data,
            args.compact_source_output.resolve(),
            compact=True,
        )
    if args.half_source_output is not None:
        draw_source_families(
            source_data,
            args.half_source_output.resolve(),
            half=True,
        )
    if args.narrow_source_output is not None:
        draw_source_families(
            source_data,
            args.narrow_source_output.resolve(),
            narrow=True,
        )
    if args.source_275_output is not None:
        draw_source_families(
            source_data,
            args.source_275_output.resolve(),
            twenty_seven_five=True,
        )
    if args.source_25_output is not None:
        draw_source_families(
            source_data,
            args.source_25_output.resolve(),
            quarter=True,
        )
    if args.compact_reliability_output is not None:
        draw_reliability(
            reliability_data,
            args.compact_reliability_output.resolve(),
            layout="compact",
        )
    if args.half_reliability_output is not None:
        draw_reliability(
            reliability_data,
            args.half_reliability_output.resolve(),
            layout="half",
        )
    if args.bar_reliability_output is not None:
        draw_reliability_bar(
            reliability_data,
            args.bar_reliability_output.resolve(),
        )
    if args.compact_bar_reliability_output is not None:
        draw_reliability_bar(
            reliability_data,
            args.compact_bar_reliability_output.resolve(),
            layout="compact",
        )
    if args.half_bar_reliability_output is not None:
        draw_reliability_bar(
            reliability_data,
            args.half_bar_reliability_output.resolve(),
            layout="half",
        )
    if args.narrow_bar_reliability_output is not None:
        draw_reliability_bar(
            reliability_data,
            args.narrow_bar_reliability_output.resolve(),
            layout="narrow",
        )
    if args.bar_reliability_275_output is not None:
        draw_reliability_bar(
            reliability_data,
            args.bar_reliability_275_output.resolve(),
            layout="twenty_seven_five",
        )
    if args.bar_reliability_25_output is not None:
        draw_reliability_bar(
            reliability_data,
            args.bar_reliability_25_output.resolve(),
            layout="quarter",
        )
    for output in outputs:
        print(output)
    if args.wide_source_output is not None:
        print(args.wide_source_output.resolve())
    if args.compact_source_output is not None:
        print(args.compact_source_output.resolve())
    if args.half_source_output is not None:
        print(args.half_source_output.resolve())
    if args.narrow_source_output is not None:
        print(args.narrow_source_output.resolve())
    if args.source_275_output is not None:
        print(args.source_275_output.resolve())
    if args.source_25_output is not None:
        print(args.source_25_output.resolve())
    if args.compact_reliability_output is not None:
        print(args.compact_reliability_output.resolve())
    if args.half_reliability_output is not None:
        print(args.half_reliability_output.resolve())
    if args.bar_reliability_output is not None:
        print(args.bar_reliability_output.resolve())
    if args.compact_bar_reliability_output is not None:
        print(args.compact_bar_reliability_output.resolve())
    if args.half_bar_reliability_output is not None:
        print(args.half_bar_reliability_output.resolve())
    if args.narrow_bar_reliability_output is not None:
        print(args.narrow_bar_reliability_output.resolve())
    if args.bar_reliability_275_output is not None:
        print(args.bar_reliability_275_output.resolve())
    if args.bar_reliability_25_output is not None:
        print(args.bar_reliability_25_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
