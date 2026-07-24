#!/usr/bin/env python3
"""Build a meeting-ready, explicitly preliminary pruning micro-pilot summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TEAL = "#73b3ab"
ORANGE = "#d4651a"
GRAY = "#8c8c8c"
RED = "#b14a4a"
LABEL = (
    "Preliminary six-layer, seed-5, N=8 weight-pruning micro-pilot — "
    "not the preregistered full experiment"
)
STRONG_CONDITION = "incorrect_suggestion_strong"


def _read_metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return dict(value)


def _metric_map(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if frame.empty or "metric" not in frame or "estimate" not in frame:
        raise ValueError(f"metric summary is missing metric/estimate columns: {path}")
    duplicates = frame["metric"].astype(str).duplicated(keep=False)
    if bool(duplicates.any()):
        names = sorted(frame.loc[duplicates, "metric"].astype(str).unique())
        raise ValueError(f"metric summary contains duplicate metrics: {names}")
    return {
        str(row.metric): float(row.estimate)
        for row in frame.itertuples(index=False)
    }


def _json_metrics(metrics: Mapping[str, float]) -> dict[str, float | None]:
    return {
        str(name): float(value) if math.isfinite(float(value)) else None
        for name, value in metrics.items()
    }


def _strict_transition_summary(paired_path: Path) -> dict[str, float | int]:
    frame = pd.read_csv(paired_path)
    strong = frame[frame["condition"].astype(str).eq(STRONG_CONDITION)].copy()
    if strong.empty:
        raise ValueError(f"no {STRONG_CONDITION!r} rows in {paired_path}")
    strict = strong[
        strong["baseline_neutral_status_category"].eq("valid")
        & strong["baseline_biased_status_category"].eq("valid")
        & strong["baseline_neutral_choice"].eq(strong["correct_letter"])
        & strong["baseline_biased_choice"].eq(strong["suggested_letter"])
    ].copy()
    if strict.empty:
        raise ValueError(
            f"no live q=0 strict flips remain in the preliminary cohort: {paired_path}"
        )
    candidate_status = strict["candidate_biased_status_category"].astype(str)
    candidate_choice = strict["candidate_biased_choice"].astype(str)
    correct = strict["correct_letter"].astype(str)
    suggested = strict["suggested_letter"].astype(str)
    valid = candidate_status.eq("valid")
    transitions = {
        "b_to_c": valid & candidate_choice.eq(correct),
        "b_to_b": valid & candidate_choice.eq(suggested),
        "b_to_other_wrong": (
            valid
            & ~candidate_choice.eq(correct)
            & ~candidate_choice.eq(suggested)
        ),
        "b_to_malformed": candidate_status.eq("malformed"),
        "b_to_invalid_or_refusal": candidate_status.isin(["invalid", "refusal"]),
    }
    count = len(strict)
    baseline_other = (
        1.0
        - strict["baseline_p_biased_c"].astype(float)
        - strict["baseline_p_biased_b"].astype(float)
    )
    candidate_other = (
        1.0
        - strict["candidate_p_biased_c"].astype(float)
        - strict["candidate_p_biased_b"].astype(float)
    )
    result: dict[str, float | int] = {
        "n_live_baseline_strict_flips": int(count),
        "delta_p_c": float(
            (
                strict["candidate_p_biased_c"].astype(float)
                - strict["baseline_p_biased_c"].astype(float)
            ).mean()
        ),
        "delta_p_b": float(
            (
                strict["candidate_p_biased_b"].astype(float)
                - strict["baseline_p_biased_b"].astype(float)
            ).mean()
        ),
        "delta_other_probability": float((candidate_other - baseline_other).mean()),
        "delta_nonvalid_rate": float((~valid).mean()),
    }
    for name, values in transitions.items():
        result[f"{name}_count"] = int(values.sum())
        result[f"{name}_rate"] = float(values.mean())
    return result


def _finite_metric(metrics: Mapping[str, float], name: str) -> float:
    value = float(metrics.get(name, float("nan")))
    if not math.isfinite(value):
        raise ValueError(f"required preliminary metric {name!r} is not finite")
    return value


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _transition_plot(summary: Mapping[str, float | int], output_dir: Path) -> None:
    rows = pd.DataFrame(
        {
            "Transition": [
                "b → c",
                "b → b",
                "b → other wrong",
                "b → malformed",
                "b → invalid/refusal",
            ],
            "Rate": [
                summary["b_to_c_rate"],
                summary["b_to_b_rate"],
                summary["b_to_other_wrong_rate"],
                summary["b_to_malformed_rate"],
                summary["b_to_invalid_or_refusal_rate"],
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    sns.barplot(
        data=rows,
        x="Transition",
        y="Rate",
        hue="Transition",
        palette=[TEAL, ORANGE, GRAY, RED, "#6c5b7b"],
        legend=False,
        ax=axis,
    )
    axis.set_title("What happens to the user-backed wrong answer after pruning?", fontsize=20)
    axis.set_xlabel("Answer transition on live baseline strict flips", fontsize=15)
    axis.set_ylabel("Fraction of examples", fontsize=15)
    axis.tick_params(axis="both", labelsize=12)
    axis.set_ylim(0.0, 1.0)
    axis.text(
        0.5,
        -0.26,
        LABEL,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    _save_figure(figure, output_dir, "answer_transitions")


def _probability_plot(summary: Mapping[str, float | int], output_dir: Path) -> None:
    rows = pd.DataFrame(
        {
            "Metric": [
                "Δ P(correct)",
                "Δ P(suggested wrong)",
                "Δ P(other)",
                "Δ non-valid rate",
            ],
            "Change": [
                summary["delta_p_c"],
                summary["delta_p_b"],
                summary["delta_other_probability"],
                summary["delta_nonvalid_rate"],
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    colors = [TEAL if value >= 0 else ORANGE for value in rows["Change"]]
    sns.barplot(
        data=rows,
        x="Metric",
        y="Change",
        hue="Metric",
        palette=colors,
        legend=False,
        ax=axis,
    )
    axis.axhline(0.0, color="#333333", linewidth=1)
    axis.set_title("Does pruning restore the correct answer rather than damage output?", fontsize=20)
    axis.set_xlabel("", fontsize=15)
    axis.set_ylabel("Mean change from live base model", fontsize=15)
    axis.tick_params(axis="both", labelsize=12)
    axis.text(
        0.5,
        -0.26,
        LABEL,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    _save_figure(figure, output_dir, "probability_changes")


def _preservation_plot(metrics: Mapping[str, float], output_dir: Path) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "Metric": [
                "Neutral accuracy",
                "Neutral P(correct)",
                "Corrective accuracy",
                "Correct-suggestion agreement",
            ],
            "Change": [
                _finite_metric(metrics, "neutral_accuracy_change"),
                _finite_metric(metrics, "neutral_p_c_change"),
                _finite_metric(metrics, "strong_biased_accuracy_change"),
                _finite_metric(metrics, "correct_suggestion_agreement_change"),
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    sns.pointplot(
        data=rows,
        x="Change",
        y="Metric",
        color=TEAL,
        linestyle="none",
        markers="o",
        markersize=9,
        ax=axis,
    )
    axis.axvline(0.0, color="#333333", linewidth=1)
    axis.set_title("Preservation checks in the preliminary held-out cohort", fontsize=20)
    axis.set_xlabel("Change from live base model", fontsize=15)
    axis.set_ylabel("", fontsize=15)
    axis.tick_params(axis="both", labelsize=12)
    axis.text(
        0.5,
        -0.22,
        LABEL,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    _save_figure(figure, output_dir, "preservation_deltas")
    return rows


def _control_plot(
    targeted: Mapping[str, float | int],
    random_control: Mapping[str, float | int],
    output_dir: Path,
) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "Intervention": ["Targeted attribution mask", "Magnitude-matched random"],
            "b → c recovery": [
                targeted["b_to_c_rate"],
                random_control["b_to_c_rate"],
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    sns.barplot(
        data=rows,
        x="Intervention",
        y="b → c recovery",
        hue="Intervention",
        palette=[TEAL, ORANGE],
        legend=False,
        ax=axis,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Is recovery specific to the selected weights?", fontsize=20)
    axis.set_xlabel("", fontsize=15)
    axis.set_ylabel("Live b → c recovery rate", fontsize=15)
    axis.tick_params(axis="both", labelsize=12)
    axis.text(
        0.5,
        -0.24,
        LABEL,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    _save_figure(figure, output_dir, "targeted_vs_random")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targeted-evaluation-dir", type=Path, required=True)
    parser.add_argument("--random-evaluation-dir", type=Path)
    parser.add_argument("--targeted-mask-metadata", type=Path, required=True)
    parser.add_argument("--random-mask-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    targeted_dir = args.targeted_evaluation_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("white")

    targeted = _strict_transition_summary(targeted_dir / "paired_items.csv")
    targeted_metrics = _metric_map(targeted_dir / "metric_summary.csv")
    targeted_mask = _read_metadata(args.targeted_mask_metadata.expanduser().resolve())
    has_random = (
        args.random_evaluation_dir is not None
        and args.random_mask_metadata is not None
    )
    if (args.random_evaluation_dir is None) != (args.random_mask_metadata is None):
        raise ValueError(
            "--random-evaluation-dir and --random-mask-metadata must be supplied together"
        )
    random_control = None
    random_metrics = None
    random_mask = None
    if has_random:
        random_dir = args.random_evaluation_dir.expanduser().resolve()
        random_control = _strict_transition_summary(random_dir / "paired_items.csv")
        random_metrics = _metric_map(random_dir / "metric_summary.csv")
        random_mask = _read_metadata(args.random_mask_metadata.expanduser().resolve())

    _transition_plot(targeted, output_dir)
    _probability_plot(targeted, output_dir)
    preservation = _preservation_plot(targeted_metrics, output_dir)
    controls = None
    if random_control is not None:
        controls = _control_plot(targeted, random_control, output_dir)

    transition_table = pd.DataFrame(
        [
            {
                "transition": name,
                "count": targeted[f"{name}_count"],
                "rate": targeted[f"{name}_rate"],
            }
            for name in (
                "b_to_c",
                "b_to_b",
                "b_to_other_wrong",
                "b_to_malformed",
                "b_to_invalid_or_refusal",
            )
        ]
    )
    transition_table.to_csv(output_dir / "answer_transitions.csv", index=False)
    preservation.to_csv(output_dir / "preservation_deltas.csv", index=False)
    if controls is not None:
        controls.to_csv(output_dir / "targeted_vs_random.csv", index=False)

    summary = {
        "label": LABEL,
        "scope": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "calibration_seed": 5,
            "calibration_rows_per_role": 8,
            "transformer_blocks": [3, 8, 13, 18, 23, 27],
            "p": 1e-5,
            "q": 5e-5,
            "primary_format": "raw",
            "loss": "completion_nll",
        },
        "targeted": {
            **targeted,
            "actual_mask_count": int(targeted_mask["surviving_count"]),
            "metrics": _json_metrics(targeted_metrics),
        },
        "random_magnitude": None,
    }
    if random_control is not None and random_mask is not None and random_metrics is not None:
        summary["random_magnitude"] = {
            **random_control,
            "actual_mask_count": int(random_mask["surviving_count"]),
            "metrics": _json_metrics(random_metrics),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Preliminary weight-pruning micro-pilot",
        "",
        f"> {LABEL}.",
        "",
        f"- Live baseline strict flips evaluated: {targeted['n_live_baseline_strict_flips']}",
        (
            "- Targeted mask: "
            f"{targeted['b_to_c_count']} b→c, "
            f"{targeted['b_to_b_count']} b→b, "
            f"{targeted['b_to_other_wrong_count']} b→other wrong, "
            f"{targeted['b_to_malformed_count']} malformed, "
            f"{targeted['b_to_invalid_or_refusal_count']} invalid/refusal."
        ),
        (
            "- Mean probability movement on those flips: "
            f"ΔP(c)={float(targeted['delta_p_c']):+.4f}, "
            f"ΔP(b)={float(targeted['delta_p_b']):+.4f}."
        ),
    ]
    if random_control is not None and random_mask is not None:
        lines.append(
            "- b→c recovery: "
            f"targeted={float(targeted['b_to_c_rate']):.1%}, "
            f"magnitude-matched random={float(random_control['b_to_c_rate']):.1%}."
        )
        lines.append(
            "- Actual mask counts: "
            f"targeted={int(targeted_mask['surviving_count']):,}, "
            f"random={int(random_mask['surviving_count']):,}."
        )
    else:
        lines.extend(
            [
                f"- b→c recovery: targeted={float(targeted['b_to_c_rate']):.1%}.",
                f"- Actual targeted-mask count: {int(targeted_mask['surviving_count']):,}.",
            ]
        )
    lines.extend(
        [
            "",
            "These results are useful for a directional meeting check only. They do not replace "
            "the full two-model, three-seed, all-layer, validation-selected experiment.",
        ]
    )
    (output_dir / "meeting_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
