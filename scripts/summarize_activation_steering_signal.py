#!/usr/bin/env python3
"""Summarize the lean exploratory W-N signal experiment."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _paired_differences(
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    field: str,
) -> list[float]:
    positive_by_question = {
        str(row["stable_question_key"]): float(row[field]) for row in positive
    }
    negative_by_question = {
        str(row["stable_question_key"]): float(row[field]) for row in negative
    }
    if set(positive_by_question) != set(negative_by_question):
        raise ValueError(
            "Paired summary inputs do not contain identical stable question keys."
        )
    keys = sorted(positive_by_question)
    return [
        positive_by_question[key] - negative_by_question[key] for key in keys
    ]


def _paired_mean_absolute_damage(
    zero: list[dict[str, Any]],
    doses: list[dict[str, Any]],
    field: str,
) -> float | None:
    zero_by_question = {
        str(row["stable_question_key"]): float(row[field]) for row in zero
    }
    missing = sorted(
        {
            str(row["stable_question_key"])
            for row in doses
            if str(row["stable_question_key"]) not in zero_by_question
        }
    )
    if missing:
        raise ValueError(
            "Dose rows are missing alpha-zero pairs for stable question keys: "
            + ", ".join(missing[:5])
        )
    values = [
        abs(float(row[field]) - zero_by_question[str(row["stable_question_key"])])
        for row in doses
    ]
    return mean(values) if values else None


def _paired_bootstrap_summary(
    values: list[float],
    *,
    seed: int,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    if not values:
        return {
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "n_questions": 0,
        }
    rng = random.Random(int(seed))
    count = len(values)
    estimates = sorted(
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(int(n_bootstrap))
    )
    return {
        "mean": mean(values),
        "ci_low": estimates[int(0.025 * len(estimates))],
        "ci_high": estimates[int(0.975 * len(estimates)) - 1],
        "n_questions": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected_layers = {
        str(row["model_name"]): int(row["selected_layer"])
        for row in selection["selections"]
    }
    rows = _read_rows(args.input)
    results: list[dict[str, Any]] = []
    for model_index, (model_name, layer) in enumerate(
        sorted(selected_layers.items())
    ):
        model_rows = [
            row
            for row in rows
            if row.get("model_name") == model_name
            and int(row.get("layer", -1)) == layer
            and row.get("scoring_mode") == "strict_choice"
        ]
        learned = [
            row
            for row in model_rows
            if row.get("treatment_type") == "learned"
            and row.get("direction_name") == "wn"
            and row.get("scale_convention") == "native"
            and row.get("probe_correct_rank") is None
        ]
        available = {float(row["alpha"]) for row in learned}
        magnitudes = sorted(
            {abs(value) for value in available if value > 0 and -value in available}
        )
        magnitude = max(magnitudes) if magnitudes else 0.0
        wrong_pos = [
            row
            for row in learned
            if row.get("condition") == "incorrect_suggestion"
            and float(row["alpha"]) == magnitude
        ]
        wrong_neg = [
            row
            for row in learned
            if row.get("condition") == "incorrect_suggestion"
            and float(row["alpha"]) == -magnitude
        ]
        neutral_zero = [
            row
            for row in learned
            if row.get("condition") == "neutral"
            and float(row["alpha"]) == 0.0
        ]
        neutral_pos = [
            row
            for row in learned
            if row.get("condition") == "neutral"
            and float(row["alpha"]) == magnitude
        ]
        neutral_neg = [
            row
            for row in learned
            if row.get("condition") == "neutral"
            and float(row["alpha"]) == -magnitude
        ]
        signed_effect_summary = _paired_bootstrap_summary(
            _paired_differences(wrong_pos, wrong_neg, "p_endorsed"),
            seed=20260726 + model_index,
        )
        alpha_one_positive = [
            row
            for row in learned
            if row.get("condition") == "incorrect_suggestion"
            and float(row["alpha"]) == 1.0
        ]
        alpha_one_negative = [
            row
            for row in learned
            if row.get("condition") == "incorrect_suggestion"
            and float(row["alpha"]) == -1.0
        ]
        alpha_one_summary = _paired_bootstrap_summary(
            _paired_differences(
                alpha_one_positive,
                alpha_one_negative,
                "p_endorsed",
            ),
            seed=20260736 + model_index,
        )
        neutral_probability_damage = _paired_mean_absolute_damage(
            neutral_zero,
            neutral_pos + neutral_neg,
            "p_correct",
        )
        neutral_accuracy_damage = _paired_mean_absolute_damage(
            neutral_zero,
            neutral_pos + neutral_neg,
            "is_correct",
        )
        dataset_effects = {}
        for dataset in sorted({str(row["dataset"]) for row in wrong_pos}):
            dataset_effects[dataset] = _paired_bootstrap_summary(
                _paired_differences(
                    [
                        row
                        for row in wrong_pos
                        if str(row["dataset"]) == dataset
                    ],
                    [
                        row
                        for row in wrong_neg
                        if str(row["dataset"]) == dataset
                    ],
                    "p_endorsed",
                ),
                seed=20260746 + model_index,
            )
        control_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in model_rows:
            if row.get("treatment_type") == "control":
                control_groups[
                    (str(row["direction_name"]), int(row["control_seed"]))
                ].append(row)
        control_scores: list[dict[str, Any]] = []
        for (name, seed), control_rows in sorted(control_groups.items()):
            positive = [
                row
                for row in control_rows
                if row.get("condition") == "incorrect_suggestion"
                and float(row["alpha"]) == magnitude
            ]
            negative = [
                row
                for row in control_rows
                if row.get("condition") == "incorrect_suggestion"
                and float(row["alpha"]) == -magnitude
            ]
            pos = _mean(positive, "p_endorsed")
            neg = _mean(negative, "p_endorsed")
            if pos is not None and neg is not None:
                control_scores.append(
                    {"direction_name": name, "seed": seed, "signed_effect": pos - neg}
                )
        probe_rows = [
            row
            for row in model_rows
            if row.get("probe_correct_rank") is not None
            and row.get("treatment_type") == "learned"
            and row.get("direction_name") == "wn"
        ]
        probe_by_alpha = []
        for alpha in (-4.0, 0.0, 4.0):
            dose = [row for row in probe_rows if float(row["alpha"]) == alpha]
            probe_by_alpha.append(
                {
                    "alpha": alpha,
                    "correct_top1_rate": _mean(dose, "probe_correct_top1"),
                    "correct_minus_endorsed_margin": _mean(
                        dose, "probe_margin_correct_minus_endorsed"
                    ),
                    "n_rows": len(dose),
                }
            )
        max_control = max(
            (row["signed_effect"] for row in control_scores),
            default=None,
        )
        results.append(
            {
                "model_name": model_name,
                "selected_layer": layer,
                "comparison_magnitude": magnitude,
                "signed_wrong_pressure_effect": signed_effect_summary["mean"],
                "signed_wrong_pressure_effect_ci_low": (
                    signed_effect_summary["ci_low"]
                ),
                "signed_wrong_pressure_effect_ci_high": (
                    signed_effect_summary["ci_high"]
                ),
                "signed_wrong_pressure_effect_n_questions": (
                    signed_effect_summary["n_questions"]
                ),
                "alpha_1_signed_wrong_pressure_effect": alpha_one_summary,
                "mean_neutral_probability_damage": neutral_probability_damage,
                "mean_neutral_accuracy_damage": neutral_accuracy_damage,
                "per_dataset_signed_effect_at_comparison_magnitude": (
                    dataset_effects
                ),
                "max_control_signed_effect": max_control,
                "effect_exceeds_all_sampled_controls": (
                    signed_effect_summary["mean"] is not None
                    and max_control is not None
                    and signed_effect_summary["mean"] > max_control
                ),
                "effect_minus_max_sampled_control": (
                    signed_effect_summary["mean"] - max_control
                    if signed_effect_summary["mean"] is not None
                    and max_control is not None
                    else None
                ),
                "control_scores": control_scores,
                "fixed_probe": probe_by_alpha,
            }
        )
    payload = {
        "study_scope": "exploratory_benchmark_label_signal_v1_20260726",
        "human_semantic_approval": False,
        "confirmatory_claim_allowed": False,
        "selection_uses_test_results": False,
        "models": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# Exploratory prompt-only W−N signal",
        "",
        "Benchmark answer keys define wrongness; this is not a human-semantic confirmatory result.",
        "",
    ]
    for row in results:
        markdown.extend(
            [
                f"## {row['model_name']}",
                "",
                f"- Selected layer: {row['selected_layer']}",
                (
                    "- Signed W pressure effect at "
                    f"|α|={row['comparison_magnitude']}: "
                    f"{row['signed_wrong_pressure_effect']} "
                    f"[{row['signed_wrong_pressure_effect_ci_low']}, "
                    f"{row['signed_wrong_pressure_effect_ci_high']}]"
                ),
                (
                    "- Signed W pressure effect at |α|=1: "
                    f"{row['alpha_1_signed_wrong_pressure_effect']['mean']} "
                    f"[{row['alpha_1_signed_wrong_pressure_effect']['ci_low']}, "
                    f"{row['alpha_1_signed_wrong_pressure_effect']['ci_high']}]"
                ),
                (
                    "- Paired mean absolute neutral probability damage: "
                    f"{row['mean_neutral_probability_damage']}"
                ),
                (
                    "- Paired mean absolute neutral accuracy damage: "
                    f"{row['mean_neutral_accuracy_damage']}"
                ),
                f"- Maximum sampled-control effect: {row['max_control_signed_effect']}",
                f"- Exceeds every sampled control: {row['effect_exceeds_all_sampled_controls']}",
                (
                    "- Effect minus maximum sampled control: "
                    f"{row['effect_minus_max_sampled_control']}"
                ),
                "",
            ]
        )
    args.output.with_suffix(".md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
