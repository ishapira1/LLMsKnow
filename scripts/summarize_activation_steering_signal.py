#!/usr/bin/env python3
"""Summarize the lean exploratory W-N signal experiment."""

from __future__ import annotations

import argparse
import json
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
    for model_name, layer in sorted(selected_layers.items()):
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
        p_wrong_pos = _mean(wrong_pos, "p_endorsed")
        p_wrong_neg = _mean(wrong_neg, "p_endorsed")
        p_neutral_zero = _mean(neutral_zero, "p_correct")
        neutral_damage_values = []
        for dose_rows in (neutral_pos, neutral_neg):
            value = _mean(dose_rows, "p_correct")
            if value is not None and p_neutral_zero is not None:
                neutral_damage_values.append(abs(value - p_neutral_zero))
        signed_effect = (
            p_wrong_pos - p_wrong_neg
            if p_wrong_pos is not None and p_wrong_neg is not None
            else None
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
                "signed_wrong_pressure_effect": signed_effect,
                "mean_neutral_probability_damage": (
                    mean(neutral_damage_values) if neutral_damage_values else None
                ),
                "max_control_signed_effect": max_control,
                "effect_exceeds_all_sampled_controls": (
                    signed_effect is not None
                    and max_control is not None
                    and signed_effect > max_control
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
                f"- Signed W pressure effect: {row['signed_wrong_pressure_effect']}",
                f"- Mean neutral probability damage: {row['mean_neutral_probability_damage']}",
                f"- Maximum sampled-control effect: {row['max_control_signed_effect']}",
                f"- Exceeds every sampled control: {row['effect_exceeds_all_sampled_controls']}",
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
