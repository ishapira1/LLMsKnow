#!/usr/bin/env python3
"""Coordinate-overlap, nesting, composition, and structural-null analyses."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

import campaign
from core import (
    DEFAULT_CONFIG,
    atomic_json,
    atomic_text,
    load_config,
    mask_coordinates,
    overlap,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
)


NULL_REPLICATES = 10_000
NULL_SEED = 20260918
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")


class WeightAnalysisError(campaign.CampaignError):
    pass


def _indices(path: Path) -> Mapping[str, Any]:
    return campaign._load_indices(path / "indices.pt")


def _question_ids(path: Path) -> set[str]:
    return {str(row["question_key"]) for row in read_jsonl(path)}


def _question_overlap(left: set[str], right: set[str]) -> Mapping[str, Any]:
    intersection = len(left & right)
    union = len(left | right)
    return {
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": intersection,
        "union_count": union,
        "jaccard": intersection / union if union else 1.0,
    }


def _composition(indices: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    counts = {}
    for name, values in indices.items():
        match = _LAYER_RE.search(name)
        if match is None:
            raise WeightAnalysisError(f"Cannot resolve transformer layer from {name}")
        layer = int(match.group(1))
        projection = name.rsplit(".", 1)[-1]
        key = (layer, projection)
        counts[key] = counts.get(key, 0) + int(values.numel())
    return [
        {"layer": layer, "projection": projection, "count": count}
        for (layer, projection), count in sorted(counts.items())
    ]


def _structural_null(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    universe: Mapping[str, int],
    *,
    namespace: str,
    replicates: int = NULL_REPLICATES,
) -> Mapping[str, Any]:
    """Exact module-count-matched null using independent hypergeometric draws."""

    observed = len(mask_coordinates(left) & mask_coordinates(right))
    rng = np.random.default_rng(
        int(stable_hash(campaign.EXPERIMENT, "structural-null", namespace)[:16], 16)
        ^ NULL_SEED
    )
    samples = np.zeros(int(replicates), dtype=np.int64)
    names = sorted(set(left) | set(right))
    for name in names:
        population = int(universe[name])
        left_count = int(left.get(name, np.empty(0)).numel())
        right_count = int(right.get(name, np.empty(0)).numel())
        if left_count > population or right_count > population:
            raise WeightAnalysisError(f"Mask count exceeds structural universe in {name}")
        samples += rng.hypergeometric(
            left_count,
            population - left_count,
            right_count,
            size=int(replicates),
        )
    expected = float(samples.mean())
    return {
        "replicates": int(replicates),
        "null": "independent_coordinate_draws_conditioned_on_each_mask's_exact_module_counts",
        "observed_intersection": observed,
        "expected_intersection": expected,
        "enrichment": observed / expected if expected > 0 else None,
        "null_sd": float(samples.std(ddof=1)),
        "null_interval_95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "upper_tail_p": float((1 + np.sum(samples >= observed)) / (int(replicates) + 1)),
        "seed": NULL_SEED,
    }


def analyze_model(args: argparse.Namespace) -> None:
    load_config(args.config)
    root = Path(args.result_root)
    mask_root = root / "masks" / args.model_key
    masks = {
        mask_id: _indices(mask_root / mask_id)
        for mask_id in (
            "n1_mechanism",
            "n1_seed17",
            "n1_seed29",
            "n1_prefix_250",
            "n1_prefix_500",
            "n1_prefix_1000",
            "source_all",
            "source_false",
        )
    }
    score_metadata = read_json(
        root / "scores" / args.model_key / "n1_seed5_prune" / "metadata.json"
    )
    universe = {
        name: int(row["numel"]) for name, row in score_metadata["tensors"].items()
    }
    pair_specs = (
        ("seed5_vs_seed17", "n1_mechanism", "n1_seed17", True),
        ("seed5_vs_seed29", "n1_mechanism", "n1_seed29", True),
        ("seed17_vs_seed29", "n1_seed17", "n1_seed29", True),
        ("user_vs_all_reliable_source", "n1_mechanism", "source_all", True),
        ("user_vs_false_source_only", "n1_mechanism", "source_false", True),
        ("size250_vs_1000", "n1_prefix_250", "n1_prefix_1000", False),
        ("size500_vs_1000", "n1_prefix_500", "n1_prefix_1000", False),
    )
    overlaps = []
    for pair_id, left_id, right_id, needs_null in pair_specs:
        row = {
            "model_key": args.model_key,
            "pair_id": pair_id,
            "left_mask": left_id,
            "right_mask": right_id,
            **overlap(masks[left_id], masks[right_id]),
            "analysis_role": (
                "sparsity_nesting_not_independent_stability"
                if pair_id.startswith("size")
                else "non_nested_mask_overlap"
            ),
        }
        if needs_null:
            row["structural_null"] = _structural_null(
                masks[left_id],
                masks[right_id],
                universe,
                namespace=f"{args.model_key}:{pair_id}",
            )
        overlaps.append(row)
    coordinates_250 = mask_coordinates(masks["n1_prefix_250"])
    coordinates_500 = mask_coordinates(masks["n1_prefix_500"])
    coordinates_1000 = mask_coordinates(masks["n1_prefix_1000"])
    if not coordinates_250 < coordinates_500 < coordinates_1000:
        raise WeightAnalysisError("N1 size masks are not exactly nested")

    manifest_root = root / "manifests" / args.model_key
    question_sets = {
        seed: _question_ids(manifest_root / f"n1_seed{seed}" / "prune.jsonl")
        for seed in (5, 17, 29)
    }
    question_overlaps = [
        {
            "model_key": args.model_key,
            "pair_id": f"seed{left}_vs_seed{right}",
            **_question_overlap(question_sets[left], question_sets[right]),
        }
        for left, right in ((5, 17), (5, 29), (17, 29))
    ]
    composition = {
        mask_id: _composition(indices)
        for mask_id, indices in masks.items()
    }
    payload = {
        "status": "complete",
        "model_key": args.model_key,
        "coordinate_namespace": (
            "model_local_only; coordinates are never intersected across architectures"
        ),
        "overlaps": overlaps,
        "question_set_overlaps": question_overlaps,
        "composition": composition,
        "exact_nesting": True,
        "structural_null_replicates": NULL_REPLICATES,
    }
    destination = root / "weight_analysis" / args.model_key
    atomic_json(destination / "analysis.json", payload)
    flat_rows = []
    for row in overlaps:
        flat = {key: value for key, value in row.items() if key != "structural_null"}
        for key, value in dict(row.get("structural_null", {})).items():
            flat[f"null_{key}"] = canonical_value(value)
        flat_rows.append(flat)
    headers = sorted({key for row in flat_rows for key in row})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=headers)
    writer.writeheader()
    writer.writerows(flat_rows)
    atomic_text(destination / "overlaps.csv", stream.getvalue())
    latex_rows = [
        "Model & Pair & Intersection & Jaccard & Enrichment \\\\",
        "\\midrule",
    ]
    for row in overlaps:
        enrichment = row.get("structural_null", {}).get("enrichment")
        escaped_pair = str(row["pair_id"]).replace("_", "\\_")
        enrichment_text = "--" if enrichment is None else f"{enrichment:.2f}"
        latex_rows.append(
            f"{args.model_key} & {escaped_pair} & "
            f"{row['intersection_count']} & {row['jaccard']:.3f} & "
            f"{enrichment_text} \\\\")
    atomic_text(
        destination / "overlaps.tex",
        "\\begin{tabular}{llrrr}\n" + "\n".join(latex_rows) + "\n\\bottomrule\n\\end{tabular}\n",
    )
    receipt = {
        "status": "complete",
        "model_key": args.model_key,
        "analysis_sha256": sha256_file(destination / "analysis.json"),
        "csv_sha256": sha256_file(destination / "overlaps.csv"),
        "latex_sha256": sha256_file(destination / "overlaps.tex"),
    }
    atomic_json(destination / "COMPLETE.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def canonical_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.result_root)
    models = {}
    for model_key in campaign.MODEL_KEYS:
        complete = read_json(root / "weight_analysis" / model_key / "COMPLETE.json")
        analysis_path = root / "weight_analysis" / model_key / "analysis.json"
        if complete.get("analysis_sha256") != sha256_file(analysis_path):
            raise WeightAnalysisError(f"Weight analysis changed for {model_key}")
        models[model_key] = read_json(analysis_path)
    receipt = {
        "status": "complete",
        "models": models,
        "cross_architecture_intersections": 0,
    }
    atomic_json(root / "weight_analysis" / "COMPLETE.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("analyze-model")
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--model-key", choices=campaign.MODEL_KEYS, required=True)
    command.set_defaults(func=analyze_model)
    command = subparsers.add_parser("aggregate")
    command.add_argument("--result-root", type=Path, required=True)
    command.set_defaults(func=aggregate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
