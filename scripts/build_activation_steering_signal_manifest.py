#!/usr/bin/env python3
"""Freeze a deterministic benchmark-label cohort for an exploratory signal run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _rank(seed: int, row: dict[str, Any]) -> str:
    key = f"{row['dataset']}::{row['source_example_id']}"
    return hashlib.sha256(f"{seed}::{key}".encode()).hexdigest()


def _balanced_sample(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        signature = json.dumps(
            row.get("neutral_correct_by_model", {}),
            sort_keys=True,
        )
        groups.setdefault((str(row["endorsed_choice"]), signature), []).append(row)
    for values in groups.values():
        values.sort(key=lambda row: _rank(seed, row))
    keys = sorted(groups)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count:
        live = [key for key in keys if groups[key]]
        if not live:
            raise ValueError(f"Only {len(selected)} of {count} requested rows exist.")
        key = live[cursor % len(live)]
        selected.append(groups[key].pop(0))
        cursor += 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--train-per-dataset", type=int, default=90)
    parser.add_argument("--val-per-dataset", type=int, default=30)
    parser.add_argument("--test-per-dataset", type=int, default=30)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.candidate_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    quotas = {
        "train": args.train_per_dataset,
        "val": args.val_per_dataset,
        "test": args.test_per_dataset,
    }
    datasets = sorted({str(row["dataset"]) for row in rows})
    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        for split, count in quotas.items():
            cell = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["split"] == split
                and str(row["correct_choice"]) != str(row["endorsed_choice"])
            ]
            selected.extend(
                _balanced_sample(cell, count=int(count), seed=args.seed)
            )
    for row in selected:
        row["semantic_b_review_status"] = "not_requested_exploratory"
        row["semantic_b_reviewer"] = ""
        row["semantic_b_reviewed_at"] = ""
        row["semantic_b_review_note"] = (
            "Exploratory signal cohort: wrongness follows the benchmark answer "
            "key only; no per-row human semantic claim is made."
        )
        row["cohort_policy"] = "benchmark_label_wrongness_exploratory"
    selected.sort(
        key=lambda row: (
            str(row["dataset"]),
            {"train": 0, "val": 1, "test": 2}[str(row["split"])],
            str(row["endorsed_choice"]),
            str(row["source_example_id"]),
        )
    )
    stable_keys = [
        f"{row['dataset']}::{row['source_example_id']}" for row in selected
    ]
    if len(stable_keys) != len(set(stable_keys)):
        raise ValueError("Signal manifest contains duplicate stable keys.")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        raise FileExistsError(args.output_jsonl)
    with args.output_jsonl.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output_jsonl),
                "n_rows": len(selected),
                "datasets": datasets,
                "quotas_per_dataset": quotas,
                "seed": args.seed,
                "human_semantic_approval": False,
                "interpretation": "exploratory_benchmark_label_signal",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
