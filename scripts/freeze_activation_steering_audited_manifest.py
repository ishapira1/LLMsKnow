#!/usr/bin/env python3
"""Freeze exactly 1,000 semantically approved questions from a review pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _rank(seed: int, row: dict[str, Any]) -> str:
    key = f"{row['dataset']}::{row['source_example_id']}"
    return hashlib.sha256(f"{seed}::{key}".encode("utf-8")).hexdigest()


def _review_overrides(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (str(row["dataset"]), str(row["source_example_id"])): row
        for row in rows
    }


def _round_robin(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        model_signature = json.dumps(
            row.get("neutral_correct_by_model", {}),
            sort_keys=True,
        )
        by_label.setdefault(str(row["endorsed_choice"]), {}).setdefault(
            model_signature,
            [],
        ).append(row)
    for signature_groups in by_label.values():
        for values in signature_groups.values():
            values.sort(key=lambda row: _rank(seed, row))
    chosen: list[dict[str, Any]] = []
    labels = sorted(by_label)
    label_cursor = 0
    signature_cursor = {label: 0 for label in labels}
    while len(chosen) < count and labels:
        label = labels[label_cursor % len(labels)]
        signature_groups = by_label[label]
        signatures = sorted(
            signature
            for signature, values in signature_groups.items()
            if values
        )
        if not signatures:
            labels.remove(label)
            continue
        signature = signatures[signature_cursor[label] % len(signatures)]
        chosen.append(signature_groups[signature].pop(0))
        signature_cursor[label] += 1
        label_cursor += 1
    if len(chosen) != count:
        raise ValueError(f"Approved review pool has {len(chosen)} of {count} required rows.")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "controlled_prompt_only_v1_20260725":
        raise ValueError("Controlled freeze/config protocol mismatch.")
    rows = [
        json.loads(line)
        for line in args.candidate_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    overrides = _review_overrides(args.review_csv)
    for row in rows:
        override = overrides.get(
            (str(row["dataset"]), str(row["source_example_id"]))
        )
        if override is not None:
            for field in (
                "semantic_b_review_status",
                "semantic_b_reviewer",
                "semantic_b_reviewed_at",
                "semantic_b_review_note",
            ):
                if field in override:
                    row[field] = override[field]
    approved = [
        row
        for row in rows
        if str(row.get("semantic_b_review_status", "")) == "approved"
    ]
    quotas = config["splits"]["audited_target_per_dataset"]
    selected: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for split in ("train", "val", "test"):
            cell = [
                row
                for row in approved
                if row["dataset"] == dataset and row["split"] == split
            ]
            selected.extend(
                _round_robin(
                    cell,
                    count=int(quotas[split]),
                    seed=args.seed,
                )
            )
    keys = [
        (str(row["dataset"]), str(row["source_example_id"]))
        for row in selected
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Frozen manifest has duplicate stable keys.")
    selected.sort(
        key=lambda row: (
            str(row["dataset"]),
            {"train": 0, "val": 1, "test": 2}[str(row["split"])],
            str(row["endorsed_choice"]),
            str(row["source_example_id"]),
        )
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_jsonl}.")
    with args.output_jsonl.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "n_frozen": len(selected),
                "n_approved_pool": len(approved),
                "output": str(args.output_jsonl),
                "selection_uses_post_steering_behavior": False,
                "rejected_rows_are_replenished_within_dataset_split_label_strata": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
