#!/usr/bin/env python3
"""Validate the frozen full-cohort manifest without model dependencies."""

from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-full-cohort", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "controlled_prompt_only_v1_20260725":
        raise ValueError("Controlled manifest/config protocol mismatch.")
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise ValueError("Question manifest is empty.")
    keys = []
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for index, row in enumerate(rows):
        dataset = str(row.get("dataset", "") or "")
        source_id = str(row.get("source_example_id", "") or "")
        split = str(row.get("split", "") or "")
        correct = str(row.get("correct_choice", "") or "").upper()
        endorsed = str(row.get("endorsed_choice", "") or "").upper()
        if not dataset or not source_id or split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid identity/split at row {index}.")
        if correct not in "ABCDE" or endorsed not in "ABCDE" or correct == endorsed:
            raise ValueError(f"Invalid correct/endorsed choices at row {index}.")
        if row.get("source_choices") is not None:
            source_choices = [
                str(value).strip().upper() for value in row["source_choices"]
            ]
            canonical = list("ABCDE"[: len(source_choices)])
            numeric = [str(value) for value in range(1, len(source_choices) + 1)]
            if source_choices == canonical:
                choice_map = dict(zip(source_choices, canonical))
            elif source_choices == numeric:
                choice_map = dict(zip(source_choices, canonical))
            else:
                raise ValueError(f"Invalid source choices at row {index}.")
            if (
                choice_map.get(
                    str(row.get("source_correct_choice", "") or "").upper()
                )
                != correct
                or choice_map.get(
                    str(row.get("source_endorsed_choice", "") or "").upper()
                )
                != endorsed
                or (
                    row.get("choice_label_map") is not None
                    and dict(row["choice_label_map"]) != choice_map
                )
            ):
                raise ValueError(
                    f"Inconsistent source/canonical choice mapping at row {index}."
                )
        if str(row.get("semantic_b_review_status", "") or "") != "approved":
            raise ValueError(f"Semantic b is not approved at row {index}.")
        reviewer = str(row.get("semantic_b_reviewer", "") or "").strip()
        reviewed_at = str(row.get("semantic_b_reviewed_at", "") or "").strip()
        if not reviewer or not reviewed_at:
            raise ValueError(
                f"Semantic review provenance is incomplete at row {index}."
            )
        try:
            datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"Semantic review timestamp is not ISO-8601 at row {index}."
            ) from exc
        keys.append((dataset, source_id))
        counts[(dataset, split)] += 1
    if len(keys) != len(set(keys)):
        raise ValueError("Manifest has duplicate (dataset, source_example_id) keys.")

    expected: dict[tuple[str, str], int] = {}
    if args.require_full_cohort:
        per_dataset = config["splits"]["audited_target_per_dataset"]
        for dataset in config["datasets"]:
            for split in ("train", "val", "test"):
                expected[(str(dataset), split)] = int(per_dataset[split])
        if counts != collections.Counter(expected):
            raise ValueError(
                "Full-cohort quotas differ: "
                f"actual={dict(sorted(counts.items()))} "
                f"expected={dict(sorted(expected.items()))}"
            )
    print(
        json.dumps(
            {
                "status": "valid",
                "n_questions": len(rows),
                "counts": {
                    f"{dataset}::{split}": count
                    for (dataset, split), count in sorted(counts.items())
                },
                "full_cohort_required": bool(args.require_full_cohort),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
