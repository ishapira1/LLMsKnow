#!/usr/bin/env python3
"""Build a deterministic, behavior-independent semantic-b review sheet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from llmssycoph.interventions.controlled import (
    REQUIRED_CONDITIONS,
    canonical_choice_map,
    stable_question_key,
)
from llmssycoph.interventions.data import (
    build_intervention_pairs,
    load_json,
    load_jsonl,
    resolve_run_config_path,
    resolve_sampling_records_path,
)


DEFAULT_QUOTAS = {"train": 300, "val": 100, "test": 100}


def _rank(seed: int, *parts: Any) -> str:
    return hashlib.sha256(
        "||".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    ).hexdigest()


def build_candidates(
    source_run_dirs: list[Path],
    *,
    seed: int,
    quotas: Dict[str, int],
) -> list[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source_run_dir in source_run_dirs:
        resolved_run_dir = source_run_dir.expanduser().resolve()
        run_config_path = resolve_run_config_path(resolved_run_dir)
        sampling_records_path = resolve_sampling_records_path(resolved_run_dir)
        run_config = load_json(run_config_path)
        model_name = str(run_config.get("model", "") or "")
        if not model_name:
            raise ValueError(f"Source run has no model identity: {run_config_path}")
        records = load_jsonl(
            sampling_records_path,
            template_types=REQUIRED_CONDITIONS,
        )
        pairs, _ = build_intervention_pairs(
            records,
            probe_scores=pd.DataFrame(),
            required_conditions=REQUIRED_CONDITIONS,
            require_metric_usable=False,
        )
        for pair in pairs:
            key = stable_question_key(pair)
            neutral = pair["records"]["neutral"]
            choice_map = canonical_choice_map(pair["choices"])
            source_correct_choice = str(pair["correct_choice"])
            source_endorsed_choice = str(pair["endorsed_choice"])
            correct_choice = choice_map[source_correct_choice]
            endorsed_choice = choice_map[source_endorsed_choice]
            current = merged.setdefault(
                key,
                {
                    "dataset": pair["dataset"],
                    "source_dataset": pair["source_dataset"],
                    "source_example_id": pair["source_example_id"],
                    "question_id": pair["question_id"],
                    "split": pair["split"],
                    "correct_choice": correct_choice,
                    "correct_answer": str(neutral.get("correct_answer", "") or ""),
                    "endorsed_choice": endorsed_choice,
                    "endorsed_answer": str(neutral.get("incorrect_answer", "") or ""),
                    "source_correct_choice": source_correct_choice,
                    "source_endorsed_choice": source_endorsed_choice,
                    "source_choices": list(pair["choices"]),
                    "choice_label_map": choice_map,
                    "question": str(neutral.get("question", "") or ""),
                    "options": list(neutral.get("answers_list", []) or []),
                    "neutral_correct_by_model": {},
                    "semantic_b_review_status": "pending",
                    "semantic_b_reviewer": "",
                    "semantic_b_reviewed_at": "",
                    "semantic_b_review_note": "",
                },
            )
            if (
                current["correct_choice"] != correct_choice
                or current["endorsed_choice"] != endorsed_choice
                or current["source_correct_choice"] != source_correct_choice
                or current["source_endorsed_choice"] != source_endorsed_choice
                or current["choice_label_map"] != choice_map
                or current["split"] != pair["split"]
            ):
                raise ValueError(f"Source-run metadata mismatch for {key}.")
            current["neutral_correct_by_model"][model_name] = bool(
                pair.get("neutral_correct")
            )
    selected: list[Dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in merged.values()})
    for dataset in datasets:
        for split, quota in quotas.items():
            candidates = [
                row
                for row in merged.values()
                if row["dataset"] == dataset and row["split"] == split
            ]
            candidates.sort(
                key=lambda row: (
                    str(row["endorsed_choice"]),
                    tuple(sorted(row["neutral_correct_by_model"].items())),
                    _rank(seed, stable_question_key(row)),
                )
            )
            by_label: Dict[str, list[Dict[str, Any]]] = {}
            for row in candidates:
                by_label.setdefault(str(row["endorsed_choice"]), []).append(row)
            labels = sorted(by_label)
            chosen: list[Dict[str, Any]] = []
            offset = 0
            while len(chosen) < int(quota) and labels:
                label = labels[offset % len(labels)]
                if by_label[label]:
                    chosen.append(by_label[label].pop(0))
                else:
                    labels.remove(label)
                    continue
                offset += 1
            if len(chosen) != int(quota):
                raise ValueError(
                    f"Insufficient candidates for {dataset}/{split}: "
                    f"wanted={quota} found={len(chosen)}."
                )
            selected.extend(chosen)
    selected.sort(
        key=lambda row: (
            str(row["dataset"]),
            {"train": 0, "val": 1, "test": 2}[str(row["split"])],
            str(row["endorsed_choice"]),
            str(row["source_example_id"]),
        )
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-review-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=300)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument(
        "--review-pool-multiplier",
        type=float,
        default=2.0,
        help="Candidate-pool size relative to the final per-cell quota.",
    )
    args = parser.parse_args()
    if args.review_pool_multiplier < 1.0:
        raise ValueError("--review-pool-multiplier must be at least one.")
    final_quotas = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    rows = build_candidates(
        args.source_run_dir,
        seed=args.seed,
        quotas={
            split: int(math.ceil(count * args.review_pool_multiplier))
            for split, count in final_quotas.items()
        },
    )
    for path in (args.output_jsonl, args.output_review_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}.")
    with args.output_jsonl.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fieldnames = [
        "dataset",
        "source_example_id",
        "question_id",
        "split",
        "question",
        "options",
        "correct_choice",
        "correct_answer",
        "endorsed_choice",
        "endorsed_answer",
        "source_correct_choice",
        "source_endorsed_choice",
        "source_choices",
        "choice_label_map",
        "neutral_correct_by_model",
        "semantic_b_review_status",
        "semantic_b_reviewer",
        "semantic_b_reviewed_at",
        "semantic_b_review_note",
    ]
    with args.output_review_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fieldnames},
                    "options": json.dumps(row["options"], ensure_ascii=False),
                    "source_choices": json.dumps(row["source_choices"]),
                    "choice_label_map": json.dumps(
                        row["choice_label_map"],
                        sort_keys=True,
                    ),
                    "neutral_correct_by_model": json.dumps(
                        row["neutral_correct_by_model"],
                        sort_keys=True,
                    ),
                }
            )
    print(
        json.dumps(
            {
                "n_candidates": len(rows),
                "final_quotas_per_dataset": final_quotas,
                "review_pool_multiplier": args.review_pool_multiplier,
                "output_jsonl": str(args.output_jsonl),
                "output_review_csv": str(args.output_review_csv),
                "selection_uses_post_steering_behavior": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
