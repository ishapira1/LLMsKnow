#!/usr/bin/env python3
"""Create a small, deterministic, behavior-audited held-out pruning cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_CONDITIONS = (
    "neutral",
    "incorrect_suggestion_strong",
    "incorrect_suggestion",
    "suggest_correct_strong",
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"evaluation manifest is empty: {path}")
    return rows


def _question_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    try:
        draw_idx = int(row.get("draw_idx", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid draw_idx in row {row.get('example_id')!r}") from exc
    values = (
        str(row.get("dataset", "") or "").strip(),
        str(row.get("split", "") or "").strip(),
        str(row.get("question_id", "") or "").strip(),
        draw_idx,
    )
    if not all(str(value) for value in values[:3]):
        raise ValueError(f"blank held-out identity in row {row.get('example_id')!r}")
    return values


def _stable_rank(seed: int, key: tuple[str, str, str, int]) -> str:
    return hashlib.sha256(
        f"{seed}|{key[0]}|{key[1]}|{key[2]}|{key[3]}".encode("utf-8")
    ).hexdigest()


def _balanced_quotas(total: int, datasets: Sequence[str]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("--questions must be positive")
    if not datasets:
        raise ValueError("no datasets remain after filtering")
    base, remainder = divmod(total, len(datasets))
    return {
        dataset: base + int(index < remainder)
        for index, dataset in enumerate(sorted(datasets))
    }


def select_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    questions: int,
    seed: int,
    splits: Sequence[str],
    conditions: Sequence[str],
    require_baseline_strict_flip: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_splits = {str(value).strip() for value in splits if str(value).strip()}
    condition_order = tuple(
        dict.fromkeys(str(value).strip() for value in conditions if str(value).strip())
    )
    if not allowed_splits:
        raise ValueError("at least one --split is required")
    if not condition_order:
        raise ValueError("at least one --condition is required")

    grouped: dict[
        tuple[str, str, str, int], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    source_datasets: set[str] = set()
    ignored = Counter()
    for raw_row in rows:
        row = dict(raw_row)
        key = _question_key(row)
        if key[1] not in allowed_splits:
            ignored["other_split"] += 1
            continue
        condition = str(row.get("condition", "") or "").strip()
        if condition not in condition_order:
            ignored["other_condition"] += 1
            continue
        source_datasets.add(key[0])
        if condition in grouped[key]:
            raise ValueError(f"duplicate condition {condition!r} for held-out key {key}")
        grouped[key][condition] = row

    eligible: dict[str, list[tuple[str, str, str, int]]] = defaultdict(list)
    rejected = Counter()
    for key, by_condition in sorted(grouped.items()):
        missing = [condition for condition in condition_order if condition not in by_condition]
        if missing:
            rejected["missing_condition"] += 1
            continue
        strict_flags = {
            bool(by_condition[condition].get("baseline_strict_flip", False))
            for condition in condition_order
        }
        if len(strict_flags) != 1:
            raise ValueError(f"baseline_strict_flip disagrees across conditions for {key}")
        if require_baseline_strict_flip and strict_flags != {True}:
            rejected["not_baseline_strict_flip"] += 1
            continue
        eligible[key[0]].append(key)

    datasets = sorted(source_datasets)
    quotas = _balanced_quotas(int(questions), datasets)
    selected_keys: list[tuple[str, str, str, int]] = []
    for dataset in datasets:
        candidates = sorted(
            eligible[dataset],
            key=lambda key: _stable_rank(int(seed), key),
        )
        required = quotas[dataset]
        if len(candidates) < required:
            raise ValueError(
                f"insufficient eligible held-out questions for {dataset}: "
                f"need {required}, found {len(candidates)}"
            )
        selected_keys.extend(candidates[:required])
    selected_keys.sort(
        key=lambda key: (
            _stable_rank(int(seed), key),
            key,
        )
    )

    selected_rows: list[dict[str, Any]] = []
    for key in selected_keys:
        for condition in condition_order:
            selected_rows.append(dict(grouped[key][condition]))
    audit = {
        "schema_version": 1,
        "selection": {
            "questions_requested": int(questions),
            "questions_selected": len(selected_keys),
            "seed": int(seed),
            "splits": sorted(allowed_splits),
            "conditions": list(condition_order),
            "require_baseline_strict_flip": bool(require_baseline_strict_flip),
            "behavior_selection_label": (
                "baseline_strict_flip"
                if require_baseline_strict_flip
                else "unfiltered_complete_questions"
            ),
        },
        "eligible_questions_by_dataset": {
            dataset: len(keys) for dataset, keys in sorted(eligible.items())
        },
        "selected_questions_by_dataset": dict(
            sorted(Counter(key[0] for key in selected_keys).items())
        ),
        "selected_rows_by_condition": dict(
            sorted(Counter(row["condition"] for row in selected_rows).items())
        ),
        "ignored_rows": dict(sorted(ignored.items())),
        "rejected_questions": dict(sorted(rejected.items())),
    }
    return selected_rows, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic, dataset-balanced subset of complete held-out "
            "question groups for a labeled pruning micro-pilot."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--questions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Allowed held-out split; repeat as needed (default: val and validation).",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help=(
            "Condition retained for every selected question; repeat as needed. "
            "The default keeps neutral, strong/weak wrong suggestion, and strong "
            "correct suggestion."
        ),
    )
    parser.add_argument("--require-baseline-strict-flip", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    audit_destination = (
        args.audit_output.expanduser().resolve()
        if args.audit_output is not None
        else destination.with_suffix(destination.suffix + ".audit.json")
    )
    selected, audit = select_rows(
        _read_jsonl(source),
        questions=args.questions,
        seed=args.seed,
        splits=args.split or ("val", "validation"),
        conditions=args.condition or DEFAULT_CONDITIONS,
        require_baseline_strict_flip=args.require_baseline_strict_flip,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(_canonical_json(row))
            handle.write("\n")
    audit.update(
        {
            "source": str(source),
            "source_sha256": _sha256_file(source),
            "output": str(destination),
            "output_sha256": _sha256_file(destination),
            "output_rows": len(selected),
        }
    )
    audit_destination.parent.mkdir(parents=True, exist_ok=True)
    audit_destination.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
