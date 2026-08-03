#!/usr/bin/env python3
"""Freeze the cached WikiText-2 raw test split into a hash-pinned JSONL input."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import random_baseline as rb


EXPECTED_ROWS = 4358


def expected_payload(source_arrow: Path, dataset_info: Path, output: Path,
                     revision: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import Dataset

    source_arrow = source_arrow.resolve()
    dataset_info = dataset_info.resolve()
    dataset = Dataset.from_file(str(source_arrow))
    texts = list(dataset["text"])
    if len(texts) != EXPECTED_ROWS or any(not isinstance(text, str) for text in texts):
        raise ValueError(f"Expected {EXPECTED_ROWS} WikiText string rows, got {len(texts)}")
    rows = [{"row_id": index, "text": text} for index, text in enumerate(texts)]
    return rows, {
        "status": "complete",
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "revision": revision,
        "rows": EXPECTED_ROWS,
        "source_arrow_path": str(source_arrow),
        "source_arrow_sha256": rb.sha256_file(source_arrow),
        "dataset_info_path": str(dataset_info),
        "dataset_info_sha256": rb.sha256_file(dataset_info),
        "frozen_input_path": str(output.resolve()),
    }


def validate_existing(output: Path, pin_path: Path, expected: dict[str, Any]) -> None:
    if output.is_file() != pin_path.is_file():
        raise RuntimeError("WikiText frozen input/pin collision is incomplete")
    if not output.is_file():
        return
    pin = rb.read_json(pin_path)
    frozen_hash = rb.sha256_file(output)
    required = {**expected, "frozen_input_sha256": frozen_hash}
    for key, value in required.items():
        if pin.get(key) != value:
            raise ValueError(f"Existing WikiText pin drift: {key}")
    rows = rb.read_jsonl(output)
    if len(rows) != EXPECTED_ROWS or [row.get("row_id") for row in rows] != list(range(EXPECTED_ROWS)):
        raise ValueError("Existing WikiText frozen rows are incomplete or reordered")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-arrow", type=Path, required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pin-output", type=Path, required=True)
    args = parser.parse_args()

    rows, payload = expected_payload(
        args.source_arrow, args.dataset_info, args.output, args.revision
    )
    validate_existing(args.output, args.pin_output, payload)
    if args.output.is_file():
        print(f"wikitext_frozen_status=reused rows={EXPECTED_ROWS}")
        return 0
    rb.atomic_jsonl(args.output, rows)
    payload["frozen_input_sha256"] = rb.sha256_file(args.output)
    payload["completed_at"] = rb.utc_now()
    rb.atomic_json(args.pin_output, payload)
    validate_existing(args.output, args.pin_output, {
        key: value for key, value in payload.items() if key not in {"frozen_input_sha256", "completed_at"}
    })
    print(f"wikitext_frozen_status=created rows={EXPECTED_ROWS} sha256={payload['frozen_input_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
