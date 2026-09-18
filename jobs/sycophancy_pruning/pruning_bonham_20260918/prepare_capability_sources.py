#!/usr/bin/env python3
"""Freeze Bonham capability sources absent from the authenticated suite bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core import atomic_json, atomic_jsonl, sha256_file, stable_hash


SPECS = {
    "boolq": ("google/boolq", None, "validation", "35b264d03638db9f4ce671b711558bf7ff0f80d5"),
    "rte": ("nyu-mll/glue", "rte", "validation", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"),
    "hellaswag": ("Rowan/hellaswag", None, "validation", "218ec52e09a7e7462a5400043bb9a69a41d06b76"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", "01e74176c63542e6b0bcb004dcdea22d94fb67b5"),
    "triviaqa_wiki": ("mandarjoshi/trivia_qa", "rc.wikipedia", "validation", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"),
}


class SourceError(RuntimeError):
    pass


def _identity(row: Mapping[str, Any], index: int) -> str:
    for key in ("idx", "id", "ind", "question_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    encoded = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24] + f"-{index}"


def _ordered(rows: list[Mapping[str, Any]], namespace: str) -> list[tuple[str, Mapping[str, Any]]]:
    indexed = [(_identity(row, index), row) for index, row in enumerate(rows)]
    return sorted(indexed, key=lambda item: (stable_hash("bonham-capability", namespace, item[0]), item[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    args = parser.parse_args()
    from datasets import load_dataset

    receipts = {}
    for name, (repo_id, subset, split, revision) in SPECS.items():
        dataset = load_dataset(
            repo_id,
            subset,
            split=split,
            revision=revision,
            cache_dir=str(args.hf_cache),
        )
        population = [dict(row) for row in dataset]
        selected = [
            {"stable_row_id": identity, "source_index": index, **row}
            for index, (identity, row) in enumerate(_ordered(population, name)[:500])
        ]
        path = args.output_root / f"{name}.jsonl"
        atomic_jsonl(path, selected)
        receipts[name] = {
            "repo_id": repo_id,
            "config": subset,
            "split": split,
            "revision": revision,
            "population_count": len(population),
            "selected_count": len(selected),
            "selection": "outcome-independent SHA-256 rank, cap 500",
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        if name == "winogrande":
            training = [
                dict(row)
                for row in load_dataset(
                    repo_id,
                    subset,
                    split="train",
                    revision=revision,
                    cache_dir=str(args.hf_cache),
                )
            ]
            ordered = _ordered(training, "winogrande-demonstrations")
            demonstrations = []
            for label, count in (("1", 3), ("2", 2)):
                demonstrations.extend(
                    row for _identity_value, row in [item for item in ordered if str(item[1].get("answer")) == label][:count]
                )
            demonstrations.sort(
                key=lambda row: stable_hash(
                    "bonham-capability", "winogrande-demo-order", _identity(row, 0)
                )
            )
            demo_path = args.output_root / "winogrande_demonstrations.jsonl"
            atomic_jsonl(demo_path, demonstrations)
            receipts[name].update(
                {
                    "demonstrations_path": str(demo_path.resolve()),
                    "demonstrations_sha256": sha256_file(demo_path),
                    "shot_count": len(demonstrations),
                }
            )
        if name == "triviaqa_wiki":
            training = [
                dict(row)
                for row in load_dataset(
                    repo_id,
                    subset,
                    split="train",
                    revision=revision,
                    cache_dir=str(args.hf_cache),
                )
            ]
            eligible = [
                item
                for item in _ordered(training, "triviaqa-demonstrations")
                if str(item[1].get("question", "")).strip()
                and str(item[1].get("answer", {}).get("value", "")).strip()
            ][:5]
            if len(eligible) != 5:
                raise SourceError("TriviaQA has fewer than five eligible demonstrations")
            demo_path = args.output_root / "triviaqa_wiki_demonstrations.jsonl"
            atomic_jsonl(demo_path, (row for _identity_value, row in eligible))
            receipts[name].update(
                {
                    "demonstrations_path": str(demo_path.resolve()),
                    "demonstrations_sha256": sha256_file(demo_path),
                    "shot_count": 5,
                }
            )
    complete = {
        "status": "complete",
        "experiment": "pruning_bonham_20260918",
        "sources": receipts,
    }
    atomic_json(args.output_root / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
