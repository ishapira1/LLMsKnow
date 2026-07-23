#!/usr/bin/env python3
"""Materialize the pinned Stanford Alpaca source on large cluster storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import urlopen


SOURCE_COMMIT = "761dc5bfbdeeffa89b8bff5d038781a4055f796a"
SOURCE_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    f"{SOURCE_COMMIT}/alpaca_data.json"
)
EXPECTED_SHA256 = "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419"
EXPECTED_ROWS = 52_002
WIKITEXT_REPOSITORY = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "test"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_EXPECTED_ROWS = 4_358


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> None:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"Stanford Alpaca checksum mismatch: expected {EXPECTED_SHA256}, got {actual_sha}"
        )
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Stanford Alpaca row-count mismatch: expected {EXPECTED_ROWS}, "
            f"got {len(rows) if isinstance(rows, list) else type(rows).__name__}"
        )
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"Stanford Alpaca row {index} is not an object")
        if not str(row.get("instruction", "")).strip():
            raise RuntimeError(f"Stanford Alpaca row {index} has no instruction")


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        for attempt in range(1, 4):
            try:
                with urlopen(SOURCE_URL, timeout=60) as response, temporary.open(
                    "wb"
                ) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                validate(temporary)
                os.replace(temporary, destination)
                return
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                time.sleep(attempt * 2)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_wikitext(cache_dir: Path) -> int:
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            dataset = load_dataset(
                WIKITEXT_REPOSITORY,
                WIKITEXT_CONFIG,
                split=WIKITEXT_SPLIT,
                revision=WIKITEXT_REVISION,
                cache_dir=str(cache_dir),
            )
            if len(dataset) != WIKITEXT_EXPECTED_ROWS:
                raise RuntimeError(
                    "WikiText row-count mismatch: "
                    f"expected {WIKITEXT_EXPECTED_ROWS}, got {len(dataset)}"
                )
            return len(dataset)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-datasets-cache", type=Path)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.is_file():
        validate(destination)
        status = "reused"
    else:
        download(destination)
        status = "downloaded"
    result = {
        "output": str(destination),
        "rows": EXPECTED_ROWS,
        "sha256": EXPECTED_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_url": SOURCE_URL,
        "status": status,
    }
    if args.hf_datasets_cache is not None:
        wikitext_rows = prepare_wikitext(args.hf_datasets_cache.expanduser().resolve())
        result["wikitext"] = {
            "cache_dir": str(args.hf_datasets_cache.expanduser().resolve()),
            "config": WIKITEXT_CONFIG,
            "repository": WIKITEXT_REPOSITORY,
            "revision": WIKITEXT_REVISION,
            "rows": wikitext_rows,
            "split": WIKITEXT_SPLIT,
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
