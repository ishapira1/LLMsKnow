#!/usr/bin/env python3
"""Resumable local Codex judge for a blinded random_baseline packet."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import random_baseline as rb


WINNER_RE = re.compile(r'"winner"\s*:\s*"(A|B|tie)"', re.IGNORECASE)


def parse_winners(text: str, expected: int) -> list[str]:
    values = [value.upper() for value in WINNER_RE.findall(text)]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} judgments, found {len(values)}")
    return values


def judge_chunk(chunk: Sequence[Mapping[str, Any]], model: str,
                effort: str) -> list[dict[str, Any]]:
    numbered = "\n\n".join(
        f"ITEM {index + 1}\n{row['prompt']}" for index, row in enumerate(chunk))
    prompt = (
        "Judge every item independently. Return exactly one compact JSON object per line, "
        "in the same order, with only the winner field. No prose.\n\n" + numbered
    )
    command = ["codex", "exec", "--model", model, "--config",
               f"model_reasoning_effort=\"{effort}\"", "--sandbox", "read-only", "-"]
    completed = subprocess.run(command, input=prompt, text=True, capture_output=True,
                               check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-2000:])
    winners = parse_winners(completed.stdout, len(chunk))
    return [{"judge_id": row["judge_id"], "winner": winner,
             "judge_model": model, "reasoning_effort": effort}
            for row, winner in zip(chunk, winners)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    packet = rb.read_jsonl(args.packet)
    existing_rows = rb.read_jsonl(args.output) if args.output.exists() else []
    existing = {str(row["judge_id"]): row for row in existing_rows}
    pending = [row for row in packet if str(row["judge_id"]) not in existing]
    chunks = [pending[index:index + args.chunk_size]
              for index in range(0, len(pending), args.chunk_size)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_chunk, chunk, args.model, args.effort): chunk
                   for chunk in chunks}
        for future in as_completed(futures):
            for row in future.result():
                existing[row["judge_id"]] = row
            ordered = [existing[row["judge_id"]] for row in packet
                       if row["judge_id"] in existing]
            rb.atomic_jsonl(args.output, ordered)
            print(f"judged={len(ordered)}/{len(packet)}", flush=True)
    if len(existing) != len(packet):
        raise RuntimeError("Judging did not complete")
    rb.atomic_json(args.output.with_suffix(".audit.json"), {
        "status": "complete", "packet_sha256": rb.sha256_file(args.packet),
        "labels_sha256": rb.sha256_file(args.output), "rows": len(packet),
        "model": args.model, "effort": args.effort, "completed_at": rb.utc_now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
