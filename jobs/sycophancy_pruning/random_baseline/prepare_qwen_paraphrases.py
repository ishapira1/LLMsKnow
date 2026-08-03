#!/usr/bin/env python3
"""Freeze Qwen's final-cohort stem-paraphrase manifest before preflight pins it."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import random_baseline as rb


REQUIRED = ("neutral", "incorrect_suggestion", "incorrect_suggestion_strong",
            "suggest_correct_strong")


def source_uid(row: Mapping[str, Any]) -> str:
    source = str(row.get("source_example_id", "") or "").strip()
    if not source:
        raise ValueError("Missing source_example_id")
    return f"{row.get('dataset', '')}::{source}::{int(row.get('draw_idx', 0) or 0)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--sampling-record", type=Path, action="append", required=True)
    parser.add_argument("--paraphrase-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        existing, audit = __import__("llmssycoph.pruning.live_inference", fromlist=["x"]).load_and_validate_evaluation_manifest(args.output)
        if not 100 <= int(audit["question_count"]) <= 256:
            raise ValueError("Existing Qwen paraphrase coverage is outside the audited range")
        return 0
    from llmssycoph.probes.movement import (
        build_same_family_paraphrase_prompt_messages,
        load_paraphrase_artifact_lookup,
    )
    from llmssycoph.pruning.strict_manifests import build_evaluation_manifest

    final_rows = rb.read_jsonl(args.final_manifest)
    final_ids = {source_uid(row) for row in final_rows}
    if len(final_ids) != 256:
        raise ValueError(f"Expected 256 Qwen final identities, got {len(final_ids)}")
    paraphrases = load_paraphrase_artifact_lookup(str(args.paraphrase_artifact))["rows_by_key"]
    candidates = []
    missing = Counter()
    for path in args.sampling_record:
        for raw in rb.read_jsonl(path):
            if source_uid(raw) not in final_ids:
                continue
            paraphrase = paraphrases.get((str(raw.get("dataset", "")),
                                          str(raw.get("source_example_id", ""))))
            if paraphrase is None or paraphrase.get("status") != "valid":
                missing["missing_or_invalid"] += 1
                continue
            stem = str(paraphrase.get("paraphrased_stem", "")).strip()
            if not stem:
                missing["empty"] += 1
                continue
            row = dict(raw)
            rendered = build_same_family_paraphrase_prompt_messages(row, stem)
            prompt = str(rendered["prompt_text"]).rstrip("\r\n")
            row.update({"prompt_text": prompt, "raw_prompt": prompt + "\n",
                        "prompt": prompt + "\n", "prompt_messages": rendered["prompt_messages"],
                        "paraphrase_source": str(args.paraphrase_artifact),
                        "paraphrased_stem": stem})
            candidates.append(row)
    built = build_evaluation_manifest(
        candidates, model_id=rb.MODEL_SPECS["qwen"]["model_id"],
        revision=rb.MODEL_SPECS["qwen"]["revision"], suggestion_seed=0,
        calibration_question_uids=set(),
    )
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in built.rows:
        grouped[source_uid(row)].add(str(row["condition"]))
    complete = {identity for identity, conditions in grouped.items()
                if set(REQUIRED).issubset(conditions)}
    if not complete.issubset(final_ids) or len(complete) < 100:
        raise ValueError(
            f"Qwen frozen paraphrase coverage is unusable: {len(complete)}/256; "
            f"exclusions={dict(missing)}"
        )
    rows = [dict(row) for row in built.rows if source_uid(row) in complete]
    rb.atomic_jsonl(args.output, rows)
    _, audit = __import__("llmssycoph.pruning.live_inference", fromlist=["x"]).load_and_validate_evaluation_manifest(args.output)
    if int(audit["question_count"]) != len(complete):
        raise AssertionError(audit)
    rb.atomic_json(args.output.with_suffix(".audit.json"), {
        "status": "complete", "questions": len(complete), "primary_questions": 256,
        "coverage_fraction": len(complete) / 256, "missing_primary_questions": 256 - len(complete),
        "selection": "all final-cohort questions with pre-existing valid frozen stems; no reselection",
        "exclusions": dict(missing), "rows": len(rows),
        "conditions": audit["condition_counts"], "output_sha256": rb.sha256_file(args.output),
        "final_manifest_sha256": rb.sha256_file(args.final_manifest),
        "sampling_sha256": {str(path): rb.sha256_file(path) for path in args.sampling_record},
        "completed_at": rb.utc_now(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
