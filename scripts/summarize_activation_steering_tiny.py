#!/usr/bin/env python3
"""Combine tiny-run manifests into the mandatory full-compute projection."""

from __future__ import annotations

import argparse
from pathlib import Path

from llmssycoph.interventions.controlled import (
    PROTOCOL_VERSION,
    read_json,
    sha256_file,
    write_strict_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = [read_json(path) for path in args.manifest]
    if any(row.get("protocol_version") != PROTOCOL_VERSION for row in manifests):
        raise ValueError("Tiny-run protocol mismatch.")
    if any(row.get("stage") != "tiny_dry_run" for row in manifests):
        raise ValueError("Every compute-projection input must be a tiny-dry-run manifest.")
    if any(not bool(row.get("score_fixed_probe")) for row in manifests):
        raise ValueError("Compute projection requires fixed-probe scoring in every tiny cell.")

    elapsed_seconds = sum(float(row["elapsed_seconds"]) for row in manifests)
    strict_rows = sum(int(row["n_strict_choice_rows"]) for row in manifests)
    if elapsed_seconds <= 0 or strict_rows <= 0:
        raise ValueError("Tiny-run manifests have no measurable strict-choice work.")
    projection = dict(manifests[0]["compute_projection"])
    full_rows = int(projection["full_strict_choice_rows"])
    full_probe_passes = int(projection["full_fixed_probe_candidate_passes"])
    projected_gpu_hours = elapsed_seconds * full_rows / strict_rows / 3600.0
    write_strict_json(
        args.output,
        {
            "protocol_version": PROTOCOL_VERSION,
            "stage": "tiny_compute_projection",
            "status": "requires_researcher_review",
            "input_manifests": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for path in args.manifest
            ],
            "observed_elapsed_seconds": elapsed_seconds,
            "observed_strict_choice_rows": strict_rows,
            "observed_strict_rows_per_second": strict_rows / elapsed_seconds,
            "batch_policies": [row["batch_policy"] for row in manifests],
            "any_forced_batch_size_one": any(
                bool(row["batch_policy"]["forced_batch_size_one"])
                for row in manifests
            ),
            "full_strict_choice_rows": full_rows,
            "full_fixed_probe_candidate_passes": full_probe_passes,
            "projected_gpu_hours": projected_gpu_hours,
            "projection_note": (
                "Conservative row-rate scaling includes prompt extraction, no-op "
                "sentinels, fixed-probe passes, and configured generation diagnostics."
            ),
            "authorizes_full_submission": False,
        },
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
