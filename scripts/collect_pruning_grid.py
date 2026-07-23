#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llmssycoph.pruning.global_selection import (
    P_GRID,
    Q_GRID,
    select_global_configuration,
)


def _rows_agree(left: pd.Series, right: pd.Series) -> bool:
    shared = [column for column in left.index if column in right.index]
    for column in shared:
        a, b = left[column], right[column]
        try:
            a_float, b_float = float(a), float(b)
        except (TypeError, ValueError):
            if str(a) != str(b):
                return False
        else:
            if np.isnan(a_float) and np.isnan(b_float):
                continue
            if not np.isclose(a_float, b_float, rtol=1e-10, atol=1e-12):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect per-configuration validation summaries and select the feasible mask."
    )
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--calibration-seed", type=int, default=5)
    parser.add_argument(
        "--artifact-identity",
        required=True,
        help="Content identity derived from the prune, preserve, and evaluation manifests.",
    )
    args = parser.parse_args()

    root = args.grid_root.expanduser().resolve()
    if root.name != args.artifact_identity:
        raise ValueError(
            "Grid root does not match --artifact-identity: "
            f"root={root}, identity={args.artifact_identity!r}"
        )
    summary_paths = sorted(root.glob("p_*_q_*/evaluation/selection_summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"No candidate selection summaries found below {root}")

    baseline: pd.Series | None = None
    candidates = []
    sources: list[dict[str, Any]] = []
    for path in summary_paths:
        frame = pd.read_csv(path)
        base_rows = frame[(frame["p"].astype(float) == 0) & (frame["q"].astype(float) == 0)]
        candidate_rows = frame[
            ~((frame["p"].astype(float) == 0) & (frame["q"].astype(float) == 0))
        ]
        if len(base_rows) != 1 or len(candidate_rows) != 1:
            raise ValueError(
                f"{path} must contain exactly one baseline and candidate row; "
                f"found {len(base_rows)} and {len(candidate_rows)}"
            )
        current_baseline = base_rows.iloc[0]
        if baseline is None:
            baseline = current_baseline
        elif not _rows_agree(baseline, current_baseline):
            raise ValueError(f"Baseline metrics are not identical across grid files; mismatch at {path}")
        candidates.append(candidate_rows.iloc[0])
        sources.append(
            {
                "path": str(path),
                "p": float(candidate_rows.iloc[0]["p"]),
                "q": float(candidate_rows.iloc[0]["q"]),
            }
        )
    assert baseline is not None
    combined = pd.DataFrame([baseline, *candidates]).sort_values(
        ["q", "p"], kind="stable"
    )
    duplicate = combined.duplicated(["split", "calibration_seed", "p", "q"], keep=False)
    if duplicate.any():
        raise ValueError(
            "Grid contains duplicate configurations: "
            f"{combined.loc[duplicate, ['split', 'calibration_seed', 'p', 'q']].to_dict('records')}"
        )

    result, audit = select_global_configuration(
        combined,
        split=args.split,
        calibration_seed=args.calibration_seed,
    )
    observed = {
        (float(row.p), float(row.q))
        for row in combined.itertuples()
        if not (float(row.p) == 0 and float(row.q) == 0)
    }
    expected = {(p, q) for p in P_GRID for q in Q_GRID}
    skipped_paths = sorted(root.glob("p_*_q_*/skipped.json"))
    manifest = {
        "schema_version": 1,
        "grid_root": str(root),
        "artifact_identity": args.artifact_identity,
        "split": args.split,
        "calibration_seed": args.calibration_seed,
        "expected_candidate_count": len(expected),
        "observed_candidate_count": len(observed),
        "missing_configurations": [
            {"p": p, "q": q} for p, q in sorted(expected - observed, key=lambda pair: (pair[1], pair[0]))
        ],
        "early_stop_records": [
            {"path": str(path), "payload": json.loads(path.read_text(encoding="utf-8"))}
            for path in skipped_paths
        ],
        "sources": sources,
        "selection": result.to_dict(),
    }
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination / "validation_grid_summary.csv", index=False)
    audit.to_csv(destination / "selection_audit.csv", index=False)
    (destination / "selected_configuration.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
