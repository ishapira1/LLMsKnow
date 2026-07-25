#!/usr/bin/env python3
"""Aggregate fixed-probe, geometry, and Alpaca controlled-steering outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from llmssycoph.interventions.controlled import (
    PROTOCOL_VERSION,
    read_json,
    read_jsonl,
    sha256_file,
    write_strict_json,
)


def _require_protocol(
    rows: Iterable[Mapping[str, Any]],
    *,
    stage: str,
    source: Path,
) -> None:
    for index, row in enumerate(rows):
        if row.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Protocol mismatch at {source}:{index + 1}.")
        if row.get("stage") != stage:
            raise ValueError(f"Stage mismatch at {source}:{index + 1}.")


def _paired_bootstrap_summary(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    unit_column: str,
    metrics: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    output: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(list(group_columns), dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, key_values))
        units = list(group.groupby(unit_column))
        if not units:
            continue
        row["n_units"] = len(units)
        for metric in metrics:
            values = np.asarray(
                [
                    float(
                        pd.to_numeric(unit_frame[metric], errors="raise")
                        .astype(float)
                        .mean()
                    )
                    for _, unit_frame in units
                ],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {metric} in supplementary aggregation.")
            samples = rng.integers(
                0,
                len(values),
                size=(int(n_bootstrap), len(values)),
            )
            bootstrap = values[samples].mean(axis=1)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci_low"] = float(np.quantile(bootstrap, 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(bootstrap, 0.975))
        output.append(row)
    return pd.DataFrame(output)


def aggregate_fixed_probe(
    paths: Sequence[Path],
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        file_rows = read_jsonl(path)
        _require_protocol(file_rows, stage="score_fixed_probe", source=path)
        rows.extend(file_rows)
    frame = pd.DataFrame(rows)
    if "scoring_mode" in frame:
        frame = frame[frame["scoring_mode"].eq("strict_choice")].copy()
    metrics = (
        "probe_correct_top1",
        "probe_correct_rank",
        "probe_margin_correct_minus_endorsed",
        "external_probe_top1_agreement",
        "external_probe_correctness_agreement",
        "external_probe_margin_sign_agreement",
    )
    required = {
        "stable_question_key",
        "model_name",
        "dataset",
        "condition",
        "layer",
        "direction_name",
        "scale_convention",
        "control_seed",
        "alpha",
        "probe_structurally_informative",
        *metrics,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fixed-probe rows are missing fields: {missing}")
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    return _paired_bootstrap_summary(
        frame,
        group_columns=(
            "model_name",
            "dataset",
            "condition",
            "layer",
            "direction_name",
            "scale_convention",
            "control_seed",
            "alpha",
            "probe_structurally_informative",
        ),
        unit_column="stable_question_key",
        metrics=metrics,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def aggregate_alpaca(
    paths: Sequence[Path],
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        file_rows = read_jsonl(path)
        _require_protocol(file_rows, stage="alpaca_guardrail", source=path)
        rows.extend(file_rows)
    frame = pd.DataFrame(rows)
    required = {
        "example_id",
        "model_name",
        "layer",
        "alpha",
        "target_mean_nll",
        "target_perplexity",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Alpaca rows are missing fields: {missing}")
    frame["target_mean_nll"] = pd.to_numeric(
        frame["target_mean_nll"], errors="raise"
    )
    frame["target_perplexity"] = pd.to_numeric(
        frame["target_perplexity"], errors="raise"
    )
    baseline_rows = frame[frame["alpha"].eq(0.0)]
    baseline_counts = baseline_rows.groupby(
        ["model_name", "example_id"],
        dropna=False,
    ).size()
    if baseline_counts.empty or not baseline_counts.eq(1).all():
        raise ValueError("Alpaca requires exactly one alpha-zero baseline per example.")
    baseline = baseline_rows[
        ["model_name", "example_id", "target_mean_nll"]
    ]
    frame = frame.merge(
        baseline,
        on=["model_name", "example_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_alpha_zero"),
    )
    if frame["target_mean_nll_alpha_zero"].isna().any():
        raise ValueError("Alpaca rows lack paired alpha-zero baselines.")
    frame["delta_target_mean_nll"] = (
        frame["target_mean_nll"] - frame["target_mean_nll_alpha_zero"]
    )
    return _paired_bootstrap_summary(
        frame,
        group_columns=("model_name", "layer", "alpha"),
        unit_column="example_id",
        metrics=(
            "target_mean_nll",
            "target_perplexity",
            "delta_target_mean_nll",
        ),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def aggregate_geometry(paths: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    pair_frames: list[pd.DataFrame] = []
    pair_metrics = ("raw_cosine", "centered_cosine", "normalized_euclidean_distance")
    for path in paths:
        payload = read_json(path)
        if (
            payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("stage") != "run_geometry"
        ):
            raise ValueError(f"Geometry protocol/stage mismatch: {path}")
        pairs_path = path.parent / "geometry_pairs.csv"
        if (
            not pairs_path.is_file()
            or payload.get("pairs_csv_sha256") != sha256_file(pairs_path)
        ):
            raise ValueError(f"Geometry pair hash mismatch: {path}")
        for row in list(payload.get("summary", []) or []):
            summaries.append(
                {
                    "model_name": payload.get("model_name"),
                    "dataset": payload.get("dataset"),
                    "split": payload.get("split"),
                    "n_questions": payload.get("n_questions"),
                    **dict(row),
                }
            )
        pairs = pd.read_csv(pairs_path)
        missing = sorted({"layer", "group", *pair_metrics} - set(pairs.columns))
        if missing:
            raise ValueError(f"Geometry pairs are missing fields: {missing}")
        pairs.insert(0, "dataset", payload.get("dataset"))
        pairs.insert(0, "model_name", payload.get("model_name"))
        pair_frames.append(pairs)
    pair_frame = pd.concat(pair_frames, ignore_index=True)
    pair_summary_rows: list[dict[str, Any]] = []
    for group_key, group in pair_frame.groupby(
        ["model_name", "dataset", "layer", "group"],
        dropna=False,
    ):
        row = dict(zip(("model_name", "dataset", "layer", "group"), group_key))
        row["n_pairs"] = len(group)
        for metric in pair_metrics:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite geometry metric: {metric}.")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(np.median(values))
        pair_summary_rows.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(pair_summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-probe", type=Path, action="append", required=True)
    parser.add_argument("--geometry", type=Path, action="append", required=True)
    parser.add_argument("--alpaca", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    target = args.output_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    fixed_probe = aggregate_fixed_probe(
        args.fixed_probe,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    alpaca = aggregate_alpaca(
        args.alpaca,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    geometry, geometry_pairs = aggregate_geometry(args.geometry)
    outputs = {
        "fixed_probe_summary": target / "fixed_probe_summary.csv",
        "alpaca_guardrail_summary": target / "alpaca_guardrail_summary.csv",
        "geometry_summary": target / "geometry_summary.csv",
        "geometry_pair_metrics_summary": (
            target / "geometry_pair_metrics_summary.csv"
        ),
    }
    fixed_probe.to_csv(outputs["fixed_probe_summary"], index=False)
    alpaca.to_csv(outputs["alpaca_guardrail_summary"], index=False)
    geometry.to_csv(outputs["geometry_summary"], index=False)
    geometry_pairs.to_csv(outputs["geometry_pair_metrics_summary"], index=False)
    write_strict_json(
        target / "manifest.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "stage": "aggregate_supplementary",
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "inputs": {
                "fixed_probe": {
                    str(path.resolve()): sha256_file(path)
                    for path in args.fixed_probe
                },
                "geometry": {
                    str(path.resolve()): sha256_file(path)
                    for path in args.geometry
                },
                "alpaca": {
                    str(path.resolve()): sha256_file(path) for path in args.alpaca
                },
            },
            "outputs": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "rows": int(
                        {
                            "fixed_probe_summary": len(fixed_probe),
                            "alpaca_guardrail_summary": len(alpaca),
                            "geometry_summary": len(geometry),
                            "geometry_pair_metrics_summary": len(geometry_pairs),
                        }[name]
                    ),
                }
                for name, path in outputs.items()
            },
        },
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
