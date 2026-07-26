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
    aggregate_count_column: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    output: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(list(group_columns), dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, key_values))
        units = list(group.groupby(unit_column))
        if not units:
            continue
        compacted = bool(
            aggregate_count_column
            and aggregate_count_column in group
            and group[aggregate_count_column].notna().all()
        )
        weights = np.asarray(
            [
                (
                    float(unit_frame[aggregate_count_column].iloc[0])
                    if compacted and aggregate_count_column is not None
                    else 1.0
                )
                for _, unit_frame in units
            ],
            dtype=np.float64,
        )
        row["n_units"] = int(weights.sum())
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
            row[f"{metric}_mean"] = float(np.average(values, weights=weights))
            if not compacted and int(n_bootstrap) > 0:
                samples = rng.integers(
                    0,
                    len(values),
                    size=(int(n_bootstrap), len(values)),
                )
                bootstrap = values[samples].mean(axis=1)
                row[f"{metric}_ci_low"] = float(
                    np.quantile(bootstrap, 0.025)
                )
                row[f"{metric}_ci_high"] = float(
                    np.quantile(bootstrap, 0.975)
                )
            else:
                row[f"{metric}_ci_low"] = None
                row[f"{metric}_ci_high"] = None
        row["interval_status"] = (
            "paired_question_bootstrap"
            if not compacted and int(n_bootstrap) > 0
            else "not_bootstrapped_compacted_control"
            if compacted
            else "bootstrap_disabled"
        )
        output.append(row)
    return pd.DataFrame(output)


def aggregate_fixed_probe(
    paths: Sequence[Path],
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    metrics = (
        "probe_correct_top1",
        "probe_correct_rank",
        "probe_margin_correct_minus_endorsed",
        "external_probe_top1_agreement",
        "external_probe_correctness_agreement",
        "external_probe_margin_sign_agreement",
    )
    group_columns = (
        "model_name",
        "dataset",
        "split",
        "direction_fit_scope",
        "condition",
        "layer",
        "direction_name",
        "scale_convention",
        "control_seed",
        "alpha",
        "probe_structurally_informative",
        "treatment_type",
    )
    compact_frames: list[pd.DataFrame] = []
    input_wide_rows = 0
    retained_learned_rows = 0
    compacted_control_rows = 0
    for path in paths:
        file_rows = read_jsonl(path)
        _require_protocol(file_rows, stage="score_fixed_probe", source=path)
        shard = pd.DataFrame(file_rows)
        del file_rows
        input_wide_rows += len(shard)
        if "scoring_mode" in shard:
            shard = shard[shard["scoring_mode"].eq("strict_choice")].copy()
        if shard.empty:
            continue
        if "split" not in shard:
            shard["split"] = "unknown"
        if "direction_fit_scope" not in shard:
            shard["direction_fit_scope"] = "unknown"
        if "treatment_type" not in shard:
            shard["treatment_type"] = np.where(
                shard["control_seed"].notna(),
                "control",
                "learned",
            )
        else:
            inferred_treatment = pd.Series(
                np.where(
                    shard["control_seed"].notna(),
                    "control",
                    "learned",
                ),
                index=shard.index,
            )
            shard["treatment_type"] = shard["treatment_type"].fillna(
                inferred_treatment
            )
        required = {
            "stable_question_key",
            *group_columns,
            *metrics,
        }
        missing = sorted(required - set(shard.columns))
        if missing:
            raise ValueError(f"Fixed-probe rows are missing fields: {missing}")
        for metric in metrics:
            shard[metric] = pd.to_numeric(shard[metric], errors="raise")
            if not np.isfinite(shard[metric].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"Non-finite fixed-probe metric: {metric}.")
        learned = shard[shard["treatment_type"].eq("learned")].copy()
        controls = shard[shard["treatment_type"].eq("control")].copy()
        retained_learned_rows += len(learned)
        learned["aggregated_n_units"] = np.nan
        if not controls.empty:
            controls = (
                controls.groupby(
                    list(group_columns),
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    {
                        **{metric: "mean" for metric in metrics},
                        "stable_question_key": "nunique",
                    }
                )
                .rename(
                    columns={
                        "stable_question_key": "aggregated_n_units",
                    }
                )
            )
            controls["stable_question_key"] = [
                "__probe_control_summary__::"
                + "::".join(str(value) for value in row)
                for row in controls[list(group_columns)].itertuples(
                    index=False,
                    name=None,
                )
            ]
            compacted_control_rows += len(controls)
        compact_frames.append(
            pd.concat((learned, controls), ignore_index=True, sort=False)[
                [
                    "stable_question_key",
                    *group_columns,
                    *metrics,
                    "aggregated_n_units",
                ]
            ]
        )
        del shard
    if not compact_frames:
        raise ValueError("Fixed-probe inputs contain no strict-choice rows.")
    frame = pd.concat(compact_frames, ignore_index=True)
    if {"arc_challenge", "commonsense_qa"}.issubset(
        set(frame["dataset"].astype(str))
    ):
        pooled = frame.copy()
        pooled["dataset"] = "pooled_arc_csqa"
        frame = pd.concat((frame, pooled), ignore_index=True)
    summary = _paired_bootstrap_summary(
        frame,
        group_columns=group_columns,
        unit_column="stable_question_key",
        metrics=metrics,
        n_bootstrap=n_bootstrap,
        seed=seed,
        aggregate_count_column="aggregated_n_units",
    )
    summary.attrs["aggregation_memory_policy"] = {
        "raw_wide_shards_preserved": True,
        "learned_rows_retained_at_question_level": True,
        "controls_compacted_to_seed_level_weighted_means": True,
        "compacted_control_intervals": (
            "not_bootstrapped; null uncertainty is represented across seeds"
        ),
        "input_wide_rows": int(input_wide_rows),
        "retained_learned_rows": int(retained_learned_rows),
        "compacted_control_rows": int(compacted_control_rows),
    }
    return summary


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
            "fixed_probe_aggregation_memory_policy": fixed_probe.attrs.get(
                "aggregation_memory_policy",
                {},
            ),
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
