from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from .offline_evaluation import (
    DEFAULT_STRONG_FAMILY,
    DEFAULT_WEAK_FAMILY,
    GLOBAL_SELECTION_COLUMNS,
    aggregate_offline_evaluation,
    pair_item_tables,
    read_item_table,
)
from .global_selection import select_global_configuration


PACKAGE_SCHEMA_VERSION = 1
REQUIRED_SEEDS: Tuple[int, ...] = (5, 17, 29)
LOCKED_MODELS: Tuple[str, ...] = (
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)
REQUIRED_CONTROL_VARIANTS: Tuple[str, ...] = (
    "primary",
    "structure_matched",
    "opposite_sign",
    "second_slice",
    "random_magnitude",
)
REPHRASE_FAMILIES: Tuple[str, ...] = (
    "incorrect_suggestion_rephrase_1",
    "incorrect_suggestion_rephrase_2",
)

TEAL = "#73b3ab"
ORANGE = "#d4651a"
NAVY = "#355070"
PURPLE = "#8e6c8a"
GRAY = "#8a8f98"
LIGHT_GRAY = "#d9dde3"
PALETTE: Tuple[str, ...] = (TEAL, ORANGE, NAVY, PURPLE, GRAY)

FIGURE_NAMES: Tuple[str, ...] = (
    "figure_1_answer_transitions",
    "figure_2_truth_restoration",
    "figure_3_preservation_deltas",
    "figure_4_intervention_specificity",
    "figure_5_sparsity_tradeoff",
    "figure_6_generalization",
)


class ResultPackageError(ValueError):
    """Raised when a result package cannot be built without guessing."""


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    model: str
    revision: str
    calibration_seed: int
    variant: str
    split: str
    p: float
    q: float
    actual_mask_count: int
    evaluation_dir: Path
    evaluation_manifest_sha256: str
    paired: pd.DataFrame
    selection: pd.DataFrame
    candidate_evaluation_path: Path
    mask_indices_sha256: Optional[str]
    mask_metadata_sha256: Optional[str]
    mask_counts_by_module: Mapping[str, int]
    mask_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class GridRun:
    model: str
    revision: str
    artifact_identity: str
    summary_path: Path
    selection_path: Path
    summary: pd.DataFrame
    selection_payload: Mapping[str, Any]


@dataclass(frozen=True)
class PackageInputs:
    experiment_root: Path
    primary_test_runs: Tuple[EvaluationRun, ...]
    control_validation_runs: Tuple[EvaluationRun, ...]
    replication_validation_runs: Tuple[EvaluationRun, ...]
    grids: Tuple[GridRun, ...]
    selection_outcomes: Mapping[Tuple[str, str], Mapping[str, Any]]
    artifact_hashes: Mapping[str, str]


@dataclass(frozen=True)
class ResultTables:
    transitions: pd.DataFrame
    truth_restoration: pd.DataFrame
    preservation: pd.DataFrame
    controls: pd.DataFrame
    tradeoff: pd.DataFrame
    generalization: pd.DataFrame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Required result artifact is missing: {source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultPackageError(f"Invalid JSON artifact {source}: {exc}") from exc


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultPackageError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ResultPackageError(f"{field} must be finite, got {number!r}")
    return number


def _single_value(frame: pd.DataFrame, column: str, *, context: str) -> Any:
    if column not in frame.columns:
        raise ResultPackageError(f"{context} is missing required column {column!r}")
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ResultPackageError(
            f"{context} must have one {column!r} value, found {values.tolist()[:5]}"
        )
    return values.iloc[0]


def _resolve_manifest_output(
    evaluation_dir: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or name not in outputs:
        raise ResultPackageError(
            f"{evaluation_dir}/offline_evaluation_manifest.json has no output {name!r}"
        )
    recorded = Path(str(outputs[name])).expanduser()
    expected = (evaluation_dir / f"{name}.csv").resolve()
    if recorded.resolve() != expected:
        raise ResultPackageError(
            f"Offline output path identity mismatch for {name}: recorded={recorded.resolve()}, "
            f"expected={expected}"
        )
    if not expected.is_file():
        raise FileNotFoundError(f"Required offline result table is missing: {expected}")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, Mapping) or name not in output_hashes:
        raise ResultPackageError(
            f"Offline manifest has no SHA-256 provenance for output {name!r}: "
            f"{evaluation_dir}/offline_evaluation_manifest.json"
        )
    observed_hash = _sha256_file(expected)
    if str(output_hashes[name]) != observed_hash:
        raise ResultPackageError(
            f"Offline output SHA-256 mismatch for {name}: expected={output_hashes[name]}, "
            f"observed={observed_hash}"
        )
    return expected


def _validate_live_output(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    metadata_path = source.parent / "live_inference_metadata.json"
    metadata = _read_json(metadata_path)
    outputs = metadata.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ResultPackageError(f"Live metadata has no outputs mapping: {metadata_path}")
    output_name = "candidate_items" if source.name == "candidate_items.jsonl" else "baseline_items"
    recorded = outputs.get(output_name)
    if not isinstance(recorded, Mapping):
        raise ResultPackageError(f"Live metadata has no {output_name!r}: {metadata_path}")
    if Path(str(recorded.get("path", ""))).expanduser().resolve() != source:
        raise ResultPackageError(f"Live output path mismatch for {source}")
    observed_hash = _sha256_file(source)
    if str(recorded.get("sha256", "")) != observed_hash:
        raise ResultPackageError(f"Live output SHA-256 mismatch for {source}")
    try:
        claimed_rows = int(recorded["rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultPackageError(f"Invalid live output row count for {source}") from exc
    observed_rows = sum(1 for line in source.open("r", encoding="utf-8") if line.strip())
    if claimed_rows != observed_rows:
        raise ResultPackageError(
            f"Live output row-count mismatch for {source}: claimed={claimed_rows}, observed={observed_rows}"
        )
    evaluation_manifest = metadata.get("evaluation_manifest")
    if not isinstance(evaluation_manifest, Mapping):
        raise ResultPackageError(f"Live metadata has no evaluation-manifest identity: {metadata_path}")
    evaluation_source = Path(str(evaluation_manifest.get("path", ""))).expanduser().resolve()
    if not evaluation_source.is_file() or _sha256_file(evaluation_source) != str(
        evaluation_manifest.get("sha256", "")
    ):
        raise ResultPackageError(
            f"Held-out evaluation manifest is missing or changed since live inference: {evaluation_source}"
        )
    mask = metadata.get("mask")
    if not isinstance(mask, Mapping):
        raise ResultPackageError(f"Live metadata has no mask identity: {metadata_path}")
    for path_key, hash_key in (
        ("indices_path", "indices_sha256"),
        ("metadata_path", "metadata_sha256"),
    ):
        recorded_path = mask.get(path_key)
        recorded_hash = mask.get(hash_key)
        if recorded_path is None and recorded_hash is None:
            continue
        mask_source = Path(str(recorded_path)).expanduser().resolve()
        if not mask_source.is_file() or _sha256_file(mask_source) != str(recorded_hash):
            raise ResultPackageError(
                f"Mask artifact is missing or changed since live inference: {mask_source}"
            )
    return metadata


def _validate_mask_provenance(
    mask: Mapping[str, Any],
    *,
    model: str,
    revision: str,
    p: float,
    q: float,
    seed: int,
    actual_mask_count: int,
    context: str,
) -> Dict[str, str]:
    """Verify that live predictions still point to the exact mask that was applied."""

    if int(mask.get("actual_mask_count", -1)) != int(actual_mask_count):
        raise ResultPackageError(f"Live/offline mask-count mismatch in {context}")
    if q == 0:
        if actual_mask_count != 0 or str(mask.get("kind", "")) != "base_model":
            raise ResultPackageError(f"q=0 run is not an unmodified base model in {context}")
        if any(mask.get(key) is not None for key in ("indices_path", "metadata_path")):
            raise ResultPackageError(f"q=0 run unexpectedly references mask files in {context}")
        return {}

    if actual_mask_count <= 0 or str(mask.get("kind", "")) != "harm_indices":
        raise ResultPackageError(f"Masked run has no valid selected-weight mask in {context}")
    if _finite_float(mask.get("alpha"), field=f"{context}: mask alpha") != 0.0:
        raise ResultPackageError(f"Selected mask is not a zeroing intervention in {context}")

    verified: Dict[str, str] = {}
    resolved_paths: Dict[str, Path] = {}
    for path_key, hash_key in (
        ("indices_path", "indices_sha256"),
        ("metadata_path", "metadata_sha256"),
    ):
        value = mask.get(path_key)
        claimed_hash = str(mask.get(hash_key, ""))
        if not value or len(claimed_hash) != 64:
            raise ResultPackageError(
                f"Masked run is missing {path_key}/{hash_key} provenance in {context}"
            )
        source = Path(str(value)).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Mask artifact is missing: {source}")
        observed_hash = _sha256_file(source)
        if observed_hash != claimed_hash:
            raise ResultPackageError(
                f"Mask artifact SHA-256 mismatch for {source}: "
                f"claimed={claimed_hash}, observed={observed_hash}"
            )
        resolved_paths[path_key] = source
        verified[str(source)] = observed_hash

    metadata = _read_json(resolved_paths["metadata_path"])
    if not isinstance(metadata, Mapping):
        raise ResultPackageError(f"Mask metadata is not a JSON object in {context}")
    if int(metadata.get("surviving_count", -1)) != actual_mask_count:
        raise ResultPackageError(f"Mask metadata count disagrees with live inference in {context}")
    counts = metadata.get("counts_by_module")
    if not isinstance(counts, Mapping) or sum(int(value) for value in counts.values()) != actual_mask_count:
        raise ResultPackageError(f"Mask per-module counts disagree with total count in {context}")
    if not math.isclose(
        _finite_float(metadata.get("p"), field=f"{context}: mask p"),
        p,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ) or not math.isclose(
        _finite_float(metadata.get("q"), field=f"{context}: mask q"),
        q,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ResultPackageError(f"Mask metadata p/q disagrees with live inference in {context}")
    identity = metadata.get("score_identity")
    if not isinstance(identity, Mapping):
        raise ResultPackageError(f"Mask metadata has no score identity in {context}")
    if (
        str(identity.get("model", "")) != model
        or str(identity.get("revision", "")) != revision
        or int(identity.get("seed", -1)) != seed
    ):
        raise ResultPackageError(f"Mask model/revision/seed identity mismatch in {context}")
    return verified


def _compare_persisted_summary(
    observed: pd.DataFrame,
    persisted: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    context: str,
) -> None:
    missing = [column for column in (*key_columns, *value_columns) if column not in persisted.columns]
    if missing:
        raise ResultPackageError(f"{context} is missing required columns: {missing}")
    left = observed[list(key_columns) + list(value_columns)].copy()
    right = persisted[list(key_columns) + list(value_columns)].copy()
    merged = left.merge(right, on=list(key_columns), how="outer", suffixes=("_fresh", "_saved"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ResultPackageError(f"{context} keys disagree with freshly aggregated predictions")
    for column in value_columns:
        a = pd.to_numeric(merged[f"{column}_fresh"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(merged[f"{column}_saved"], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True):
            raise ResultPackageError(
                f"{context} column {column!r} disagrees with freshly aggregated predictions"
            )


def _variant_from_evaluation_dir(evaluation_dir: Path) -> str:
    config_dir = evaluation_dir.parent
    identity_dir = config_dir.parent
    variant_dir = identity_dir.parent
    selected_dir = variant_dir.parent
    if selected_dir.name == "selected":
        return variant_dir.name
    if selected_dir.name == "main" and variant_dir.name == "primary":
        return "primary"
    else:
        raise ResultPackageError(
            f"Evaluation directory has unexpected selected/grid layout: {evaluation_dir}"
        )


def _validate_variant_contract(
    variant: str,
    *,
    calibration_seed: int,
    metadata: Mapping[str, Any],
    context: str,
) -> None:
    expected = {
        "primary": (5, True, False, "none"),
        "primary_seed17": (17, True, False, "none"),
        "primary_seed29": (29, True, False, "none"),
        "structure_matched": (5, True, False, "structure_matched"),
        "opposite_sign": (5, False, False, "none"),
        "second_slice": (5, True, True, "none"),
        "random_magnitude": (5, True, False, "random_magnitude"),
    }
    if variant not in expected:
        raise ResultPackageError(f"Unsupported result-package variant {variant!r} in {context}")
    expected_seed, expected_negative, expected_second_slice, expected_control = expected[variant]
    if calibration_seed != expected_seed:
        raise ResultPackageError(f"Variant {variant!r} has the wrong calibration seed in {context}")
    if (
        bool(metadata.get("neg_prune")) != expected_negative
        or bool(metadata.get("freeze_first_top_q")) != expected_second_slice
        or str(metadata.get("control", "")) != expected_control
    ):
        raise ResultPackageError(f"Mask semantics do not match variant {variant!r} in {context}")

    identity = metadata.get("score_identity")
    if not isinstance(identity, Mapping):
        raise ResultPackageError(f"Mask has no score identity in {context}")
    expected_identity = {
        "score_format": "raw",
        "loss_mode": "completion_nll",
        "attribution_variant": "paper",
        "no_abs": True,
        "abs_prune": False,
        "abs_preserve": True,
    }
    mismatches = {
        field: (identity.get(field), value)
        for field, value in expected_identity.items()
        if identity.get(field) != value
    }
    if mismatches:
        raise ResultPackageError(
            f"Mask score identity does not match the primary paper contract in {context}: "
            f"{mismatches}"
        )

    random_audit = metadata.get("random_magnitude_match")
    if variant != "random_magnitude":
        if random_audit is not None:
            raise ResultPackageError(f"Non-random variant has a random-match audit in {context}")
        return
    counts = metadata.get("counts_by_module")
    if not isinstance(counts, Mapping) or not isinstance(random_audit, Mapping):
        raise ResultPackageError(f"Random control is missing per-module match audit in {context}")
    if set(counts) != set(random_audit):
        raise ResultPackageError(f"Random-control audit modules disagree with mask modules in {context}")
    for module, raw_count in counts.items():
        audit = random_audit[module]
        if not isinstance(audit, Mapping):
            raise ResultPackageError(f"Invalid random-control audit for {module!r} in {context}")
        target_bins = [int(value) for value in audit.get("target_bin_counts", [])]
        random_bins = [int(value) for value in audit.get("random_bin_counts", [])]
        count = int(raw_count)
        if (
            int(audit.get("numel", -1)) != count
            or sum(target_bins) != count
            or target_bins != random_bins
            or audit.get("exact_bin_match") is not True
            or audit.get("disjoint") is not True
        ):
            raise ResultPackageError(
                f"Random-control magnitude-bin audit failed for {module!r} in {context}"
            )


def load_evaluation_run(evaluation_dir: Path) -> Tuple[EvaluationRun, Dict[str, str]]:
    """Load one offline evaluation and verify it against its immutable live outputs."""

    directory = Path(evaluation_dir).expanduser().resolve()
    manifest_path = directory / "offline_evaluation_manifest.json"
    manifest = _read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != 1:
        raise ResultPackageError(f"Unsupported offline evaluation schema in {manifest_path}")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ResultPackageError(f"Offline manifest has no metadata mapping: {manifest_path}")
    baseline_path = Path(str(metadata.get("baseline", ""))).expanduser().resolve()
    candidate_path = Path(str(metadata.get("candidate", ""))).expanduser().resolve()
    if baseline_path.name != "candidate_items.jsonl" or candidate_path.name != "candidate_items.jsonl":
        raise ResultPackageError(
            "Canonical pruning comparisons must pair the replayed q=0 candidate_items.jsonl "
            "with the masked candidate_items.jsonl"
        )
    if not baseline_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError(
            f"Offline manifest points to missing live predictions: {baseline_path}, {candidate_path}"
        )
    for key, source in (("baseline_sha256", baseline_path), ("candidate_sha256", candidate_path)):
        claimed_hash = metadata.get(key)
        if not isinstance(claimed_hash, str) or claimed_hash != _sha256_file(source):
            raise ResultPackageError(
                f"Offline input SHA-256 mismatch for {key}: {manifest_path}"
            )
    baseline_live = _validate_live_output(baseline_path)
    candidate_live = _validate_live_output(candidate_path)

    guardrail_sources = metadata.get("guardrail_sources")
    if not isinstance(guardrail_sources, Mapping) or set(guardrail_sources) != {
        "baseline",
        "candidate",
    }:
        raise ResultPackageError(
            f"Offline manifest lacks hashed baseline/candidate guardrail sources: {manifest_path}"
        )
    guardrail_payloads: Dict[str, Mapping[str, Any]] = {}
    guardrail_paths: Dict[str, Path] = {}
    guardrail_hashes: Dict[str, str] = {}
    for role in ("baseline", "candidate"):
        source_record = guardrail_sources[role]
        if not isinstance(source_record, Mapping):
            raise ResultPackageError(f"Invalid {role} guardrail source in {manifest_path}")
        source = Path(str(source_record.get("path", ""))).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{role.title()} guardrail source is missing: {source}")
        observed_hash = _sha256_file(source)
        if observed_hash != str(source_record.get("sha256", "")):
            raise ResultPackageError(f"{role.title()} guardrail-source SHA-256 mismatch")
        payload = _read_json(source)
        if not isinstance(payload, Mapping):
            raise ResultPackageError(f"{role.title()} guardrail source is not a JSON object")
        for scalar in ("preservation_loss", "wikitext_perplexity"):
            persisted_value = _finite_float(
                source_record.get(scalar), field=f"{role} persisted {scalar}"
            )
            source_value = _finite_float(payload.get(scalar), field=f"{role} source {scalar}")
            if not math.isclose(persisted_value, source_value, rel_tol=1e-10, abs_tol=1e-12):
                raise ResultPackageError(
                    f"{role.title()} persisted {scalar} disagrees with its hashed source"
                )
        guardrail_payloads[role] = payload
        guardrail_paths[role] = source
        guardrail_hashes[str(source)] = observed_hash

    persisted_paths = {
        name: _resolve_manifest_output(directory, manifest, name)
        for name in ("paired_items", "family_summary", "metric_summary", "selection_summary")
    }
    saved_selection = pd.read_csv(persisted_paths["selection_summary"])
    saved_baseline = saved_selection[
        np.isclose(saved_selection["p"].astype(float), 0.0)
        & np.isclose(saved_selection["q"].astype(float), 0.0)
    ]
    if len(saved_baseline) != 1:
        raise ResultPackageError(
            f"Offline selection must contain one effective q=0 baseline: {persisted_paths['selection_summary']}"
        )
    baseline = read_item_table(baseline_path)
    candidate = read_item_table(candidate_path)
    # Sensitivity/replication jobs may reuse the common q=0 behavioral replay
    # while supplying a separately matched preservation/PPL baseline. Reapply
    # the values only from their independently hashed evaluation artifacts.
    for role, frame in (("baseline", baseline), ("candidate", candidate)):
        for scalar in ("preservation_loss", "wikitext_perplexity"):
            frame[scalar] = _finite_float(
                guardrail_payloads[role].get(scalar), field=f"{role} source {scalar}"
            )
    paired = pair_item_tables(baseline, candidate, calibration_seed=None)
    fresh = aggregate_offline_evaluation(paired, n_bootstrap=0)
    _compare_persisted_summary(
        fresh.selection_summary,
        saved_selection,
        key_columns=("split", "calibration_seed", "p", "q"),
        value_columns=(
            "actual_mask_count",
            "wrong_probability_uplift",
            "biased_correct_probability",
            "neutral_accuracy",
            "neutral_correct_probability",
            "correction_accuracy",
            "agreement_accuracy",
            "preservation_loss",
            "wikitext_perplexity",
            "other_wrong_invalid_rate",
            "b_to_c_recovery_rate",
        ),
        context=str(persisted_paths["selection_summary"]),
    )

    model_payload = candidate_live.get("model")
    baseline_model_payload = baseline_live.get("model")
    if not isinstance(model_payload, Mapping) or not isinstance(baseline_model_payload, Mapping):
        raise ResultPackageError("Live inference metadata is missing model identity")
    model = str(model_payload.get("model_id", ""))
    revision = str(model_payload.get("revision", ""))
    if not model or not revision:
        raise ResultPackageError("Candidate live inference has an empty model/revision identity")
    if (
        str(baseline_model_payload.get("model_id", "")) != model
        or str(baseline_model_payload.get("revision", "")) != revision
    ):
        raise ResultPackageError(f"Baseline/candidate model identity mismatch in {directory}")
    for role, payload in guardrail_payloads.items():
        if str(payload.get("model", "")) != model or str(payload.get("revision", "")) != revision:
            raise ResultPackageError(
                f"{role.title()} guardrail-source model identity mismatch in {directory}"
            )
    candidate_manifest = candidate_live.get("evaluation_manifest")
    baseline_manifest = baseline_live.get("evaluation_manifest")
    if not isinstance(candidate_manifest, Mapping) or not isinstance(baseline_manifest, Mapping):
        raise ResultPackageError("Live inference metadata is missing evaluation-manifest identity")
    evaluation_sha = str(candidate_manifest.get("sha256", ""))
    if not evaluation_sha or str(baseline_manifest.get("sha256", "")) != evaluation_sha:
        raise ResultPackageError(f"Baseline/candidate evaluation-manifest mismatch in {directory}")
    evaluation_manifest_hashes: Dict[str, str] = {}
    for role, manifest_identity in (
        ("baseline", baseline_manifest),
        ("candidate", candidate_manifest),
    ):
        source = Path(str(manifest_identity.get("path", ""))).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{role.title()} evaluation manifest is missing: {source}")
        observed_hash = _sha256_file(source)
        if observed_hash != str(manifest_identity.get("sha256", "")):
            raise ResultPackageError(
                f"{role.title()} evaluation-manifest SHA-256 mismatch in {directory}"
            )
        evaluation_manifest_hashes[str(source)] = observed_hash

    baseline_configuration = baseline_live.get("configuration")
    if not isinstance(baseline_configuration, Mapping):
        raise ResultPackageError(f"Baseline live metadata has no configuration: {directory}")
    baseline_p = _finite_float(
        baseline_configuration.get("p"), field=f"{directory}: baseline p"
    )
    baseline_q = _finite_float(
        baseline_configuration.get("q"), field=f"{directory}: baseline q"
    )
    baseline_seed_float = _finite_float(
        baseline_configuration.get("calibration_seed"),
        field=f"{directory}: baseline calibration_seed",
    )
    baseline_seed = int(baseline_seed_float)
    if baseline_seed_float != baseline_seed or baseline_p != 0.0 or baseline_q != 0.0:
        raise ResultPackageError(
            f"Canonical baseline must be the direct p=0, q=0 checkpoint in {directory}"
        )
    baseline_mask = baseline_live.get("mask")
    if not isinstance(baseline_mask, Mapping):
        raise ResultPackageError(f"Baseline live metadata has no mask provenance in {directory}")
    _validate_mask_provenance(
        baseline_mask,
        model=model,
        revision=revision,
        p=0.0,
        q=0.0,
        seed=baseline_seed,
        actual_mask_count=0,
        context=f"{directory} baseline",
    )

    configuration = candidate_live.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ResultPackageError(f"Candidate live metadata has no configuration: {directory}")
    p = _finite_float(configuration.get("p"), field=f"{directory}: p")
    q = _finite_float(configuration.get("q"), field=f"{directory}: q")
    seed_float = _finite_float(
        configuration.get("calibration_seed"), field=f"{directory}: calibration_seed"
    )
    seed = int(seed_float)
    if seed_float != seed:
        raise ResultPackageError(f"Non-integer calibration seed in {directory}")
    baseline_guardrail = guardrail_payloads["baseline"]
    candidate_guardrail = guardrail_payloads["candidate"]
    if (
        _finite_float(baseline_guardrail.get("p"), field="baseline guardrail p") != 0.0
        or _finite_float(baseline_guardrail.get("q"), field="baseline guardrail q") != 0.0
        or int(baseline_guardrail.get("seed", -1)) != seed
        or not math.isclose(
            _finite_float(candidate_guardrail.get("p"), field="candidate guardrail p"),
            p,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            _finite_float(candidate_guardrail.get("q"), field="candidate guardrail q"),
            q,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or int(candidate_guardrail.get("seed", -1)) != seed
    ):
        raise ResultPackageError(f"Guardrail-source p/q/seed identity mismatch in {directory}")
    split = str(_single_value(fresh.paired_items, "split", context=str(directory)))
    paired_p = _finite_float(_single_value(fresh.paired_items, "p", context=str(directory)), field="p")
    paired_q = _finite_float(_single_value(fresh.paired_items, "q", context=str(directory)), field="q")
    paired_seed = int(_single_value(fresh.paired_items, "calibration_seed", context=str(directory)))
    if not math.isclose(p, paired_p, rel_tol=1e-12, abs_tol=1e-15) or not math.isclose(
        q, paired_q, rel_tol=1e-12, abs_tol=1e-15
    ) or seed != paired_seed:
        raise ResultPackageError(f"Live/offline p/q/seed identity mismatch in {directory}")
    actual_mask_count = int(
        _single_value(fresh.paired_items, "actual_mask_count", context=str(directory))
    )
    mask = candidate_live.get("mask")
    if not isinstance(mask, Mapping):
        raise ResultPackageError(f"Candidate live metadata has no mask provenance in {directory}")
    mask_hashes = _validate_mask_provenance(
        mask,
        model=model,
        revision=revision,
        p=p,
        q=q,
        seed=seed,
        actual_mask_count=actual_mask_count,
        context=str(directory),
    )
    mask_metadata_path = mask.get("metadata_path")
    candidate_evaluation_path = (
        Path(str(mask_metadata_path)).expanduser().resolve().parent / "evaluation.json"
        if mask_metadata_path
        else Path()
    )
    if q > 0 and not candidate_evaluation_path.is_file():
        raise FileNotFoundError(
            f"Candidate utility/evaluation artifact is missing: {candidate_evaluation_path}"
        )
    candidate_guardrail_origin = Path(
        str(guardrail_sources["candidate"].get("original_path", ""))
    ).expanduser().resolve()
    if q > 0 and candidate_guardrail_origin != candidate_evaluation_path:
        raise ResultPackageError(
            f"Candidate guardrail snapshot has the wrong original evaluation path in {directory}"
        )
    variant = _variant_from_evaluation_dir(directory)
    mask_metadata: Mapping[str, Any] = {}
    if q > 0:
        mask_metadata = _read_json(Path(str(mask_metadata_path)))
        if not isinstance(mask_metadata, Mapping):
            raise ResultPackageError(f"Mask metadata is not a JSON object in {directory}")
        _validate_variant_contract(
            variant,
            calibration_seed=seed,
            metadata=mask_metadata,
            context=str(directory),
        )
    run_id = f"{model}|{revision}|seed={seed}|{variant}|{split}|p={p:.12g}|q={q:.12g}"
    hashes = {
        str(manifest_path): _sha256_file(manifest_path),
        str(baseline_path): _sha256_file(baseline_path),
        str(candidate_path): _sha256_file(candidate_path),
        str(baseline_path.parent / "live_inference_metadata.json"): _sha256_file(
            baseline_path.parent / "live_inference_metadata.json"
        ),
        str(candidate_path.parent / "live_inference_metadata.json"): _sha256_file(
            candidate_path.parent / "live_inference_metadata.json"
        ),
    }
    hashes.update(evaluation_manifest_hashes)
    hashes.update(mask_hashes)
    hashes.update(guardrail_hashes)
    hashes.update({str(path): _sha256_file(path) for path in persisted_paths.values()})
    return (
        EvaluationRun(
            run_id=run_id,
            model=model,
            revision=revision,
            calibration_seed=seed,
            variant=variant,
            split=split,
            p=p,
            q=q,
            actual_mask_count=actual_mask_count,
            evaluation_dir=directory,
            evaluation_manifest_sha256=evaluation_sha,
            paired=fresh.paired_items,
            selection=fresh.selection_summary,
            candidate_evaluation_path=candidate_evaluation_path,
            mask_indices_sha256=(str(mask.get("indices_sha256")) if q > 0 else None),
            mask_metadata_sha256=(str(mask.get("metadata_sha256")) if q > 0 else None),
            mask_counts_by_module={
                str(name): int(count)
                for name, count in (mask.get("counts_by_module", {}) or {}).items()
            },
            mask_metadata=dict(mask_metadata),
        ),
        hashes,
    )


def _model_label(model: str) -> str:
    lowered = model.lower()
    if "qwen" in lowered:
        return "Qwen2.5-7B-Instruct"
    if "llama" in lowered:
        return "Llama-3.1-8B-Instruct"
    return model.rsplit("/", 1)[-1]


def _variant_label(variant: str) -> str:
    return {
        "primary": "Targeted pruning",
        "structure_matched": "Correction-target",
        "opposite_sign": "Opposite sign",
        "second_slice": "Second slice",
        "random_magnitude": "Magnitude-matched random",
    }.get(variant, variant.replace("_", " ").title())


def _load_grid(selection_path: Path) -> Tuple[GridRun, Dict[str, str]]:
    path = Path(selection_path).expanduser().resolve()
    payload = _read_json(path)
    if str(payload.get("artifact_identity", "")) != path.parent.name:
        raise ResultPackageError(f"Selection artifact identity/path mismatch: {path}")
    summary_path = path.parent / "validation_grid_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Grid summary is missing: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {
        "p",
        "q",
        "split",
        "calibration_seed",
        "actual_mask_count",
        "wrong_probability_uplift",
        "neutral_accuracy",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ResultPackageError(f"Grid summary {summary_path} is missing columns: {missing}")
    if summary.duplicated(["split", "calibration_seed", "p", "q"]).any():
        raise ResultPackageError(f"Grid summary contains duplicate configurations: {summary_path}")
    if set(summary["split"].astype(str)) != {"val"} or set(
        pd.to_numeric(summary["calibration_seed"], errors="raise").astype(int)
    ) != {5}:
        raise ResultPackageError(f"Grid summary must be seed-5 validation only: {summary_path}")
    source_records = payload.get("sources")
    if not isinstance(source_records, list) or not source_records:
        raise ResultPackageError(f"Selection manifest has no sources list: {path}")
    source_hashes_from_manifest: Dict[str, str] = {}
    source_hashes: Dict[str, str] = {}
    source_runs: list[EvaluationRun] = []
    candidate_rows: list[pd.Series] = []
    baseline_row: Optional[pd.Series] = None
    for record in source_records:
        if not isinstance(record, Mapping):
            raise ResultPackageError(f"Invalid selection source record in {path}")
        source = Path(str(record.get("path", ""))).expanduser().resolve()
        if not source.is_file() or source.name != "selection_summary.csv":
            raise FileNotFoundError(f"Selection source no longer exists: {source}")
        source_hashes_from_manifest[str(source)] = _sha256_file(source)
        source_run, run_hashes = load_evaluation_run(source.parent)
        source_hashes.update(run_hashes)
        if source_run.variant != "primary" or source_run.split != "val" or source_run.calibration_seed != 5:
            raise ResultPackageError(
                f"Grid source is not a seed-5 primary validation run: {source}"
            )
        record_p = _finite_float(record.get("p"), field=f"{source}: recorded p")
        record_q = _finite_float(record.get("q"), field=f"{source}: recorded q")
        if not math.isclose(record_p, source_run.p, rel_tol=1e-12, abs_tol=1e-15) or not math.isclose(
            record_q, source_run.q, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ResultPackageError(f"Grid source p/q record disagrees with live inference: {source}")
        current_base = source_run.selection[
            np.isclose(source_run.selection["p"].astype(float), 0.0)
            & np.isclose(source_run.selection["q"].astype(float), 0.0)
        ]
        current_candidate = source_run.selection[
            np.isclose(source_run.selection["p"].astype(float), source_run.p)
            & np.isclose(source_run.selection["q"].astype(float), source_run.q)
        ]
        if len(current_base) != 1 or len(current_candidate) != 1:
            raise ResultPackageError(
                f"Grid source must contain one baseline and candidate row: {source}"
            )
        if baseline_row is None:
            baseline_row = current_base.iloc[0]
        else:
            _compare_persisted_summary(
                pd.DataFrame([baseline_row]),
                current_base,
                key_columns=("split", "calibration_seed", "p", "q"),
                value_columns=GLOBAL_SELECTION_COLUMNS[4:],
                context=f"baseline identity across grid sources at {source}",
            )
        candidate_rows.append(current_candidate.iloc[0])
        source_runs.append(source_run)

    assert baseline_row is not None
    source_identities = {
        (run.model, run.revision, run.evaluation_manifest_sha256) for run in source_runs
    }
    if len(source_identities) != 1:
        raise ResultPackageError(
            f"Grid sources disagree on model/revision/held-out identity: {source_identities}"
        )
    source_run = source_runs[0]
    reconstructed = pd.DataFrame([baseline_row, *candidate_rows]).reset_index(drop=True)
    _compare_persisted_summary(
        reconstructed,
        summary,
        key_columns=("split", "calibration_seed", "p", "q"),
        value_columns=GLOBAL_SELECTION_COLUMNS[4:],
        context=f"reconstructed validation grid {summary_path}",
    )

    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ResultPackageError(f"Selection manifest has no selection mapping: {path}")
    recomputed_selection, recomputed_audit = select_global_configuration(
        reconstructed,
        split="val",
        calibration_seed=5,
    )
    expected_selection = recomputed_selection.to_dict()
    for field in ("status", "actual_mask_count", "reason"):
        if selection.get(field) != expected_selection[field]:
            raise ResultPackageError(
                f"Persisted selection field {field!r} disagrees with recomputation in {path}"
            )
    for field in ("selected_p", "selected_q", "b_to_c_recovery_rate"):
        observed = selection.get(field)
        expected = expected_selection[field]
        if observed is None or expected is None:
            if observed is not expected:
                raise ResultPackageError(
                    f"Persisted selection field {field!r} disagrees with recomputation in {path}"
                )
        elif not math.isclose(float(observed), float(expected), rel_tol=1e-10, abs_tol=1e-12):
            raise ResultPackageError(
                f"Persisted selection field {field!r} disagrees with recomputation in {path}"
            )

    audit_path = path.parent / "selection_audit.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Selection audit is missing: {audit_path}")
    persisted_audit = pd.read_csv(audit_path).sort_values(["q", "p"], kind="stable").reset_index(drop=True)
    expected_audit = recomputed_audit.sort_values(["q", "p"], kind="stable").reset_index(drop=True)
    required_audit_columns = (
        "p",
        "q",
        "actual_mask_count",
        "b_to_c_recovery_rate",
        "feasible",
        "failure_reasons",
    )
    if any(column not in persisted_audit for column in required_audit_columns) or len(
        persisted_audit
    ) != len(expected_audit):
        raise ResultPackageError(f"Persisted selection audit has the wrong schema/size: {audit_path}")
    for column in required_audit_columns[:-1]:
        left = pd.to_numeric(persisted_audit[column], errors="raise").to_numpy(dtype=float)
        right = pd.to_numeric(expected_audit[column], errors="raise").to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-10, atol=1e-12):
            raise ResultPackageError(
                f"Persisted selection audit column {column!r} disagrees with recomputation"
            )
    if persisted_audit["failure_reasons"].fillna("").astype(str).tolist() != expected_audit[
        "failure_reasons"
    ].fillna("").astype(str).tolist():
        raise ResultPackageError("Persisted selection failure reasons disagree with recomputation")

    hashes = {
        str(path): _sha256_file(path),
        str(summary_path): _sha256_file(summary_path),
        str(audit_path): _sha256_file(audit_path),
    }
    hashes.update(source_hashes_from_manifest)
    hashes.update(source_hashes)
    return (
        GridRun(
            model=source_run.model,
            revision=source_run.revision,
            artifact_identity=str(payload["artifact_identity"]),
            summary_path=summary_path,
            selection_path=path,
            summary=summary,
            selection_payload=payload,
        ),
        hashes,
    )


def discover_package_inputs(experiment_root: Path) -> PackageInputs:
    root = Path(experiment_root).expanduser().resolve()
    analysis_root = root / "analysis"
    if not analysis_root.is_dir():
        raise FileNotFoundError(f"Experiment analysis directory is missing: {analysis_root}")

    selection_paths = sorted(
        analysis_root.glob(
            "*/revision_*/seed_5/main/selection/*/selected_configuration.json"
        )
    )
    if not selection_paths:
        raise FileNotFoundError(f"No seed-5 validation selections found below {analysis_root}")
    grids: list[GridRun] = []
    hashes: Dict[str, str] = {}
    for path in selection_paths:
        grid, grid_hashes = _load_grid(path)
        grids.append(grid)
        hashes.update(grid_hashes)
    grid_identities = [(grid.model, grid.revision) for grid in grids]
    if len(grids) != 2 or len(set(grid_identities)) != 2:
        raise ResultPackageError(
            f"Expected exactly one selection for each locked model, found {grid_identities}"
        )
    if {model for model, _revision in grid_identities} != set(LOCKED_MODELS):
        raise ResultPackageError(
            f"Selection models do not match the locked experiment: {grid_identities}"
        )
    outcomes: Dict[Tuple[str, str], Mapping[str, Any]] = {
        (grid.model, grid.revision): dict(grid.selection_payload["selection"])
        for grid in grids
    }
    selected_identities = {
        identity
        for identity, outcome in outcomes.items()
        if str(outcome.get("status", "")) == "selected"
    }

    selected_manifests = sorted(
        analysis_root.glob(
            "*/revision_*/seed_*/main/selected/*/*/p_*_q_*/evaluation_*/offline_evaluation_manifest.json"
        )
    )
    primary: list[EvaluationRun] = []
    controls: list[EvaluationRun] = []
    replication_validation: list[EvaluationRun] = []
    relevant_variants = set(REQUIRED_CONTROL_VARIANTS) | {
        "primary_seed17",
        "primary_seed29",
    }
    for manifest_path in selected_manifests:
        path_variant = manifest_path.parents[3].name
        if path_variant not in relevant_variants:
            continue
        run, run_hashes = load_evaluation_run(manifest_path.parent)
        hashes.update(run_hashes)
        if run.split == "test" and run.variant in {
            "primary",
            "primary_seed17",
            "primary_seed29",
        }:
            primary.append(run)
        if run.split == "val" and run.variant in REQUIRED_CONTROL_VARIANTS:
            controls.append(run)
        if run.split == "val" and run.variant in {"primary_seed17", "primary_seed29"}:
            replication_validation.append(run)

    observed_primary_identities = {(run.model, run.revision) for run in primary}
    unexpected = observed_primary_identities.difference(selected_identities)
    if unexpected:
        raise ResultPackageError(
            f"Found selected-mask test artifacts for no-feasible model identities: {sorted(unexpected)}"
        )
    missing_selected = selected_identities.difference(observed_primary_identities)
    if missing_selected:
        raise ResultPackageError(
            f"Selected models are missing final-test artifacts: {sorted(missing_selected)}"
        )
    for identity in sorted(selected_identities):
        runs = [run for run in primary if (run.model, run.revision) == identity]
        seeds = sorted(run.calibration_seed for run in runs)
        if seeds != list(REQUIRED_SEEDS):
            raise ResultPackageError(
                f"Primary test runs for {identity[0]} must contain seeds {REQUIRED_SEEDS}, found {seeds}"
            )
        if len({(run.p, run.q) for run in runs}) != 1:
            raise ResultPackageError(f"Replications changed selected (p,q) for {identity[0]}")
        if len({run.evaluation_manifest_sha256 for run in runs}) != 1:
            raise ResultPackageError(f"Replications changed held-out manifest for {identity[0]}")
        reference = [run for run in runs if run.calibration_seed == 5 and run.variant == "primary"]
        if len(reference) != 1:
            raise ResultPackageError(f"Expected one seed-5 primary test run for {identity[0]}")

        outcome = outcomes[identity]
        selected_p = _finite_float(outcome.get("selected_p"), field=f"{identity[0]} selected_p")
        selected_q = _finite_float(outcome.get("selected_q"), field=f"{identity[0]} selected_q")
        if any(
            not math.isclose(run.p, selected_p, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(run.q, selected_q, rel_tol=1e-12, abs_tol=1e-15)
            for run in runs
        ):
            raise ResultPackageError(
                f"Primary test p/q does not match seed-5 validation selection for {identity[0]}"
            )
        selected_count = int(outcome.get("actual_mask_count", -1))
        if reference[0].actual_mask_count != selected_count:
            raise ResultPackageError(
                f"Seed-5 test mask count does not match validation selection for {identity[0]}: "
                f"test={reference[0].actual_mask_count}, selected={selected_count}"
            )

        control_runs = [run for run in controls if (run.model, run.revision) == identity]
        by_variant = {run.variant: run for run in control_runs}
        missing_controls = sorted(set(REQUIRED_CONTROL_VARIANTS).difference(by_variant))
        if missing_controls:
            raise ResultPackageError(
                f"Missing seed-5 validation controls for {identity[0]}: {missing_controls}"
            )
        if len(control_runs) != len(by_variant):
            raise ResultPackageError(f"Duplicate validation control variants for {identity[0]}")
        control_identity = {
            (run.calibration_seed, run.split, run.p, run.q, run.evaluation_manifest_sha256)
            for run in control_runs
        }
        if len(control_identity) != 1:
            raise ResultPackageError(
                f"Control comparisons do not share seed/split/(p,q)/held-out identity for {identity[0]}"
            )
        if control_runs[0].evaluation_manifest_sha256 != reference[0].evaluation_manifest_sha256:
            raise ResultPackageError(
                f"Validation controls and final test do not share the fixed held-out manifest for {identity[0]}"
            )
        for run in control_runs:
            if not math.isclose(run.p, selected_p, rel_tol=1e-12, abs_tol=1e-15) or not math.isclose(
                run.q, selected_q, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ResultPackageError(
                    f"Control {run.variant!r} p/q does not match selection for {identity[0]}"
                )
        if by_variant["primary"].actual_mask_count != selected_count:
            raise ResultPackageError(
                f"Seed-5 validation primary mask count does not match selection for {identity[0]}"
            )
        random_run = by_variant["random_magnitude"]
        if random_run.mask_counts_by_module != by_variant["primary"].mask_counts_by_module:
            raise ResultPackageError(
                f"Magnitude-matched random control does not preserve exact per-module counts for {identity[0]}"
            )

        replication_runs = [
            run for run in replication_validation if (run.model, run.revision) == identity
        ]
        by_seed = {run.calibration_seed: run for run in replication_runs}
        if len(replication_runs) != len(by_seed) or set(by_seed) != {17, 29}:
            raise ResultPackageError(
                f"Validation replications for {identity[0]} must contain exactly seeds 17 and 29"
            )
        validation_by_seed = {5: by_variant["primary"], **by_seed}
        for test_run in runs:
            validation_run = validation_by_seed[test_run.calibration_seed]
            if (
                validation_run.evaluation_manifest_sha256
                != test_run.evaluation_manifest_sha256
                or not math.isclose(validation_run.p, test_run.p, rel_tol=1e-12, abs_tol=1e-15)
                or not math.isclose(validation_run.q, test_run.q, rel_tol=1e-12, abs_tol=1e-15)
                or validation_run.mask_indices_sha256 != test_run.mask_indices_sha256
                or validation_run.mask_metadata_sha256 != test_run.mask_metadata_sha256
                or validation_run.mask_counts_by_module != test_run.mask_counts_by_module
            ):
                raise ResultPackageError(
                    f"Validation/test mask identity mismatch for {identity[0]} seed "
                    f"{test_run.calibration_seed}"
                )

    return PackageInputs(
        experiment_root=root,
        primary_test_runs=tuple(sorted(primary, key=lambda run: (run.model, run.calibration_seed))),
        control_validation_runs=tuple(sorted(controls, key=lambda run: (run.model, run.variant))),
        replication_validation_runs=tuple(
            sorted(replication_validation, key=lambda run: (run.model, run.calibration_seed))
        ),
        grids=tuple(sorted(grids, key=lambda grid: grid.model)),
        selection_outcomes=outcomes,
        artifact_hashes=dict(sorted(hashes.items())),
    )


def _strict_flip_subset(
    paired: pd.DataFrame,
    *,
    family: str,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    subset = paired[paired["condition"].astype(str).eq(str(family))].copy()
    if dataset is not None:
        subset = subset[subset["dataset"].astype(str).eq(str(dataset))].copy()
    if subset.empty:
        qualifier = f", dataset={dataset!r}" if dataset is not None else ""
        raise ResultPackageError(f"No rows for family={family!r}{qualifier}")
    strict = (
        subset["baseline_neutral_status_category"].eq("valid")
        & subset["baseline_biased_status_category"].eq("valid")
        & subset["baseline_neutral_choice"].eq(subset["correct_letter"])
        & subset["baseline_biased_choice"].eq(subset["suggested_letter"])
    )
    eligible = subset[strict].copy()
    if eligible.empty:
        qualifier = f", dataset={dataset!r}" if dataset is not None else ""
        raise ResultPackageError(
            f"No baseline strict flips for family={family!r}{qualifier}; recovery is undefined"
        )
    return eligible


def _cluster_indicator_table(frame: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "cluster_id": frame["cluster_id"].astype(str).to_numpy(),
            "value": pd.to_numeric(values, errors="coerce").to_numpy(dtype=float),
        }
    )
    if not np.isfinite(work["value"].to_numpy(dtype=float)).all():
        raise ResultPackageError("Non-finite item contribution in result table")
    return work.groupby("cluster_id", sort=True, as_index=False)["value"].mean()


def _bootstrap_cluster_mean(
    cluster_values: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    values = cluster_values["value"].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ResultPackageError("Cannot bootstrap an empty or non-finite cluster table")
    estimate = float(values.mean())
    if values.size == 1 or n_bootstrap <= 0:
        return estimate, estimate, estimate
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_bootstrap), dtype=float)
    for start in range(0, int(n_bootstrap), 256):
        stop = min(start + 256, int(n_bootstrap))
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        draws[start:stop] = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return estimate, float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def transition_table(
    run: EvaluationRun,
    *,
    family: str = DEFAULT_STRONG_FAMILY,
    dataset: Optional[str] = None,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 5,
) -> pd.DataFrame:
    eligible = _strict_flip_subset(run.paired, family=family, dataset=dataset)
    status = eligible["candidate_biased_status_category"].astype(str)
    choice = eligible["candidate_biased_choice"].astype(str)
    valid = status.eq("valid")
    known = pd.Series(
        [selected in set(choices) for selected, choices in zip(choice, eligible["choice_letters"])],
        index=eligible.index,
        dtype=bool,
    )
    indicators = {
        "b → c": valid & choice.eq(eligible["correct_letter"]),
        "b → b": valid & choice.eq(eligible["suggested_letter"]),
        "b → other wrong": (
            valid
            & known
            & ~choice.eq(eligible["correct_letter"])
            & ~choice.eq(eligible["suggested_letter"])
        ),
        "b → invalid/refusal": status.isin(["invalid", "refusal", "malformed"]),
    }
    coverage = sum(indicator.astype(int) for indicator in indicators.values())
    if not coverage.eq(1).all():
        bad = eligible.loc[~coverage.eq(1), ["question_id", "candidate_biased_choice", "candidate_biased_status_category"]]
        raise ResultPackageError(
            f"Transition categories are not exhaustive/exclusive for {run.run_id}: "
            f"{bad.head(3).to_dict('records')}"
        )
    rows: list[dict[str, Any]] = []
    for index, (transition, indicator) in enumerate(indicators.items()):
        clusters = _cluster_indicator_table(eligible, indicator.astype(float))
        estimate, ci_low, ci_high = _bootstrap_cluster_mean(
            clusters,
            seed=int(bootstrap_seed) + index,
            n_bootstrap=n_bootstrap,
        )
        rows.append(
            {
                "run_id": run.run_id,
                "model": run.model,
                "model_label": _model_label(run.model),
                "calibration_seed": run.calibration_seed,
                "variant": run.variant,
                "split": run.split,
                "family": family,
                "dataset": dataset or "all",
                "phase": "Pruned",
                "transition": transition,
                "rate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_questions": int(clusters.shape[0]),
                "n_rows": int(len(eligible)),
            }
        )
    return pd.DataFrame(rows)


def _metric_delta_rows(
    run: EvaluationRun,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    eligible = _strict_flip_subset(run.paired, family=DEFAULT_STRONG_FAMILY)
    baseline_invalid = ~eligible["baseline_biased_status_category"].eq("valid")
    candidate_invalid = ~eligible["candidate_biased_status_category"].eq("valid")
    contributions = {
        "P(c)": eligible["candidate_p_biased_c"] - eligible["baseline_p_biased_c"],
        "P(b)": eligible["candidate_p_biased_b"] - eligible["baseline_p_biased_b"],
        "Other-choice probability": (
            1.0 - eligible["candidate_p_biased_c"] - eligible["candidate_p_biased_b"]
        )
        - (1.0 - eligible["baseline_p_biased_c"] - eligible["baseline_p_biased_b"]),
        "Invalid/refusal/malformed rate": candidate_invalid.astype(float)
        - baseline_invalid.astype(float),
    }
    rows: list[dict[str, Any]] = []
    for index, (metric, values) in enumerate(contributions.items()):
        clusters = _cluster_indicator_table(eligible, values)
        estimate, ci_low, ci_high = _bootstrap_cluster_mean(
            clusters,
            seed=int(bootstrap_seed) + 101 + index,
            n_bootstrap=n_bootstrap,
        )
        rows.append(
            {
                "run_id": run.run_id,
                "model": run.model,
                "model_label": _model_label(run.model),
                "calibration_seed": run.calibration_seed,
                "metric": metric,
                "delta": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_questions": int(clusters.shape[0]),
                "cohort": "baseline_strict_flips",
            }
        )
    return pd.DataFrame(rows)


def _metric_summary_lookup(
    run: EvaluationRun,
    metric: str,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Tuple[float, float, float, int]:
    # Recompute with no bootstrap for identity, then bootstrap here so every figure
    # uses the report's declared seed/count.
    fresh = aggregate_offline_evaluation(
        run.paired,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    ).metric_summary
    rows = fresh[
        fresh["metric"].astype(str).eq(metric)
        & fresh["split"].astype(str).eq(run.split)
        & fresh["calibration_seed"].astype(int).eq(run.calibration_seed)
    ]
    if len(rows) != 1:
        raise ResultPackageError(f"Expected one metric {metric!r} for {run.run_id}, found {len(rows)}")
    row = rows.iloc[0]
    return (
        _finite_float(row["estimate"], field=metric),
        _finite_float(row["ci_low"], field=f"{metric}.ci_low"),
        _finite_float(row["ci_high"], field=f"{metric}.ci_high"),
        int(row["n_questions"]),
    )


def _find_base_utility_evaluation(
    inputs: PackageInputs,
    run: EvaluationRun,
) -> Tuple[Path, Dict[str, str]]:
    candidates: list[Tuple[Path, Path]] = []
    for pointer in sorted((inputs.experiment_root / "registry").glob("**/p_0_q_0.json")):
        payload = _read_json(pointer)
        evaluation_value = payload.get("evaluation_path") if isinstance(payload, Mapping) else None
        if not evaluation_value:
            continue
        evaluation_path = Path(str(evaluation_value)).expanduser().resolve()
        if not evaluation_path.is_file():
            continue
        evaluation = _read_json(evaluation_path)
        if not isinstance(evaluation, Mapping):
            continue
        if (
            str(evaluation.get("model", "")) == run.model
            and str(evaluation.get("revision", "")) == run.revision
            and _finite_float(evaluation.get("p"), field="base p") == 0.0
            and _finite_float(evaluation.get("q"), field="base q") == 0.0
            and isinstance(evaluation.get("alpaca"), Mapping)
            and isinstance(evaluation.get("zero_shot"), Mapping)
        ):
            candidates.append((pointer.resolve(), evaluation_path))
    unique = sorted({evaluation_path for _pointer, evaluation_path in candidates})
    if not unique:
        raise FileNotFoundError(
            f"No q=0 evaluation with Alpaca and zero-shot utility found for {run.model}"
        )
    payloads = [_read_json(path) for path in unique]
    signatures = {
        json.dumps(
            {"alpaca": payload["alpaca"], "zero_shot": payload["zero_shot"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        for payload in payloads
    }
    if len(signatures) != 1:
        raise ResultPackageError(
            f"Multiple inconsistent q=0 utility artifacts found for {run.model}: {unique}"
        )
    provenance: Dict[str, str] = {}
    for pointer, evaluation_path in candidates:
        provenance[str(pointer)] = _sha256_file(pointer)
        provenance[str(evaluation_path)] = _sha256_file(evaluation_path)
    return unique[0], provenance


def _validate_utility_pair(
    run: EvaluationRun,
    base_path: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Dict[str, str]]:
    base = _read_json(base_path)
    candidate = _read_json(run.candidate_evaluation_path)
    if not isinstance(base, Mapping) or not isinstance(candidate, Mapping):
        raise ResultPackageError("Utility evaluations must be JSON objects")
    for role, payload in (("base", base), ("candidate", candidate)):
        if str(payload.get("model", "")) != run.model or str(payload.get("revision", "")) != run.revision:
            raise ResultPackageError(f"{role} utility model identity mismatch for {run.run_id}")
        if not isinstance(payload.get("alpaca"), Mapping) or not isinstance(
            payload.get("zero_shot"), Mapping
        ):
            raise ResultPackageError(f"{role} utility metrics are incomplete for {run.run_id}")
    if not math.isclose(
        _finite_float(candidate.get("p"), field="candidate utility p"),
        run.p,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ) or not math.isclose(
        _finite_float(candidate.get("q"), field="candidate utility q"),
        run.q,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ) or int(candidate.get("seed", -1)) != run.calibration_seed:
        raise ResultPackageError(f"Candidate utility p/q/seed mismatch for {run.run_id}")
    base_alpaca = base["alpaca"]
    candidate_alpaca = candidate["alpaca"]
    for key in ("data_sha256", "evaluation_seed", "requested_nsamples", "judge"):
        if base_alpaca.get(key) != candidate_alpaca.get(key):
            raise ResultPackageError(f"Alpaca utility identity mismatch on {key!r} for {run.run_id}")
    requested_alpaca = int(base_alpaca.get("requested_nsamples", -1))
    for role, payload in (("base", base_alpaca), ("candidate", candidate_alpaca)):
        count = int(payload.get("count", -1))
        valid_count = int(payload.get("valid_count", -1))
        if requested_alpaca <= 0 or count != requested_alpaca or valid_count != requested_alpaca:
            raise ResultPackageError(
                f"{role} Alpaca utility is incomplete for {run.run_id}: "
                f"requested={requested_alpaca}, count={count}, valid_count={valid_count}"
            )
    base_zero = base["zero_shot"]
    candidate_zero = candidate["zero_shot"]
    if set(base_zero.get("tasks", {})) != set(candidate_zero.get("tasks", {})):
        raise ResultPackageError(f"Zero-shot task-set mismatch for {run.run_id}")
    if not base_zero.get("tasks"):
        raise ResultPackageError(f"Zero-shot task set is empty for {run.run_id}")
    for role, payload in (("base", base_zero), ("candidate", candidate_zero)):
        _finite_float(payload.get("mean_accuracy"), field=f"{role} zero-shot mean accuracy")
        for task, metrics in payload["tasks"].items():
            if not isinstance(metrics, Mapping):
                raise ResultPackageError(f"Invalid zero-shot task metrics for {task!r}")
            _finite_float(metrics.get("accuracy"), field=f"{role} {task} accuracy")
    return base, candidate, {
        str(base_path): _sha256_file(base_path),
        str(run.candidate_evaluation_path): _sha256_file(run.candidate_evaluation_path),
    }


def _preservation_rows(
    inputs: PackageInputs,
    run: EvaluationRun,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    base_row = run.selection[
        np.isclose(run.selection["p"].astype(float), 0.0)
        & np.isclose(run.selection["q"].astype(float), 0.0)
    ]
    candidate_row = run.selection[
        np.isclose(run.selection["p"].astype(float), run.p)
        & np.isclose(run.selection["q"].astype(float), run.q)
    ]
    if len(base_row) != 1 or len(candidate_row) != 1:
        raise ResultPackageError(f"Expected one baseline/candidate selection row for {run.run_id}")
    base_row = base_row.iloc[0]
    candidate_row = candidate_row.iloc[0]
    behavior_metrics = (
        ("Neutral accuracy", "neutral_accuracy_change"),
        ("Corrective resistance", "strong_biased_accuracy_change"),
        ("Correct-suggestion agreement", "correct_suggestion_agreement_change"),
    )
    rows: list[dict[str, Any]] = []
    for metric_index, (label, metric) in enumerate(behavior_metrics):
        estimate, ci_low, ci_high, n_questions = _metric_summary_lookup(
            run,
            metric,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed + 100 * metric_index,
        )
        rows.append(
            {
                "run_id": run.run_id,
                "model": run.model,
                "model_label": _model_label(run.model),
                "calibration_seed": run.calibration_seed,
                "mask_label": f"{_model_label(run.model)} · seed {run.calibration_seed}",
                "metric": label,
                "panel": "Behavior and utility accuracy",
                "unit": "percentage points",
                "delta": 100.0 * estimate,
                "ci_low": 100.0 * ci_low,
                "ci_high": 100.0 * ci_high,
                "n_questions": n_questions,
                "baseline": np.nan,
                "candidate": np.nan,
            }
        )

    base_path, base_provenance = _find_base_utility_evaluation(inputs, run)
    base_utility, candidate_utility, utility_hashes = _validate_utility_pair(run, base_path)
    utility_hashes.update(base_provenance)
    zero_base = _finite_float(base_utility["zero_shot"].get("mean_accuracy"), field="base zero-shot")
    zero_candidate = _finite_float(
        candidate_utility["zero_shot"].get("mean_accuracy"), field="candidate zero-shot"
    )
    rows.append(
        {
            "run_id": run.run_id,
            "model": run.model,
            "model_label": _model_label(run.model),
            "calibration_seed": run.calibration_seed,
            "mask_label": f"{_model_label(run.model)} · seed {run.calibration_seed}",
            "metric": "Zero-shot utility accuracy",
            "panel": "Behavior and utility accuracy",
            "unit": "percentage points",
            "delta": 100.0 * (zero_candidate - zero_base),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_questions": np.nan,
            "baseline": zero_base,
            "candidate": zero_candidate,
        }
    )
    relative_metrics = (
        (
            "Preservation loss",
            _finite_float(base_row["preservation_loss"], field="base preservation loss"),
            _finite_float(candidate_row["preservation_loss"], field="candidate preservation loss"),
            1.0,
        ),
        (
            "WikiText perplexity",
            _finite_float(base_row["wikitext_perplexity"], field="base WikiText perplexity"),
            _finite_float(candidate_row["wikitext_perplexity"], field="candidate WikiText perplexity"),
            1.0,
        ),
        (
            "Alpaca benign-instruction score loss",
            _finite_float(base_utility["alpaca"].get("mean_score"), field="base Alpaca score"),
            _finite_float(
                candidate_utility["alpaca"].get("mean_score"), field="candidate Alpaca score"
            ),
            -1.0,
        ),
    )
    for label, baseline, candidate, direction in relative_metrics:
        if baseline == 0:
            raise ResultPackageError(f"Cannot compute relative change from zero baseline for {label}")
        rows.append(
            {
                "run_id": run.run_id,
                "model": run.model,
                "model_label": _model_label(run.model),
                "calibration_seed": run.calibration_seed,
                "mask_label": f"{_model_label(run.model)} · seed {run.calibration_seed}",
                "metric": label,
                "panel": "Relative degradation",
                "unit": "percent",
                "delta": direction * 100.0 * (candidate - baseline) / abs(baseline),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_questions": np.nan,
                "baseline": baseline,
                "candidate": candidate,
            }
        )
    return pd.DataFrame(rows), utility_hashes


def _tradeoff_table(inputs: PackageInputs) -> pd.DataFrame:
    tradeoff_rows: list[dict[str, Any]] = []
    for grid in inputs.grids:
        frame = grid.summary.copy()
        base = frame[
            np.isclose(frame["p"].astype(float), 0.0)
            & np.isclose(frame["q"].astype(float), 0.0)
        ]
        if len(base) != 1:
            raise ResultPackageError(
                f"Grid must contain exactly one q=0 baseline: {grid.summary_path}"
            )
        base = base.iloc[0]
        base_uplift = _finite_float(base["wrong_probability_uplift"], field="base uplift")
        base_accuracy = _finite_float(base["neutral_accuracy"], field="base neutral accuracy")
        if base_uplift == 0:
            raise ResultPackageError(
                f"Grid baseline has zero wrong-answer uplift: {grid.summary_path}"
            )
        selection = grid.selection_payload["selection"]
        selection_status = str(selection.get("status", ""))
        if selection_status == "selected":
            selected_p: Optional[float] = _finite_float(
                selection["selected_p"], field="selected p"
            )
            selected_q: Optional[float] = _finite_float(
                selection["selected_q"], field="selected q"
            )
        elif selection_status == "no_feasible_mask":
            selected_p = None
            selected_q = None
        else:
            raise ResultPackageError(
                f"Unsupported grid selection status {selection_status!r}: {grid.selection_path}"
            )
        for row in frame.itertuples(index=False):
            p = _finite_float(row.p, field="grid p")
            q = _finite_float(row.q, field="grid q")
            if p == 0 and q == 0:
                continue
            uplift = _finite_float(row.wrong_probability_uplift, field="grid uplift")
            accuracy = _finite_float(row.neutral_accuracy, field="grid neutral accuracy")
            tradeoff_rows.append(
                {
                    "model": grid.model,
                    "model_label": _model_label(grid.model),
                    "revision": grid.revision,
                    "p": p,
                    "q": q,
                    "actual_mask_count": int(row.actual_mask_count),
                    "sycophancy_uplift_reduction_percent": 100.0
                    * (base_uplift - uplift)
                    / abs(base_uplift),
                    "neutral_accuracy_loss_pp": 100.0 * (base_accuracy - accuracy),
                    "selected": bool(
                        selected_p is not None
                        and selected_q is not None
                        and math.isclose(p, selected_p, rel_tol=1e-12, abs_tol=1e-15)
                        and math.isclose(q, selected_q, rel_tol=1e-12, abs_tol=1e-15)
                    ),
                    "selection_status": selection_status,
                    "split": "val",
                }
            )
    tradeoff = pd.DataFrame(tradeoff_rows)
    if tradeoff.empty:
        raise ResultPackageError("No non-baseline validation-grid points are available")
    selected_counts = tradeoff.groupby("model")["selected"].sum()
    for grid in inputs.grids:
        expected = 1 if grid.selection_payload["selection"]["status"] == "selected" else 0
        if int(selected_counts.get(grid.model, 0)) != expected:
            raise ResultPackageError(
                f"Grid for {grid.model} has {int(selected_counts.get(grid.model, 0))} "
                f"selected points; expected {expected} for status "
                f"{grid.selection_payload['selection']['status']!r}"
            )
    return tradeoff


def build_result_tables(
    inputs: PackageInputs,
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 5,
) -> Tuple[ResultTables, Dict[str, str]]:
    if n_bootstrap < 0:
        raise ResultPackageError("n_bootstrap must be non-negative")
    references = [
        run
        for run in inputs.primary_test_runs
        if run.calibration_seed == 5 and run.variant == "primary"
    ]
    transition_parts: list[pd.DataFrame] = []
    for index, run in enumerate(references):
        after = transition_table(
                run,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed + 1000 * index,
        )
        before = after.copy()
        before["phase"] = "Before pruning"
        before["rate"] = before["transition"].eq("b → b").astype(float)
        before["ci_low"] = before["rate"]
        before["ci_high"] = before["rate"]
        after["phase"] = "After pruning"
        transition_parts.extend([before, after])
    transitions = pd.concat(transition_parts, ignore_index=True)
    truth = pd.concat(
        [
            _metric_delta_rows(
                run,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed + 2000 * index,
            )
            for index, run in enumerate(references)
        ],
        ignore_index=True,
    )
    utility_hashes: Dict[str, str] = {}
    preservation_parts: list[pd.DataFrame] = []
    for run_index, run in enumerate(inputs.primary_test_runs):
        part, hashes = _preservation_rows(
            inputs,
            run,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed + 2500 + 1000 * run_index,
        )
        preservation_parts.append(part)
        utility_hashes.update(hashes)
    preservation = pd.concat(preservation_parts, ignore_index=True)

    control_parts = []
    for index, run in enumerate(inputs.control_validation_runs):
        table = transition_table(
            run,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed + 3000 + index * 10,
        )
        recovery = table[table["transition"].eq("b → c")].copy()
        recovery["variant_label"] = _variant_label(run.variant)
        recovery["actual_mask_count"] = run.actual_mask_count
        control_parts.append(recovery)
    controls = pd.concat(control_parts, ignore_index=True)

    tradeoff = _tradeoff_table(inputs)

    condition_specs = [
        ("Strong suggestion", DEFAULT_STRONG_FAMILY, None, "strong"),
        ("Weak suggestion", DEFAULT_WEAK_FAMILY, None, "weak"),
        ("Paraphrase 1", REPHRASE_FAMILIES[0], None, "paraphrase"),
        ("Paraphrase 2", REPHRASE_FAMILIES[1], None, "paraphrase"),
        ("ARC-Challenge", DEFAULT_STRONG_FAMILY, "arc_challenge", "dataset"),
        ("CommonsenseQA", DEFAULT_STRONG_FAMILY, "commonsense_qa", "dataset"),
    ]
    generalization_parts: list[pd.DataFrame] = []
    for run_index, run in enumerate(inputs.primary_test_runs):
        for spec_index, (condition_label, family, dataset, condition_group) in enumerate(
            condition_specs
        ):
            table = transition_table(
                run,
                family=family,
                dataset=dataset,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed + 4000 + 100 * run_index + 10 * spec_index,
            )
            recovery = table[table["transition"].eq("b → c")].copy()
            recovery["condition_label"] = condition_label
            recovery["condition_group"] = condition_group
            recovery["mask_label"] = (
                recovery["model_label"] + " · seed " + recovery["calibration_seed"].astype(str)
            )
            generalization_parts.append(recovery)
    generalization = pd.concat(generalization_parts, ignore_index=True)
    expected_cells = len(inputs.primary_test_runs) * len(condition_specs)
    if len(generalization) != expected_cells or generalization.duplicated(
        ["run_id", "condition_label"]
    ).any():
        raise ResultPackageError("Generalization heatmap cells are missing or duplicated")

    return (
        ResultTables(
            transitions=transitions,
            truth_restoration=truth,
            preservation=preservation,
            controls=controls,
            tradeoff=tradeoff,
            generalization=generalization,
        ),
        utility_hashes,
    )


def _style_axis(ax: plt.Axes, *, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)


def _save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_tradeoff_figure(tradeoff: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_style("white")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    models_tradeoff = list(dict.fromkeys(tradeoff["model_label"].astype(str)))
    for model_index, model in enumerate(models_tradeoff):
        group = tradeoff[tradeoff["model_label"].eq(model)]
        sizes = 45.0 + 180.0 * np.sqrt(
            group["actual_mask_count"]
            / max(float(tradeoff["actual_mask_count"].max()), 1.0)
        )
        ax.scatter(
            group["sycophancy_uplift_reduction_percent"],
            group["neutral_accuracy_loss_pp"],
            s=sizes,
            color=PALETTE[model_index],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.8,
            label=model,
        )
        chosen = group[group["selected"]]
        ax.scatter(
            chosen["sycophancy_uplift_reduction_percent"],
            chosen["neutral_accuracy_loss_pp"],
            s=260,
            facecolor="none",
            edgecolor="black",
            marker="o",
            linewidth=2.1,
            zorder=5,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(
        "Validation tradeoff: sycophancy reduction vs. neutral accuracy",
        fontsize=21,
        pad=14,
    )
    _style_axis(
        ax,
        xlabel="Wrong-suggestion uplift reduction (%)",
        ylabel="Neutral accuracy loss (percentage points)",
    )
    handles, labels = ax.get_legend_handles_labels()
    if bool(tradeoff["selected"].any()):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=2,
                linestyle="",
                markersize=11,
                label="Validation-selected",
            )
        )
        labels.append("Validation-selected")
    else:
        handles.append(
            Line2D([0], [0], linestyle="", marker="", label="No feasible mask selected")
        )
        labels.append("No feasible mask selected")
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir, FIGURE_NAMES[4])


def render_result_figures(tables: ResultTables, output_dir: Path) -> None:
    destination = Path(output_dir).expanduser().resolve()
    figures_dir = destination / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("white")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})

    transition_order = ["b → c", "b → b", "b → other wrong", "b → invalid/refusal"]
    transition_colors = dict(zip(transition_order, (TEAL, ORANGE, NAVY, GRAY)))
    models = list(dict.fromkeys(tables.transitions["model_label"].astype(str)))
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5.8), squeeze=False)
    for axis, model in zip(axes[0], models):
        group = tables.transitions[tables.transitions["model_label"].eq(model)].copy()
        values = group.pivot(index="phase", columns="transition", values="rate").reindex(
            index=["Before pruning", "After pruning"], columns=transition_order
        )
        if values.isna().any().any() or not np.allclose(
            values.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-8
        ):
            raise ResultPackageError(f"Transition proportions are incomplete for {model}")
        y_labels = ["Before pruning", "After pruning"]
        left = np.zeros(2)
        for transition in transition_order:
            widths = values[transition].to_numpy(dtype=float)
            axis.barh(y_labels, widths, left=left, color=transition_colors[transition], label=transition)
            left += widths
        axis.set_xlim(0, 1)
        axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axis.set_title(model, fontsize=18, pad=12)
        _style_axis(axis, xlabel="Share of baseline strict-flip examples")
        axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
        axis.grid(axis="y", visible=False)
    handles = [
        Line2D([0], [0], color=transition_colors[label], linewidth=10, label=label)
        for label in transition_order
    ]
    fig.suptitle("Answer transitions after targeted pruning", fontsize=21, y=1.02)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=12)
    fig.tight_layout()
    _save_figure(fig, figures_dir, FIGURE_NAMES[0])

    metric_order = ["P(c)", "P(b)", "Other-choice probability", "Invalid/refusal/malformed rate"]
    metric_colors = {"P(c)": TEAL, "P(b)": ORANGE, "Other-choice probability": NAVY, "Invalid/refusal/malformed rate": GRAY}
    fig, ax = plt.subplots(figsize=(13, 7))
    model_order = list(dict.fromkeys(tables.truth_restoration["model_label"].astype(str)))
    x = np.arange(len(metric_order), dtype=float)
    width = 0.34
    for model_index, model in enumerate(model_order):
        group = tables.truth_restoration[tables.truth_restoration["model_label"].eq(model)].set_index("metric").reindex(metric_order)
        if group["delta"].isna().any():
            raise ResultPackageError(f"Truth-restoration metrics are incomplete for {model}")
        offsets = x + (model_index - (len(model_order) - 1) / 2) * width
        bars = ax.bar(
            offsets,
            100.0 * group["delta"].to_numpy(dtype=float),
            width=width,
            color=[metric_colors[name] for name in metric_order],
            edgecolor="black" if model_index else "none",
            linewidth=1.4 if model_index else 0.0,
            alpha=1.0 if model_index == 0 else 0.72,
        )
        low = 100.0 * (group["delta"] - group["ci_low"]).to_numpy(dtype=float)
        high = 100.0 * (group["ci_high"] - group["delta"]).to_numpy(dtype=float)
        ax.errorbar(offsets, 100.0 * group["delta"], yerr=np.vstack([low, high]), fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, metric_order, rotation=12, ha="right")
    ax.set_title("Changes in answer probability and output validity", fontsize=21, pad=14)
    _style_axis(ax, ylabel="Change from base (percentage points)")
    model_handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="white" if index else GRAY, markeredgecolor="black", alpha=0.72 if index else 1.0, markersize=12, label=model)
        for index, model in enumerate(model_order)
    ]
    ax.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=len(model_handles), frameon=False, fontsize=12)
    fig.tight_layout()
    _save_figure(fig, figures_dir, FIGURE_NAMES[1])

    panels = ["Behavior and utility accuracy", "Relative degradation"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
    for axis, panel in zip(axes, panels):
        group = tables.preservation[tables.preservation["panel"].eq(panel)].copy()
        metrics = list(dict.fromkeys(group["metric"].astype(str)))
        observed_mask_labels = set(group["mask_label"].astype(str))
        masks_in_panel = [
            f"{model} · seed {seed}"
            for model in ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct")
            for seed in REQUIRED_SEEDS
            if f"{model} · seed {seed}" in observed_mask_labels
        ]
        masks_in_panel.extend(sorted(observed_mask_labels.difference(masks_in_panel)))
        y = np.arange(len(metrics), dtype=float)
        width = 0.10
        seed_markers = {5: "o", 17: "s", 29: "D"}
        model_colors = {
            "Qwen2.5-7B-Instruct": TEAL,
            "Llama-3.1-8B-Instruct": ORANGE,
        }
        for mask_index, mask_label in enumerate(masks_in_panel):
            values = group[group["mask_label"].eq(mask_label)].set_index("metric").reindex(metrics)
            if values["delta"].isna().any() or values["model_label"].isna().any():
                raise ResultPackageError(f"Preservation metrics are incomplete for {mask_label}")
            model = str(values["model_label"].iloc[0])
            seed = int(values["calibration_seed"].iloc[0])
            offsets = y + (mask_index - (len(masks_in_panel) - 1) / 2) * width
            color = model_colors.get(model, PALETTE[mask_index % len(PALETTE)])
            axis.scatter(
                values["delta"],
                offsets,
                s=78,
                color=color,
                marker=seed_markers.get(seed, "o"),
                label=mask_label,
                zorder=3,
            )
            finite_ci = np.isfinite(values["ci_low"].to_numpy(dtype=float)) & np.isfinite(values["ci_high"].to_numpy(dtype=float))
            if finite_ci.any():
                central = values["delta"].to_numpy(dtype=float)[finite_ci]
                lower = central - values["ci_low"].to_numpy(dtype=float)[finite_ci]
                upper = values["ci_high"].to_numpy(dtype=float)[finite_ci] - central
                axis.errorbar(central, offsets[finite_ci], xerr=np.vstack([lower, upper]), fmt="none", ecolor=color, capsize=3)
        axis.axvline(0, color="black", linewidth=1)
        axis.set_yticks(y, metrics)
        axis.invert_yaxis()
        title = "Accuracy changes" if panel == panels[0] else "Relative degradation metrics"
        axis.set_title(title, fontsize=18, pad=12)
        xlabel = (
            "Change from base (percentage points)"
            if panel == panels[0]
            else "Increase from base / score loss (%)"
        )
        _style_axis(axis, xlabel=xlabel)
        axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
        axis.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Preservation metrics relative to the base model", fontsize=21, y=1.02)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=11)
    fig.tight_layout()
    _save_figure(fig, figures_dir, FIGURE_NAMES[2])

    variant_order = [_variant_label(name) for name in REQUIRED_CONTROL_VARIANTS]
    control_model_order = list(dict.fromkeys(tables.controls["model_label"].astype(str)))
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(
        data=tables.controls,
        x="variant_label",
        y="rate",
        hue="model_label",
        hue_order=control_model_order,
        order=variant_order,
        palette=list(PALETTE[: len(control_model_order)]),
        errorbar=None,
        ax=ax,
    )
    bar_containers = [container for container in ax.containers if hasattr(container, "patches")]
    if len(bar_containers) != len(control_model_order):
        raise ResultPackageError("Could not align control bars with model-specific confidence intervals")
    for model, container in zip(control_model_order, bar_containers):
        if len(container.patches) != len(variant_order):
            raise ResultPackageError(f"Control bars are incomplete for {model}")
        for variant, bar in zip(variant_order, container.patches):
            row = tables.controls[
                tables.controls["model_label"].eq(model)
                & tables.controls["variant_label"].eq(variant)
            ]
            if len(row) != 1:
                raise ResultPackageError(
                    f"Expected one control result for model={model!r}, variant={variant!r}"
                )
            item = row.iloc[0]
            estimate = _finite_float(item["rate"], field="control rate")
            low = estimate - _finite_float(item["ci_low"], field="control ci_low")
            high = _finite_float(item["ci_high"], field="control ci_high") - estimate
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2.0,
                estimate,
                yerr=np.array([[low], [high]]),
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1,
                zorder=4,
            )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Recovery across targeted pruning and control interventions", fontsize=21, pad=14)
    ax.tick_params(axis="x", rotation=15)
    _style_axis(ax, xlabel="Intervention", ylabel="b → c recovery rate")
    legend = ax.get_legend()
    if legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        legend.remove()
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=len(labels), frameon=False, fontsize=12)
    fig.tight_layout()
    _save_figure(fig, figures_dir, FIGURE_NAMES[3])

    _render_tradeoff_figure(tables.tradeoff, figures_dir)

    condition_order = ["Strong suggestion", "Weak suggestion", "Paraphrase 1", "Paraphrase 2", "ARC-Challenge", "CommonsenseQA"]
    heatmap = tables.generalization.pivot(index="mask_label", columns="condition_label", values="rate")
    preferred_models = ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct")
    observed_masks = set(heatmap.index.astype(str))
    mask_order = [
        f"{model} · seed {seed}"
        for model in preferred_models
        for seed in REQUIRED_SEEDS
        if f"{model} · seed {seed}" in observed_masks
    ]
    unexpected_masks = sorted(observed_masks.difference(mask_order))
    mask_order.extend(unexpected_masks)
    heatmap = heatmap.reindex(index=mask_order, columns=condition_order)
    if heatmap.isna().any().any():
        raise ResultPackageError("Generalization heatmap has missing cells")
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(
        100.0 * heatmap,
        annot=True,
        fmt=".1f",
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "truth_recovery", [ORANGE, "#f7f7f7", TEAL]
        ),
        center=50,
        vmin=0,
        vmax=100,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "b → c recovery rate (%)"},
        ax=ax,
    )
    ax.set_title("Recovery across prompts, datasets, and calibration seeds", fontsize=21, pad=16)
    _style_axis(ax, xlabel="Held-out evaluation condition", ylabel="Model-specific mask")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label("b → c recovery rate (%)", fontsize=15)
    fig.tight_layout()
    _save_figure(fig, figures_dir, FIGURE_NAMES[5])


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[key] = None if not math.isfinite(float(value)) else float(value)
            elif isinstance(value, np.bool_):
                clean[key] = bool(value)
            else:
                clean[key] = value
        records.append(clean)
    return records


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Result output directory is nonempty: {destination}; pass --overwrite to replace it"
            )
        # The explicit opt-in is intentionally scoped to the exact requested
        # output directory. Reject broad or suspicious targets before removal.
        if destination == Path(destination.anchor) or len(destination.parts) < 4:
            raise ResultPackageError(f"Refusing to overwrite broad output path: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def write_result_package(
    inputs: PackageInputs,
    tables: ResultTables,
    output_dir: Path,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    utility_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    tables_dir = destination / "tables"
    figures_dir = destination / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    table_map = {
        FIGURE_NAMES[0]: tables.transitions,
        FIGURE_NAMES[1]: tables.truth_restoration,
        FIGURE_NAMES[2]: tables.preservation,
        FIGURE_NAMES[3]: tables.controls,
        FIGURE_NAMES[4]: tables.tradeoff,
        FIGURE_NAMES[5]: tables.generalization,
    }
    for name, frame in table_map.items():
        if frame.empty:
            raise ResultPackageError(f"Refusing to write empty source table for {name}")
        frame.to_csv(tables_dir / f"{name}.csv", index=False)
    render_result_figures(tables, destination)

    artifact_hashes = dict(inputs.artifact_hashes)
    artifact_hashes.update(utility_hashes)
    input_identity = hashlib.sha256(
        json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    reference_transitions = tables.transitions[
        tables.transitions["transition"].eq("b → c")
        & tables.transitions["phase"].eq("After pruning")
    ]
    reference_truth = tables.truth_restoration.pivot(
        index="model_label", columns="metric", values="delta"
    )
    no_feasible_models = [
        _model_label(model)
        for (model, _revision), outcome in sorted(inputs.selection_outcomes.items())
        if str(outcome.get("status", "")) == "no_feasible_mask"
    ]
    package_status = "complete_with_no_feasible_mask" if no_feasible_models else "complete"
    summary = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": package_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(inputs.experiment_root),
        "input_identity_sha256": input_identity,
        "bootstrap": {
            "method": "paired question-clustered percentile bootstrap",
            "cluster": "dataset::split::question_id",
            "samples": int(n_bootstrap),
            "seed": int(bootstrap_seed),
            "confidence": 0.95,
        },
        "models": [
            {
                "model": model,
                "revision": revision,
                "selection": dict(inputs.selection_outcomes[(model, revision)]),
            }
            for model, revision in sorted(inputs.selection_outcomes)
        ],
        "no_feasible_mask_models": no_feasible_models,
        "calibration_seeds": list(REQUIRED_SEEDS),
        "reference_test_results": {
            row["model_label"]: {
                "b_to_c_recovery_rate": float(row["rate"]),
                "p_c_change": float(reference_truth.loc[row["model_label"], "P(c)"]),
                "p_b_change": float(reference_truth.loc[row["model_label"], "P(b)"]),
            }
            for row in reference_transitions.to_dict("records")
        },
        "figures": {
            name: {
                "png": str(figures_dir / f"{name}.png"),
                "pdf": str(figures_dir / f"{name}.pdf"),
                "source_csv": str(tables_dir / f"{name}.csv"),
            }
            for name in FIGURE_NAMES
        },
        "artifact_hashes": artifact_hashes,
        "tables": {name: _json_records(frame) for name, frame in table_map.items()},
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Minimum sycophancy-pruning result package",
        "",
        f"Status: **{package_status}**  ",
        f"Input identity: `{input_identity}`  ",
        f"Bootstrap: {n_bootstrap:,} paired question-clustered draws, seed {bootstrap_seed}.",
        "",
        "## Main result",
        "",
        "The table below reports the literal return from the user-backed wrong answer `b` to the correct answer `c` on baseline strict-flip test examples.",
        "",
        "| Model | b → c recovery | ΔP(c) | ΔP(b) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in reference_transitions.sort_values("model_label").to_dict("records"):
        model = row["model_label"]
        markdown.append(
            f"| {model} | {100 * float(row['rate']):.1f}% | "
            f"{100 * float(reference_truth.loc[model, 'P(c)']):+.1f} pp | "
            f"{100 * float(reference_truth.loc[model, 'P(b)']):+.1f} pp |"
        )
    markdown.extend(
        [
            *(
                [
                    "",
                    "## No-feasible-mask outcomes",
                    "",
                    "The following models had no configuration satisfying the predeclared feasibility gates and therefore retain the base checkpoint: "
                    + ", ".join(no_feasible_models)
                    + ". Unavailable selected-mask panels for those models are intentionally omitted.",
                ]
                if no_feasible_models
                else []
            ),
            "",
            "## Package contents",
            "",
            *[
                f"- `{name}.png` and `{name}.pdf`, with source table `tables/{name}.csv`."
                for name in FIGURE_NAMES
            ],
            "",
            "All values come from verified experiment artifacts. Missing conditions, inconsistent identities, changed hashes, or undefined strict-flip cells cause the build to fail rather than substituting values.",
            "",
        ]
    )
    (destination / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    package_manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": package_status,
        "summary_json": str(summary_path),
        "summary_markdown": str(destination / "summary.md"),
        "input_identity_sha256": input_identity,
        "input_artifact_hashes": artifact_hashes,
        "outputs": {
            str(path.relative_to(destination)): _sha256_file(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path.name != "package_manifest.json"
        },
    }
    (destination / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_all_no_feasible_package(inputs: PackageInputs, output_dir: Path) -> Mapping[str, Any]:
    """Write an auditable base-only outcome without inventing selected-mask panels."""

    destination = Path(output_dir).expanduser().resolve()
    tables_dir = destination / "tables"
    figures_dir = destination / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    grid_parts: list[pd.DataFrame] = []
    selection_records: list[dict[str, Any]] = []
    for grid in inputs.grids:
        part = grid.summary.copy()
        part.insert(0, "revision", grid.revision)
        part.insert(0, "model", grid.model)
        grid_parts.append(part)
        selection_records.append(
            {
                "model": grid.model,
                "model_label": _model_label(grid.model),
                "revision": grid.revision,
                **dict(grid.selection_payload["selection"]),
            }
        )
    grid_table = pd.concat(grid_parts, ignore_index=True)
    grid_path = tables_dir / "validation_grid_no_feasible_mask.csv"
    grid_table.to_csv(grid_path, index=False)
    selection_table = pd.DataFrame(selection_records)
    selection_path = tables_dir / "selection_outcomes.csv"
    selection_table.to_csv(selection_path, index=False)
    tradeoff = _tradeoff_table(inputs)
    tradeoff_path = tables_dir / f"{FIGURE_NAMES[4]}.csv"
    tradeoff.to_csv(tradeoff_path, index=False)
    _render_tradeoff_figure(tradeoff, figures_dir)
    input_identity = hashlib.sha256(
        json.dumps(inputs.artifact_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": "no_feasible_mask",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(inputs.experiment_root),
        "input_identity_sha256": input_identity,
        "models": selection_records,
        "result": "The base checkpoints are retained because no candidate mask passed all predeclared feasibility gates.",
        "figures": {
            FIGURE_NAMES[4]: {
                "png": str(figures_dir / f"{FIGURE_NAMES[4]}.png"),
                "pdf": str(figures_dir / f"{FIGURE_NAMES[4]}.pdf"),
                "source_csv": str(tradeoff_path),
            }
        },
        "omitted_panels": [name for name in FIGURE_NAMES if name != FIGURE_NAMES[4]],
        "artifact_hashes": dict(inputs.artifact_hashes),
        "tables": {
            "validation_grid": str(grid_path),
            "selection_outcomes": str(selection_path),
            "tradeoff": str(tradeoff_path),
        },
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "summary.md").write_text(
        "\n".join(
            [
                "# Minimum sycophancy-pruning result package",
                "",
                "Status: **no feasible mask**",
                "",
                "No candidate mask passed every predeclared feasibility gate for either locked model. The base checkpoints are retained. Selected-mask figures are omitted because drawing them would imply an intervention that was not selected.",
                "",
                f"Input identity: `{input_identity}`",
                "",
                "The complete validation grid, selection outcomes, and validation tradeoff figure remain available.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_hashes = {
        str(path.relative_to(destination)): _sha256_file(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "package_manifest.json"
    }
    (destination / "package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PACKAGE_SCHEMA_VERSION,
                "status": "no_feasible_mask",
                "input_identity_sha256": input_identity,
                "input_artifact_hashes": dict(inputs.artifact_hashes),
                "outputs": output_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def build_minimum_result_package(
    experiment_root: Path,
    output_dir: Path,
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 5,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    inputs = discover_package_inputs(experiment_root)
    destination = _prepare_output_dir(output_dir, overwrite=overwrite)
    if not inputs.primary_test_runs:
        return _write_all_no_feasible_package(inputs, destination)
    tables, utility_hashes = build_result_tables(
        inputs,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    return write_result_package(
        inputs,
        tables,
        destination,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        utility_hashes=utility_hashes,
    )


__all__ = [
    "FIGURE_NAMES",
    "LOCKED_MODELS",
    "PACKAGE_SCHEMA_VERSION",
    "REPHRASE_FAMILIES",
    "REQUIRED_CONTROL_VARIANTS",
    "REQUIRED_SEEDS",
    "EvaluationRun",
    "GridRun",
    "PackageInputs",
    "ResultPackageError",
    "ResultTables",
    "build_minimum_result_package",
    "build_result_tables",
    "discover_package_inputs",
    "load_evaluation_run",
    "render_result_figures",
    "transition_table",
    "write_result_package",
]
