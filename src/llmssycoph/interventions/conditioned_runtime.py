from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .activations import (
    resolve_prompt_suffix_mask,
    score_repeated_prompt_without_hook,
    score_with_residual_additions,
)
from .conditioned_audit import AUDIT_PROTOCOL_VERSION, CANONICAL_LABELS
from .controlled import (
    PROTOCOL_VERSION,
    assert_noop_contract,
    canonical_choice_map,
    canonicalize_choice_mapping,
    load_controlled_direction_artifact,
    make_controlled_result_row,
    read_json,
    sha256_file,
    stable_question_key,
    write_strict_json,
    write_strict_jsonl,
)
from .controlled_runtime import (
    _load_sources_and_pairs,
    _model_revision,
    _read_controlled_config,
    _semantic_approval_required,
    load_controlled_runtime,
)
from .data import (
    build_intervention_pairs,
    load_source_bundle,
)


CONDITIONED_STAGE_B_VERSION = "conditioned_arc_causal_gate_v1_20260726"
STAGE_B_CONDITIONS = ("neutral", "incorrect_suggestion", "suggest_correct")
RATIO_GRID = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)
_ARC_DIRECTION_CACHE: Dict[tuple[str, int], Dict[str, np.ndarray]] = {}


def _conditioned_training_metadata(artifact: Any) -> Dict[str, np.ndarray]:
    required = (
        "training_dataset",
        "training_endorsed_choice",
        "training_correct_choice",
        "training_belief_class",
    )
    if all(name in artifact.arrays for name in required):
        return {
            name: np.asarray(artifact.arrays[name]).astype(str) for name in required
        }
    directory_name = artifact.path.parent.name
    if not directory_name.startswith("conditioned_directions_model_"):
        raise ValueError("Cannot infer legacy conditioned-audit metadata path.")
    model_index = directory_name.rsplit("_", 1)[-1]
    metadata_path = artifact.path.parent.parent / f"question_metadata_model_{model_index}.csv"
    if (
        not metadata_path.is_file()
        or sha256_file(metadata_path)
        != str(artifact.metadata.get("question_metadata_sha256", ""))
    ):
        raise ValueError("Conditioned training metadata identity mismatch.")
    frame = pd.read_csv(metadata_path, dtype=str)
    return {
        "training_dataset": frame["dataset"].to_numpy(dtype=str),
        "training_endorsed_choice": frame["endorsed_choice"].to_numpy(dtype=str),
        "training_correct_choice": frame["correct_choice"].to_numpy(dtype=str),
        "training_belief_class": frame["belief_class"].to_numpy(dtype=str),
    }


def _derived_arc_directions(artifact: Any, layer: int) -> Dict[str, np.ndarray]:
    key = (str(artifact.path), int(layer))
    cached = _ARC_DIRECTION_CACHE.get(key)
    if cached is not None:
        return cached
    source_artifact = load_controlled_direction_artifact(
        Path(str(artifact.metadata["source_direction_artifact"]))
    )
    layer_index = source_artifact.layer_index(layer)
    wrong = np.asarray(
        source_artifact.arrays["training_states_incorrect_suggestion"][
            :, layer_index, :
        ],
        dtype=np.float64,
    )
    neutral = np.asarray(
        source_artifact.arrays["training_states_neutral"][:, layer_index, :],
        dtype=np.float64,
    )
    correct_states = np.asarray(
        source_artifact.arrays["training_states_suggest_correct"][
            :, layer_index, :
        ],
        dtype=np.float64,
    )
    metadata = _conditioned_training_metadata(artifact)
    datasets = metadata["training_dataset"]
    labels = metadata["training_endorsed_choice"]
    belief = metadata["training_belief_class"]
    arc = datasets == "arc_challenge"
    wc = wrong - correct_states
    label_bank = np.stack(
        [
            wc[arc & (labels == label)].mean(axis=0)
            if np.any(arc & (labels == label))
            else wc[arc].mean(axis=0)
            for label in CANONICAL_LABELS
        ],
        axis=0,
    )
    eligible = arc & (belief != "neutral_is_other")
    oriented = np.where((belief == "neutral_is_c")[:, None], wc, -wc)
    cached = {
        "label_bank": label_bank.astype(np.float32),
        "belief": oriented[eligible].mean(axis=0).astype(np.float32),
        "wc": wc[arc].mean(axis=0).astype(np.float32),
        "wn": (wrong[arc] - neutral[arc]).mean(axis=0).astype(np.float32),
    }
    _ARC_DIRECTION_CACHE[key] = cached
    return cached


def _strict_equal_rows(
    left_probabilities: Sequence[Mapping[str, float]],
    right_probabilities: Sequence[Mapping[str, float]],
    left_scores: Sequence[Mapping[str, float]],
    right_scores: Sequence[Mapping[str, float]],
) -> None:
    assert_noop_contract(
        left_probabilities,
        right_probabilities,
        exact=True,
        max_probability_error=0.0,
    )
    if len(left_scores) != len(right_scores):
        raise AssertionError("No-op score row counts differ.")
    for left, right in zip(left_scores, right_scores):
        if dict(left) != dict(right):
            raise AssertionError(
                f"Alpha-zero option log scores differ: {dict(left)} != {dict(right)}"
            )


def _balanced_take(
    pairs: Sequence[Mapping[str, Any]],
    *,
    maximum: int,
    labels: Sequence[str] = ("A", "B", "C", "D"),
) -> list[Mapping[str, Any]]:
    groups = {
        label: sorted(
            [
                pair
                for pair in pairs
                if str(pair["canonical_endorsed_choice"]) == label
            ],
            key=lambda pair: str(pair["stable_question_key"]),
        )
        for label in labels
    }
    selected: list[Mapping[str, Any]] = []
    index = 0
    while len(selected) < int(maximum):
        changed = False
        for label in labels:
            if index < len(groups[label]) and len(selected) < int(maximum):
                selected.append(groups[label][index])
                changed = True
        if not changed:
            break
        index += 1
    return sorted(selected, key=lambda pair: str(pair["stable_question_key"]))


def build_conditioned_arc_cohort(
    *,
    source_run_dir: Path,
    training_manifest_path: Path,
    output_path: Path,
    maximum_per_split: int = 120,
) -> Path:
    """Freeze a model-specific source-neutral-correct ARC val/test cohort."""

    source = load_source_bundle(
        source_run_dir,
        record_conditions=(
            "neutral",
            "incorrect_suggestion",
            "incorrect_suggestion_strong",
            "suggest_correct",
        ),
        require_probe=False,
    )
    if source.dataset_name != "arc_challenge":
        raise ValueError("The conditioned causal cohort is ARC-only.")
    pairs, _ = build_intervention_pairs(
        source.records,
        probe_scores=source.probe_scores,
        required_conditions=(
            "neutral",
            "incorrect_suggestion",
            "incorrect_suggestion_strong",
            "suggest_correct",
        ),
        allowed_splits=("val", "test"),
        require_metric_usable=False,
    )
    training_keys = {
        stable_question_key(row)
        for row in json.loads(
            json.dumps(
                [
                    json.loads(line)
                    for line in Path(training_manifest_path)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            )
        )
        if str(row.get("split", "")) == "train"
    }
    candidates: Dict[str, list[Dict[str, Any]]] = {"val": [], "test": []}
    for pair in pairs:
        key = stable_question_key(pair)
        if key in training_keys:
            raise ValueError(f"ARC causal pool overlaps direction training key {key}.")
        choice_map = canonical_choice_map(pair["choices"])
        correct = choice_map[str(pair["correct_choice"])]
        endorsed = choice_map[str(pair["endorsed_choice"])]
        neutral_choice = choice_map.get(str(pair["neutral_selected_choice"]), "")
        if (
            neutral_choice != correct
            or endorsed == correct
            or endorsed not in {"A", "B", "C", "D"}
        ):
            continue
        pair_copy = dict(pair)
        pair_copy["stable_question_key"] = key
        pair_copy["choice_label_map"] = choice_map
        pair_copy["canonical_correct_choice"] = correct
        pair_copy["canonical_endorsed_choice"] = endorsed
        candidates[str(pair["split"])].append(pair_copy)

    selected = {
        split: _balanced_take(values, maximum=int(maximum_per_split))
        for split, values in candidates.items()
    }
    rows: list[Dict[str, Any]] = []
    for split in ("val", "test"):
        for pair in selected[split]:
            neutral = pair["records"]["neutral"]
            rows.append(
                {
                    "dataset": "arc_challenge",
                    "source_dataset": pair["source_dataset"],
                    "source_example_id": pair["source_example_id"],
                    "split": split,
                    "question_id": pair["question_id"],
                    "question": neutral.get("question", ""),
                    "options": neutral.get("answers_list", []),
                    "correct_answer": neutral.get("correct_answer", ""),
                    "endorsed_answer": pair["records"]["incorrect_suggestion"].get(
                        "suggested_answer",
                        pair["records"]["incorrect_suggestion"].get(
                            "incorrect_answer", ""
                        ),
                    ),
                    "correct_choice": pair["canonical_correct_choice"],
                    "endorsed_choice": pair["canonical_endorsed_choice"],
                    "source_choices": list(pair["choices"]),
                    "source_correct_choice": pair["correct_choice"],
                    "source_endorsed_choice": pair["endorsed_choice"],
                    "choice_label_map": pair["choice_label_map"],
                    "source_neutral_choice": pair["neutral_selected_choice"],
                    "source_neutral_correct": True,
                    "cohort_policy": (
                        "model_specific_source_neutral_correct_arc_exploratory"
                    ),
                    "semantic_b_review_status": "not_requested_exploratory",
                    "semantic_b_review_note": (
                        "Wrongness follows the ARC benchmark answer key; no new "
                        "per-row semantic review was requested."
                    ),
                }
            )
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite cohort manifest: {target}")
    write_strict_jsonl(target, rows)
    counts = pd.DataFrame(rows).groupby(["split", "endorsed_choice"]).size()
    write_strict_json(
        target.with_suffix(".summary.json"),
        {
            "stage_b_version": CONDITIONED_STAGE_B_VERSION,
            "model_name": source.model_name,
            "source_run_dir": str(source.run_dir),
            "source_sampling_sha256": sha256_file(source.sampling_records_path),
            "training_manifest_sha256": sha256_file(training_manifest_path),
            "manifest_sha256": sha256_file(target),
            "maximum_per_split": int(maximum_per_split),
            "counts": [
                {"split": split, "endorsed_choice": label, "count": int(count)}
                for (split, label), count in counts.items()
            ],
        },
    )
    return target


def _validate_conditioned_artifact(
    artifact: Any,
    *,
    config: Mapping[str, Any],
    source: Any,
) -> None:
    if int(artifact.metadata.get("artifact_schema_version", 1)) < 2:
        raise ValueError("Stage B requires a conditioned direction artifact schema v2.")
    if (
        str(artifact.metadata.get("conditioned_audit_protocol_version", ""))
        != AUDIT_PROTOCOL_VERSION
    ):
        raise ValueError("Conditioned audit protocol mismatch.")
    if str(artifact.metadata.get("model_name", "")) != str(source.model_name):
        raise ValueError("Conditioned artifact/source model mismatch.")
    expected_revision = _model_revision(config, source.model_name)
    if str(artifact.metadata.get("configured_model_revision", "")) != str(
        expected_revision
    ):
        raise ValueError("Conditioned artifact model revision mismatch.")
    source_artifact = Path(str(artifact.metadata.get("source_direction_artifact", "")))
    expected_hash = str(artifact.metadata.get("source_direction_sha256", ""))
    if not source_artifact.is_file() or sha256_file(source_artifact) != expected_hash:
        raise ValueError("Conditioned artifact source-direction identity mismatch.")


def _median_residual_norm(artifact: Any, layer: int) -> float:
    diagnostics = list(artifact.metadata.get("diagnostics", []) or [])
    value = float(diagnostics[artifact.layer_index(layer)]["median_residual_norm"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid median residual norm at layer {layer}: {value}")
    return value


def _arc_direction(
    artifact: Any,
    *,
    family: str,
    layer: int,
    endorsed_choice: str,
) -> tuple[np.ndarray, str]:
    derived = None
    if family == "b_conditioned_wc":
        if "arc_b_conditioned_wc_bank" in artifact.arrays:
            vector = artifact.conditioned_direction(
                "arc_b_conditioned_wc",
                layer,
                conditioning_key=endorsed_choice,
            )
        else:
            derived = _derived_arc_directions(artifact, layer)
            vector = derived["label_bank"][
                list(CANONICAL_LABELS).index(endorsed_choice)
            ]
        return vector, endorsed_choice
    if family == "belief_conflict":
        vector = (
            artifact.conditioned_direction("arc_belief_conflict", layer)
            if "arc_belief_conflict_direction" in artifact.arrays
            else _derived_arc_directions(artifact, layer)["belief"]
        )
        return vector, "neutral_is_c"
    if family == "global_wc":
        vector = (
            artifact.raw_direction("arc_wc", layer)
            if "arc_wc_raw" in artifact.arrays
            else _derived_arc_directions(artifact, layer)["wc"]
        )
        return vector, "global"
    if family == "global_wn":
        vector = (
            artifact.raw_direction("arc_wn", layer)
            if "arc_wn_raw" in artifact.arrays
            else _derived_arc_directions(artifact, layer)["wn"]
        )
        return vector, "global"
    raise KeyError(f"Unknown Stage-B direction family {family!r}.")


def _matched_controls(
    artifact: Any,
    *,
    layer: int,
    primary_family: str,
    seeds: Sequence[int],
) -> tuple[Dict[tuple[str, int, str], np.ndarray], Dict[str, np.ndarray]]:
    source_artifact = load_controlled_direction_artifact(
        Path(str(artifact.metadata["source_direction_artifact"]))
    )
    layer_index = source_artifact.layer_index(layer)
    wrong = np.asarray(
        source_artifact.arrays["training_states_incorrect_suggestion"][
            :, layer_index, :
        ],
        dtype=np.float64,
    )
    correct = np.asarray(
        source_artifact.arrays["training_states_suggest_correct"][:, layer_index, :],
        dtype=np.float64,
    )
    deltas = wrong - correct
    training_metadata = _conditioned_training_metadata(artifact)
    datasets = training_metadata["training_dataset"]
    labels = training_metadata["training_endorsed_choice"]
    belief = training_metadata["training_belief_class"]
    arc = datasets == "arc_challenge"
    controls: Dict[tuple[str, int, str], np.ndarray] = {}
    saved: Dict[str, np.ndarray] = {}
    if primary_family == "b_conditioned_wc":
        groups = [
            (
                label,
                deltas[arc & (labels == label)],
                _arc_direction(
                    artifact,
                    family="b_conditioned_wc",
                    layer=layer,
                    endorsed_choice=label,
                )[0],
            )
            for label in CANONICAL_LABELS
        ]
    elif primary_family == "belief_conflict":
        eligible = arc & (belief != "neutral_is_other")
        oriented = np.where(
            (belief == "neutral_is_c")[:, None],
            deltas,
            -deltas,
        )
        groups = [
            (
                "global",
                oriented[eligible],
                _arc_direction(
                    artifact,
                    family="belief_conflict",
                    layer=layer,
                    endorsed_choice="A",
                )[0],
            )
        ]
    else:
        raise ValueError(f"Unsupported primary control family {primary_family!r}.")
    for label_index, (label, members, target_raw) in enumerate(groups):
        if not len(members):
            continue
        target = np.asarray(target_raw, dtype=np.float64)
        target_norm = float(np.linalg.norm(target))
        for seed in seeds:
            rng = np.random.default_rng(
                17011 + 1009 * int(seed) + 31 * int(layer) + label_index
            )
            signs = np.ones(len(members), dtype=np.float64)
            signs[: len(members) // 2] = -1.0
            if len(members) % 2:
                signs[-1] = 0.0
            signs = rng.permutation(signs)
            item = np.mean(members * signs[:, None], axis=0)
            item_norm = float(np.linalg.norm(item))
            if item_norm <= np.finfo(np.float64).tiny:
                raise FloatingPointError("Degenerate item-sign control.")
            item = item * (target_norm / item_norm)
            isotropic = rng.standard_normal(target.shape[0])
            isotropic *= target_norm / float(np.linalg.norm(isotropic))
            controls[("item_sign_matched", int(seed), label)] = item.astype(
                np.float32
            )
            controls[("isotropic_matched", int(seed), label)] = isotropic.astype(
                np.float32
            )
            saved[f"item_sign_matched_seed_{int(seed)}_label_{label}"] = item.astype(
                np.float32
            )
            saved[f"isotropic_matched_seed_{int(seed)}_label_{label}"] = (
                isotropic.astype(np.float32)
            )
    return controls, saved


def _position_metadata(
    tokenizer: Any,
    pair: Mapping[str, Any],
    condition: str,
) -> Dict[str, Any]:
    record = pair["records"][condition]
    suffix = resolve_prompt_suffix_mask(
        tokenizer,
        record["prompt_messages"],
        neutral_messages=pair["records"]["neutral"]["prompt_messages"],
        condition=condition,
    )
    prompt_count = int(suffix["prompt_token_count"])
    boundary_mask = np.zeros(prompt_count, dtype=np.float32)
    boundary_mask[-1] = 1.0
    return {
        **suffix,
        "boundary_mask": boundary_mask,
    }


def _addition_for_ratio(
    direction: np.ndarray,
    *,
    ratio: float,
    median_residual_norm: float,
    position_mode: str,
    suffix_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    vector = np.asarray(direction, dtype=np.float64)
    raw_norm = float(np.linalg.norm(vector))
    if raw_norm <= np.finfo(np.float64).tiny:
        raise FloatingPointError("Cannot steer with a zero conditioned direction.")
    total_target_norm = abs(float(ratio)) * float(median_residual_norm)
    signed_boundary = (
        np.sign(float(ratio)) * total_target_norm * vector / raw_norm
        if float(ratio) != 0.0
        else np.zeros_like(vector)
    )
    if position_mode == "boundary_only":
        mask = np.zeros(len(suffix_mask), dtype=np.float32)
        mask[-1] = 1.0
        per_token = signed_boundary
    elif position_mode == "suffix_energy_matched":
        mask = np.asarray(suffix_mask, dtype=np.float32)
        count = int(np.count_nonzero(mask))
        if count <= 0:
            raise ValueError("Energy-matched suffix mask is empty.")
        per_token = signed_boundary / math.sqrt(count)
    elif position_mode == "suffix_same_per_position":
        mask = np.asarray(suffix_mask, dtype=np.float32)
        per_token = signed_boundary
        total_target_norm *= math.sqrt(int(np.count_nonzero(mask)))
    else:
        raise KeyError(f"Unknown position mode {position_mode!r}.")
    count = int(np.count_nonzero(mask))
    per_token_norm = float(np.linalg.norm(per_token))
    total_norm = per_token_norm * math.sqrt(count)
    actual_ratio = total_norm / float(median_residual_norm)
    if actual_ratio > 0.2000001:
        raise ValueError(
            f"Total injected/residual ratio {actual_ratio} exceeds the 0.20 cap."
        )
    return (
        per_token.astype(np.float32),
        mask,
        {
            "raw_direction_norm": raw_norm,
            "per_token_injected_norm": per_token_norm,
            "total_injected_norm": total_norm,
            "actual_total_injected_residual_ratio": actual_ratio,
            "native_alpha_per_token": (
                math.copysign(per_token_norm / raw_norm, float(ratio))
                if float(ratio) != 0
                else 0.0
            ),
            "boundary_equivalent_native_alpha": (
                math.copysign(
                    abs(float(ratio)) * float(median_residual_norm) / raw_norm,
                    float(ratio),
                )
                if float(ratio) != 0
                else 0.0
            ),
        },
    )


def run_conditioned_arc_steering(
    *,
    config_path: Path,
    source_run_dir: Path,
    question_manifest_path: Path,
    directions_path: Path,
    output_dir: Path,
    split: str,
    layers: Sequence[int],
    primary_family: str,
    direction_families: Sequence[str],
    position_modes: Sequence[str],
    ratios: Sequence[float],
    minimum_neutral_correct: int,
    maximum_live_questions: Optional[int] = None,
    control_seeds: Sequence[int] = (),
    control_ratio: Optional[float] = None,
    device: str = "cuda",
    device_map_auto: bool = False,
    hf_cache_dir: Optional[str] = None,
    torch_dtype: Optional[str] = "auto",
    progress_every: int = 10,
) -> Path:
    """Run one model/split scientific shard at batch size one."""

    if split not in {"val", "test"}:
        raise ValueError("Stage-B split must be val or test.")
    config = _read_controlled_config(config_path)
    sources, pairs, manifest_summary = _load_sources_and_pairs(
        [source_run_dir],
        manifest_path=question_manifest_path,
        splits=(split,),
        require_human_approval=_semantic_approval_required(config),
        require_probe=False,
    )
    source = sources[0]
    if source.dataset_name != "arc_challenge":
        raise ValueError("Stage B is ARC-only.")
    artifact = load_controlled_direction_artifact(directions_path)
    _validate_conditioned_artifact(artifact, config=config, source=source)
    layer_values = sorted({int(layer) for layer in layers})
    if not layer_values or any(layer not in artifact.layers for layer in layer_values):
        raise ValueError("Stage-B layer absent from conditioned artifact.")
    if any(abs(float(ratio)) > 0.2000001 for ratio in ratios):
        raise ValueError("Stage-B injected/residual ratio exceeds 0.20.")
    if any(mode not in {"boundary_only", "suffix_energy_matched", "suffix_same_per_position"} for mode in position_modes):
        raise ValueError("Unknown Stage-B position mode.")

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    model, tokenizer, runtime = load_controlled_runtime(
        source,
        config,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    start_time = time.monotonic()
    forward_calls = 0

    # Establish the live alpha-zero cohort before constructing any intervention.
    neutral_baselines: Dict[str, tuple[Dict[str, float], Dict[str, float], str]] = {}
    for pair in pairs:
        probabilities, scores = score_repeated_prompt_without_hook(
            model,
            tokenizer,
            pair["records"]["neutral"]["prompt_messages"],
            choices=pair["choices"],
            batch_size=1,
        )
        forward_calls += 1
        predicted_source = max(probabilities[0], key=probabilities[0].get)
        neutral_baselines[pair["stable_question_key"]] = (
            probabilities[0],
            scores[0],
            predicted_source,
        )
    live_pairs = [
        pair
        for pair in pairs
        if neutral_baselines[pair["stable_question_key"]][2]
        == pair["source_correct_choice"]
    ]
    if maximum_live_questions is not None:
        live_pairs = list(
            _balanced_take(
                live_pairs,
                maximum=int(maximum_live_questions),
            )
        )
    if len(live_pairs) < int(minimum_neutral_correct):
        stopped = {
            "stage_b_version": CONDITIONED_STAGE_B_VERSION,
            "status": "stopped_before_steering",
            "reason": "same_shard_neutral_correct_below_minimum",
            "model_name": source.model_name,
            "split": split,
            "source_manifest_questions": len(pairs),
            "same_shard_neutral_correct_questions": len(live_pairs),
            "minimum_required": int(minimum_neutral_correct),
            "forward_calls": forward_calls,
            "elapsed_seconds": time.monotonic() - start_time,
        }
        write_strict_json(target / "manifest.json", stopped)
        return target / "manifest.json"

    controls: Dict[tuple[str, int, str], np.ndarray] = {}
    control_arrays: Dict[str, np.ndarray] = {}
    if control_seeds:
        if len(layer_values) != 1:
            raise ValueError("Control shards must contain exactly the selected layer.")
        controls, control_arrays = _matched_controls(
            artifact,
            layer=layer_values[0],
            primary_family=primary_family,
            seeds=control_seeds,
        )
        np.savez_compressed(target / "control_vectors.npz", **control_arrays)

    result_rows: list[Dict[str, Any]] = []
    no_op_rows: list[Dict[str, Any]] = []
    maximum_suffix_token_count = 0
    for layer in layer_values:
        residual_norm = _median_residual_norm(artifact, layer)
        for pair_index, pair in enumerate(live_pairs, start=1):
            stable_key = pair["stable_question_key"]
            source_neutral_choice = pair["choice_label_map"].get(
                pair["neutral_selected_choice"], ""
            )
            same_shard_neutral_source = neutral_baselines[stable_key][2]
            same_shard_neutral_choice = pair["choice_label_map"][
                same_shard_neutral_source
            ]
            baseline_by_condition: Dict[
                str, tuple[Dict[str, float], Dict[str, float]]
            ] = {}
            position_by_condition = {
                condition: _position_metadata(tokenizer, pair, condition)
                for condition in STAGE_B_CONDITIONS
            }
            maximum_suffix_token_count = max(
                maximum_suffix_token_count,
                *[
                    int(value["suffix_token_count"])
                    for value in position_by_condition.values()
                ],
            )
            for condition in STAGE_B_CONDITIONS:
                if condition == "neutral":
                    baseline_probabilities, baseline_scores, _ = neutral_baselines[
                        stable_key
                    ]
                else:
                    probability_rows, score_rows = score_repeated_prompt_without_hook(
                        model,
                        tokenizer,
                        pair["records"][condition]["prompt_messages"],
                        choices=pair["choices"],
                        batch_size=1,
                    )
                    forward_calls += 1
                    baseline_probabilities = probability_rows[0]
                    baseline_scores = score_rows[0]
                baseline_by_condition[condition] = (
                    baseline_probabilities,
                    baseline_scores,
                )
                # Check every scientific position mask against the same-shape
                # disabled-hook baseline.  The addition is exactly zero.
                for position_mode in sorted(set(position_modes)):
                    suffix = position_by_condition[condition]
                    mask = (
                        suffix["boundary_mask"]
                        if position_mode == "boundary_only"
                        else suffix["token_mask"]
                    )
                    hidden = artifact.raw_direction("arc_wc", layer).shape[0]
                    zero_probabilities, zero_scores = score_with_residual_additions(
                        model,
                        tokenizer,
                        pair["records"][condition]["prompt_messages"],
                        choices=pair["choices"],
                        residual_layer=layer,
                        addition_vectors=np.zeros((1, hidden), dtype=np.float32),
                        token_masks=np.asarray(mask, dtype=np.float32)[None, :],
                        max_batch_size=1,
                    )
                    forward_calls += 1
                    _strict_equal_rows(
                        [baseline_probabilities],
                        zero_probabilities,
                        [baseline_scores],
                        zero_scores,
                    )
                    no_op_rows.append(
                        {
                            "stable_question_key": stable_key,
                            "condition": condition,
                            "layer": layer,
                            "position_mode": position_mode,
                            "exact": True,
                        }
                    )

            for condition in STAGE_B_CONDITIONS:
                baseline_probabilities, baseline_scores = baseline_by_condition[
                    condition
                ]
                suffix = position_by_condition[condition]
                for family in direction_families:
                    direction, conditioning_key = _arc_direction(
                        artifact,
                        family=family,
                        layer=layer,
                        endorsed_choice=pair["canonical_endorsed_choice"],
                    )
                    for position_mode in position_modes:
                        for ratio in ratios:
                            addition, mask, norm_metadata = _addition_for_ratio(
                                direction,
                                ratio=float(ratio),
                                median_residual_norm=residual_norm,
                                position_mode=position_mode,
                                suffix_mask=suffix["token_mask"],
                            )
                            if float(ratio) == 0.0:
                                probability_row = baseline_probabilities
                                score_row = baseline_scores
                            else:
                                probability_rows, score_rows = score_with_residual_additions(
                                    model,
                                    tokenizer,
                                    pair["records"][condition]["prompt_messages"],
                                    choices=pair["choices"],
                                    residual_layer=layer,
                                    addition_vectors=addition[None, :],
                                    token_masks=mask[None, :],
                                    max_batch_size=1,
                                )
                                forward_calls += 1
                                probability_row = probability_rows[0]
                                score_row = score_rows[0]
                            metadata = {
                                "protocol_version": PROTOCOL_VERSION,
                                "stage_b_version": CONDITIONED_STAGE_B_VERSION,
                                "stage": "validation" if split == "val" else "heldout_test",
                                "stable_question_key": stable_key,
                                "question_id": pair["question_id"],
                                "source_example_id": pair["source_example_id"],
                                "dataset": "arc_challenge",
                                "split": split,
                                "condition": condition,
                                "model_name": source.model_name,
                                "layer": layer,
                                "stream": "post_block_residual",
                                "conditioning_family": family,
                                "conditioning_key": conditioning_key,
                                "belief_class": "neutral_is_c",
                                "position_mode": position_mode,
                                "suffix_start_index": int(suffix["suffix_start_index"]),
                                "suffix_end_index": int(suffix["suffix_end_index"]),
                                "suffix_token_count": int(suffix["suffix_token_count"]),
                                "prompt_token_count": int(suffix["prompt_token_count"]),
                                "per_token_injected_norm": norm_metadata[
                                    "per_token_injected_norm"
                                ],
                                "total_injected_norm": norm_metadata[
                                    "total_injected_norm"
                                ],
                                "injected_norm": norm_metadata["total_injected_norm"],
                                "raw_direction_norm": norm_metadata[
                                    "raw_direction_norm"
                                ],
                                "native_alpha_per_token": norm_metadata[
                                    "native_alpha_per_token"
                                ],
                                "boundary_equivalent_native_alpha": norm_metadata[
                                    "boundary_equivalent_native_alpha"
                                ],
                                "injected_residual_ratio_target": float(ratio),
                                "actual_total_injected_residual_ratio": norm_metadata[
                                    "actual_total_injected_residual_ratio"
                                ],
                                "control_type": None,
                                "control_seed": None,
                                "treatment_type": "learned",
                                "source_neutral_choice": source_neutral_choice,
                                "same_shard_neutral_choice": same_shard_neutral_choice,
                                "same_shard_neutral_correct": True,
                                "alpha_zero_noop_exact": True,
                                "use_cache": False,
                                "scoring_mode": "strict_choice",
                            }
                            result_rows.append(
                                make_controlled_result_row(
                                    metadata=metadata,
                                    probabilities=canonicalize_choice_mapping(
                                        probability_row, pair["choice_label_map"]
                                    ),
                                    log_scores=canonicalize_choice_mapping(
                                        score_row, pair["choice_label_map"]
                                    ),
                                    baseline_probabilities=canonicalize_choice_mapping(
                                        baseline_probabilities,
                                        pair["choice_label_map"],
                                    ),
                                    baseline_log_scores=canonicalize_choice_mapping(
                                        baseline_scores, pair["choice_label_map"]
                                    ),
                                    correct_choice=pair["canonical_correct_choice"],
                                    endorsed_choice=pair[
                                        "canonical_endorsed_choice"
                                    ],
                                    median_residual_norm=residual_norm,
                                )
                            )

                if controls:
                    if control_ratio is None or float(control_ratio) <= 0:
                        raise ValueError("Control shards require a positive control_ratio.")
                    for (control_type, control_seed, label), direction in controls.items():
                        if label not in {
                            "global",
                            pair["canonical_endorsed_choice"],
                        }:
                            continue
                        for position_mode in position_modes:
                            for signed_ratio in (
                                -abs(float(control_ratio)),
                                abs(float(control_ratio)),
                            ):
                                addition, mask, norm_metadata = _addition_for_ratio(
                                    direction,
                                    ratio=signed_ratio,
                                    median_residual_norm=residual_norm,
                                    position_mode=position_mode,
                                    suffix_mask=suffix["token_mask"],
                                )
                                probability_rows, score_rows = score_with_residual_additions(
                                    model,
                                    tokenizer,
                                    pair["records"][condition]["prompt_messages"],
                                    choices=pair["choices"],
                                    residual_layer=layer,
                                    addition_vectors=addition[None, :],
                                    token_masks=mask[None, :],
                                    max_batch_size=1,
                                )
                                forward_calls += 1
                                metadata = {
                                    "protocol_version": PROTOCOL_VERSION,
                                    "stage_b_version": CONDITIONED_STAGE_B_VERSION,
                                    "stage": "heldout_test_control",
                                    "stable_question_key": stable_key,
                                    "question_id": pair["question_id"],
                                    "source_example_id": pair["source_example_id"],
                                    "dataset": "arc_challenge",
                                    "split": split,
                                    "condition": condition,
                                    "model_name": source.model_name,
                                    "layer": layer,
                                    "stream": "post_block_residual",
                                    "conditioning_family": primary_family,
                                    "conditioning_key": label,
                                    "belief_class": "neutral_is_c",
                                    "position_mode": position_mode,
                                    "suffix_start_index": int(
                                        suffix["suffix_start_index"]
                                    ),
                                    "suffix_end_index": int(
                                        suffix["suffix_end_index"]
                                    ),
                                    "suffix_token_count": int(
                                        suffix["suffix_token_count"]
                                    ),
                                    "prompt_token_count": int(
                                        suffix["prompt_token_count"]
                                    ),
                                    "per_token_injected_norm": norm_metadata[
                                        "per_token_injected_norm"
                                    ],
                                    "total_injected_norm": norm_metadata[
                                        "total_injected_norm"
                                    ],
                                    "injected_norm": norm_metadata[
                                        "total_injected_norm"
                                    ],
                                    "raw_direction_norm": norm_metadata[
                                        "raw_direction_norm"
                                    ],
                                    "native_alpha_per_token": norm_metadata[
                                        "native_alpha_per_token"
                                    ],
                                    "boundary_equivalent_native_alpha": norm_metadata[
                                        "boundary_equivalent_native_alpha"
                                    ],
                                    "injected_residual_ratio_target": signed_ratio,
                                    "actual_total_injected_residual_ratio": (
                                        norm_metadata[
                                            "actual_total_injected_residual_ratio"
                                        ]
                                    ),
                                    "control_type": control_type,
                                    "control_seed": control_seed,
                                    "treatment_type": "control",
                                    "source_neutral_choice": source_neutral_choice,
                                    "same_shard_neutral_choice": (
                                        same_shard_neutral_choice
                                    ),
                                    "same_shard_neutral_correct": True,
                                    "alpha_zero_noop_exact": True,
                                    "use_cache": False,
                                    "scoring_mode": "strict_choice",
                                }
                                result_rows.append(
                                    make_controlled_result_row(
                                        metadata=metadata,
                                        probabilities=canonicalize_choice_mapping(
                                            probability_rows[0],
                                            pair["choice_label_map"],
                                        ),
                                        log_scores=canonicalize_choice_mapping(
                                            score_rows[0], pair["choice_label_map"]
                                        ),
                                        baseline_probabilities=canonicalize_choice_mapping(
                                            baseline_probabilities,
                                            pair["choice_label_map"],
                                        ),
                                        baseline_log_scores=canonicalize_choice_mapping(
                                            baseline_scores,
                                            pair["choice_label_map"],
                                        ),
                                        correct_choice=pair[
                                            "canonical_correct_choice"
                                        ],
                                        endorsed_choice=pair[
                                            "canonical_endorsed_choice"
                                        ],
                                        median_residual_norm=residual_norm,
                                    )
                                )
            if progress_every > 0 and (
                pair_index % int(progress_every) == 0
                or pair_index == len(live_pairs)
            ):
                print(
                    f"[conditioned-stage-b] layer={layer} "
                    f"questions={pair_index}/{len(live_pairs)} "
                    f"forward_calls={forward_calls}",
                    flush=True,
                )

    results_path = target / "question_results.jsonl"
    write_strict_jsonl(results_path, result_rows)
    write_strict_jsonl(target / "noop_rows.jsonl", no_op_rows)
    elapsed = time.monotonic() - start_time
    manifest = {
        "stage_b_version": CONDITIONED_STAGE_B_VERSION,
        "status": "complete",
        "model_name": source.model_name,
        "split": split,
        "layers": layer_values,
        "primary_family": primary_family,
        "direction_families": list(direction_families),
        "position_modes": list(position_modes),
        "ratios": [float(value) for value in ratios],
        "control_seeds": [int(value) for value in control_seeds],
        "control_ratio": control_ratio,
        "source_manifest_questions": len(pairs),
        "same_shard_neutral_correct_questions": len(live_pairs),
        "maximum_live_questions": maximum_live_questions,
        "question_results_rows": len(result_rows),
        "forward_calls": forward_calls,
        "elapsed_seconds": elapsed,
        "seconds_per_forward": elapsed / max(1, forward_calls),
        "alpha_zero_noop_exact": all(row["exact"] for row in no_op_rows),
        "nonfinite_failures": 0,
        "maximum_suffix_token_count": maximum_suffix_token_count,
        "runtime": runtime,
        "config_sha256": sha256_file(config_path),
        "question_manifest_sha256": sha256_file(question_manifest_path),
        "directions_sha256": sha256_file(artifact.path),
        "question_results_sha256": sha256_file(results_path),
        "manifest_validation": manifest_summary,
    }
    if control_arrays:
        manifest["control_vectors_sha256"] = sha256_file(
            target / "control_vectors.npz"
        )
    write_strict_json(target / "manifest.json", manifest)
    return results_path


def _paired_bootstrap(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Paired bootstrap requires finite values.")
    observed = float(array.mean())
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_bootstrap), dtype=np.float64)
    for index in range(int(n_bootstrap)):
        draws[index] = float(
            array[rng.integers(0, len(array), size=len(array))].mean()
        )
    return observed, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def select_conditioned_validation(
    *,
    input_paths: Sequence[Path],
    cpu_decision_path: Path,
    output_path: Path,
    n_bootstrap: int = 2000,
    seed: int = 5,
) -> Path:
    frames = [
        pd.DataFrame(
            [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        for path in input_paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    if set(frame["split"]) != {"val"}:
        raise ValueError("Validation selection accepts val rows only.")
    decision = read_json(cpu_decision_path)
    selections: list[Dict[str, Any]] = []
    candidate_rows: list[Dict[str, Any]] = []
    for model_name, model_frame in frame.groupby("model_name"):
        expected_family = decision["models"][model_name]["primary_family"]
        learned = model_frame[
            model_frame["treatment_type"].eq("learned")
            & model_frame["conditioning_family"].eq(expected_family)
        ]
        candidates: list[Dict[str, Any]] = []
        for (layer, position_mode), cell in learned.groupby(
            ["layer", "position_mode"]
        ):
            magnitudes = sorted(
                {
                    abs(float(value))
                    for value in cell["injected_residual_ratio_target"]
                    if float(value) != 0
                }
            )
            for magnitude in magnitudes:
                by_key = {}
                for condition in STAGE_B_CONDITIONS:
                    for ratio in (0.0, -magnitude, magnitude):
                        subset = cell[
                            cell["condition"].eq(condition)
                            & np.isclose(
                                cell["injected_residual_ratio_target"].astype(float),
                                ratio,
                            )
                        ].set_index("stable_question_key")
                        by_key[(condition, ratio)] = subset
                keys = sorted(
                    set.intersection(
                        *[set(subset.index) for subset in by_key.values()]
                    )
                )
                if not keys:
                    continue
                w0 = by_key[("incorrect_suggestion", 0.0)].loc[keys]
                wn = by_key[("incorrect_suggestion", -magnitude)].loc[keys]
                wp = by_key[("incorrect_suggestion", magnitude)].loc[keys]
                n0 = by_key[("neutral", 0.0)].loc[keys]
                nn = by_key[("neutral", -magnitude)].loc[keys]
                c0 = by_key[("suggest_correct", 0.0)].loc[keys]
                cn = by_key[("suggest_correct", -magnitude)].loc[keys]
                did_values = (
                    (w0["p_endorsed"].to_numpy() - wn["p_endorsed"].to_numpy())
                    - (n0["p_endorsed"].to_numpy() - nn["p_endorsed"].to_numpy())
                )
                did, did_low, did_high = _paired_bootstrap(
                    did_values,
                    n_bootstrap=n_bootstrap,
                    seed=seed + int(layer) * 100 + int(round(magnitude * 100)),
                )
                wrong_top1_reduction = float(
                    w0["equals_endorsed"].mean() - wn["equals_endorsed"].mean()
                )
                neutral_accuracy_damage = float(
                    n0["is_correct"].mean() - nn["is_correct"].mean()
                )
                correct_suggestion_damage = float(
                    c0["is_correct"].mean() - cn["is_correct"].mean()
                )
                positive_increases = float(
                    wp["p_endorsed"].mean() - w0["p_endorsed"].mean()
                )
                negative_reduces = float(
                    w0["p_endorsed"].mean() - wn["p_endorsed"].mean()
                )
                neutral_p_correct_damage = float(
                    np.mean(
                        np.abs(
                            n0["p_correct"].to_numpy()
                            - nn["p_correct"].to_numpy()
                        )
                    )
                )
                eligible = bool(
                    positive_increases > 0
                    and negative_reduces > 0
                    and wrong_top1_reduction >= 0.05
                    and neutral_accuracy_damage <= 0.02
                    and correct_suggestion_damage <= 0.02
                    and did_low > 0
                    and magnitude <= 0.20
                )
                candidates.append(
                    {
                        "model_name": model_name,
                        "conditioning_family": expected_family,
                        "layer": int(layer),
                        "position_mode": position_mode,
                        "ratio_magnitude": magnitude,
                        "n_questions": len(keys),
                        "difference_in_differences": did,
                        "did_ci_low": did_low,
                        "did_ci_high": did_high,
                        "wrong_top1_endorsement_reduction": wrong_top1_reduction,
                        "neutral_accuracy_damage": neutral_accuracy_damage,
                        "correct_suggestion_accuracy_damage": (
                            correct_suggestion_damage
                        ),
                        "positive_wrong_p_endorsed_increase": positive_increases,
                        "negative_wrong_p_endorsed_reduction": negative_reduces,
                        "mean_absolute_neutral_p_correct_damage": (
                            neutral_p_correct_damage
                        ),
                        "selection_score": did - neutral_p_correct_damage,
                        "eligible": eligible,
                    }
                )
        candidate_rows.extend(candidates)
        eligible_candidates = [row for row in candidates if row["eligible"]]
        if not eligible_candidates:
            selections.append(
                {
                    "model_name": model_name,
                    "status": "no_eligible_validation_candidate",
                    "selected": None,
                }
            )
            continue
        selected = sorted(
            eligible_candidates,
            key=lambda row: (
                -row["selection_score"],
                row["ratio_magnitude"],
                0 if row["position_mode"] == "boundary_only" else 1,
                row["layer"],
            ),
        )[0]
        other_layers = sorted(
            {int(row["layer"]) for row in eligible_candidates}
            - {int(selected["layer"])},
            key=lambda layer: (
                abs(layer - int(selected["layer"])),
                layer,
            ),
        )
        selected["neighbor_layer"] = (
            other_layers[0]
            if other_layers
            else int(selected["layer"]) - 1
            if int(selected["layer"]) > 1
            else int(selected["layer"]) + 1
        )
        selections.append(
            {
                "model_name": model_name,
                "status": "selected",
                "selected": selected,
            }
        )
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite validation selection: {target}")
    write_strict_json(
        target,
        {
            "stage_b_version": CONDITIONED_STAGE_B_VERSION,
            "cpu_decision_sha256": sha256_file(cpu_decision_path),
            "selections": selections,
            "candidate_table": candidate_rows,
            "all_models_have_eligible_candidate": all(
                value["status"] == "selected" for value in selections
            ),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
        },
    )
    return target


def project_conditioned_compute(
    *,
    benchmark_manifests: Sequence[Path],
    validation_questions_per_model: int,
    test_questions_per_model: int,
    output_path: Path,
) -> Path:
    measured = [read_json(path) for path in benchmark_manifests]
    seconds_per_forward = {
        str(row["model_name"]): float(row["seconds_per_forward"])
        for row in measured
    }
    if len(seconds_per_forward) != 2:
        raise ValueError("Compute projection requires one benchmark per model.")

    def forward_count(
        *,
        include_wn: bool,
        neighbor_full_curve: bool,
        cohort: int,
    ) -> int:
        learned_families = 3 if include_wn else 2
        # Every shard first scores the source-neutral prompt. Within every layer
        # it then scores two non-neutral disabled baselines, six exact-noop
        # masks (N/W/C × two positions), and all nonzero learned treatments.
        validation = cohort * (
            1 + 3 * (2 + 3 * 2 + 3 * 2 * 6 * learned_families)
        )
        selected_learned = 1 + 2 + 3 * 2 + 3 * 2 * 6
        neighbor_learned = (
            1 + 2 + 3 * 2 + 3 * 2 * 6
            if neighbor_full_curve
            else 1 + 2 + 3 * 2 + 3 * 2 * 2
        )
        # Each of the 20 control-seed shards contains both matched control
        # families at ±rho, selected layer, both positions, and all conditions.
        controls = 20 * (1 + 2 + 3 * 2 + 2 * 2 * 2 * 3)
        # The same-per-position sensitivity is a separate shard with one mode.
        sensitivity = 1 + 2 + 3 + 2 * 3
        heldout_cohort = min(int(test_questions_per_model), int(cohort))
        test = heldout_cohort * (
            selected_learned + neighbor_learned + controls + sensitivity
        )
        # The eight-question BF16/no-op gate uses one layer, two modes, and
        # {-0.05, 0, +0.05} for the primary family.
        bf16_gate = 8 * (1 + 2 + 3 * 2 + 3 * 2 * 2)
        return bf16_gate + validation + test

    reductions = []
    include_wn = True
    neighbor_full = True
    cohort = int(validation_questions_per_model)
    while True:
        count = forward_count(
            include_wn=include_wn,
            neighbor_full_curve=neighbor_full,
            cohort=cohort,
        )
        projected = sum(
            seconds_per_forward[model] * count / 3600.0
            for model in seconds_per_forward
        )
        if projected <= 48.0:
            break
        if include_wn:
            include_wn = False
            reductions.append("drop_global_wn")
        elif neighbor_full:
            neighbor_full = False
            reductions.append("reduce_neighbor_to_selected_dose_triplet")
        elif cohort > 100:
            cohort = 100
            reductions.append("reduce_each_cohort_from_120_to_100")
        else:
            break
    authorized = projected <= 48.0
    result = {
        "stage_b_version": CONDITIONED_STAGE_B_VERSION,
        "seconds_per_forward_by_model": seconds_per_forward,
        "include_global_wn": include_wn,
        "neighbor_full_curve": neighbor_full,
        "validation_questions_per_model": cohort,
        "test_questions_per_model": min(int(test_questions_per_model), cohort),
        "projected_forward_calls_per_model": count,
        "projected_accelerator_hours_total": projected,
        "maximum_accelerator_hours": 48.0,
        "reductions_applied": reductions,
        "gpu_submission_authorized": authorized,
        "stop_reason": None if authorized else "projection_exceeds_48_gpu_hours",
    }
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite compute projection: {target}")
    write_strict_json(target, result)
    return target


def _cell_contrast(
    learned: pd.DataFrame,
    *,
    layer: int,
    position_mode: str,
    magnitude: float,
) -> Dict[str, Any]:
    cell = learned[
        learned["layer"].astype(int).eq(int(layer))
        & learned["position_mode"].eq(position_mode)
    ]
    by_condition_ratio: Dict[tuple[str, float], pd.DataFrame] = {}
    for condition in ("neutral", "incorrect_suggestion", "suggest_correct"):
        for ratio in (0.0, -abs(float(magnitude)), abs(float(magnitude))):
            by_condition_ratio[(condition, ratio)] = cell[
                cell["condition"].eq(condition)
                & np.isclose(
                    cell["injected_residual_ratio_target"].astype(float), ratio
                )
            ].set_index("stable_question_key")
    keys = sorted(
        set.intersection(
            *[set(value.index) for value in by_condition_ratio.values()]
        )
    )
    if not keys:
        raise ValueError(
            f"Incomplete held-out cell layer={layer} mode={position_mode} "
            f"magnitude={magnitude}."
        )
    w0 = by_condition_ratio[("incorrect_suggestion", 0.0)].loc[keys]
    wn = by_condition_ratio[
        ("incorrect_suggestion", -abs(float(magnitude)))
    ].loc[keys]
    wp = by_condition_ratio[
        ("incorrect_suggestion", abs(float(magnitude)))
    ].loc[keys]
    n0 = by_condition_ratio[("neutral", 0.0)].loc[keys]
    nn = by_condition_ratio[("neutral", -abs(float(magnitude)))].loc[keys]
    c0 = by_condition_ratio[("suggest_correct", 0.0)].loc[keys]
    cn = by_condition_ratio[
        ("suggest_correct", -abs(float(magnitude)))
    ].loc[keys]
    did_values = (
        (w0["p_endorsed"].to_numpy() - wn["p_endorsed"].to_numpy())
        - (n0["p_endorsed"].to_numpy() - nn["p_endorsed"].to_numpy())
    )
    return {
        "keys": keys,
        "did_values": did_values,
        "wrong_probability_correction": float(
            (w0["p_endorsed"] - wn["p_endorsed"]).mean()
        ),
        "positive_wrong_probability_change": float(
            (wp["p_endorsed"] - w0["p_endorsed"]).mean()
        ),
        "wrong_top1_endorsement_reduction": float(
            w0["equals_endorsed"].mean() - wn["equals_endorsed"].mean()
        ),
        "neutral_accuracy_damage": float(
            n0["is_correct"].mean() - nn["is_correct"].mean()
        ),
        "correct_suggestion_accuracy_damage": float(
            c0["is_correct"].mean() - cn["is_correct"].mean()
        ),
        "mean_absolute_neutral_p_correct_damage": float(
            np.mean(
                np.abs(
                    n0["p_correct"].to_numpy() - nn["p_correct"].to_numpy()
                )
            )
        ),
    }


def aggregate_conditioned_test(
    *,
    input_paths: Sequence[Path],
    selection_path: Path,
    output_dir: Path,
    n_bootstrap: int = 2000,
    seed: int = 5,
) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    frames = [
        pd.DataFrame(
            [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        for path in input_paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    if set(frame["split"]) != {"test"}:
        raise ValueError("Held-out aggregation accepts test rows only.")
    selection = read_json(selection_path)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_results: list[Dict[str, Any]] = []
    null_rows: list[Dict[str, Any]] = []
    for model_selection in selection["selections"]:
        model_name = model_selection["model_name"]
        selected = model_selection.get("selected")
        if not selected:
            model_results.append(
                {
                    "model_name": model_name,
                    "status": "validation_had_no_eligible_candidate",
                    "robust": False,
                }
            )
            continue
        model_frame = frame[frame["model_name"].eq(model_name)]
        family = str(selected["conditioning_family"])
        layer = int(selected["layer"])
        neighbor = int(selected["neighbor_layer"])
        mode = str(selected["position_mode"])
        magnitude = float(selected["ratio_magnitude"])
        learned = model_frame[
            model_frame["treatment_type"].eq("learned")
            & model_frame["conditioning_family"].eq(family)
        ]
        primary = _cell_contrast(
            learned,
            layer=layer,
            position_mode=mode,
            magnitude=magnitude,
        )
        did, did_low, did_high = _paired_bootstrap(
            primary["did_values"],
            n_bootstrap=n_bootstrap,
            seed=seed + layer,
        )
        neighbor_result = _cell_contrast(
            learned,
            layer=neighbor,
            position_mode=mode,
            magnitude=magnitude,
        )
        neighbor_did = float(np.mean(neighbor_result["did_values"]))

        controls = model_frame[
            model_frame["treatment_type"].eq("control")
            & model_frame["layer"].astype(int).eq(layer)
            & model_frame["position_mode"].eq(mode)
        ]
        null_dids: Dict[str, list[float]] = {
            "item_sign_matched": [],
            "isotropic_matched": [],
        }
        for (control_type, control_seed), control_cell in controls.groupby(
            ["control_type", "control_seed"]
        ):
            condition_rows = {
                condition: control_cell[
                    control_cell["condition"].eq(condition)
                    & np.isclose(
                        control_cell[
                            "injected_residual_ratio_target"
                        ].astype(float),
                        -magnitude,
                    )
                ].set_index("stable_question_key")
                for condition in ("neutral", "incorrect_suggestion")
            }
            keys = sorted(
                set(condition_rows["neutral"].index)
                & set(condition_rows["incorrect_suggestion"].index)
            )
            if not keys:
                continue
            # Each control row carries its own same-shard disabled baseline via
            # delta_p_endorsed = steered - baseline.  Using that delta avoids a
            # cross-job comparison to the learned shard's alpha-zero rows.
            values = (
                -condition_rows["incorrect_suggestion"].loc[keys][
                    "delta_p_endorsed"
                ].to_numpy()
                + condition_rows["neutral"].loc[keys][
                    "delta_p_endorsed"
                ].to_numpy()
            )
            null_value = float(np.mean(values))
            null_dids[str(control_type)].append(null_value)
            null_rows.append(
                {
                    "model_name": model_name,
                    "control_type": str(control_type),
                    "control_seed": int(control_seed),
                    "difference_in_differences": null_value,
                }
            )
        item_null = np.asarray(null_dids["item_sign_matched"], dtype=np.float64)
        isotropic_null = np.asarray(
            null_dids["isotropic_matched"], dtype=np.float64
        )
        if len(item_null) != 20 or len(isotropic_null) != 20:
            raise ValueError(
                f"Expected 20 controls per family for {model_name}, got "
                f"item={len(item_null)} isotropic={len(isotropic_null)}."
            )
        item_p95 = float(np.quantile(item_null, 0.95))
        isotropic_p95 = float(np.quantile(isotropic_null, 0.95))

        boundary = _cell_contrast(
            learned,
            layer=layer,
            position_mode="boundary_only",
            magnitude=magnitude,
        )
        suffix = _cell_contrast(
            learned,
            layer=layer,
            position_mode="suffix_energy_matched",
            magnitude=magnitude,
        )
        common_keys = sorted(set(boundary["keys"]) & set(suffix["keys"]))
        boundary_map = dict(zip(boundary["keys"], boundary["did_values"]))
        suffix_map = dict(zip(suffix["keys"], suffix["did_values"]))
        position_values = np.asarray(
            [suffix_map[key] - boundary_map[key] for key in common_keys],
            dtype=np.float64,
        )
        position_difference, position_low, position_high = _paired_bootstrap(
            position_values,
            n_bootstrap=n_bootstrap,
            seed=seed + 1000 + layer,
        )
        position_supported = bool(
            position_low > 0
            and suffix["neutral_accuracy_damage"]
            <= boundary["neutral_accuracy_damage"]
        )
        no_op_exact = bool(
            model_frame["alpha_zero_noop_exact"].fillna(False).all()
        )
        no_nonfinite = bool(
            (~model_frame["nonfinite_failure"].fillna(True).astype(bool)).all()
        )
        robust = bool(
            did_low > 0
            and primary["wrong_top1_endorsement_reduction"] >= 0.05
            and did > item_p95
            and primary["neutral_accuracy_damage"] <= 0.02
            and primary["mean_absolute_neutral_p_correct_damage"]
            < primary["wrong_probability_correction"]
            and neighbor_did > 0
            and no_op_exact
            and no_nonfinite
        )
        model_results.append(
            {
                "model_name": model_name,
                "status": "complete",
                "conditioning_family": family,
                "selected_layer": layer,
                "neighbor_layer": neighbor,
                "selected_position_mode": mode,
                "selected_ratio_magnitude": magnitude,
                "difference_in_differences": did,
                "did_ci_low": did_low,
                "did_ci_high": did_high,
                "wrong_probability_correction": primary[
                    "wrong_probability_correction"
                ],
                "wrong_top1_endorsement_reduction": primary[
                    "wrong_top1_endorsement_reduction"
                ],
                "neutral_accuracy_damage": primary["neutral_accuracy_damage"],
                "correct_suggestion_accuracy_damage": primary[
                    "correct_suggestion_accuracy_damage"
                ],
                "mean_absolute_neutral_p_correct_damage": primary[
                    "mean_absolute_neutral_p_correct_damage"
                ],
                "neighbor_difference_in_differences": neighbor_did,
                "item_sign_null_p95": item_p95,
                "isotropic_null_p95": isotropic_p95,
                "effect_above_item_sign_null_p95": did > item_p95,
                "alpha_zero_noop_exact": no_op_exact,
                "no_nonfinite_values": no_nonfinite,
                "robust": robust,
                "suffix_minus_boundary_did": position_difference,
                "suffix_minus_boundary_ci_low": position_low,
                "suffix_minus_boundary_ci_high": position_high,
                "position_hypothesis_supported": position_supported,
            }
        )

    all_robust = len(model_results) == 2 and all(
        value.get("robust", False) for value in model_results
    )
    robust_count = sum(bool(value.get("robust", False)) for value in model_results)
    conclusion = (
        "robust_both_models"
        if all_robust
        else "model_specific"
        if robust_count == 1
        else "reject_one_vector_correction_for_this_setup"
    )
    pd.DataFrame(null_rows).to_csv(output / "control_nulls.csv", index=False)
    pd.DataFrame(model_results).to_csv(
        output / "model_results.csv", index=False
    )

    sns.set_style("white")
    learned_plot = frame[
        frame["treatment_type"].eq("learned")
        & frame["conditioning_family"].isin(
            [
                value.get("conditioning_family")
                for value in model_results
                if value.get("conditioning_family")
            ]
        )
    ].copy()
    learned_plot["ratio"] = learned_plot[
        "injected_residual_ratio_target"
    ].astype(float)
    plot_summary = (
        learned_plot.groupby(
            [
                "model_name",
                "layer",
                "position_mode",
                "condition",
                "ratio",
            ],
            as_index=False,
        )["p_endorsed"]
        .mean()
    )
    grid = sns.relplot(
        data=plot_summary,
        x="ratio",
        y="p_endorsed",
        hue="position_mode",
        style="condition",
        col="model_name",
        kind="line",
        marker="o",
        palette={
            "boundary_only": "#73b3ab",
            "suffix_energy_matched": "#d4651a",
        },
        height=5.5,
        aspect=1.1,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels(
        "Injected / residual norm ratio", "Mean P(endorsed answer)"
    )
    grid.set_titles("{col_name}", size=17)
    for axis in grid.axes.flat:
        axis.tick_params(labelsize=12)
        axis.xaxis.label.set_size(15)
        axis.yaxis.label.set_size(15)
    if grid.legend is not None:
        grid.legend.set_bbox_to_anchor((0.5, -0.08))
        grid.legend.set_loc("upper center")
        grid.legend.set_ncols(3)
    grid.figure.suptitle(
        "Conditioned steering held-out dose response", fontsize=20, y=1.03
    )
    grid.figure.savefig(
        output / "conditioned_dose_response.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(grid.figure)

    report = {
        "stage_b_version": CONDITIONED_STAGE_B_VERSION,
        "conclusion": conclusion,
        "all_models_robust": all_robust,
        "models": model_results,
        "selection_sha256": sha256_file(selection_path),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "control_nulls_sha256": sha256_file(output / "control_nulls.csv"),
        "model_results_sha256": sha256_file(output / "model_results.csv"),
        "plot_sha256": sha256_file(output / "conditioned_dose_response.png"),
    }
    write_strict_json(output / "final_decision.json", report)
    lines = [
        "# Conditioned steering held-out gate",
        "",
        f"Conclusion: **{conclusion.replace('_', ' ')}**.",
        "",
        "| Model | DID [95% CI] | Wrong top-1 reduction | Neutral accuracy damage | Item-null p95 | Neighbor DID | Robust |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_results:
        if row.get("status") != "complete":
            lines.append(
                f"| {row['model_name']} | n/a | n/a | n/a | n/a | n/a | no |"
            )
            continue
        lines.append(
            f"| {row['model_name']} | {row['difference_in_differences']:.3f} "
            f"[{row['did_ci_low']:.3f}, {row['did_ci_high']:.3f}] | "
            f"{row['wrong_top1_endorsement_reduction']:.3f} | "
            f"{row['neutral_accuracy_damage']:.3f} | "
            f"{row['item_sign_null_p95']:.3f} | "
            f"{row['neighbor_difference_in_differences']:.3f} | "
            f"{'yes' if row['robust'] else 'no'} |"
        )
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "final_decision.json"


__all__ = [
    "CONDITIONED_STAGE_B_VERSION",
    "RATIO_GRID",
    "STAGE_B_CONDITIONS",
    "build_conditioned_arc_cohort",
    "aggregate_conditioned_test",
    "project_conditioned_compute",
    "run_conditioned_arc_steering",
    "select_conditioned_validation",
]
