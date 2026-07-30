from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .activations import (
    score_repeated_prompt_without_hook,
    score_with_multilayer_residual_additions,
)
from .conditioned_runtime import (
    STAGE_B_CONDITIONS,
    _arc_direction,
    _balanced_take,
    _conditioned_training_metadata,
    _median_residual_norm,
    _paired_bootstrap,
    _position_metadata,
    _strict_equal_rows,
    _validate_conditioned_artifact,
)
from .controlled import (
    PROTOCOL_VERSION,
    canonicalize_choice_mapping,
    load_controlled_direction_artifact,
    make_controlled_result_row,
    sha256_file,
    write_strict_json,
    write_strict_jsonl,
)
from .controlled_runtime import (
    _load_sources_and_pairs,
    _read_controlled_config,
    _semantic_approval_required,
    load_controlled_runtime,
)


MULTILAYER_PROTOCOL_VERSION = "conditioned_multilayer_gate_v1_20260729"
LAYER_MODES = ("all_nonterminal", "selected_single")
POSITION_MODES = ("boundary_only", "suffix_energy_matched")


def _arc_residual_norms(
    artifact: Any,
    layers: Sequence[int],
) -> Dict[int, float]:
    """Load the source activation tensor once and derive all layer references."""

    source_artifact = load_controlled_direction_artifact(
        Path(str(artifact.metadata["source_direction_artifact"]))
    )
    neutral = np.asarray(
        source_artifact.arrays["training_states_neutral"],
        dtype=np.float64,
    )
    datasets = _conditioned_training_metadata(artifact)["training_dataset"]
    arc = datasets == "arc_challenge"
    if len(datasets) != len(neutral) or not np.any(arc):
        raise ValueError("Cannot compute ARC residual norms for multilayer steering.")
    values: Dict[int, float] = {}
    for layer in layers:
        layer_index = source_artifact.layer_index(int(layer))
        value = float(
            np.median(np.linalg.norm(neutral[arc, layer_index, :], axis=1))
        )
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid median residual norm at layer {layer}: {value}")
        values[int(layer)] = value
    return values


def _multilayer_additions(
    artifact: Any,
    *,
    layers: Sequence[int],
    family: str,
    endorsed_choice: str,
    aggregate_ratio: float,
    position_mode: str,
    suffix_mask: np.ndarray,
    direction_cache: Optional[
        Mapping[tuple[int, str], tuple[np.ndarray, str]]
    ] = None,
    residual_norms: Optional[Mapping[int, float]] = None,
) -> tuple[Dict[int, np.ndarray], np.ndarray, Dict[str, Any]]:
    """Create layer-specific additions with controlled aggregate normalized energy."""

    layer_values = sorted({int(layer) for layer in layers})
    if not layer_values:
        raise ValueError("At least one steering layer is required.")
    if abs(float(aggregate_ratio)) > 0.2000001:
        raise ValueError("Aggregate normalized ratio exceeds the preregistered 0.20 cap.")
    per_layer_ratio = float(aggregate_ratio) / math.sqrt(len(layer_values))
    suffix = np.asarray(suffix_mask, dtype=np.float32)
    if position_mode == "boundary_only":
        mask = np.zeros(len(suffix), dtype=np.float32)
        mask[-1] = 1.0
        token_scale = 1.0
    elif position_mode == "suffix_energy_matched":
        mask = suffix
        count = int(np.count_nonzero(mask))
        if count <= 0:
            raise ValueError("Energy-matched suffix mask is empty.")
        token_scale = 1.0 / math.sqrt(count)
    else:
        raise KeyError(f"Unknown multilayer position mode {position_mode!r}.")

    additions: Dict[int, np.ndarray] = {}
    layer_metadata: Dict[str, Any] = {}
    raw_total_squared = 0.0
    normalized_ratios = []
    conditioning_keys = set()
    for layer in layer_values:
        cached_direction = (
            direction_cache.get((layer, endorsed_choice))
            if direction_cache is not None
            else None
        )
        if cached_direction is None:
            direction, conditioning_key = _arc_direction(
                artifact,
                family=family,
                layer=layer,
                endorsed_choice=endorsed_choice,
            )
        else:
            direction, conditioning_key = cached_direction
        conditioning_keys.add(str(conditioning_key))
        vector = np.asarray(direction, dtype=np.float64)
        raw_direction_norm = float(np.linalg.norm(vector))
        if raw_direction_norm <= np.finfo(np.float64).tiny:
            raise FloatingPointError(f"Zero steering direction at layer {layer}.")
        residual_norm = (
            float(residual_norms[layer])
            if residual_norms is not None
            else _median_residual_norm(artifact, layer)
        )
        boundary_norm = abs(per_layer_ratio) * residual_norm
        boundary_vector = (
            np.sign(per_layer_ratio)
            * boundary_norm
            * vector
            / raw_direction_norm
            if per_layer_ratio != 0.0
            else np.zeros_like(vector)
        )
        addition = boundary_vector * token_scale
        per_token_norm = float(np.linalg.norm(addition))
        token_count = int(np.count_nonzero(mask))
        total_layer_norm = per_token_norm * math.sqrt(token_count)
        actual_layer_ratio = total_layer_norm / residual_norm
        additions[layer] = addition.astype(np.float32)
        raw_total_squared += total_layer_norm**2
        normalized_ratios.append(actual_layer_ratio)
        layer_metadata[str(layer)] = {
            "raw_direction_norm": raw_direction_norm,
            "residual_norm_reference_value": residual_norm,
            "native_alpha_per_token": (
                math.copysign(per_token_norm / raw_direction_norm, per_layer_ratio)
                if per_layer_ratio != 0.0
                else 0.0
            ),
            "per_token_injected_norm": per_token_norm,
            "total_injected_norm": total_layer_norm,
            "actual_injected_residual_ratio": actual_layer_ratio,
        }
    if len(conditioning_keys) != 1:
        raise AssertionError("Conditioning keys unexpectedly differ across layers.")
    actual_aggregate_ratio = float(np.linalg.norm(normalized_ratios))
    if not math.isclose(
        actual_aggregate_ratio,
        abs(float(aggregate_ratio)),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise AssertionError(
            "Aggregate normalized energy mismatch: "
            f"target={aggregate_ratio} actual={actual_aggregate_ratio}"
        )
    return additions, mask, {
        "conditioning_key": next(iter(conditioning_keys)),
        "aggregate_ratio_target": float(aggregate_ratio),
        "actual_aggregate_normalized_ratio": actual_aggregate_ratio,
        "per_layer_ratio_target": per_layer_ratio,
        "aggregate_raw_injected_norm": math.sqrt(raw_total_squared),
        "layer_metadata": layer_metadata,
    }


def run_multilayer_arc_steering(
    *,
    config_path: Path,
    source_run_dir: Path,
    question_manifest_path: Path,
    directions_path: Path,
    output_dir: Path,
    split: str,
    selected_layer: int,
    primary_family: str,
    ratios: Sequence[float],
    position_modes: Sequence[str] = POSITION_MODES,
    layer_modes: Sequence[str] = LAYER_MODES,
    minimum_neutral_correct: int = 100,
    maximum_live_questions: Optional[int] = None,
    device: str = "cuda",
    device_map_auto: bool = False,
    hf_cache_dir: Optional[str] = None,
    torch_dtype: Optional[str] = "auto",
    progress_every: int = 10,
) -> Path:
    """Run all-layer and same-shard single-layer conditioned ARC steering."""

    if split != "val":
        raise ValueError("This bounded multilayer gate is validation-only.")
    if primary_family not in {"belief_conflict", "b_conditioned_wc"}:
        raise ValueError("Multilayer gate requires a CPU-passing conditioned family.")
    if any(mode not in LAYER_MODES for mode in layer_modes):
        raise ValueError(f"Unknown layer mode in {list(layer_modes)!r}.")
    if any(mode not in POSITION_MODES for mode in position_modes):
        raise ValueError(f"Unknown position mode in {list(position_modes)!r}.")
    if any(abs(float(value)) > 0.2000001 for value in ratios):
        raise ValueError("Aggregate normalized ratio exceeds 0.20.")
    if 0.0 not in {float(value) for value in ratios}:
        raise ValueError("The ratio grid must contain alpha zero.")

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
        raise ValueError("The multilayer gate is ARC-only.")
    artifact = load_controlled_direction_artifact(directions_path)
    _validate_conditioned_artifact(artifact, config=config, source=source)
    all_layers = sorted(int(layer) for layer in artifact.layers)
    if int(selected_layer) not in all_layers:
        raise ValueError("Selected layer is absent from the conditioned artifact.")
    layer_sets = {
        "all_nonterminal": all_layers,
        "selected_single": [int(selected_layer)],
    }
    residual_norms = _arc_residual_norms(artifact, all_layers)
    endorsed_labels = sorted(
        {str(pair["canonical_endorsed_choice"]) for pair in pairs}
    )
    direction_cache = {
        (layer, label): _arc_direction(
            artifact,
            family=primary_family,
            layer=layer,
            endorsed_choice=label,
        )
        for layer in all_layers
        for label in endorsed_labels
    }

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
    started = time.monotonic()
    forward_calls = 0

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
        prediction = max(probabilities[0], key=probabilities[0].get)
        neutral_baselines[pair["stable_question_key"]] = (
            probabilities[0],
            scores[0],
            prediction,
        )
    live_pairs = [
        pair
        for pair in pairs
        if neutral_baselines[pair["stable_question_key"]][2]
        == pair["source_correct_choice"]
    ]
    if maximum_live_questions is not None:
        live_pairs = list(
            _balanced_take(live_pairs, maximum=int(maximum_live_questions))
        )
    if len(live_pairs) < int(minimum_neutral_correct):
        write_strict_json(
            target / "manifest.json",
            {
                "multilayer_protocol_version": MULTILAYER_PROTOCOL_VERSION,
                "status": "stopped_before_steering",
                "reason": "same_shard_neutral_correct_below_minimum",
                "model_name": source.model_name,
                "same_shard_neutral_correct_questions": len(live_pairs),
                "minimum_required": int(minimum_neutral_correct),
                "forward_calls": forward_calls,
            },
        )
        return target / "manifest.json"

    result_rows: list[Dict[str, Any]] = []
    no_op_rows: list[Dict[str, Any]] = []
    for pair_index, pair in enumerate(live_pairs, start=1):
        stable_key = pair["stable_question_key"]
        source_neutral_choice = pair["choice_label_map"].get(
            pair["neutral_selected_choice"], ""
        )
        same_shard_neutral_source = neutral_baselines[stable_key][2]
        same_shard_neutral_choice = pair["choice_label_map"][
            same_shard_neutral_source
        ]
        position_by_condition = {
            condition: _position_metadata(tokenizer, pair, condition)
            for condition in STAGE_B_CONDITIONS
        }
        baseline_by_condition: Dict[
            str, tuple[Dict[str, float], Dict[str, float]]
        ] = {}
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

            # Exercise the complete simultaneous hook stack at alpha zero for
            # every scientific token mask and compare with the same forward shape.
            for position_mode in position_modes:
                zero_additions = {
                    layer: np.zeros(
                        direction_cache[
                            (layer, pair["canonical_endorsed_choice"])
                        ][0].shape[0],
                        dtype=np.float32,
                    )[None, :]
                    for layer in all_layers
                }
                suffix = position_by_condition[condition]
                mask = (
                    suffix["boundary_mask"]
                    if position_mode == "boundary_only"
                    else suffix["token_mask"]
                )
                zero_probabilities, zero_scores = (
                    score_with_multilayer_residual_additions(
                        model,
                        tokenizer,
                        pair["records"][condition]["prompt_messages"],
                        choices=pair["choices"],
                        addition_vectors_by_layer=zero_additions,
                        token_masks=np.asarray(mask, dtype=np.float32)[None, :],
                        max_batch_size=1,
                    )
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
                        "layer_mode": "all_nonterminal",
                        "position_mode": position_mode,
                        "exact": True,
                    }
                )

        for layer_mode in layer_modes:
            steering_layers = layer_sets[layer_mode]
            for condition in STAGE_B_CONDITIONS:
                baseline_probabilities, baseline_scores = baseline_by_condition[
                    condition
                ]
                suffix = position_by_condition[condition]
                for position_mode in position_modes:
                    for ratio in ratios:
                        additions, mask, norm_metadata = _multilayer_additions(
                            artifact,
                            layers=steering_layers,
                            family=primary_family,
                            endorsed_choice=pair["canonical_endorsed_choice"],
                            aggregate_ratio=float(ratio),
                            position_mode=position_mode,
                            suffix_mask=suffix["token_mask"],
                            direction_cache=direction_cache,
                            residual_norms=residual_norms,
                        )
                        if float(ratio) == 0.0:
                            probability_row = baseline_probabilities
                            score_row = baseline_scores
                        else:
                            probability_rows, score_rows = (
                                score_with_multilayer_residual_additions(
                                    model,
                                    tokenizer,
                                    pair["records"][condition]["prompt_messages"],
                                    choices=pair["choices"],
                                    addition_vectors_by_layer={
                                        layer: vector[None, :]
                                        for layer, vector in additions.items()
                                    },
                                    token_masks=mask[None, :],
                                    max_batch_size=1,
                                )
                            )
                            forward_calls += 1
                            probability_row = probability_rows[0]
                            score_row = score_rows[0]
                        metadata = {
                            "protocol_version": PROTOCOL_VERSION,
                            "multilayer_protocol_version": MULTILAYER_PROTOCOL_VERSION,
                            "stage": "multilayer_validation",
                            "stable_question_key": stable_key,
                            "question_id": pair["question_id"],
                            "source_example_id": pair["source_example_id"],
                            "dataset": "arc_challenge",
                            "split": split,
                            "condition": condition,
                            "model_name": source.model_name,
                            "layer": (
                                int(selected_layer)
                                if layer_mode == "selected_single"
                                else -1
                            ),
                            "layer_mode": layer_mode,
                            "layers": steering_layers,
                            "n_steered_layers": len(steering_layers),
                            "stream": "post_block_residual",
                            "conditioning_family": primary_family,
                            "conditioning_key": norm_metadata["conditioning_key"],
                            "belief_class": "neutral_is_c",
                            "position_mode": position_mode,
                            "suffix_start_index": int(suffix["suffix_start_index"]),
                            "suffix_end_index": int(suffix["suffix_end_index"]),
                            "suffix_token_count": int(suffix["suffix_token_count"]),
                            "prompt_token_count": int(suffix["prompt_token_count"]),
                            "aggregate_ratio_target": float(ratio),
                            "injected_residual_ratio_target": float(ratio),
                            "per_layer_ratio_target": norm_metadata[
                                "per_layer_ratio_target"
                            ],
                            "actual_aggregate_normalized_ratio": norm_metadata[
                                "actual_aggregate_normalized_ratio"
                            ],
                            "aggregate_raw_injected_norm": norm_metadata[
                                "aggregate_raw_injected_norm"
                            ],
                            "layer_injection_metadata": norm_metadata[
                                "layer_metadata"
                            ],
                            "injected_norm": norm_metadata[
                                "aggregate_raw_injected_norm"
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
                        result = make_controlled_result_row(
                            metadata=metadata,
                            probabilities=canonicalize_choice_mapping(
                                probability_row, pair["choice_label_map"]
                            ),
                            log_scores=canonicalize_choice_mapping(
                                score_row, pair["choice_label_map"]
                            ),
                            baseline_probabilities=canonicalize_choice_mapping(
                                baseline_probabilities, pair["choice_label_map"]
                            ),
                            baseline_log_scores=canonicalize_choice_mapping(
                                baseline_scores, pair["choice_label_map"]
                            ),
                            correct_choice=pair["canonical_correct_choice"],
                            endorsed_choice=pair["canonical_endorsed_choice"],
                            median_residual_norm=1.0,
                        )
                        result["injected_norm_ratio"] = norm_metadata[
                            "actual_aggregate_normalized_ratio"
                        ]
                        result_rows.append(result)
        if progress_every > 0 and (
            pair_index % int(progress_every) == 0
            or pair_index == len(live_pairs)
        ):
            print(
                f"[multilayer-gate] model={source.model_name} "
                f"questions={pair_index}/{len(live_pairs)} "
                f"forward_calls={forward_calls}",
                flush=True,
            )

    results_path = target / "question_results.jsonl"
    write_strict_jsonl(results_path, result_rows)
    write_strict_jsonl(target / "noop_rows.jsonl", no_op_rows)
    elapsed = time.monotonic() - started
    write_strict_json(
        target / "manifest.json",
        {
            "multilayer_protocol_version": MULTILAYER_PROTOCOL_VERSION,
            "status": "complete",
            "model_name": source.model_name,
            "split": split,
            "all_nonterminal_layers": all_layers,
            "selected_single_layer": int(selected_layer),
            "primary_family": primary_family,
            "layer_modes": list(layer_modes),
            "position_modes": list(position_modes),
            "ratios": [float(value) for value in ratios],
            "aggregate_scaling": "root_sum_square_of_per_layer_normalized_ratios",
            "residual_norm_reference": "arc_training_neutral_median_per_layer",
            "residual_norm_reference_by_layer": {
                str(layer): value for layer, value in residual_norms.items()
            },
            "source_manifest_questions": len(pairs),
            "same_shard_neutral_correct_questions": len(live_pairs),
            "minimum_neutral_correct": int(minimum_neutral_correct),
            "maximum_live_questions": maximum_live_questions,
            "question_results_rows": len(result_rows),
            "forward_calls": forward_calls,
            "elapsed_seconds": elapsed,
            "seconds_per_forward": elapsed / max(1, forward_calls),
            "alpha_zero_noop_exact": all(row["exact"] for row in no_op_rows),
            "nonfinite_failures": 0,
            "runtime": runtime,
            "config_sha256": sha256_file(config_path),
            "question_manifest_sha256": sha256_file(question_manifest_path),
            "directions_sha256": sha256_file(artifact.path),
            "question_results_sha256": sha256_file(results_path),
            "manifest_validation": manifest_summary,
        },
    )
    return results_path


def select_multilayer_validation(
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    n_bootstrap: int = 2000,
    seed: int = 5,
) -> Path:
    """Aggregate the bounded validation and compare all layers with one layer."""

    frames = []
    for path in input_paths:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        frames.append(pd.DataFrame(rows))
    frame = pd.concat(frames, ignore_index=True)
    if set(frame["split"]) != {"val"}:
        raise ValueError("Multilayer selection accepts validation rows only.")

    candidates: list[Dict[str, Any]] = []
    per_question: Dict[tuple[str, str, str, float], pd.Series] = {}
    for (model_name, layer_mode, position_mode), cell in frame.groupby(
        ["model_name", "layer_mode", "position_mode"]
    ):
        magnitudes = sorted(
            {
                abs(float(value))
                for value in cell["aggregate_ratio_target"]
                if float(value) != 0.0
            }
        )
        for magnitude in magnitudes:
            indexed = {}
            for condition in STAGE_B_CONDITIONS:
                for ratio in (0.0, -magnitude, magnitude):
                    indexed[(condition, ratio)] = cell[
                        cell["condition"].eq(condition)
                        & np.isclose(
                            cell["aggregate_ratio_target"].astype(float), ratio
                        )
                    ].set_index("stable_question_key")
            keys = sorted(
                set.intersection(*[set(value.index) for value in indexed.values()])
            )
            if not keys:
                continue
            w0 = indexed[("incorrect_suggestion", 0.0)].loc[keys]
            wn = indexed[("incorrect_suggestion", -magnitude)].loc[keys]
            wp = indexed[("incorrect_suggestion", magnitude)].loc[keys]
            n0 = indexed[("neutral", 0.0)].loc[keys]
            nn = indexed[("neutral", -magnitude)].loc[keys]
            c0 = indexed[("suggest_correct", 0.0)].loc[keys]
            cn = indexed[("suggest_correct", -magnitude)].loc[keys]
            did_values = pd.Series(
                (w0["p_endorsed"].to_numpy() - wn["p_endorsed"].to_numpy())
                - (n0["p_endorsed"].to_numpy() - nn["p_endorsed"].to_numpy()),
                index=keys,
            )
            per_question[(model_name, layer_mode, position_mode, magnitude)] = (
                did_values
            )
            did, did_low, did_high = _paired_bootstrap(
                did_values.to_numpy(),
                n_bootstrap=n_bootstrap,
                seed=(
                    seed
                    + (0 if layer_mode == "all_nonterminal" else 10000)
                    + (0 if position_mode == "boundary_only" else 1000)
                    + int(round(magnitude * 100))
                ),
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
            positive_increase = float(
                wp["p_endorsed"].mean() - w0["p_endorsed"].mean()
            )
            negative_reduction = float(
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
                positive_increase > 0
                and negative_reduction > 0
                and wrong_top1_reduction >= 0.05
                and neutral_accuracy_damage <= 0.02
                and correct_suggestion_damage <= 0.02
                and did_low > 0
                and magnitude <= 0.20
            )
            candidates.append(
                {
                    "model_name": model_name,
                    "layer_mode": layer_mode,
                    "position_mode": position_mode,
                    "ratio_magnitude": magnitude,
                    "n_questions": len(keys),
                    "difference_in_differences": did,
                    "did_ci_low": did_low,
                    "did_ci_high": did_high,
                    "wrong_top1_endorsement_reduction": wrong_top1_reduction,
                    "neutral_accuracy_damage": neutral_accuracy_damage,
                    "correct_suggestion_accuracy_damage": correct_suggestion_damage,
                    "positive_wrong_p_endorsed_increase": positive_increase,
                    "negative_wrong_p_endorsed_reduction": negative_reduction,
                    "mean_absolute_neutral_p_correct_damage": (
                        neutral_p_correct_damage
                    ),
                    "selection_score": did - neutral_p_correct_damage,
                    "eligible": eligible,
                }
            )

    comparisons = []
    for model_name in sorted(frame["model_name"].unique()):
        for position_mode in POSITION_MODES:
            magnitudes = sorted(
                {
                    key[3]
                    for key in per_question
                    if key[:3] == (
                        model_name,
                        "all_nonterminal",
                        position_mode,
                    )
                }
            )
            for magnitude in magnitudes:
                all_key = (
                    model_name,
                    "all_nonterminal",
                    position_mode,
                    magnitude,
                )
                single_key = (
                    model_name,
                    "selected_single",
                    position_mode,
                    magnitude,
                )
                if single_key not in per_question:
                    continue
                left = per_question[all_key]
                right = per_question[single_key]
                keys = sorted(set(left.index) & set(right.index))
                difference = left.loc[keys].to_numpy() - right.loc[keys].to_numpy()
                observed, low, high = _paired_bootstrap(
                    difference,
                    n_bootstrap=n_bootstrap,
                    seed=seed + 20000 + int(round(magnitude * 100)),
                )
                comparisons.append(
                    {
                        "model_name": model_name,
                        "position_mode": position_mode,
                        "ratio_magnitude": magnitude,
                        "n_questions": len(keys),
                        "all_layers_minus_single_layer_did": observed,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )

    selections = []
    for model_name in sorted(frame["model_name"].unique()):
        eligible = [
            row
            for row in candidates
            if row["model_name"] == model_name
            and row["layer_mode"] == "all_nonterminal"
            and row["eligible"]
        ]
        selected = (
            sorted(
                eligible,
                key=lambda row: (
                    -row["selection_score"],
                    row["ratio_magnitude"],
                    0 if row["position_mode"] == "boundary_only" else 1,
                ),
            )[0]
            if eligible
            else None
        )
        selections.append(
            {
                "model_name": model_name,
                "status": "selected" if selected else "no_eligible_candidate",
                "selected": selected,
            }
        )

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite multilayer decision: {target}")
    write_strict_json(
        target,
        {
            "multilayer_protocol_version": MULTILAYER_PROTOCOL_VERSION,
            "candidate_table": candidates,
            "all_layers_vs_single_layer": comparisons,
            "selections": selections,
            "both_models_pass": all(
                value["status"] == "selected" for value in selections
            ),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
        },
    )
    return target


__all__ = [
    "MULTILAYER_PROTOCOL_VERSION",
    "_multilayer_additions",
    "run_multilayer_arc_steering",
    "select_multilayer_validation",
]
