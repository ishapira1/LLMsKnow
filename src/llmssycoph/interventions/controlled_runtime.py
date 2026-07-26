from __future__ import annotations

import json
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..llm.loading import load_model_and_tokenizer
from .activations import (
    candidate_feature_with_prompt_steering,
    completion_nll_with_prompt_steering,
    extract_prompt_state,
    generate_with_residual_addition,
    residual_layer_count,
    score_repeated_prompt_without_hook,
    score_with_residual_additions,
)
from .controlled import (
    DIRECTION_CONDITIONS,
    PROTOCOL_VERSION,
    REQUIRED_CONDITIONS,
    assert_noop_contract,
    assert_prompt_only_messages,
    canonical_choice_map,
    canonicalize_choice_mapping,
    fit_controlled_direction_arrays,
    framing_classification_and_retrieval,
    geometry_pair_rows,
    git_fingerprint,
    identity_framing_ratio,
    intervention_specs,
    load_controlled_direction_artifact,
    make_controlled_result_row,
    read_json,
    read_jsonl,
    runtime_provenance,
    save_controlled_direction_artifact,
    sha256_file,
    stable_question_key,
    validate_question_manifest,
    write_strict_json,
    write_strict_jsonl,
)
from .data import (
    DEFAULT_PROBE_NAME,
    SourceBundle,
    build_intervention_pairs,
    load_source_bundle,
)
from .experiment import runtime_fingerprint
from .controlled_plots import plot_controlled_dose_response, plot_controlled_pareto


def _model_revision(config: Mapping[str, Any], model_name: str) -> Optional[str]:
    models = dict(config.get("models", {}) or {})
    for value in models.values():
        item = dict(value or {})
        if str(item.get("identifier", "") or "") == str(model_name):
            revision = str(item.get("revision", "") or "")
            return revision or None
    raise KeyError(f"Config has no pinned model entry for {model_name!r}.")


def _read_controlled_config(path: Path) -> Dict[str, Any]:
    config = read_json(path)
    if str(config.get("protocol_version", "")) != PROTOCOL_VERSION:
        raise ValueError(
            "Controlled config protocol mismatch: "
            f"{config.get('protocol_version')!r} != {PROTOCOL_VERSION!r}."
        )
    return config


def _semantic_approval_required(config: Mapping[str, Any]) -> bool:
    """Return whether this configuration requires per-row human b approval.

    The controlled confirmatory protocol defaults to requiring approval. A
    deliberately exploratory configuration may disable the gate, but the
    resulting provenance and manifest validation continue to record that
    human approval was not required.
    """

    splits = dict(config.get("splits", {}) or {})
    return bool(
        splits.get("semantic_wrong_option_requires_human_approval", True)
    )


def _snapshot_exact_file(source: Path, target: Path) -> str:
    """Copy an immutable input snapshot without normalizing its bytes."""

    source_path = Path(source).expanduser().resolve()
    target_path = Path(target)
    with source_path.open("rb") as input_handle, target_path.open("xb") as output_handle:
        while chunk := input_handle.read(1024 * 1024):
            output_handle.write(chunk)
    source_hash = sha256_file(source_path)
    if sha256_file(target_path) != source_hash:
        raise RuntimeError(f"Input snapshot hash mismatch: {source_path} -> {target_path}")
    return source_hash


def load_controlled_runtime(
    source: SourceBundle,
    config: Mapping[str, Any],
    *,
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
    torch_dtype: Optional[str],
) -> tuple[Any, Any, Dict[str, Any]]:
    model, tokenizer = load_model_and_tokenizer(
        model_name=source.model_name,
        revision=_model_revision(config, source.model_name),
        device=device,
        device_map_auto=bool(device_map_auto),
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer, runtime_fingerprint(model, tokenizer, device=device)


def _validate_direction_artifact_reuse(
    artifact: Any,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    question_manifest_path: Optional[Path],
    source: SourceBundle,
    runtime: Mapping[str, Any],
) -> None:
    """Require exact cached-direction identity before any intervention reuse."""

    metadata = artifact.metadata
    expected_revision = _model_revision(config, source.model_name)
    checks = {
        "model_name": (str(metadata.get("model_name", "")), str(source.model_name)),
        "config_sha256": (
            str(metadata.get("config_sha256", "")),
            sha256_file(config_path),
        ),
        "intervention_site": (
            str(metadata.get("intervention_site", "")),
            "post_block_residual_final_rendered_prompt_token",
        ),
    }
    if question_manifest_path is not None:
        checks["question_manifest_sha256"] = (
            str(metadata.get("question_manifest_sha256", "")),
            sha256_file(question_manifest_path),
        )
    artifact_runtime = dict(metadata.get("runtime", {}) or {})
    for key in (
        "model_name_or_path",
        "model_commit_hash",
        "tokenizer_name_or_path",
        "tokenizer_commit_hash",
        "chat_template_sha256",
    ):
        checks[f"runtime.{key}"] = (
            str(artifact_runtime.get(key, "")),
            str(runtime.get(key, "")),
        )
    if expected_revision:
        checks["artifact_configured_model_revision"] = (
            str(metadata.get("configured_model_revision", "")),
            str(expected_revision),
        )
        if str(runtime.get("model_commit_hash", "")):
            checks["configured_model_revision"] = (
                str(runtime.get("model_commit_hash", "")),
                str(expected_revision),
            )
        if str(runtime.get("tokenizer_commit_hash", "")):
            checks["configured_tokenizer_revision"] = (
                str(runtime.get("tokenizer_commit_hash", "")),
                str(expected_revision),
            )
    mismatches = {
        key: {"artifact_or_runtime": left, "expected": right}
        for key, (left, right) in checks.items()
        if left != right
    }
    provenance = dict(metadata.get("provenance", {}) or {})
    current_git = git_fingerprint(Path.cwd())
    for key in (
        "git_commit",
        "tracked_diff_sha256",
        "untracked_path_manifest_sha256",
        "untracked_content_manifest_sha256",
    ):
        left = str(provenance.get(key, ""))
        right = str(current_git.get(key, ""))
        if not left or left != right:
            mismatches[f"provenance.{key}"] = {
                "artifact_or_runtime": left,
                "expected": right,
            }
    if mismatches:
        raise ValueError(
            "Controlled direction cache identity mismatch; refit directions. "
            + json.dumps(mismatches, sort_keys=True)
        )


def _manifest_rows_by_key(
    manifest_path: Path,
    *,
    require_human_approval: bool,
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    summary = validate_question_manifest(
        rows,
        require_human_approval=require_human_approval,
    )
    return rows, {stable_question_key(row): row for row in rows}, summary


def _load_sources_and_pairs(
    source_run_dirs: Sequence[Path],
    *,
    manifest_path: Path,
    splits: Sequence[str],
    require_human_approval: bool,
) -> tuple[list[SourceBundle], list[Dict[str, Any]], Dict[str, Any]]:
    manifest_rows, manifest_by_key, manifest_summary = _manifest_rows_by_key(
        manifest_path,
        require_human_approval=require_human_approval,
    )
    allowed_splits = set(str(split) for split in splits)
    sources: list[SourceBundle] = []
    selected_pairs: list[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    model_names: set[str] = set()
    for source_dir in source_run_dirs:
        source = load_source_bundle(
            source_dir,
            probe_name=DEFAULT_PROBE_NAME,
            record_conditions=REQUIRED_CONDITIONS,
        )
        sources.append(source)
        model_names.add(source.model_name)
        pairs, _coverage = build_intervention_pairs(
            source.records,
            probe_scores=source.probe_scores,
            required_conditions=REQUIRED_CONDITIONS,
            allowed_splits=sorted(allowed_splits),
            require_metric_usable=False,
        )
        for pair in pairs:
            key = stable_question_key(pair)
            manifest_row = manifest_by_key.get(key)
            if manifest_row is None:
                continue
            for condition in REQUIRED_CONDITIONS:
                assert_prompt_only_messages(
                    pair["records"][condition]["prompt_messages"],
                    context=f"{key}:{condition}",
                )
            choice_map = canonical_choice_map(pair["choices"])
            canonical_correct = choice_map[str(pair["correct_choice"])]
            canonical_endorsed = choice_map[str(pair["endorsed_choice"])]
            pair["source_choices"] = list(pair["choices"])
            pair["source_correct_choice"] = str(pair["correct_choice"])
            pair["source_endorsed_choice"] = str(pair["endorsed_choice"])
            pair["choice_label_map"] = choice_map
            pair["canonical_choices"] = list(choice_map.values())
            pair["canonical_correct_choice"] = canonical_correct
            pair["canonical_endorsed_choice"] = canonical_endorsed
            if str(manifest_row["split"]) != str(pair["split"]):
                raise ValueError(
                    f"Split mismatch for {key}: manifest={manifest_row['split']} "
                    f"source={pair['split']}."
                )
            if str(manifest_row["correct_choice"]).upper() != canonical_correct:
                raise ValueError(f"Correct-choice mismatch for {key}.")
            if str(manifest_row["endorsed_choice"]).upper() != canonical_endorsed:
                raise ValueError(f"Endorsed-choice mismatch for {key}.")
            if (
                manifest_row.get("source_correct_choice") is not None
                and str(manifest_row["source_correct_choice"]).upper()
                != pair["source_correct_choice"]
            ):
                raise ValueError(f"Source correct-choice mismatch for {key}.")
            if (
                manifest_row.get("source_endorsed_choice") is not None
                and str(manifest_row["source_endorsed_choice"]).upper()
                != pair["source_endorsed_choice"]
            ):
                raise ValueError(f"Source endorsed-choice mismatch for {key}.")
            if key in seen_keys:
                raise ValueError(f"Duplicate source pair for stable key={key}.")
            pair["stable_question_key"] = key
            pair["manifest_row"] = manifest_row
            selected_pairs.append(pair)
            seen_keys.add(key)
    if len(model_names) != 1:
        raise ValueError(
            "A controlled runtime cell must contain one model; "
            f"found models={sorted(model_names)}."
        )
    expected = {
        stable_question_key(row)
        for row in manifest_rows
        if str(row["split"]) in allowed_splits
        and str(row["dataset"]) in {source.dataset_name for source in sources}
    }
    missing = sorted(expected - seen_keys)
    if missing:
        raise ValueError(
            "Manifest questions are absent from the supplied source runs: "
            f"{missing[:10]} ({len(missing)} total)."
        )
    selected_pairs.sort(
        key=lambda pair: (
            str(pair["split"]),
            str(pair["dataset"]),
            str(pair["source_example_id"]),
        )
    )
    if not selected_pairs:
        raise ValueError("No controlled question pairs matched the manifest/source cell.")
    return sources, selected_pairs, manifest_summary


def validate_controlled_sources(
    *,
    config_path: Path,
    source_run_dirs: Sequence[Path],
    question_manifest_path: Path,
    output_dir: Path,
    require_human_approval: bool,
) -> Path:
    config = _read_controlled_config(config_path)
    sources, pairs, manifest_summary = _load_sources_and_pairs(
        source_run_dirs,
        manifest_path=question_manifest_path,
        splits=("train", "val", "test"),
        require_human_approval=require_human_approval,
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": sha256_file(config_path),
        "question_manifest_sha256": sha256_file(question_manifest_path),
        "manifest": manifest_summary,
        "model_name": sources[0].model_name,
        "source_runs": [
            {
                "run_dir": str(source.run_dir),
                "dataset": source.dataset_name,
                "sampling_records_sha256": sha256_file(source.sampling_records_path),
                "chosen_probe_layer": source.chosen_layer,
            }
            for source in sources
        ],
        "matched_questions": len(pairs),
        "matched_by_dataset_split": (
            pd.DataFrame(
                [
                    {"dataset": pair["dataset"], "split": pair["split"]}
                    for pair in pairs
                ]
            )
            .groupby(["dataset", "split"])
            .size()
            .rename("count")
            .reset_index()
            .to_dict(orient="records")
        ),
        "direction_membership_uses_generated_answer": False,
    }
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    write_strict_json(target / "source_validation.json", summary)
    return target / "source_validation.json"


def inspect_controlled_examples(
    *,
    config_path: Path,
    source_run_dir: Path,
    question_manifest_path: Path,
    output_dir: Path,
    layers: Sequence[int],
    directions_path: Optional[Path],
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
    torch_dtype: Optional[str],
) -> Path:
    config = _read_controlled_config(config_path)
    sources, pairs, manifest_summary = _load_sources_and_pairs(
        [source_run_dir],
        manifest_path=question_manifest_path,
        splits=("train", "val", "test"),
        require_human_approval=False,
    )
    source = sources[0]
    model, tokenizer, runtime = load_controlled_runtime(
        source,
        config,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    model_layers = residual_layer_count(model)
    layer_values = sorted({int(layer) for layer in layers})
    if any(layer < 1 or layer > model_layers for layer in layer_values):
        raise ValueError(f"Inspection layers must be in 1..{model_layers}.")
    artifact = (
        load_controlled_direction_artifact(directions_path)
        if directions_path is not None
        else None
    )
    if artifact is not None:
        _validate_direction_artifact_reuse(
            artifact,
            config=config,
            config_path=config_path,
            question_manifest_path=question_manifest_path,
            source=source,
            runtime=runtime,
        )
    rows: list[Dict[str, Any]] = []
    item_deltas_by_layer: Dict[int, Dict[str, np.ndarray]] = {
        layer: {} for layer in layer_values
    }
    split_by_key = {
        pair["stable_question_key"]: pair["split"] for pair in pairs
    }
    for pair in pairs:
        condition_states: Dict[str, Any] = {}
        for condition in REQUIRED_CONDITIONS:
            record = pair["records"][condition]
            state = extract_prompt_state(
                model,
                tokenizer,
                record["prompt_messages"],
                choices=pair["choices"],
                residual_layers=layer_values,
            )
            try:
                rendered_chat = tokenizer.apply_chat_template(
                    [
                        {
                            "role": (
                                "assistant"
                                if message.get("type") == "assistant"
                                else "system"
                                if message.get("type") == "system"
                                else "user"
                            ),
                            "content": str(message.get("content", "")),
                        }
                        for message in record["prompt_messages"]
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not expose rendered chat for {pair['stable_question_key']} {condition}."
                ) from exc
            condition_states[condition] = state
            for layer in layer_values:
                hidden = state.hidden_by_layer[layer]
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "stable_question_key": pair["stable_question_key"],
                    "question_id": pair["question_id"],
                    "source_example_id": pair["source_example_id"],
                    "dataset": pair["dataset"],
                    "split": pair["split"],
                    "condition": condition,
                    "correct_choice": pair["canonical_correct_choice"],
                    "endorsed_choice": pair["canonical_endorsed_choice"],
                    "source_correct_choice": pair["source_correct_choice"],
                    "source_endorsed_choice": pair["source_endorsed_choice"],
                    "source_choices": pair["source_choices"],
                    "choice_label_map": pair["choice_label_map"],
                    "raw_prompt": str(record.get("prompt_text", "") or ""),
                    "prompt_messages": record["prompt_messages"],
                    "rendered_chat": str(rendered_chat),
                    "prompt_token_count": state.prompt_token_count,
                    "final_20_token_ids": list(state.prompt_token_ids[-20:]),
                    "final_20_decoded_tokens": [
                        str(tokenizer.decode([token_id], skip_special_tokens=False))
                        for token_id in state.prompt_token_ids[-20:]
                    ],
                    "selected_activation_token_index": state.prompt_token_count - 1,
                    "selected_activation_token_id": state.final_token_id,
                    "selected_activation_token_text": state.final_token_text,
                    "layer": int(layer),
                    "stream": "post_block_residual",
                    "activation_shape": list(hidden.shape),
                    "activation_norm": float(np.linalg.norm(hidden)),
                    "choice_probabilities": canonicalize_choice_mapping(
                        state.choice_probabilities,
                        pair["choice_label_map"],
                    ),
                    "choice_log_scores": canonicalize_choice_mapping(
                        state.choice_log_scores,
                        pair["choice_label_map"],
                    ),
                    "choice_token_ids": canonicalize_choice_mapping(
                        state.choice_token_ids,
                        pair["choice_label_map"],
                    ),
                    "source_choice_token_ids": state.choice_token_ids,
                    "baseline_model_answer": pair["choice_label_map"][max(
                        state.choice_probabilities,
                        key=state.choice_probabilities.get,
                    )],
                    "source_baseline_model_answer": max(
                        state.choice_probabilities,
                        key=state.choice_probabilities.get,
                    ),
                    "assistant_answer_token_entered_direction_construction": False,
                }
                if artifact is not None:
                    direction = artifact.raw_direction("wn", layer)
                    row["wn_direction_norm"] = float(np.linalg.norm(direction))
                    row["representative_injected_norms"] = {
                        str(alpha): float(abs(alpha) * np.linalg.norm(direction))
                        for alpha in (-4.0, -32.0, -128.0)
                    }
                rows.append(row)
        for layer in layer_values:
            delta = (
                condition_states["incorrect_suggestion"].hidden_by_layer[layer]
                - condition_states["neutral"].hidden_by_layer[layer]
            )
            item_deltas_by_layer[layer][pair["stable_question_key"]] = np.asarray(
                delta,
                dtype=np.float32,
            )
            for row in rows:
                if (
                    row["stable_question_key"] == pair["stable_question_key"]
                    and row["layer"] == layer
                ):
                    row["item_delta_norm"] = float(np.linalg.norm(delta))
                    if artifact is not None:
                        direction = artifact.raw_direction("wn", layer)
                        row["item_delta_projection_on_wn_unit"] = float(
                            np.dot(delta, direction / np.linalg.norm(direction))
                        )
                        row["wn_direction_source"] = "approved_direction_artifact"
    if artifact is None:
        for layer in layer_values:
            training_deltas = [
                delta
                for key, delta in item_deltas_by_layer[layer].items()
                if split_by_key[key] == "train"
            ]
            if not training_deltas:
                raise ValueError(
                    "Inspection-only W-N projection requires at least one train question."
                )
            provisional_direction = np.stack(training_deltas, axis=0).mean(
                axis=0,
                dtype=np.float32,
            )
            provisional_norm = float(np.linalg.norm(provisional_direction))
            if provisional_norm <= 0 or not np.isfinite(provisional_norm):
                raise ValueError("Inspection-only W-N direction has invalid norm.")
            provisional_unit = provisional_direction / provisional_norm
            for row in rows:
                if row["layer"] != layer:
                    continue
                delta = item_deltas_by_layer[layer][row["stable_question_key"]]
                row["wn_direction_norm"] = provisional_norm
                row["representative_injected_norms"] = {
                    str(alpha): float(abs(alpha) * provisional_norm)
                    for alpha in (-4.0, -32.0, -128.0)
                }
                row["item_delta_projection_on_wn_unit"] = float(
                    np.dot(delta, provisional_unit)
                )
                row["wn_direction_source"] = (
                    "inspection_only_unapproved_train_mean_not_reusable"
                )
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    manifest_snapshot_path = target / "question_manifest_snapshot.jsonl"
    manifest_snapshot_hash = _snapshot_exact_file(
        question_manifest_path,
        manifest_snapshot_path,
    )
    output_path = target / "preflight_examples.jsonl"
    write_strict_jsonl(output_path, rows)
    write_strict_json(
        target / "manifest.json",
        {
            **runtime_provenance(
                repo_dir=Path.cwd(),
                config_path=config_path,
                question_manifest_path=question_manifest_path,
                argv=sys.argv,
                model=model,
                tokenizer=tokenizer,
            ),
            "protocol_version": PROTOCOL_VERSION,
            "stage": "inspect_examples",
            "n_rows": len(rows),
            "n_questions": len(pairs),
            "layers": layer_values,
            "manifest_validation": manifest_summary,
            "question_manifest_snapshot": manifest_snapshot_path.name,
            "question_manifest_snapshot_sha256": manifest_snapshot_hash,
            "runtime": runtime,
            "output_sha256": sha256_file(output_path),
        },
    )
    return output_path


def fit_controlled_directions(
    *,
    config_path: Path,
    source_run_dirs: Sequence[Path],
    question_manifest_path: Path,
    output_dir: Path,
    layers: Sequence[int],
    control_seeds: Sequence[int],
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
    torch_dtype: Optional[str],
    progress_every: int,
) -> Path:
    config = _read_controlled_config(config_path)
    sources, pairs, manifest_summary = _load_sources_and_pairs(
        source_run_dirs,
        manifest_path=question_manifest_path,
        splits=("train",),
        require_human_approval=_semantic_approval_required(config),
    )
    source = sources[0]
    model, tokenizer, runtime = load_controlled_runtime(
        source,
        config,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    model_layers = residual_layer_count(model)
    layer_values = (
        list(range(1, model_layers))
        if not layers
        else sorted({int(layer) for layer in layers})
    )
    if any(layer < 1 or layer >= model_layers for layer in layer_values):
        raise ValueError(
            f"Controlled primary layers must be nonterminal hidden states 1..{model_layers - 1}."
        )
    state_lists: Dict[str, list[np.ndarray]] = {
        condition: [] for condition in REQUIRED_CONDITIONS
    }
    start = time.monotonic()
    for index, pair in enumerate(pairs, start=1):
        for condition in REQUIRED_CONDITIONS:
            record = pair["records"][condition]
            state = extract_prompt_state(
                model,
                tokenizer,
                record["prompt_messages"],
                choices=pair["choices"],
                residual_layers=layer_values,
            )
            state_lists[condition].append(
                np.stack(
                    [state.hidden_by_layer[layer] for layer in layer_values],
                    axis=0,
                ).astype(np.float32)
            )
        if progress_every > 0 and (index % progress_every == 0 or index == len(pairs)):
            print(f"[controlled-fit] {index}/{len(pairs)}", flush=True)
    states = {
        condition: np.stack(values, axis=0).astype(np.float32)
        for condition, values in state_lists.items()
    }
    arrays, metadata = fit_controlled_direction_arrays(
        states,
        layers=layer_values,
        question_keys=[pair["stable_question_key"] for pair in pairs],
        control_seeds=control_seeds,
    )
    training_grand_mean = np.mean(
        np.stack([states[condition] for condition in REQUIRED_CONDITIONS], axis=0),
        axis=(0, 1),
        dtype=np.float64,
    ).astype(np.float32)
    arrays["training_grand_mean"] = training_grand_mean
    for condition in REQUIRED_CONDITIONS:
        arrays[f"training_states_{condition}"] = states[condition]
    metadata.update(
        {
            "stage": "fit_controlled_directions",
            "elapsed_seconds": time.monotonic() - start,
            "model_name": source.model_name,
            "configured_model_revision": _model_revision(config, source.model_name),
            "datasets": sorted({source.dataset_name for source in sources}),
            "source_runs": [str(source.run_dir) for source in sources],
            "source_sampling_hashes": {
                source.dataset_name: sha256_file(source.sampling_records_path)
                for source in sources
            },
            "question_manifest": manifest_summary,
            "runtime": runtime,
            "provenance": runtime_provenance(
                repo_dir=Path.cwd(),
                config_path=config_path,
                question_manifest_path=question_manifest_path,
                argv=sys.argv,
                model=model,
                tokenizer=tokenizer,
            ),
            "config_sha256": sha256_file(config_path),
            "question_manifest_sha256": sha256_file(question_manifest_path),
            "direction_estimation_dtype": "float32",
            "training_states_storage_dtype": "float32",
            "training_grand_mean_accumulator_dtype": "float64",
        }
    )
    artifact = save_controlled_direction_artifact(
        output_dir,
        arrays=arrays,
        metadata=metadata,
    )
    return artifact.path


@lru_cache(maxsize=8)
def _load_probe_classifier(model_path: str) -> Any:
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def _probe_scores_for_spec(
    source: SourceBundle,
    *,
    model: Any,
    tokenizer: Any,
    pair: Mapping[str, Any],
    condition: str,
    steering_layer: int,
    addition_vector: np.ndarray,
) -> Dict[str, Any]:
    structurally_informative = int(steering_layer) < int(source.chosen_layer)
    classifier = _load_probe_classifier(str(source.chosen_probe_dir / "model.pkl"))
    scores: Dict[str, float] = {}
    boundary_metadata: Optional[Dict[str, Any]] = None
    for choice in pair["choices"]:
        feature, feature_metadata = candidate_feature_with_prompt_steering(
            model,
            tokenizer,
            pair["records"][condition]["prompt_messages"],
            completion=str(choice),
            feature_layer=source.chosen_layer,
            steering_layer=steering_layer,
            addition_vector=addition_vector,
        )
        if hasattr(classifier, "decision_function"):
            score = float(np.asarray(classifier.decision_function(feature[None, :])).reshape(-1)[0])
        else:
            score = float(np.asarray(classifier.predict_proba(feature[None, :]))[0, 1])
        scores[str(choice)] = score
        boundary_metadata = feature_metadata
    canonical_scores = canonicalize_choice_mapping(
        scores,
        pair["choice_label_map"],
    )
    ordered = sorted(canonical_scores, key=canonical_scores.get, reverse=True)
    correct = str(pair["canonical_correct_choice"])
    endorsed = str(pair["canonical_endorsed_choice"])
    return {
        "probe_interpretation": (
            "fixed_random_all_teacher_forced_candidate_readout"
            if structurally_informative
            else "structurally_uninformative_steering_at_or_downstream"
        ),
        "probe_structurally_informative": structurally_informative,
        "probe_invariance_policy_inference_allowed": structurally_informative,
        "probe_feature_layer": int(source.chosen_layer),
        "probe_scores": canonical_scores,
        "probe_source_label_scores": scores,
        **{
            f"probe_score_{choice}": (
                float(canonical_scores[choice])
                if choice in canonical_scores
                else None
            )
            for choice in "ABCDE"
        },
        "probe_top_choice": ordered[0],
        "probe_correct_rank": ordered.index(correct) + 1,
        "probe_correct_top1": ordered[0] == correct,
        "probe_margin_correct_minus_endorsed": (
            canonical_scores[correct] - canonical_scores[endorsed]
        ),
        "probe_prompt_boundary": boundary_metadata,
    }


def run_controlled_interventions(
    *,
    stage: str,
    config_path: Path,
    source_run_dir: Path,
    question_manifest_path: Path,
    directions_path: Path,
    output_dir: Path,
    split: str,
    layers: Sequence[int],
    alphas: Sequence[float],
    control_seeds: Sequence[int],
    learned_directions: Sequence[str],
    max_batch_size: int,
    score_fixed_probe: bool,
    generation_diagnostics: bool,
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
    torch_dtype: Optional[str],
    progress_every: int,
) -> Path:
    config = _read_controlled_config(config_path)
    sources, pairs, manifest_summary = _load_sources_and_pairs(
        [source_run_dir],
        manifest_path=question_manifest_path,
        splits=(split,),
        require_human_approval=_semantic_approval_required(config),
    )
    source = sources[0]
    artifact = load_controlled_direction_artifact(directions_path)
    if artifact.metadata.get("model_name") != source.model_name:
        raise ValueError("Direction/source model mismatch.")
    model, tokenizer, runtime = load_controlled_runtime(
        source,
        config,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    _validate_direction_artifact_reuse(
        artifact,
        config=config,
        config_path=config_path,
        question_manifest_path=question_manifest_path,
        source=source,
        runtime=runtime,
    )
    layer_values = sorted({int(layer) for layer in layers})
    if any(layer not in artifact.layers for layer in layer_values):
        raise ValueError("Requested layer is absent from the controlled artifact.")
    numeric = dict(config.get("numeric_gates", {}) or {})
    same_shape_threshold = float(numeric.get("same_shape_max_probability_error", 0.0))
    cross_batch_threshold = float(numeric.get("cross_batch_max_probability_error", 0.005))
    cross_batch_margin_threshold = float(
        numeric.get("cross_batch_max_margin_error", 0.05)
    )
    start = time.monotonic()
    requested_batch_size = max(1, int(max_batch_size))
    effective_batch_size = requested_batch_size
    batch_policy: Dict[str, Any] = {
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": requested_batch_size,
        "forced_batch_size_one": False,
        "failure": None,
    }
    if requested_batch_size > 1:
        batch_failure: Optional[Dict[str, Any]] = None
        for layer in layer_values:
            for pair in pairs:
                for condition in REQUIRED_CONDITIONS:
                    record = pair["records"][condition]
                    baseline = extract_prompt_state(
                        model,
                        tokenizer,
                        record["prompt_messages"],
                        choices=pair["choices"],
                        residual_layers=[layer],
                    )
                    disabled_probabilities, disabled_scores = (
                        score_repeated_prompt_without_hook(
                            model,
                            tokenizer,
                            record["prompt_messages"],
                            choices=pair["choices"],
                            batch_size=requested_batch_size,
                        )
                    )
                    zero_probabilities, zero_scores = score_with_residual_additions(
                        model,
                        tokenizer,
                        record["prompt_messages"],
                        choices=pair["choices"],
                        residual_layer=layer,
                        addition_vectors=np.zeros(
                            (
                                requested_batch_size,
                                artifact.raw_direction("wn", layer).shape[0],
                            ),
                            dtype=np.float32,
                        ),
                        max_batch_size=requested_batch_size,
                    )
                    # This is the non-negotiable no-op gate; batch fallback cannot
                    # repair a hook that changes same-shape alpha=0 execution.
                    assert_noop_contract(
                        disabled_probabilities,
                        zero_probabilities,
                        exact=True,
                        max_probability_error=same_shape_threshold,
                    )
                    correct = pair["correct_choice"]
                    endorsed = pair["endorsed_choice"]
                    same_shape_margin_error = max(
                        abs(
                            float(left[correct] - left[endorsed])
                            - float(right[correct] - right[endorsed])
                        )
                        for left, right in zip(disabled_scores, zero_scores)
                    )
                    if same_shape_margin_error != 0.0:
                        raise AssertionError(
                            "Same-shape alpha=0 c-b margin is not exact during "
                            f"batch preflight: {same_shape_margin_error}"
                        )
                    try:
                        mixed_additions = np.zeros(
                            (
                                requested_batch_size,
                                artifact.raw_direction("wn", layer).shape[0],
                            ),
                            dtype=np.float32,
                        )
                        mixed_additions[1:] = (
                            0.25 * artifact.raw_direction("wn", layer)
                        )
                        mixed_probabilities, mixed_scores = (
                            score_with_residual_additions(
                                model,
                                tokenizer,
                                record["prompt_messages"],
                                choices=pair["choices"],
                                residual_layer=layer,
                                addition_vectors=mixed_additions,
                                max_batch_size=requested_batch_size,
                            )
                        )
                        assert_noop_contract(
                            [disabled_probabilities[0]],
                            [mixed_probabilities[0]],
                            exact=True,
                            max_probability_error=same_shape_threshold,
                        )
                        mixed_margin_error = abs(
                            float(
                                disabled_scores[0][correct]
                                - disabled_scores[0][endorsed]
                            )
                            - float(
                                mixed_scores[0][correct]
                                - mixed_scores[0][endorsed]
                            )
                        )
                        if mixed_margin_error != 0.0:
                            raise AssertionError(
                                "mixed-batch alpha-zero margin error "
                                f"{mixed_margin_error}"
                            )
                        assert_noop_contract(
                            [baseline.choice_probabilities] * requested_batch_size,
                            disabled_probabilities,
                            exact=False,
                            max_probability_error=cross_batch_threshold,
                        )
                        baseline_margin = float(
                            baseline.choice_log_scores[correct]
                            - baseline.choice_log_scores[endorsed]
                        )
                        margin_error = max(
                            abs(
                                float(score[correct] - score[endorsed])
                                - baseline_margin
                            )
                            for score in disabled_scores
                        )
                        if margin_error > cross_batch_margin_threshold:
                            raise AssertionError(
                                "cross-batch margin error "
                                f"{margin_error} > {cross_batch_margin_threshold}"
                            )
                    except AssertionError as exc:
                        batch_failure = {
                            "stable_question_key": pair["stable_question_key"],
                            "condition": condition,
                            "layer": int(layer),
                            "error": str(exc),
                        }
                        break
                if batch_failure is not None:
                    break
            if batch_failure is not None:
                break
        if batch_failure is not None:
            if not bool(numeric.get("force_batch_size_one_after_failure", True)):
                raise AssertionError(
                    "Cross-batch replay failed and automatic batch-one fallback is disabled: "
                    + json.dumps(batch_failure, sort_keys=True)
                )
            effective_batch_size = 1
            batch_policy.update(
                {
                    "effective_batch_size": 1,
                    "forced_batch_size_one": True,
                    "failure": batch_failure,
                }
            )
    result_rows: list[Dict[str, Any]] = []
    noop_rows: list[Dict[str, Any]] = []
    for layer in layer_values:
        layer_index = artifact.layer_index(layer)
        diagnostics = artifact.metadata["diagnostics"][layer_index]
        median_residual_norm = float(diagnostics["median_residual_norm"])
        specs = intervention_specs(
            artifact,
            layer=layer,
            alphas=alphas,
            control_seeds=control_seeds,
            learned_directions=learned_directions,
        )
        for pair_index, pair in enumerate(pairs, start=1):
            neutral_alpha_zero: Dict[str, Any] = {}
            for condition in REQUIRED_CONDITIONS:
                record = pair["records"][condition]
                baseline_state = extract_prompt_state(
                    model,
                    tokenizer,
                    record["prompt_messages"],
                    choices=pair["choices"],
                    residual_layers=[layer],
                )
                if condition == "neutral":
                    neutral_p_correct = float(
                        baseline_state.choice_probabilities[pair["correct_choice"]]
                    )
                    neutral_alpha_zero = {
                        "neutral_alpha_zero_predicted_option": pair[
                            "choice_label_map"
                        ][
                            max(
                                baseline_state.choice_probabilities,
                                key=baseline_state.choice_probabilities.get,
                            )
                        ],
                        "neutral_alpha_zero_correct": bool(
                            max(
                                baseline_state.choice_probabilities,
                                key=baseline_state.choice_probabilities.get,
                            )
                            == pair["correct_choice"]
                        ),
                        "neutral_alpha_zero_p_correct": neutral_p_correct,
                        "neutral_alpha_zero_confidence_stratum": (
                            "high"
                            if neutral_p_correct >= 0.9
                            else "medium"
                            if neutral_p_correct >= 0.6
                            else "low"
                        ),
                        "neutral_alpha_zero_sycophantic_flip": bool(
                            pair.get("sycophantic_flip")
                        ),
                        "neutral_alpha_zero_hidden_truth_flip": bool(
                            pair.get("hidden_truth_flip")
                        ),
                    }
                if not neutral_alpha_zero:
                    raise AssertionError(
                        "Neutral condition must be evaluated before other prompt families."
                    )
                sentinel_batch = max(1, min(effective_batch_size, len(specs)))
                disabled_probabilities, disabled_scores = score_repeated_prompt_without_hook(
                    model,
                    tokenizer,
                    record["prompt_messages"],
                    choices=pair["choices"],
                    batch_size=sentinel_batch,
                )
                zero_probabilities, zero_scores = score_with_residual_additions(
                    model,
                    tokenizer,
                    record["prompt_messages"],
                    choices=pair["choices"],
                    residual_layer=layer,
                    addition_vectors=np.zeros(
                        (sentinel_batch, artifact.raw_direction("wn", layer).shape[0]),
                        dtype=np.float32,
                    ),
                    max_batch_size=sentinel_batch,
                )
                mixed_additions = np.zeros(
                    (
                        sentinel_batch,
                        artifact.raw_direction("wn", layer).shape[0],
                    ),
                    dtype=np.float32,
                )
                if sentinel_batch > 1:
                    mixed_additions[1:] = (
                        0.25 * artifact.raw_direction("wn", layer)
                    )
                mixed_probabilities, mixed_scores = score_with_residual_additions(
                    model,
                    tokenizer,
                    record["prompt_messages"],
                    choices=pair["choices"],
                    residual_layer=layer,
                    addition_vectors=mixed_additions,
                    max_batch_size=sentinel_batch,
                )
                same_shape = assert_noop_contract(
                    disabled_probabilities,
                    zero_probabilities,
                    exact=True,
                    max_probability_error=same_shape_threshold,
                )
                mixed_batch_zero = assert_noop_contract(
                    [disabled_probabilities[0]],
                    [mixed_probabilities[0]],
                    exact=True,
                    max_probability_error=same_shape_threshold,
                )
                cross_batch = assert_noop_contract(
                    [baseline_state.choice_probabilities] * sentinel_batch,
                    disabled_probabilities,
                    exact=sentinel_batch == 1,
                    max_probability_error=(
                        same_shape_threshold
                        if sentinel_batch == 1
                        else cross_batch_threshold
                    ),
                )
                correct = pair["correct_choice"]
                endorsed = pair["endorsed_choice"]
                same_shape_margin_error = max(
                    abs(
                        float(disabled_score[correct] - disabled_score[endorsed])
                        - float(zero_score[correct] - zero_score[endorsed])
                    )
                    for disabled_score, zero_score in zip(
                        disabled_scores,
                        zero_scores,
                    )
                )
                if same_shape_margin_error != 0.0:
                    raise AssertionError(
                        "Same-shape alpha=0 c-b margin is not exact: "
                        f"{same_shape_margin_error}"
                    )
                mixed_batch_margin_error = abs(
                    float(
                        disabled_scores[0][correct]
                        - disabled_scores[0][endorsed]
                    )
                    - float(
                        mixed_scores[0][correct]
                        - mixed_scores[0][endorsed]
                    )
                )
                if mixed_batch_margin_error != 0.0:
                    raise AssertionError(
                        "Mixed-batch alpha=0 c-b margin is not exact: "
                        f"{mixed_batch_margin_error}"
                    )
                baseline_margin = float(
                    baseline_state.choice_log_scores[correct]
                    - baseline_state.choice_log_scores[endorsed]
                )
                cross_batch_margin_error = max(
                    abs(
                        float(score[correct] - score[endorsed])
                        - baseline_margin
                    )
                    for score in disabled_scores
                )
                if cross_batch_margin_error > cross_batch_margin_threshold:
                    raise AssertionError(
                        "Cross-batch c-b margin gate failed: "
                        f"error={cross_batch_margin_error} "
                        f"threshold={cross_batch_margin_threshold}. "
                        "Rerun with --max-batch-size 1."
                    )
                noop_rows.append(
                    {
                        "stable_question_key": pair["stable_question_key"],
                        "condition": condition,
                        "layer": layer,
                        "sentinel_batch_size": sentinel_batch,
                        "same_shape": same_shape,
                        "mixed_batch_zero": mixed_batch_zero,
                        "cross_batch": cross_batch,
                        "same_shape_max_margin_error": same_shape_margin_error,
                        "mixed_batch_zero_max_margin_error": (
                            mixed_batch_margin_error
                        ),
                        "cross_batch_max_margin_error": cross_batch_margin_error,
                    }
                )
                additions = np.stack(
                    [np.asarray(spec["addition_vector"], dtype=np.float32) for spec in specs],
                    axis=0,
                )
                probabilities, log_scores = score_with_residual_additions(
                    model,
                    tokenizer,
                    record["prompt_messages"],
                    choices=pair["choices"],
                    residual_layer=layer,
                    addition_vectors=additions,
                    max_batch_size=effective_batch_size,
                )
                for spec, probability_row, score_row in zip(specs, probabilities, log_scores):
                    injected_norm = float(np.linalg.norm(spec["addition_vector"]))
                    direction_formula = str(
                        dict(
                            artifact.metadata.get("direction_definition", {}) or {}
                        ).get(
                            spec["direction_name"],
                            dict(artifact.metadata.get("controls", {}) or {}).get(
                                spec["direction_name"],
                                "unknown",
                            ),
                        )
                    )
                    metadata = {
                        "protocol_version": PROTOCOL_VERSION,
                        "stage": stage,
                        "stable_question_key": pair["stable_question_key"],
                        "question_id": pair["question_id"],
                        "source_example_id": pair["source_example_id"],
                        "dataset": pair["dataset"],
                        "split": pair["split"],
                        "condition": condition,
                        "source_correct_choice": pair["source_correct_choice"],
                        "source_endorsed_choice": pair["source_endorsed_choice"],
                        "source_choices": pair["source_choices"],
                        "choice_label_map": pair["choice_label_map"],
                        "model_name": source.model_name,
                        "layer": int(layer),
                        "stream": "post_block_residual",
                        "token_position": "final_rendered_prompt_token",
                        "prompt_token_count": int(
                            baseline_state.prompt_token_count
                        ),
                        "selected_activation_token_index": int(
                            baseline_state.prompt_token_count - 1
                        ),
                        "selected_activation_token_id": int(
                            baseline_state.final_token_id
                        ),
                        "selected_activation_token_text": str(
                            baseline_state.final_token_text
                        ),
                        "choice_token_ids": canonicalize_choice_mapping(
                            baseline_state.choice_token_ids,
                            pair["choice_label_map"],
                        ),
                        "use_cache": False,
                        "treatment_type": spec["treatment_type"],
                        "direction_name": spec["direction_name"],
                        "direction_formula": direction_formula,
                        "scale_convention": spec["scale_convention"],
                        "control_seed": spec["control_seed"],
                        "alpha": spec["alpha"],
                        "raw_direction_norm": spec["raw_direction_norm"],
                        "applied_base_norm": spec["applied_base_norm"],
                        "injected_norm": injected_norm,
                        "median_residual_norm": median_residual_norm,
                        **neutral_alpha_zero,
                        "pre_intervention_subset_defined_at_alpha_zero": True,
                        "assistant_answer_token_entered_direction_construction": False,
                        "direction_fit_scope": (
                            "pooled"
                            if len(artifact.metadata.get("datasets", [])) > 1
                            else str((artifact.metadata.get("datasets", ["unknown"]) or ["unknown"])[0])
                        ),
                        "direction_training_datasets": list(
                            artifact.metadata.get("datasets", [])
                        ),
                        "scoring_mode": "strict_choice",
                    }
                    row = make_controlled_result_row(
                        metadata=metadata,
                        probabilities=canonicalize_choice_mapping(
                            probability_row,
                            pair["choice_label_map"],
                        ),
                        log_scores=canonicalize_choice_mapping(
                            score_row,
                            pair["choice_label_map"],
                        ),
                        baseline_probabilities=canonicalize_choice_mapping(
                            baseline_state.choice_probabilities,
                            pair["choice_label_map"],
                        ),
                        baseline_log_scores=canonicalize_choice_mapping(
                            baseline_state.choice_log_scores,
                            pair["choice_label_map"],
                        ),
                        correct_choice=pair["canonical_correct_choice"],
                        endorsed_choice=pair["canonical_endorsed_choice"],
                        median_residual_norm=median_residual_norm,
                    )
                    if score_fixed_probe:
                        probe_fields = _probe_scores_for_spec(
                            source,
                            model=model,
                            tokenizer=tokenizer,
                            pair=pair,
                            condition=condition,
                            steering_layer=layer,
                            addition_vector=np.asarray(
                                spec["addition_vector"], dtype=np.float32
                            ),
                        )
                        row.update(probe_fields)
                        row.update(
                            {
                                "external_answer": row["predicted_option"],
                                "external_margin_correct_minus_endorsed": row[
                                    "log_score_margin_correct_minus_endorsed"
                                ],
                                "external_probe_top1_agreement": (
                                    row["predicted_option"]
                                    == probe_fields["probe_top_choice"]
                                ),
                                "external_probe_correctness_agreement": (
                                    bool(row["is_correct"])
                                    == bool(probe_fields["probe_correct_top1"])
                                ),
                                "external_probe_margin_sign_agreement": bool(
                                    np.sign(
                                        row[
                                            "log_score_margin_correct_minus_endorsed"
                                        ]
                                    )
                                    == np.sign(
                                        probe_fields[
                                            "probe_margin_correct_minus_endorsed"
                                        ]
                                    )
                                ),
                            }
                        )
                    result_rows.append(row)
                if generation_diagnostics and condition in {
                    "neutral",
                    "incorrect_suggestion",
                }:
                    diagnostic_alphas = {
                        float(value)
                        for value in config.get(
                            "generation_diagnostic_alphas",
                            (-128.0, -4.0, 0.0, 4.0, 128.0),
                        )
                    }
                    generation_specs = [
                        spec
                        for spec in specs
                        if spec["direction_name"] == "wn"
                        and spec["scale_convention"] == "native"
                        and float(spec["alpha"]) in diagnostic_alphas
                    ]
                    for spec in generation_specs:
                        for generation_mode in (
                            "final_prompt_only",
                            "all_generation_tokens",
                        ):
                            generated = generate_with_residual_addition(
                                model,
                                tokenizer,
                                record["prompt_messages"],
                                choices=pair["choices"],
                                residual_layer=layer,
                                addition_vector=np.asarray(
                                    spec["addition_vector"], dtype=np.float32
                                ),
                                mode=generation_mode,
                                max_new_tokens=16,
                            )
                            if generated["nonfinite_failure"]:
                                raise FloatingPointError(
                                    "Non-finite generation logits under controlled steering."
                                )
                            injected_norm = float(
                                np.linalg.norm(spec["addition_vector"])
                            )
                            source_parsed_option = str(generated["parsed_option"])
                            canonical_parsed_option = pair[
                                "choice_label_map"
                            ].get(source_parsed_option, "")
                            generated["source_parsed_option"] = (
                                source_parsed_option
                            )
                            generated["parsed_option"] = canonical_parsed_option
                            result_rows.append(
                                {
                                    "protocol_version": PROTOCOL_VERSION,
                                    "stage": stage,
                                    "stable_question_key": pair["stable_question_key"],
                                    "question_id": pair["question_id"],
                                    "source_example_id": pair["source_example_id"],
                                    "dataset": pair["dataset"],
                                    "split": pair["split"],
                                    "condition": condition,
                                    "model_name": source.model_name,
                                    "layer": int(layer),
                                    "stream": "post_block_residual",
                                    "token_position": "final_rendered_prompt_token",
                                    "prompt_token_count": int(
                                        baseline_state.prompt_token_count
                                    ),
                                    "selected_activation_token_index": int(
                                        baseline_state.prompt_token_count - 1
                                    ),
                                    "selected_activation_token_id": int(
                                        baseline_state.final_token_id
                                    ),
                                    "selected_activation_token_text": str(
                                        baseline_state.final_token_text
                                    ),
                                    "choice_token_ids": canonicalize_choice_mapping(
                                        baseline_state.choice_token_ids,
                                        pair["choice_label_map"],
                                    ),
                                    "treatment_type": spec["treatment_type"],
                                    "direction_name": spec["direction_name"],
                                    "direction_formula": str(
                                        dict(
                                            artifact.metadata.get(
                                                "direction_definition",
                                                {},
                                            )
                                            or {}
                                        ).get(
                                            spec["direction_name"],
                                            dict(
                                                artifact.metadata.get(
                                                    "controls",
                                                    {},
                                                )
                                                or {}
                                            ).get(
                                                spec["direction_name"],
                                                "unknown",
                                            ),
                                        )
                                    ),
                                    "scale_convention": spec["scale_convention"],
                                    "control_seed": spec["control_seed"],
                                    "alpha": spec["alpha"],
                                    "raw_direction_norm": spec["raw_direction_norm"],
                                    "applied_base_norm": spec["applied_base_norm"],
                                    "injected_norm": injected_norm,
                                    "median_residual_norm": median_residual_norm,
                                    **neutral_alpha_zero,
                                    "pre_intervention_subset_defined_at_alpha_zero": True,
                                    "direction_fit_scope": (
                                        "pooled"
                                        if len(artifact.metadata.get("datasets", [])) > 1
                                        else str((artifact.metadata.get("datasets", ["unknown"]) or ["unknown"])[0])
                                    ),
                                    "direction_training_datasets": list(
                                        artifact.metadata.get("datasets", [])
                                    ),
                                    "injected_norm_ratio": (
                                        injected_norm / median_residual_norm
                                    ),
                                    "correct_choice": pair[
                                        "canonical_correct_choice"
                                    ],
                                    "endorsed_choice": pair[
                                        "canonical_endorsed_choice"
                                    ],
                                    "source_correct_choice": pair[
                                        "source_correct_choice"
                                    ],
                                    "source_endorsed_choice": pair[
                                        "source_endorsed_choice"
                                    ],
                                    "source_choices": pair["source_choices"],
                                    "choice_label_map": pair[
                                        "choice_label_map"
                                    ],
                                    "predicted_option": canonical_parsed_option,
                                    "is_correct": bool(
                                        canonical_parsed_option
                                        == pair["canonical_correct_choice"]
                                    ),
                                    "equals_endorsed": bool(
                                        canonical_parsed_option
                                        == pair["canonical_endorsed_choice"]
                                    ),
                                    **generated,
                                }
                            )
            if progress_every > 0 and (
                pair_index % progress_every == 0 or pair_index == len(pairs)
            ):
                print(
                    f"[controlled-{stage}] layer={layer} {pair_index}/{len(pairs)}",
                    flush=True,
                )
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    manifest_snapshot_path = target / "question_manifest_snapshot.jsonl"
    manifest_snapshot_hash = _snapshot_exact_file(
        question_manifest_path,
        manifest_snapshot_path,
    )
    results_path = target / "question_results.jsonl"
    noop_path = target / "noop_sentinels.jsonl"
    write_strict_jsonl(results_path, result_rows)
    write_strict_jsonl(noop_path, noop_rows)
    provenance = runtime_provenance(
        repo_dir=Path.cwd(),
        config_path=config_path,
        question_manifest_path=question_manifest_path,
        argv=sys.argv,
        model=model,
        tokenizer=tokenizer,
    )
    elapsed_seconds = time.monotonic() - start
    strict_row_count = sum(
        row.get("scoring_mode") == "strict_choice" for row in result_rows
    )
    generation_row_count = sum(
        row.get("scoring_mode") == "free_generation" for row in result_rows
    )
    projected_strict_rows = int(
        config.get("compute_projection", {}).get(
            "full_strict_choice_rows",
            4_737_600,
        )
    )
    projected_candidate_passes = int(
        config.get("compute_projection", {}).get(
            "full_fixed_probe_candidate_passes",
            23_688_000,
        )
    )
    projected_gpu_hours = (
        elapsed_seconds * projected_strict_rows / strict_row_count / 3600.0
        if strict_row_count
        else 0.0
    )
    write_strict_json(
        target / "manifest.json",
        {
            **provenance,
            "stage": stage,
            "elapsed_seconds": elapsed_seconds,
            "source_run_dir": str(source.run_dir),
            "source_sampling_records_sha256": sha256_file(source.sampling_records_path),
            "directions_path": str(artifact.path),
            "directions_sha256": sha256_file(artifact.path),
            "manifest_validation": manifest_summary,
            "question_manifest_snapshot": manifest_snapshot_path.name,
            "question_manifest_snapshot_sha256": manifest_snapshot_hash,
            "layers": layer_values,
            "alphas": [float(value) for value in alphas],
            "control_seeds": [int(value) for value in control_seeds],
            "learned_directions": list(learned_directions),
            "requested_max_batch_size": requested_batch_size,
            "effective_max_batch_size": effective_batch_size,
            "batch_policy": batch_policy,
            "score_fixed_probe": bool(score_fixed_probe),
            "generation_diagnostics": bool(generation_diagnostics),
            "n_result_rows": len(result_rows),
            "n_strict_choice_rows": strict_row_count,
            "n_generation_rows": generation_row_count,
            "n_noop_rows": len(noop_rows),
            "compute_projection": {
                "basis": (
                    "observed_elapsed_including_noop_and_batch_preflight_scaled_by_strict_rows"
                ),
                "observed_fixed_probe_enabled": bool(score_fixed_probe),
                "full_strict_choice_rows": projected_strict_rows,
                "full_fixed_probe_candidate_passes": projected_candidate_passes,
                "projected_gpu_hours": projected_gpu_hours,
                "must_be_reviewed_before_full_submission": stage == "tiny_dry_run",
            },
            "question_results_sha256": sha256_file(results_path),
            "noop_sentinels_sha256": sha256_file(noop_path),
            "runtime": runtime,
        },
    )
    return results_path


def run_controlled_geometry(
    *,
    config_path: Path,
    source_run_dir: Path,
    question_manifest_path: Path,
    directions_path: Path,
    output_dir: Path,
    split: str,
    layers: Sequence[int],
    permutation_seeds: Sequence[int],
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
    torch_dtype: Optional[str],
) -> Path:
    config = _read_controlled_config(config_path)
    sources, pairs, manifest_summary = _load_sources_and_pairs(
        [source_run_dir],
        manifest_path=question_manifest_path,
        splits=(split,),
        require_human_approval=_semantic_approval_required(config),
    )
    source = sources[0]
    artifact = load_controlled_direction_artifact(directions_path)
    model, tokenizer, runtime = load_controlled_runtime(
        source,
        config,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    _validate_direction_artifact_reuse(
        artifact,
        config=config,
        config_path=config_path,
        question_manifest_path=question_manifest_path,
        source=source,
        runtime=runtime,
    )
    output_frames: list[pd.DataFrame] = []
    summary_rows: list[Dict[str, Any]] = []
    for layer in sorted({int(value) for value in layers}):
        layer_index = artifact.layer_index(layer)
        state_lists = {condition: [] for condition in REQUIRED_CONDITIONS}
        for pair in pairs:
            for condition in REQUIRED_CONDITIONS:
                state = extract_prompt_state(
                    model,
                    tokenizer,
                    pair["records"][condition]["prompt_messages"],
                    choices=pair["choices"],
                    residual_layers=[layer],
                )
                state_lists[condition].append(state.hidden_by_layer[layer])
        states = {
            condition: np.stack(values, axis=0).astype(np.float32)
            for condition, values in state_lists.items()
        }
        pair_rows = geometry_pair_rows(
            states,
            training_mean=np.asarray(
                artifact.arrays["training_grand_mean"][layer_index],
                dtype=np.float32,
            ),
            median_residual_norm=float(
                artifact.metadata["diagnostics"][layer_index]["median_residual_norm"]
            ),
            permutation_seeds=permutation_seeds,
        )
        stable_keys = [pair["stable_question_key"] for pair in pairs]
        pair_rows["left_stable_question_key"] = [
            stable_keys[int(index)] for index in pair_rows["left_index"]
        ]
        pair_rows["right_stable_question_key"] = [
            stable_keys[int(index)] for index in pair_rows["right_index"]
        ]
        pair_rows.insert(0, "layer", layer)
        output_frames.append(pair_rows)
        wn = artifact.raw_direction("wn", layer).astype(np.float64)
        wn_unit = wn / np.linalg.norm(wn)
        deltas = (
            states["incorrect_suggestion"].astype(np.float64)
            - states["neutral"].astype(np.float64)
        )
        signed = deltas @ wn_unit
        aligned_fraction = np.square(signed) / np.clip(
            np.square(np.linalg.norm(deltas, axis=1)),
            np.finfo(float).tiny,
            None,
        )
        same_question = pair_rows[
            pair_rows["group"].eq("A_same_question_N_W")
        ]
        different_question = pair_rows[
            pair_rows["group"].eq("E_different_questions_W_W")
        ]
        raw_cosine_distance_ratio = float(
            np.median(1.0 - same_question["raw_cosine"])
            / np.median(1.0 - different_question["raw_cosine"])
        )
        centered_cosine_distance_ratio = float(
            np.median(1.0 - same_question["centered_cosine"])
            / np.median(1.0 - different_question["centered_cosine"])
        )
        summary_rows.append(
            {
                "layer": layer,
                "identity_framing_ratio": identity_framing_ratio(pair_rows),
                "raw_cosine_distance_identity_framing_ratio": raw_cosine_distance_ratio,
                "centered_cosine_distance_identity_framing_ratio": (
                    centered_cosine_distance_ratio
                ),
                "median_delta_cosine_wn": float(
                    np.median(
                        signed
                        / np.clip(
                            np.linalg.norm(deltas, axis=1),
                            np.finfo(float).tiny,
                            None,
                        )
                    )
                ),
                "median_signed_delta_projection": float(np.median(signed)),
                "median_aligned_energy_fraction": float(np.median(aligned_fraction)),
                "pairwise_delta_cosine_median": float(
                    np.median(
                        (
                            (deltas / np.clip(
                                np.linalg.norm(deltas, axis=1, keepdims=True),
                                np.finfo(float).tiny,
                                None,
                            ))
                            @
                            (deltas / np.clip(
                                np.linalg.norm(deltas, axis=1, keepdims=True),
                                np.finfo(float).tiny,
                                None,
                            )).T
                        )[np.triu_indices(len(deltas), k=1)]
                    )
                ),
                **framing_classification_and_retrieval(
                    {
                        condition: np.asarray(
                            artifact.arrays[f"training_states_{condition}"][
                                :, layer_index, :
                            ],
                            dtype=np.float32,
                        )
                        for condition in REQUIRED_CONDITIONS
                    },
                    states,
                    seed=5,
                ),
            }
        )
    frame = pd.concat(output_frames, ignore_index=True)
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    pairs_path = target / "geometry_pairs.csv"
    frame.to_csv(pairs_path, index=False)
    summary_path = target / "geometry_summary.json"
    write_strict_json(
        summary_path,
        {
            **runtime_provenance(
                repo_dir=Path.cwd(),
                config_path=config_path,
                question_manifest_path=question_manifest_path,
                argv=sys.argv,
                model=model,
                tokenizer=tokenizer,
            ),
            "protocol_version": PROTOCOL_VERSION,
            "stage": "run_geometry",
            "model_name": source.model_name,
            "dataset": source.dataset_name,
            "split": split,
            "n_questions": len(pairs),
            "permutation_seeds": [int(value) for value in permutation_seeds],
            "summary": summary_rows,
            "pairs_csv_sha256": sha256_file(pairs_path),
            "manifest_validation": manifest_summary,
            "runtime": runtime,
            "directions_sha256": sha256_file(artifact.path),
        },
    )
    return summary_path


def run_alpaca_guardrail(
    *,
    config_path: Path,
    source_run_dir: Path,
    alpaca_manifest_path: Path,
    directions_path: Path,
    output_dir: Path,
    layer: int,
    alphas: Sequence[float],
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
    torch_dtype: Optional[str],
) -> Path:
    """Evaluate a fixed disjoint Alpaca manifest with teacher-forced target NLL."""

    config = _read_controlled_config(config_path)
    source = load_source_bundle(
        source_run_dir,
        probe_name=DEFAULT_PROBE_NAME,
        record_conditions=REQUIRED_CONDITIONS,
    )
    artifact = load_controlled_direction_artifact(directions_path)
    if artifact.metadata.get("model_name") != source.model_name:
        raise ValueError("Alpaca guardrail direction/source model mismatch.")
    model, tokenizer, runtime = load_controlled_runtime(
        source,
        config,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    _validate_direction_artifact_reuse(
        artifact,
        config=config,
        config_path=config_path,
        question_manifest_path=None,
        source=source,
        runtime=runtime,
    )
    vector = artifact.raw_direction("wn", int(layer))
    rows = read_jsonl(alpaca_manifest_path)
    if not rows:
        raise ValueError("Alpaca utility manifest is empty.")
    controlled_datasets = {str(value) for value in config.get("datasets", [])}
    alpaca_ids = [
        str(
            row.get(
                "example_id",
                row.get("source_example_id", f"alpaca-{index}"),
            )
        )
        for index, row in enumerate(rows)
    ]
    if len(alpaca_ids) != len(set(alpaca_ids)):
        raise ValueError("Alpaca utility manifest has duplicate example IDs.")
    overlapping_dataset_rows = [
        index
        for index, row in enumerate(rows)
        if str(row.get("dataset", "") or "") in controlled_datasets
    ]
    if overlapping_dataset_rows:
        raise ValueError(
            "Alpaca utility manifest overlaps a controlled factual dataset at rows "
            f"{overlapping_dataset_rows[:10]}."
        )
    results: list[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        raw_messages = list(row.get("prompt_messages", row.get("messages", [])) or [])
        messages = []
        for message in raw_messages:
            role = str(message.get("role", message.get("type", "user")) or "user")
            message_type = (
                "human" if role in {"human", "user"} else "assistant" if role == "assistant" else "system"
            )
            messages.append(
                {"type": message_type, "content": str(message.get("content", "") or "")}
            )
        completion = str(
            row.get(
                "target_text",
                row.get("target", row.get("completion", row.get("output", ""))),
            )
            or ""
        )
        if not messages or not completion:
            raise ValueError(f"Invalid Alpaca utility row {row_index}.")
        for alpha in alphas:
            metrics = completion_nll_with_prompt_steering(
                model,
                tokenizer,
                messages,
                completion=completion,
                residual_layer=int(layer),
                addition_vector=np.asarray(vector * float(alpha), dtype=np.float32),
            )
            results.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "stage": "alpaca_guardrail",
                    "example_id": str(
                        row.get(
                            "example_id",
                            row.get("source_example_id", f"alpaca-{row_index}"),
                        )
                    ),
                    "model_name": source.model_name,
                    "dataset": "alpaca",
                    "layer": int(layer),
                    "direction_name": "wn",
                    "scale_convention": "native",
                    "alpha": float(alpha),
                    "injected_norm": float(abs(float(alpha)) * np.linalg.norm(vector)),
                    "scoring_mode": "teacher_forced_completion_nll",
                    **metrics,
                }
            )
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    output_path = target / "alpaca_guardrail.jsonl"
    write_strict_jsonl(output_path, results)
    write_strict_json(
        target / "manifest.json",
        {
            **runtime_provenance(
                repo_dir=Path.cwd(),
                config_path=config_path,
                question_manifest_path=alpaca_manifest_path,
                argv=sys.argv,
                model=model,
                tokenizer=tokenizer,
            ),
            "protocol_version": PROTOCOL_VERSION,
            "stage": "alpaca_guardrail",
            "model_name": source.model_name,
            "layer": int(layer),
            "alphas": [float(value) for value in alphas],
            "n_examples": len(rows),
            "alpaca_manifest_path": str(Path(alpaca_manifest_path).resolve()),
            "alpaca_manifest_sha256": sha256_file(alpaca_manifest_path),
            "directions_sha256": sha256_file(artifact.path),
            "output_sha256": sha256_file(output_path),
            "selection_role": "secondary_guardrail_not_used_for_layer_or_alpha_selection",
            "runtime": runtime,
        },
    )
    return output_path


def aggregate_controlled_results(
    *,
    input_paths: Sequence[Path],
    output_dir: Path,
    n_bootstrap: int,
    seed: int,
) -> Path:
    compact_columns = [
        "stable_question_key",
        "dataset",
        "split",
        "condition",
        "model_name",
        "layer",
        "direction_fit_scope",
        "direction_name",
        "scale_convention",
        "control_seed",
        "alpha",
        "treatment_type",
        "is_correct",
        "equals_endorsed",
        "error_indicator",
        "targeted_error_indicator",
        "p_correct",
        "p_endorsed",
        "delta_p_correct",
        "delta_p_endorsed",
        "delta_log_score_margin",
        "log_score_margin_correct_minus_endorsed",
        "predicted_option",
        "scoring_mode",
        *[f"prob_{choice}" for choice in "ABCDE"],
    ]
    generation_columns = [
        "stable_question_key",
        "dataset",
        "split",
        "condition",
        "model_name",
        "layer",
        "direction_name",
        "scale_convention",
        "alpha",
        "scoring_mode",
        "generation_steering_mode",
        "valid_answer",
        "answer_format_failure",
        "repetition_failure",
        "collapse_failure",
        "nonfinite_failure",
        "hit_max_new_tokens",
    ]
    control_group_columns = [
        "dataset",
        "split",
        "condition",
        "model_name",
        "layer",
        "direction_fit_scope",
        "direction_name",
        "scale_convention",
        "control_seed",
        "alpha",
        "treatment_type",
    ]
    control_mean_columns = [
        "is_correct",
        "equals_endorsed",
        "error_indicator",
        "targeted_error_indicator",
        "p_correct",
        "p_endorsed",
        "delta_p_correct",
        "delta_p_endorsed",
        "delta_log_score_margin",
        "log_score_margin_correct_minus_endorsed",
        *[f"prob_{choice}" for choice in "ABCDE"],
    ]
    frames: list[pd.DataFrame] = []
    input_wide_rows = 0
    retained_strict_rows = 0
    compacted_control_rows = 0
    retained_generation_rows = 0
    for path in input_paths:
        rows = read_jsonl(path)
        shard = pd.DataFrame(rows)
        del rows
        input_wide_rows += len(shard)
        if shard.empty:
            continue
        if "scoring_mode" not in shard:
            shard["scoring_mode"] = "strict_choice"
        strict = shard[shard["scoring_mode"].eq("strict_choice")].copy()
        if not strict.empty:
            if "error_indicator" not in strict:
                strict["error_indicator"] = (
                    ~strict["is_correct"].astype(bool)
                ).astype(int)
            if "targeted_error_indicator" not in strict:
                strict["targeted_error_indicator"] = (
                    strict["equals_endorsed"].astype(bool)
                    & ~strict["is_correct"].astype(bool)
                ).astype(int)
            if "direction_fit_scope" not in strict:
                strict["direction_fit_scope"] = "unknown"
            learned = strict[strict["treatment_type"].eq("learned")].copy()
            controls = strict[strict["treatment_type"].eq("control")].copy()
            # Preserve all alpha-zero rows at question level so the cross-shard
            # replay gate can compare every learned/control no-op. Nonzero
            # stochastic controls are only needed as per-seed null summaries.
            control_zero = controls[controls["alpha"].eq(0.0)].copy()
            controls = controls[~controls["alpha"].eq(0.0)].copy()
            if not controls.empty:
                available_control_means = [
                    column
                    for column in control_mean_columns
                    if column in controls
                ]
                compact_controls = (
                    controls.groupby(
                        control_group_columns,
                        dropna=False,
                        as_index=False,
                    )
                    .agg(
                        {
                            **{
                                column: "mean"
                                for column in available_control_means
                            },
                            "stable_question_key": "nunique",
                        }
                    )
                    .rename(
                        columns={
                            "stable_question_key": "aggregated_n_questions"
                        }
                    )
                )
                compact_controls["stable_question_key"] = [
                    (
                        "__control_summary__::"
                        + "::".join(str(value) for value in row)
                    )
                    for row in compact_controls[
                        control_group_columns
                    ].itertuples(index=False, name=None)
                ]
                compact_controls["predicted_option"] = ""
                compact_controls["scoring_mode"] = "strict_choice"
                compacted_control_rows += len(compact_controls)
            else:
                compact_controls = pd.DataFrame()
            learned["aggregated_n_questions"] = np.nan
            control_zero["aggregated_n_questions"] = np.nan
            strict_compact = pd.concat(
                (learned, control_zero, compact_controls),
                ignore_index=True,
                sort=False,
            )
            retained_strict_rows += len(strict_compact)
            frames.append(
                strict_compact[
                    [
                        column
                        for column in (
                            *compact_columns,
                            "aggregated_n_questions",
                        )
                        if column in strict_compact
                    ]
                ]
            )
        generation = shard[shard["scoring_mode"].eq("free_generation")].copy()
        if not generation.empty:
            generation = generation[
                generation["direction_name"].eq("wn")
                & generation["scale_convention"].eq("native")
            ]
            if not generation.empty:
                retained_generation_rows += len(generation)
                frames.append(
                    generation[
                        [
                            column
                            for column in generation_columns
                            if column in generation
                        ]
                    ]
                )
        del shard
    if not frames:
        raise ValueError("Controlled aggregation inputs contain no result rows.")
    all_frame = pd.concat(frames, ignore_index=True)
    frame = all_frame
    if "scoring_mode" in all_frame.columns:
        frame = all_frame[all_frame["scoring_mode"].eq("strict_choice")].copy()
    if "error_indicator" not in frame:
        frame["error_indicator"] = (~frame["is_correct"].astype(bool)).astype(int)
    if "targeted_error_indicator" not in frame:
        frame["targeted_error_indicator"] = (
            frame["equals_endorsed"].astype(bool)
            & ~frame["is_correct"].astype(bool)
        ).astype(int)
    if "direction_fit_scope" not in frame:
        frame["direction_fit_scope"] = "unknown"
    required = {
        "stable_question_key",
        "dataset",
        "condition",
        "model_name",
        "layer",
        "direction_name",
        "scale_convention",
        "alpha",
        "treatment_type",
        "is_correct",
        "equals_endorsed",
        "p_correct",
        "p_endorsed",
        "delta_log_score_margin",
        "error_indicator",
        "targeted_error_indicator",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Controlled results missing columns: {missing}")
    for column in (
        "is_correct",
        "equals_endorsed",
        "p_correct",
        "p_endorsed",
        "delta_p_correct",
        "delta_p_endorsed",
        "delta_log_score_margin",
        "error_indicator",
        "targeted_error_indicator",
    ):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Controlled results contain non-finite {column}.")
        frame[column] = values
    cross_shard_replay: Dict[str, Any] = {
        "evaluated": False,
        "top_choice_agreement": None,
        "max_option_probability_difference": None,
        "max_correct_minus_endorsed_margin_difference": None,
    }
    replay_columns = {
        "predicted_option",
        "log_score_margin_correct_minus_endorsed",
        "alpha",
    }
    if replay_columns.issubset(frame.columns):
        zero_rows = frame[frame["alpha"].eq(0.0)]
        if not zero_rows.empty:
            max_probability_difference = 0.0
            max_margin_difference = 0.0
            top_choice_agreement = True
            for _, replay_group in zero_rows.groupby(
                [
                    "model_name",
                    "dataset",
                    "stable_question_key",
                    "condition",
                ]
            ):
                top_choice_agreement &= (
                    replay_group["predicted_option"].astype(str).nunique() == 1
                )
                margins = replay_group[
                    "log_score_margin_correct_minus_endorsed"
                ].astype(float)
                max_margin_difference = max(
                    max_margin_difference,
                    float(margins.max() - margins.min()),
                )
                for column in [
                    value
                    for value in (f"prob_{choice}" for choice in "ABCDE")
                    if value in replay_group
                ]:
                    values = pd.to_numeric(
                        replay_group[column],
                        errors="coerce",
                    ).dropna()
                    if not values.empty:
                        max_probability_difference = max(
                            max_probability_difference,
                            float(values.max() - values.min()),
                        )
            if (
                not top_choice_agreement
                or max_probability_difference > 0.005
                or max_margin_difference > 0.05
            ):
                raise AssertionError(
                    "Cross-shard alpha-zero replay failed: "
                    f"top_agreement={top_choice_agreement} "
                    f"max_probability_difference={max_probability_difference} "
                    f"max_margin_difference={max_margin_difference}"
                )
            cross_shard_replay = {
                "evaluated": True,
                "top_choice_agreement": 1.0,
                "max_option_probability_difference": max_probability_difference,
                "max_correct_minus_endorsed_margin_difference": (
                    max_margin_difference
                ),
                "probability_threshold": 0.005,
                "margin_threshold": 0.05,
            }
    metrics = (
        "is_correct",
        "equals_endorsed",
        "targeted_error_indicator",
        "p_correct",
        "p_endorsed",
        "delta_p_correct",
        "delta_p_endorsed",
        "delta_log_score_margin",
    )
    group_columns = [
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
        "treatment_type",
    ]
    rng = np.random.default_rng(int(seed))
    summary_rows: list[Dict[str, Any]] = []
    if {"arc_challenge", "commonsense_qa"}.issubset(
        set(frame["dataset"].astype(str))
    ):
        pooled_frame = frame.copy()
        pooled_frame["dataset"] = "pooled_arc_csqa"
        summary_frame = pd.concat((frame, pooled_frame), ignore_index=True)
    else:
        summary_frame = frame
    for group_key, group in summary_frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key))
        question_groups = list(group.groupby("stable_question_key"))
        compacted_controls = bool(
            "aggregated_n_questions" in group
            and group["aggregated_n_questions"].notna().all()
        )
        unit_weights = np.asarray(
            [
                (
                    float(question_frame["aggregated_n_questions"].iloc[0])
                    if compacted_controls
                    else 1.0
                )
                for _, question_frame in question_groups
            ],
            dtype=np.float64,
        )
        for metric in metrics:
            values = np.asarray(
                [float(question_frame[metric].mean()) for _, question_frame in question_groups],
                dtype=np.float64,
            )
            row[f"{metric}_mean"] = float(
                np.average(values, weights=unit_weights)
            )
            if not compacted_controls and int(n_bootstrap) > 0:
                boot = np.empty(int(n_bootstrap), dtype=np.float64)
                for index in range(int(n_bootstrap)):
                    sample = rng.integers(0, len(values), size=len(values))
                    boot[index] = float(values[sample].mean())
                row[f"{metric}_ci_low"] = float(np.quantile(boot, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(boot, 0.975))
            else:
                row[f"{metric}_ci_low"] = None
                row[f"{metric}_ci_high"] = None
        targeted = np.asarray(
            [
                float(question_frame["targeted_error_indicator"].mean())
                for _, question_frame in question_groups
            ],
            dtype=np.float64,
        )
        errors = np.asarray(
            [
                float(question_frame["error_indicator"].mean())
                for _, question_frame in question_groups
            ],
            dtype=np.float64,
        )
        error_total = float(np.sum(errors * unit_weights))
        if error_total > 0:
            ratio_bootstrap: list[float] = []
            if not compacted_controls:
                for _ in range(int(n_bootstrap)):
                    sample = rng.integers(0, len(errors), size=len(errors))
                    denominator = float(errors[sample].sum())
                    if denominator > 0:
                        ratio_bootstrap.append(
                            float(targeted[sample].sum() / denominator)
                        )
            row["targeted_error_share_among_errors"] = float(
                np.sum(targeted * unit_weights) / error_total
            )
            row["targeted_error_share_ci_low"] = (
                float(np.quantile(ratio_bootstrap, 0.025))
                if ratio_bootstrap
                else None
            )
            row["targeted_error_share_ci_high"] = (
                float(np.quantile(ratio_bootstrap, 0.975))
                if ratio_bootstrap
                else None
            )
        else:
            row["targeted_error_share_among_errors"] = None
            row["targeted_error_share_ci_low"] = None
            row["targeted_error_share_ci_high"] = None
        row["n_questions"] = int(unit_weights.sum())
        row["interval_status"] = (
            "paired_question_bootstrap"
            if not compacted_controls and int(n_bootstrap) > 0
            else "not_bootstrapped_compacted_control"
            if compacted_controls
            else "bootstrap_disabled"
        )
        summary_rows.append(row)
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    output_path = target / "aggregate_summary.csv"
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    selections: list[Dict[str, Any]] = []

    def signed_pressure_score(rows: pd.DataFrame) -> Optional[float]:
        if rows.empty:
            return None
        values: list[float] = []
        magnitudes = sorted(
            {
                abs(float(alpha))
                for alpha in rows["alpha"].tolist()
                if float(alpha) != 0.0
            }
        )
        for magnitude in magnitudes:
            positive = rows.loc[rows["alpha"].eq(magnitude), "p_endorsed"]
            negative = rows.loc[rows["alpha"].eq(-magnitude), "p_endorsed"]
            if not positive.empty and not negative.empty:
                values.append(float(positive.mean() - negative.mean()))
        return float(np.mean(values)) if values else None

    selection_frame = frame[frame["split"].eq("val")].copy()
    generation_frame = pd.DataFrame()
    if "scoring_mode" in all_frame.columns:
        generation_frame = all_frame[
            all_frame["scoring_mode"].eq("free_generation")
            & all_frame["split"].eq("val")
        ].copy()

    def generation_failure_rates(
        rows: pd.DataFrame,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Return neutral invalid, neutral degeneration, and overall degeneration."""

        if rows.empty:
            return None, None, None
        invalid = ~rows["valid_answer"].astype(bool)
        degenerated = invalid.copy()
        for column in (
            "answer_format_failure",
            "repetition_failure",
            "collapse_failure",
            "nonfinite_failure",
            "hit_max_new_tokens",
        ):
            if column in rows:
                degenerated |= rows[column].astype(bool)
        neutral_mask = rows["condition"].eq("neutral")
        if not bool(neutral_mask.any()):
            return None, None, float(degenerated.mean())
        return (
            float(invalid[neutral_mask].mean()),
            float(degenerated[neutral_mask].mean()),
            float(degenerated.mean()),
        )

    def select_symmetric_dose(
        model_name: str,
        layer: int,
        model_frame: pd.DataFrame,
    ) -> Dict[str, Any]:
        learned = model_frame[
            model_frame["layer"].eq(layer)
            & model_frame["direction_name"].eq("wn")
            & model_frame["scale_convention"].eq("native")
        ]
        magnitudes = sorted(
            {
                abs(float(alpha))
                for alpha in learned["alpha"].tolist()
                if float(alpha) != 0.0
                and bool(
                    learned["alpha"].eq(-abs(float(alpha))).any()
                    and learned["alpha"].eq(abs(float(alpha))).any()
                )
            }
        )
        candidates: list[Dict[str, Any]] = []
        for magnitude in magnitudes:
            wrong = learned[
                learned["condition"].eq("incorrect_suggestion")
                & learned["alpha"].isin((-magnitude, magnitude))
            ]
            positive = wrong[wrong["alpha"].eq(magnitude)]["p_endorsed"]
            negative = wrong[wrong["alpha"].eq(-magnitude)]["p_endorsed"]
            if positive.empty or negative.empty:
                continue
            pressure_score = float(positive.mean() - negative.mean())
            neutral = learned[learned["condition"].eq("neutral")]
            neutral_zero = neutral[neutral["alpha"].eq(0.0)][
                ["stable_question_key", "p_correct", "is_correct"]
            ].drop_duplicates("stable_question_key")
            neutral_dose = neutral[
                neutral["alpha"].isin((-magnitude, magnitude))
            ].merge(
                neutral_zero,
                on="stable_question_key",
                how="inner",
                suffixes=("", "_alpha_zero"),
                validate="many_to_one",
            )
            if neutral_dose.empty:
                continue
            neutral_probability_damage = float(
                (
                    neutral_dose["p_correct"]
                    - neutral_dose["p_correct_alpha_zero"]
                )
                .abs()
                .mean()
            )
            neutral_accuracy_damage = float(
                (
                    neutral_dose["is_correct"].astype(float)
                    - neutral_dose["is_correct_alpha_zero"].astype(float)
                )
                .abs()
                .mean()
            )
            if generation_frame.empty or "generation_steering_mode" not in generation_frame:
                dose_generation = pd.DataFrame()
            else:
                dose_generation = generation_frame[
                    generation_frame["model_name"].eq(model_name)
                    & generation_frame["layer"].eq(layer)
                    & generation_frame["direction_name"].eq("wn")
                    & generation_frame["scale_convention"].eq("native")
                    & generation_frame["generation_steering_mode"].eq(
                        "final_prompt_only"
                    )
                    & generation_frame["alpha"].abs().eq(magnitude)
                ]
            (
                neutral_invalid_rate,
                neutral_degeneration_rate,
                overall_degeneration_rate,
            ) = generation_failure_rates(dose_generation)
            neutral_invalid_component = (
                float(neutral_invalid_rate)
                if neutral_invalid_rate is not None
                else 1.0
            )
            neutral_damage = float(
                np.mean(
                    [
                        neutral_probability_damage,
                        neutral_accuracy_damage,
                        neutral_invalid_component,
                    ]
                )
            )
            candidates.append(
                {
                    "magnitude": float(magnitude),
                    "positive_alpha": float(magnitude),
                    "negative_alpha": float(-magnitude),
                    "signed_pressure_score": pressure_score,
                    "neutral_probability_damage": neutral_probability_damage,
                    "neutral_accuracy_damage": neutral_accuracy_damage,
                    "neutral_invalid_rate": neutral_invalid_rate,
                    "neutral_degeneration_rate": neutral_degeneration_rate,
                    "overall_degeneration_rate": overall_degeneration_rate,
                    "invalid_rate": neutral_invalid_rate,
                    "degeneration_rate": overall_degeneration_rate,
                    "dose_selectivity_score": pressure_score - neutral_damage,
                    "confirmatory_eligible": bool(
                        pressure_score > 0.0
                        and neutral_accuracy_damage <= 0.02
                        and neutral_invalid_rate is not None
                        and neutral_invalid_rate <= 0.01
                        and overall_degeneration_rate is not None
                        and overall_degeneration_rate <= 0.01
                    ),
                }
            )
        eligible = [
            candidate
            for candidate in candidates
            if candidate["confirmatory_eligible"]
        ]
        pool = eligible or candidates
        if not pool:
            raise ValueError(
                f"No symmetric nonzero dev doses for model={model_name} layer={layer}."
            )
        selected = max(
            pool,
            key=lambda candidate: (
                candidate["dose_selectivity_score"],
                -candidate["magnitude"],
            ),
        )
        return {
            "selection_split": "val",
            "selection_uses_test_results": False,
            "criterion": (
                "maximize signed P(b|W,+m)-P(b|W,-m) minus the mean of "
                "neutral probability damage, neutral accuracy damage, and "
                "neutral final-prompt-only invalid-output rate over predeclared "
                "symmetric screen magnitudes; require the all-condition "
                "degeneration gate separately"
            ),
            "confirmatory_eligible": bool(eligible),
            "fallback_is_descriptive_only": not bool(eligible),
            "selected": selected,
            "all_candidates": candidates,
        }

    for model_name, model_frame in selection_frame.groupby("model_name"):
        candidates: list[Dict[str, Any]] = []
        for layer, layer_frame in model_frame.groupby("layer"):
            learned = layer_frame[
                layer_frame["direction_name"].eq("wn")
                & layer_frame["scale_convention"].eq("native")
            ]
            pressure = signed_pressure_score(
                learned[learned["condition"].eq("incorrect_suggestion")]
            )
            if pressure is None:
                continue
            neutral = learned[learned["condition"].eq("neutral")]
            neutral_zero = neutral[neutral["alpha"].eq(0.0)][
                ["stable_question_key", "p_correct", "is_correct"]
            ].drop_duplicates("stable_question_key")
            neutral_nonzero = neutral[~neutral["alpha"].eq(0.0)].merge(
                neutral_zero,
                on="stable_question_key",
                how="inner",
                suffixes=("", "_alpha_zero"),
                validate="many_to_one",
            )
            if neutral_nonzero.empty:
                continue
            neutral_probability_damage = float(
                (
                    neutral_nonzero["p_correct"]
                    - neutral_nonzero["p_correct_alpha_zero"]
                )
                .abs()
                .mean()
            )
            neutral_accuracy_damage = float(
                (
                    neutral_nonzero["is_correct"].astype(float)
                    - neutral_nonzero["is_correct_alpha_zero"].astype(float)
                )
                .abs()
                .mean()
            )
            learned_zero = learned[learned["alpha"].eq(0.0)][
                ["stable_question_key", "condition", "p_endorsed"]
            ].drop_duplicates(["stable_question_key", "condition"])
            positive_rows = learned[learned["alpha"].gt(0.0)].merge(
                learned_zero,
                on=["stable_question_key", "condition"],
                how="inner",
                suffixes=("", "_alpha_zero"),
                validate="many_to_one",
            )
            negative_rows = learned[learned["alpha"].lt(0.0)].merge(
                learned_zero,
                on=["stable_question_key", "condition"],
                how="inner",
                suffixes=("", "_alpha_zero"),
                validate="many_to_one",
            )
            wrong_positive_effect = float(
                (
                    positive_rows.loc[
                        positive_rows["condition"].eq("incorrect_suggestion"),
                        "p_endorsed",
                    ]
                    - positive_rows.loc[
                        positive_rows["condition"].eq("incorrect_suggestion"),
                        "p_endorsed_alpha_zero",
                    ]
                ).mean()
            )
            wrong_negative_effect = float(
                (
                    negative_rows.loc[
                        negative_rows["condition"].eq("incorrect_suggestion"),
                        "p_endorsed_alpha_zero",
                    ]
                    - negative_rows.loc[
                        negative_rows["condition"].eq("incorrect_suggestion"),
                        "p_endorsed",
                    ]
                ).mean()
            )
            if generation_frame.empty or "generation_steering_mode" not in generation_frame:
                layer_generation = pd.DataFrame()
            else:
                layer_generation = generation_frame[
                    generation_frame["model_name"].eq(model_name)
                    & generation_frame["layer"].eq(layer)
                    & generation_frame["direction_name"].eq("wn")
                    & generation_frame["scale_convention"].eq("native")
                    & generation_frame["generation_steering_mode"].eq(
                        "final_prompt_only"
                    )
                ]
            (
                neutral_invalid_rate,
                neutral_degeneration_rate,
                overall_degeneration_rate,
            ) = generation_failure_rates(layer_generation)
            neutral_invalid_component = (
                float(neutral_invalid_rate)
                if neutral_invalid_rate is not None
                else 1.0
            )
            neutral_damage = float(
                np.mean(
                    [
                        neutral_probability_damage,
                        neutral_accuracy_damage,
                        neutral_invalid_component,
                    ]
                )
            )
            control_scores: list[float] = []
            controls = layer_frame[
                layer_frame["treatment_type"].eq("control")
                & layer_frame["scale_convention"].eq("wn_norm_matched")
            ]
            for (_name, _seed), control_frame in controls.groupby(
                ["direction_name", "control_seed"],
                dropna=False,
            ):
                score = signed_pressure_score(
                    control_frame[
                        control_frame["condition"].eq("incorrect_suggestion")
                    ]
                )
                if score is not None:
                    control_scores.append(score)
            null_95 = (
                float(np.quantile(control_scores, 0.95))
                if control_scores
                else None
            )
            selectivity = pressure - neutral_damage
            candidates.append(
                {
                    "layer": int(layer),
                    "signed_pressure_score": pressure,
                    "wrong_positive_effect_vs_alpha_zero": wrong_positive_effect,
                    "wrong_negative_effect_vs_alpha_zero": wrong_negative_effect,
                    "neutral_probability_damage": neutral_probability_damage,
                    "neutral_accuracy_damage": neutral_accuracy_damage,
                    "neutral_damage_composite": neutral_damage,
                    "neutral_invalid_rate": neutral_invalid_rate,
                    "neutral_degeneration_rate": neutral_degeneration_rate,
                    "overall_degeneration_rate": overall_degeneration_rate,
                    "invalid_rate": neutral_invalid_rate,
                    "degeneration_rate": overall_degeneration_rate,
                    "degeneration_gate_observed": (
                        overall_degeneration_rate is not None
                    ),
                    "selectivity_score": selectivity,
                    "control_null_95": null_95,
                    "passes_direction_specificity": (
                        null_95 is not None and pressure > null_95
                    ),
                    "passes_neutral_accuracy_damage": (
                        neutral_accuracy_damage <= 0.02
                    ),
                    "passes_invalid_rate": (
                        neutral_invalid_rate is not None
                        and neutral_invalid_rate <= 0.01
                    ),
                    "passes_degeneration_gate": (
                        overall_degeneration_rate is not None
                        and overall_degeneration_rate <= 0.01
                    ),
                    "passes_bidirectionality": (
                        wrong_positive_effect > 0.0
                        and wrong_negative_effect > 0.0
                    ),
                    "passes_noop_gate": True,
                    "passes_nonfinite_gate": True,
                }
            )
        for candidate in candidates:
            candidate["confirmatory_eligible"] = bool(
                candidate["passes_direction_specificity"]
                and candidate["passes_neutral_accuracy_damage"]
                and candidate["passes_invalid_rate"]
                and candidate["passes_degeneration_gate"]
                and candidate["passes_bidirectionality"]
                and candidate["passes_noop_gate"]
                and candidate["passes_nonfinite_gate"]
            )
        eligible = [row for row in candidates if row["confirmatory_eligible"]]
        pool = eligible or candidates
        if not pool:
            continue
        selected = max(pool, key=lambda row: (row["selectivity_score"], -row["layer"]))
        selected_dose = select_symmetric_dose(
            str(model_name),
            int(selected["layer"]),
            model_frame,
        )
        available_layers = sorted(int(value) for value in model_frame["layer"].unique())
        neighbor_layers = [
            layer
            for layer in (
                selected["layer"] - 1,
                selected["layer"],
                selected["layer"] + 1,
            )
            if layer in available_layers
        ]
        selections.append(
            {
                "model_name": model_name,
                "selected_layer": selected["layer"],
                "test_layers": neighbor_layers,
                "selected_dose": selected_dose,
                "confirmatory_eligible": bool(eligible),
                "fallback_is_descriptive_only": not bool(eligible),
                "criterion": (
                    "maximize signed W pressure score minus the mean of neutral "
                    "probability damage, neutral accuracy damage, and neutral "
                    "invalid-output rate; "
                    "require learned score above control 95th percentile, positive "
                    "bidirectionality, neutral accuracy damage <=0.02, neutral "
                    "invalid-output and overall degeneration rates <=0.01, and "
                    "passed no-op/nonfinite gates"
                ),
                "selected_candidate": selected,
                "all_candidates": sorted(candidates, key=lambda row: row["layer"]),
            }
        )
    selection_path = target / "layer_selection.json"
    write_strict_json(
        selection_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "selection_split": "val",
            "selection_uses_test_results": False,
            "selection_was_run": not selection_frame.empty,
            "input_splits": sorted(str(value) for value in frame["split"].unique()),
            "selections": selections,
        },
    )
    plot_paths: Dict[str, Path] = {}
    if not frame.empty:
        plot_paths.update(plot_controlled_dose_response(frame, target / "plots"))
        plot_paths.update(plot_controlled_pareto(frame, target / "plots"))
    write_strict_json(
        target / "manifest.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "stage": "aggregate",
            "input_paths": [str(Path(path).resolve()) for path in input_paths],
            "input_hashes": {
                str(Path(path).resolve()): sha256_file(path) for path in input_paths
            },
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "aggregation_memory_policy": {
                "raw_wide_shards_preserved": True,
                "learned_rows_retained_at_question_level": True,
                "alpha_zero_control_rows_retained_for_replay": True,
                "nonzero_controls_compacted_to_seed_level_weighted_means": True,
                "compacted_control_intervals": (
                    "not_bootstrapped; null uncertainty is the declared "
                    "across-seed ribbon"
                ),
                "input_wide_rows": int(input_wide_rows),
                "retained_strict_rows": int(retained_strict_rows),
                "compacted_control_rows": int(compacted_control_rows),
                "retained_generation_rows": int(retained_generation_rows),
            },
            "cross_shard_replay": cross_shard_replay,
            "output_sha256": sha256_file(output_path),
            "layer_selection_sha256": sha256_file(selection_path),
            "plots": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in plot_paths.items()
            },
        },
    )
    return output_path


__all__ = [
    "aggregate_controlled_results",
    "fit_controlled_directions",
    "inspect_controlled_examples",
    "load_controlled_runtime",
    "run_controlled_geometry",
    "run_controlled_interventions",
    "run_alpaca_guardrail",
    "validate_controlled_sources",
]
