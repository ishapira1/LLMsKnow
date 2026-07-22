from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..llm.loading import load_model_and_tokenizer
from .activations import (
    extract_prompt_state,
    residual_layer_count,
    score_with_residual_additions,
    score_with_residual_replacements,
    top_choice,
)
from .data import (
    DEFAULT_PROBE_NAME,
    SourceBundle,
    build_intervention_pairs,
    filter_pairs,
    load_source_bundle,
    normalize_choice,
    selected_choice,
)
from .directions import (
    DirectionArtifact,
    fit_direction_arrays,
    load_direction_artifact,
    load_probe_vector,
    orthogonal_component,
    parallel_component,
    save_direction_artifact,
    unit_vector,
)
from .metrics import (
    bootstrap_difference_interval,
    expand_result_subsets,
    make_result_row,
    summarize_result_frame,
)
from .metrics import DEFAULT_SUMMARY_METRICS
from .plots import plot_selected_dose_response, plot_validation_layer_profile


EXPERIMENT_CONDITIONS = (
    "neutral",
    "incorrect_suggestion",
    "incorrect_suggestion_strong",
    "suggest_correct_strong",
)
PRIMARY_BIASED_CONDITION = "incorrect_suggestion_strong"
PRIMARY_NEUTRAL_CONDITION = "neutral"
PRIMARY_CORRECT_SUGGESTION_CONDITION = "suggest_correct_strong"
DEFAULT_ALPHAS = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class RuntimeBundle:
    model: Any
    tokenizer: Any
    device: str
    fingerprint: Dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_json(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_sanitize_json(dict(payload)), indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(target)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(_sanitize_json(dict(row)), default=_json_default, allow_nan=False) + "\n"
            )
    temporary.replace(target)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def resolve_device(requested: str, source: SourceBundle) -> str:
    value = str(requested or "auto").strip().lower()
    if value != "auto":
        return value
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def runtime_fingerprint(model: Any, tokenizer: Any, *, device: str) -> Dict[str, Any]:
    try:
        import torch

        torch_version = str(torch.__version__)
    except Exception:  # pragma: no cover - environment diagnostic
        torch_version = "unavailable"
    try:
        import transformers

        transformers_version = str(transformers.__version__)
    except Exception:  # pragma: no cover - environment diagnostic
        transformers_version = "unavailable"
    config = getattr(model, "config", None)
    init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "device": str(device),
        "model_class": model.__class__.__name__,
        "model_name_or_path": str(getattr(config, "_name_or_path", "") or ""),
        "model_commit_hash": str(getattr(config, "_commit_hash", "") or ""),
        "model_dtype": str(getattr(model, "dtype", "") or ""),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "") or ""),
        "tokenizer_commit_hash": str(init_kwargs.get("_commit_hash", "") or ""),
        "chat_template_sha256": _hash_text(chat_template),
        "checkpoint_revision_replay_note": (
            "The source run did not freeze a revision; baseline replay is the reproducibility gate."
        ),
    }


def load_runtime(
    source: SourceBundle,
    *,
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
    torch_dtype: Optional[str],
) -> RuntimeBundle:
    resolved_device = resolve_device(device, source)
    model, tokenizer = load_model_and_tokenizer(
        model_name=source.model_name,
        device=resolved_device,
        device_map_auto=bool(device_map_auto),
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    return RuntimeBundle(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        fingerprint=runtime_fingerprint(model, tokenizer, device=resolved_device),
    )


def _pair_stratum(pair: Mapping[str, Any]) -> str:
    return f"correct={pair['correct_choice']}|endorsed={pair['endorsed_choice']}"


def fit_restoration_directions(
    *,
    source_run_dir: Path,
    output_dir: Path,
    fit_split: str = "train",
    layers: Optional[Sequence[int]] = None,
    conditions: Sequence[str] = EXPERIMENT_CONDITIONS,
    max_questions: Optional[int] = None,
    seed: int = 5,
    probe_name: str = DEFAULT_PROBE_NAME,
    device: str = "auto",
    device_map_auto: bool = False,
    hf_cache_dir: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    progress_every: int = 50,
    n_control_directions: int = 20,
) -> DirectionArtifact:
    """Fit pre-answer neutral-minus-biased MeanDiff on the training split only."""

    if str(fit_split) != "train":
        raise ValueError("Confirmatory restoration directions must be fit on split='train'.")

    source = load_source_bundle(
        source_run_dir,
        probe_name=probe_name,
        record_conditions=conditions,
    )
    pairs, coverage = build_intervention_pairs(
        source.records,
        probe_scores=source.probe_scores,
        required_conditions=conditions,
        allowed_splits=[fit_split],
    )
    fit_pairs = filter_pairs(pairs, split=fit_split, max_questions=max_questions)
    if len(fit_pairs) < 2:
        raise ValueError(f"Need at least two paired {fit_split!r} questions; found {len(fit_pairs)}.")

    target_dir = Path(output_dir).expanduser().resolve()
    if (target_dir / "directions.npz").exists() or (target_dir / "manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing direction artifact in {target_dir}."
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(target_dir / "pair_coverage.csv", index=False)
    runtime = load_runtime(
        source,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    model_layers = residual_layer_count(runtime.model)
    layer_values = list(range(1, model_layers + 1)) if layers is None else sorted(
        {int(layer) for layer in layers}
    )
    invalid = [layer for layer in layer_values if layer < 1 or layer > model_layers]
    if invalid:
        raise ValueError(f"Invalid layers {invalid}; loaded model exposes layers 1..{model_layers}.")

    neutral_states: List[np.ndarray] = []
    biased_states: List[np.ndarray] = []
    for index, pair in enumerate(fit_pairs, start=1):
        choices = pair["choices"]
        neutral = pair["records"][PRIMARY_NEUTRAL_CONDITION]
        biased = pair["records"][PRIMARY_BIASED_CONDITION]
        neutral_state = extract_prompt_state(
            runtime.model,
            runtime.tokenizer,
            neutral["prompt_messages"],
            choices=choices,
            residual_layers=layer_values,
        )
        biased_state = extract_prompt_state(
            runtime.model,
            runtime.tokenizer,
            biased["prompt_messages"],
            choices=choices,
            residual_layers=layer_values,
        )
        neutral_states.append(
            np.stack([neutral_state.hidden_by_layer[layer] for layer in layer_values]).astype(
                np.float16
            )
        )
        biased_states.append(
            np.stack([biased_state.hidden_by_layer[layer] for layer in layer_values]).astype(
                np.float16
            )
        )
        if int(progress_every) > 0 and (index % int(progress_every) == 0 or index == len(fit_pairs)):
            print(
                f"[fit-directions] {index}/{len(fit_pairs)} pairs "
                f"model={source.model_name} dataset={source.dataset_name}",
                flush=True,
            )

    probe_vectors: Dict[int, np.ndarray] = {}
    probe_sources: Dict[str, str] = {}
    probe_failures: Dict[str, str] = {}
    for layer in layer_values:
        try:
            vector, _metadata, probe_dir = load_probe_vector(
                source.run_dir,
                layer=layer,
                probe_name=probe_name,
            )
            probe_vectors[layer] = vector
            probe_sources[str(layer)] = str(probe_dir)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            probe_failures[str(layer)] = str(exc)

    arrays, direction_metadata = fit_direction_arrays(
        np.stack(neutral_states, axis=0),
        np.stack(biased_states, axis=0),
        layers=layer_values,
        option_position_strata=[_pair_stratum(pair) for pair in fit_pairs],
        seed=int(seed),
        probe_vectors_by_layer=probe_vectors,
        n_control_directions=int(n_control_directions),
    )
    manifest = {
        **direction_metadata,
        "created_at": utc_now(),
        "stage": "fit_restoration_directions",
        "source_run_dir": str(source.run_dir),
        "source_run_config_path": str(source.run_config_path),
        "source_sampling_records_path": str(source.sampling_records_path),
        "source_sampling_records_sha256": sha256_file(source.sampling_records_path),
        "source_probe_scores_path": str(source.probe_scores_path),
        "source_probe_name": probe_name,
        "model_name": source.model_name,
        "dataset_name": source.dataset_name,
        "source_probe_feature_site": "last_token_of_teacher_forced_candidate_answer",
        "source_probe_direct_steering_status": "exploratory_cross_token_position_only",
        "fit_split": str(fit_split),
        "fit_question_ids_sha256": _hash_text(
            "\n".join(
                f"{pair['question_id']}::{pair['draw_idx']}" for pair in fit_pairs
            )
        ),
        "conditions": list(conditions),
        "max_questions": max_questions,
        "runtime": runtime.fingerprint,
        "probe_vector_sources": probe_sources,
        "probe_vector_load_failures": probe_failures,
    }
    return save_direction_artifact(target_dir, arrays=arrays, metadata=manifest)


def _stable_rng(*parts: Any) -> np.random.Generator:
    material = "||".join(str(part) for part in parts)
    seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(seed)


def _wrong_question_donor_indices(pairs: Sequence[Mapping[str, Any]]) -> List[int]:
    groups: Dict[tuple[str, str, int], List[int]] = {}
    for index, pair in enumerate(pairs):
        key = (
            str(pair["correct_choice"]),
            str(pair["endorsed_choice"]),
            len(pair["choices"]),
        )
        groups.setdefault(key, []).append(index)
    donors = [-1] * len(pairs)
    for indices in groups.values():
        if len(indices) > 1:
            for offset, index in enumerate(indices):
                donors[index] = indices[(offset + 1) % len(indices)]
    for index, donor in enumerate(donors):
        if donor >= 0:
            continue
        for candidate in range(len(pairs)):
            if candidate != index and len(pairs[candidate]["choices"]) == len(pairs[index]["choices"]):
                donors[index] = candidate
                break
    return donors


def _pair_metadata(pair: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "split": str(pair["split"]),
        "question_id": str(pair["question_id"]),
        "draw_idx": int(pair["draw_idx"]),
        "neutral_correct": bool(pair.get("neutral_correct")),
        "sycophantic_flip": bool(pair.get("sycophantic_flip")),
        "hidden_truth_flip": bool(pair.get("hidden_truth_flip")),
        "probe_follows_user": bool(pair.get("probe_follows_user")),
        "probe_other": bool(pair.get("probe_other")),
        "sycophantic_flip_probe_user": bool(pair.get("sycophantic_flip_probe_user")),
        "sycophantic_flip_probe_other": bool(pair.get("sycophantic_flip_probe_other")),
        "neutral_wrong_to_correct_suggestion_correct": bool(
            pair.get("neutral_wrong_to_correct_suggestion_correct")
        ),
        "high_confidence_neutral_correct": bool(pair.get("high_confidence_neutral_correct")),
        "baseline_replay_matched": bool(pair.get("baseline_replay_matched", False)),
        "probe_argmax_choice": str(pair.get("probe_argmax_choice", "") or ""),
        "probe_score_gap_correct_minus_selected": pair.get(
            "probe_score_gap_correct_minus_selected", float("nan")
        ),
        "neutral_p_correct_saved": pair.get("neutral_p_correct_saved", float("nan")),
    }


def _extract_layer_baselines(
    runtime: RuntimeBundle,
    pairs: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    conditions: Sequence[str],
    progress_every: int,
) -> tuple[
    Dict[str, np.ndarray],
    Dict[str, List[Dict[str, float]]],
    Dict[str, List[Dict[str, float]]],
    pd.DataFrame,
]:
    state_lists: Dict[str, List[np.ndarray]] = {condition: [] for condition in conditions}
    probabilities: Dict[str, List[Dict[str, float]]] = {condition: [] for condition in conditions}
    log_scores: Dict[str, List[Dict[str, float]]] = {condition: [] for condition in conditions}
    replay_rows: List[Dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs, start=1):
        for condition in conditions:
            record = pair["records"][condition]
            state = extract_prompt_state(
                runtime.model,
                runtime.tokenizer,
                record["prompt_messages"],
                choices=pair["choices"],
                residual_layers=[layer],
            )
            state_lists[condition].append(state.hidden_by_layer[layer])
            probabilities[condition].append(state.choice_probabilities)
            log_scores[condition].append(state.choice_log_scores)
            recomputed = top_choice(state.choice_probabilities)
            saved_probabilities = dict(record.get("choice_probabilities", {}) or {})
            saved_probability_choice = top_choice(
                {
                    choice: float(saved_probabilities.get(choice, 0.0))
                    for choice in pair["choices"]
                }
            )
            saved_response_choice = selected_choice(record)
            max_abs_error = max(
                (
                    abs(float(state.choice_probabilities.get(choice, 0.0)) - float(saved_probabilities.get(choice, 0.0)))
                    for choice in pair["choices"]
                ),
                default=float("nan"),
            )
            replay_rows.append(
                {
                    "split": pair["split"],
                    "question_id": pair["question_id"],
                    "draw_idx": pair["draw_idx"],
                    "condition": condition,
                    "saved_probability_choice": saved_probability_choice,
                    "saved_response_choice": saved_response_choice,
                    "recomputed_choice": recomputed,
                    "top_choice_match": bool(saved_probability_choice == recomputed),
                    "response_choice_match": bool(saved_response_choice == recomputed),
                    "max_abs_probability_error": max_abs_error,
                    "prompt_token_count": state.prompt_token_count,
                }
            )
        if int(progress_every) > 0 and (
            pair_index % int(progress_every) == 0 or pair_index == len(pairs)
        ):
            print(f"[run-layer/baseline] {pair_index}/{len(pairs)} pairs layer={layer}", flush=True)
    states = {
        condition: np.stack(values, axis=0).astype(np.float32)
        for condition, values in state_lists.items()
    }
    return states, probabilities, log_scores, pd.DataFrame(replay_rows)


def _random_matched_delta(
    reference_delta: np.ndarray,
    *,
    layer: int,
    question_id: str,
    seed: int,
) -> np.ndarray:
    rng = _stable_rng("matched-random-patch", layer, question_id, seed)
    direction = unit_vector(rng.normal(size=reference_delta.shape[0]), name="matched random patch")
    return np.asarray(direction * float(np.linalg.norm(reference_delta)), dtype=np.float32)


def run_intervention_layer(
    *,
    source_run_dir: Path,
    directions_path: Path,
    output_root: Path,
    layer: int,
    split: str,
    conditions: Sequence[str] = EXPERIMENT_CONDITIONS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    max_questions: Optional[int] = None,
    random_control_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    probe_name: str = DEFAULT_PROBE_NAME,
    include_transported_probe: bool = True,
    min_baseline_replay_agreement: float = 0.98,
    max_baseline_probability_p99_error: float = 0.01,
    max_batch_size: int = 16,
    device: str = "auto",
    device_map_auto: bool = False,
    hf_cache_dir: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    protocol: str = "patch-localize",
    selection_path: Optional[Path] = None,
    progress_every: int = 25,
) -> Path:
    """Run paired patches, MeanDiff steering, and controls at one pre-answer layer."""

    protocol_value = str(protocol).strip().replace("_", "-")
    allowed_protocols = {"patch-localize", "dose-tune", "confirm"}
    if protocol_value not in allowed_protocols:
        raise ValueError(
            f"Unknown protocol={protocol!r}; expected one of {sorted(allowed_protocols)}."
        )
    expected_split = "test" if protocol_value == "confirm" else "val"
    if str(split) != expected_split:
        raise ValueError(
            f"Protocol {protocol_value!r} is restricted to split={expected_split!r}; got {split!r}."
        )
    direction_artifact = load_direction_artifact(directions_path)
    if protocol_value == "confirm" and direction_artifact.metadata.get("max_questions") is not None:
        raise ValueError(
            "Held-out confirmation is blocked because the MeanDiff artifact was fit as a capped pilot."
        )
    layer_value = int(layer)
    direction_artifact.layer_index(layer_value)
    source = load_source_bundle(
        source_run_dir,
        probe_name=probe_name,
        record_conditions=conditions,
    )
    direction_source = Path(
        str(direction_artifact.metadata.get("source_run_dir", "") or "")
    ).expanduser()
    if not direction_source or direction_source.resolve() != source.run_dir:
        raise ValueError(
            "Direction/source mismatch: directions were fit from "
            f"{direction_source}, but this shard requested {source.run_dir}."
        )
    if str(direction_artifact.metadata.get("model_name", "")) != source.model_name:
        raise ValueError("Direction manifest model_name does not match the source run.")
    if str(direction_artifact.metadata.get("dataset_name", "")) != source.dataset_name:
        raise ValueError("Direction manifest dataset_name does not match the source run.")
    if list(direction_artifact.metadata.get("conditions", [])) != list(conditions):
        raise ValueError("Direction manifest conditions do not match the requested conditions.")
    frozen_selection: Optional[Dict[str, Any]] = None
    if protocol_value == "confirm":
        if selection_path is None:
            raise ValueError("Confirmatory test execution requires selection_path.")
        frozen_selection_path = Path(selection_path).expanduser().resolve()
        frozen_selection = json.loads(frozen_selection_path.read_text(encoding="utf-8"))
        if not bool(frozen_selection.get("test_confirmation_allowed", False)):
            raise ValueError("Frozen selection is marked as a pilot and cannot unlock test confirmation.")
        expected_fields = {
            "source_run_dir": str(source.run_dir),
            "directions_manifest_sha256": sha256_file(direction_artifact.metadata_path),
            "directions_npz_sha256": sha256_file(direction_artifact.path),
            "model_name": source.model_name,
            "dataset_name": source.dataset_name,
        }
        for field, expected_value in expected_fields.items():
            if frozen_selection.get(field) != expected_value:
                raise ValueError(f"Frozen selection mismatch for field {field!r}.")
        if int(frozen_selection["selected_layer"]) != layer_value:
            raise ValueError("Frozen selection layer does not match requested confirmation layer.")
        if [float(value) for value in frozen_selection["test_alphas"]] != [
            float(value) for value in alphas
        ]:
            raise ValueError("Confirmation alphas do not exactly match the frozen selection.")
    pairs, coverage = build_intervention_pairs(
        source.records,
        probe_scores=source.probe_scores,
        required_conditions=conditions,
        allowed_splits=[split],
    )
    selected_pairs = filter_pairs(pairs, split=split, max_questions=max_questions)
    if not selected_pairs:
        raise ValueError(f"No usable paired questions for split={split!r}.")

    shard_dir = (
        Path(output_root).expanduser().resolve()
        / "layers"
        / f"layer_{layer_value:03d}"
        / str(split)
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    protocol_slug = protocol_value.replace("-", "_")
    item_path = shard_dir / f"item_results_{protocol_slug}.jsonl"
    manifest_path = shard_dir / f"manifest_{protocol_slug}.json"
    if item_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing intervention shard. Use a new RUN_ID/output root "
            f"or remove the exact incomplete shard after inspection: {item_path}"
        )
    coverage.to_csv(shard_dir / "pair_coverage.csv", index=False)
    runtime = load_runtime(
        source,
        device=device,
        device_map_auto=device_map_auto,
        hf_cache_dir=hf_cache_dir,
        torch_dtype=torch_dtype,
    )
    model_layers = residual_layer_count(runtime.model)
    if layer_value > model_layers:
        raise ValueError(f"Requested layer={layer_value}, loaded model has only {model_layers} layers.")

    states, baseline_probabilities, baseline_log_scores, replay = _extract_layer_baselines(
        runtime,
        selected_pairs,
        layer=layer_value,
        conditions=conditions,
        progress_every=progress_every,
    )
    replay_path = shard_dir / f"baseline_replay_{protocol_slug}.csv"
    replay.to_csv(replay_path, index=False)
    condition_replay = {
        str(condition): {
            "n": int(len(group)),
            "top_choice_agreement": float(group["top_choice_match"].mean()),
            "probability_error_p99": float(group["max_abs_probability_error"].quantile(0.99)),
        }
        for condition, group in replay.groupby("condition")
    }
    replay_agreement = min(
        values["top_choice_agreement"] for values in condition_replay.values()
    )
    replay_probability_p99 = max(
        values["probability_error_p99"] for values in condition_replay.values()
    )
    for pair in selected_pairs:
        pair_rows = replay[
            (replay["question_id"].astype(str) == str(pair["question_id"]))
            & (replay["draw_idx"].astype(int) == int(pair["draw_idx"]))
        ]
        pair["baseline_replay_matched"] = bool(
            len(pair_rows) == len(conditions)
            and pair_rows["top_choice_match"].all()
            and (pair_rows["max_abs_probability_error"] <= float(max_baseline_probability_p99_error)).all()
        )
    replay_report = {
        "n_rows": int(len(replay)),
        "condition_results": condition_replay,
        "minimum_condition_top_choice_agreement": replay_agreement,
        "maximum_condition_probability_error_p99": replay_probability_p99,
        "median_max_abs_probability_error": float(replay["max_abs_probability_error"].median()),
        "max_abs_probability_error": float(replay["max_abs_probability_error"].max()),
        "required_primary_top_choice_agreement": float(min_baseline_replay_agreement),
        "required_maximum_probability_error_p99": float(max_baseline_probability_p99_error),
        "passed": bool(
            replay_agreement >= float(min_baseline_replay_agreement)
            and replay_probability_p99 <= float(max_baseline_probability_p99_error)
        ),
    }
    write_json(shard_dir / f"baseline_replay_summary_{protocol_slug}.json", replay_report)
    if not replay_report["passed"]:
        raise RuntimeError(
            "Baseline replay failed before intervention: "
            f"minimum condition agreement={replay_agreement:.3f}, "
            f"maximum condition p99 probability error={replay_probability_p99:.4g}. "
            f"See {replay_path}."
        )

    restoration_unit = direction_artifact.vector("restoration_unit", layer_value)
    restoration_scale = direction_artifact.scalar("restoration_scale", layer_value)
    available_control_seeds = set(
        np.asarray(direction_artifact.arrays["control_seeds"], dtype=int).tolist()
    )
    missing_control_seeds = sorted(
        set(int(value) for value in random_control_seeds) - available_control_seeds
    )
    if missing_control_seeds:
        raise ValueError(
            f"Requested unavailable steering control seeds {missing_control_seeds}; "
            f"artifact provides {sorted(available_control_seeds)}."
        )
    control_directions = {
        int(control_seed): {
            "null_unit": direction_artifact.control_vector(
                "null_unit", layer_value, int(control_seed)
            ),
            "null_native_scale": direction_artifact.control_scalar(
                "null_scale", layer_value, int(control_seed)
            ),
            "random_unit": direction_artifact.control_vector(
                "random_unit", layer_value, int(control_seed)
            ),
            "random_native_scale": direction_artifact.control_scalar(
                "random_scale", layer_value, int(control_seed)
            ),
        }
        for control_seed in random_control_seeds
    }
    probe_available = bool(direction_artifact.scalar("probe_available", layer_value))
    probe_unit = (
        direction_artifact.vector("probe_unit", layer_value) if probe_available else None
    )
    probe_scale = (
        direction_artifact.scalar("probe_scale", layer_value) if probe_available else float("nan")
    )
    donor_indices = _wrong_question_donor_indices(selected_pairs)
    result_rows: List[Dict[str, Any]] = []

    def score_specs(
        *,
        pair_index: int,
        condition: str,
        specs: Sequence[tuple[Dict[str, Any], np.ndarray]],
    ) -> None:
        pair = selected_pairs[pair_index]
        record = pair["records"][condition]
        for operation in ("add", "replace"):
            operation_specs = [
                spec for spec in specs if str(spec[0].get("intervention_operation", "add")) == operation
            ]
            if not operation_specs:
                continue
            vectors = np.stack(
                [np.asarray(vector, dtype=np.float32) for _, vector in operation_specs], axis=0
            )
            scorer = (
                score_with_residual_additions
                if operation == "add"
                else score_with_residual_replacements
            )
            vector_argument = (
                {"addition_vectors": vectors}
                if operation == "add"
                else {"replacement_vectors": vectors}
            )
            scored, scored_log_scores = scorer(
                runtime.model,
                runtime.tokenizer,
                record["prompt_messages"],
                choices=pair["choices"],
                residual_layer=layer_value,
                max_batch_size=max_batch_size,
                **vector_argument,
            )
            for (spec_metadata, _vector), probabilities, log_scores in zip(
                operation_specs, scored, scored_log_scores
            ):
                metadata = {
                    **_pair_metadata(pair),
                    **spec_metadata,
                    "model_name": source.model_name,
                    "dataset_name": source.dataset_name,
                    "layer": layer_value,
                    "model_layer_count": int(model_layers),
                    "is_terminal_layer": bool(layer_value == model_layers),
                    "condition": condition,
                    "protocol": protocol_value,
                }
                result_rows.append(
                    make_result_row(
                        probabilities=probabilities,
                        baseline_probabilities=baseline_probabilities[condition][pair_index],
                        log_scores=log_scores,
                        baseline_log_scores=baseline_log_scores[condition][pair_index],
                        neutral_baseline_probabilities=baseline_probabilities[
                            PRIMARY_NEUTRAL_CONDITION
                        ][pair_index],
                        neutral_baseline_log_scores=baseline_log_scores[
                            PRIMARY_NEUTRAL_CONDITION
                        ][pair_index],
                        correct_choice=pair["correct_choice"],
                        endorsed_choice=pair["endorsed_choice"],
                        condition_suggested_choice=normalize_choice(
                            record.get("suggested_label")
                        ),
                        metadata=metadata,
                    )
                )

    for pair_index, pair in enumerate(selected_pairs):
        neutral_state = states[PRIMARY_NEUTRAL_CONDITION][pair_index]
        biased_state = states[PRIMARY_BIASED_CONDITION][pair_index]
        paired_delta = neutral_state - biased_state
        zero = np.zeros_like(paired_delta)
        for condition in conditions:
            specs: List[tuple[Dict[str, Any], np.ndarray]] = [
                (
                    {
                        "intervention": "no_op",
                        "direction_family": "none",
                        "alpha": 0.0,
                        "control_seed": None,
                        "confirmatory_status": "control",
                        "intervention_operation": "add",
                    },
                    zero,
                )
            ]
            if protocol_value in {"dose-tune", "confirm"}:
                for alpha in alphas:
                    specs.append(
                        (
                            {
                                "intervention": "steer_restoration_meandiff",
                                "direction_family": "preanswer_restoration_meandiff",
                                "alpha": float(alpha),
                                "control_seed": None,
                                "confirmatory_status": "confirmatory",
                                "intervention_operation": "add",
                            },
                            float(alpha) * restoration_scale * restoration_unit,
                        )
                    )
                    for control_seed in random_control_seeds:
                        control = control_directions[int(control_seed)]
                        specs.extend(
                            [
                                (
                                    {
                                        "intervention": "steer_rademacher_null",
                                        "direction_family": "train_pair_label_sign_null",
                                        "alpha": float(alpha),
                                        "control_seed": int(control_seed),
                                        "confirmatory_status": "null_control",
                                        "intervention_operation": "add",
                                        "native_projection_sd": control["null_native_scale"],
                                        "applied_scale": restoration_scale,
                                    },
                                    float(alpha) * restoration_scale * control["null_unit"],
                                ),
                                (
                                    {
                                        "intervention": "steer_random_direction",
                                        "direction_family": "treatment_norm_matched_random",
                                        "alpha": float(alpha),
                                        "control_seed": int(control_seed),
                                        "confirmatory_status": "random_control",
                                        "intervention_operation": "add",
                                        "native_projection_sd": control["random_native_scale"],
                                        "applied_scale": restoration_scale,
                                    },
                                    float(alpha) * restoration_scale * control["random_unit"],
                                ),
                            ]
                        )
                    if include_transported_probe and probe_available and probe_unit is not None:
                        specs.append(
                            (
                                {
                                    "intervention": "steer_transported_random_all_probe",
                                    "direction_family": "random_all_probe_cross_token",
                                    "alpha": float(alpha),
                                    "control_seed": None,
                                    "confirmatory_status": "exploratory_cross_token",
                                    "intervention_operation": "add",
                                },
                                float(alpha) * probe_scale * probe_unit,
                            )
                        )

            if (
                protocol_value in {"patch-localize", "confirm"}
                and condition == PRIMARY_BIASED_CONDITION
            ):
                specs.extend(
                    [
                        (
                            {
                                "intervention": "patch_paired_full",
                                "direction_family": "item_neutral_minus_biased",
                                "alpha": 1.0,
                                "control_seed": None,
                                "confirmatory_status": "confirmatory",
                                "intervention_operation": "replace",
                            },
                            neutral_state,
                        ),
                        (
                            {
                                "intervention": "patch_paired_reverse_sign",
                                "direction_family": "item_biased_minus_neutral",
                                "alpha": -1.0,
                                "control_seed": None,
                                "confirmatory_status": "sign_control",
                                "intervention_operation": "add",
                            },
                            -paired_delta,
                        ),
                    ]
                )
                donor_index = donor_indices[pair_index]
                if donor_index >= 0:
                    specs.append(
                        (
                            {
                                "intervention": "patch_wrong_question",
                                "direction_family": "other_item_neutral_minus_current_biased",
                                "alpha": 1.0,
                                "control_seed": None,
                                "donor_question_id": selected_pairs[donor_index]["question_id"],
                                "confirmatory_status": "donor_control",
                                "intervention_operation": "replace",
                            },
                            states[PRIMARY_NEUTRAL_CONDITION][donor_index],
                        )
                    )
                for control_seed in random_control_seeds:
                    specs.append(
                        (
                            {
                                "intervention": "patch_random_matched",
                                "direction_family": "isotropic_item_norm_matched",
                                "alpha": 1.0,
                                "control_seed": int(control_seed),
                                "confirmatory_status": "random_control",
                                "intervention_operation": "add",
                            },
                            _random_matched_delta(
                                paired_delta,
                                layer=layer_value,
                                question_id=str(pair["question_id"]),
                                seed=int(control_seed),
                            ),
                        )
                    )
                if probe_available and probe_unit is not None:
                    specs.extend(
                        [
                            (
                                {
                                    "intervention": "patch_transported_probe_parallel",
                                    "direction_family": "random_all_probe_cross_token_parallel",
                                    "alpha": 1.0,
                                    "control_seed": None,
                                    "confirmatory_status": "exploratory_cross_token",
                                    "intervention_operation": "add",
                                },
                                parallel_component(paired_delta, probe_unit),
                            ),
                            (
                                {
                                    "intervention": "patch_transported_probe_orthogonal",
                                    "direction_family": "random_all_probe_cross_token_orthogonal",
                                    "alpha": 1.0,
                                    "control_seed": None,
                                    "confirmatory_status": "exploratory_cross_token",
                                    "intervention_operation": "add",
                                },
                                orthogonal_component(paired_delta, probe_unit),
                            ),
                        ]
                    )
            elif (
                protocol_value in {"patch-localize", "confirm"}
                and condition == PRIMARY_NEUTRAL_CONDITION
            ):
                specs.append(
                    (
                        {
                            "intervention": "patch_reverse_full",
                            "direction_family": "item_biased_minus_neutral",
                            "alpha": 1.0,
                            "control_seed": None,
                            "confirmatory_status": "confirmatory_bidirectional",
                            "intervention_operation": "replace",
                        },
                        biased_state,
                    )
                )
            score_specs(pair_index=pair_index, condition=condition, specs=specs)

        completed = pair_index + 1
        if int(progress_every) > 0 and (
            completed % int(progress_every) == 0 or completed == len(selected_pairs)
        ):
            print(
                f"[run-layer/intervene] {completed}/{len(selected_pairs)} pairs "
                f"layer={layer_value} rows={len(result_rows)}",
                flush=True,
            )

    no_op = [row for row in result_rows if row["intervention"] == "no_op"]
    no_op_max_error = max(
        (float(row["total_variation_from_baseline"]) for row in no_op),
        default=float("nan"),
    )
    for row in result_rows:
        row.pop("probabilities", None)
        row.pop("baseline_probabilities", None)
    write_jsonl(item_path, result_rows)
    manifest = {
        "created_at": utc_now(),
        "stage": "run_intervention_layer",
        "protocol": protocol_value,
        "source_run_dir": str(source.run_dir),
        "directions_path": str(direction_artifact.path),
        "directions_manifest_sha256": sha256_file(direction_artifact.metadata_path),
        "directions_npz_sha256": sha256_file(direction_artifact.path),
        "selection_path": str(Path(selection_path).resolve()) if selection_path else None,
        "selection_sha256": sha256_file(Path(selection_path)) if selection_path else None,
        "output_item_results": str(item_path),
        "model_name": source.model_name,
        "dataset_name": source.dataset_name,
        "split": str(split),
        "layer": layer_value,
        "model_layer_count": int(model_layers),
        "conditions": list(conditions),
        "alphas": [float(alpha) for alpha in alphas],
        "random_control_seeds": [int(seed) for seed in random_control_seeds],
        "n_pairs": int(len(selected_pairs)),
        "max_questions": max_questions,
        "n_result_rows": int(len(result_rows)),
        "include_transported_probe": bool(include_transported_probe),
        "transported_probe_interpretation": "exploratory_cross_token_position_not_confirmatory",
        "baseline_replay": replay_report,
        "no_op_max_abs_probability_error": float(no_op_max_error),
        "runtime": runtime.fingerprint,
        "argv": list(sys.argv),
    }
    write_json(manifest_path, manifest)
    return item_path


def _result_paths(
    output_root: Path,
    *,
    split: Optional[str] = None,
    protocol: Optional[str] = None,
) -> List[Path]:
    protocol_slug = str(protocol).replace("-", "_") if protocol else "*"
    split_pattern = str(split) if split else "*"
    pattern = f"layers/layer_*/{split_pattern}/item_results_{protocol_slug}.jsonl"
    return sorted(Path(output_root).glob(pattern))


def _manifest_for_item_path(path: Path) -> Path:
    suffix = path.name.removeprefix("item_results_").removesuffix(".jsonl")
    return path.with_name(f"manifest_{suffix}.json")


def _validate_result_manifests(
    paths: Sequence[Path],
    *,
    expected_protocol: Optional[str] = None,
) -> Dict[str, Any]:
    if not paths:
        raise FileNotFoundError("No intervention result shards were found.")
    manifests = []
    for path in paths:
        manifest_path = _manifest_for_item_path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing shard manifest for {path}: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_protocol and str(payload.get("protocol")) != str(expected_protocol):
            raise ValueError(
                f"Shard protocol mismatch in {manifest_path}: {payload.get('protocol')!r}."
            )
        manifests.append(payload)
    coherence_fields = (
        "source_run_dir",
        "directions_manifest_sha256",
        "directions_npz_sha256",
        "model_name",
        "dataset_name",
        "conditions",
        "max_questions",
        "model_layer_count",
    )
    reference = manifests[0]
    for payload in manifests[1:]:
        for field in coherence_fields:
            if payload.get(field) != reference.get(field):
                raise ValueError(f"Mixed shard manifests: field {field!r} is not coherent.")
    layers = [int(payload["layer"]) for payload in manifests]
    if len(layers) != len(set(layers)):
        raise ValueError(f"Duplicate layer shards detected: {layers}")
    return {**reference, "validated_layers": sorted(layers), "n_validated_shards": len(paths)}


def _read_result_tree(
    output_root: Path,
    *,
    split: Optional[str] = None,
    protocol: Optional[str] = None,
    interventions: Optional[Sequence[str]] = None,
    layers: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    paths = _result_paths(output_root, split=split, protocol=protocol)
    allowed_interventions = set(interventions) if interventions else None
    allowed_layers = set(int(layer) for layer in layers) if layers else None
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if allowed_interventions is not None and row.get("intervention") not in allowed_interventions:
                    continue
                if allowed_layers is not None and int(row.get("layer", -1)) not in allowed_layers:
                    continue
                rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"No matching item results below {output_root} for split={split!r}, "
            f"protocol={protocol!r}."
        )
    return pd.DataFrame(rows)


def select_validation_layers(
    *,
    output_root: Path,
    split: str = "val",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Choose a small causal layer window using validation patches only."""

    if str(split) != "val":
        raise ValueError("Confirmatory layer selection is restricted to split='val'.")
    root = Path(output_root).expanduser().resolve()
    if (root / "selected_intervention.json").exists() or _result_paths(
        root, split="test", protocol="confirm"
    ):
        raise FileExistsError("Refusing to reselect layers after a test selection/run exists.")
    paths = _result_paths(root, split=split, protocol="patch-localize")
    coherence = _validate_result_manifests(paths, expected_protocol="patch-localize")
    model_layer_count = int(coherence.get("model_layer_count", max(coherence["validated_layers"]) + 1))
    expected_layers = list(range(1, model_layer_count))
    if coherence["validated_layers"] != expected_layers:
        raise ValueError(
            "Patch-localization layer set is incomplete: "
            f"expected={expected_layers}, found={coherence['validated_layers']}."
        )
    frame = _read_result_tree(
        root,
        split=split,
        protocol="patch-localize",
        interventions=(
            "patch_paired_full",
            "patch_reverse_full",
            "patch_random_matched",
            "patch_wrong_question",
        ),
    )
    selection_population = (
        frame["neutral_correct"].astype(bool)
        & ~frame["hidden_truth_flip"].astype(bool)
    )
    full_patch = frame[
        (frame["intervention"] == "patch_paired_full")
        & (frame["condition"] == PRIMARY_BIASED_CONDITION)
        & selection_population
    ]
    reverse_patch = frame[
        (frame["intervention"] == "patch_reverse_full")
        & (frame["condition"] == PRIMARY_NEUTRAL_CONDITION)
        & selection_population
    ]
    if full_patch.empty or reverse_patch.empty:
        raise ValueError("Validation results are missing paired full or reverse patches.")
    forward = full_patch.groupby("layer")["delta_margin"].mean()
    reverse = reverse_patch.groupby("layer")["delta_margin"].mean()
    random_patch = frame[
        (frame["intervention"] == "patch_random_matched")
        & (frame["condition"] == PRIMARY_BIASED_CONDITION)
        & selection_population
    ]
    random_by_item = random_patch.groupby(
        ["layer", "question_id", "draw_idx"], as_index=False
    )["delta_margin"].mean()
    random_mean = random_by_item.groupby("layer")["delta_margin"].mean()
    wrong_mean = frame[
        (frame["intervention"] == "patch_wrong_question")
        & (frame["condition"] == PRIMARY_BIASED_CONDITION)
        & selection_population
    ].groupby("layer")["delta_margin"].mean()
    layer_table = pd.DataFrame(
        {
            "forward_delta_margin": forward,
            "reverse_delta_margin": reverse,
            "matched_random_delta_margin": random_mean,
            "wrong_question_delta_margin": wrong_mean,
        }
    )
    layer_table["bidirectional_objective"] = (
        layer_table["forward_delta_margin"] - layer_table["reverse_delta_margin"]
    ) / 2.0
    layer_table = layer_table.dropna().reset_index().sort_values(
        ["bidirectional_objective", "layer"], ascending=[False, True]
    )
    layer_table["forward_minus_matched_random"] = (
        layer_table["forward_delta_margin"] - layer_table["matched_random_delta_margin"]
    )
    layer_table["forward_minus_wrong_question"] = (
        layer_table["forward_delta_margin"] - layer_table["wrong_question_delta_margin"]
    )
    layer_table["eligible"] = (
        (layer_table["bidirectional_objective"] > 0.0)
        & (layer_table["forward_minus_matched_random"] > 0.0)
        & (layer_table["forward_minus_wrong_question"] > 0.0)
    )
    if layer_table.empty:
        raise ValueError("No layer has both forward and reverse validation estimates.")
    eligible_layers = layer_table[layer_table["eligible"]].head(int(top_k))
    layer_table.to_csv(root / "validation_layer_selection.csv", index=False)
    if eligible_layers.empty:
        no_go = {
            "created_at": utc_now(),
            "stage": "layer_selection",
            "reason": "no_layer_beats_bidirectional_and_patch_controls",
            "frozen_test_blocked": True,
        }
        write_json(root / "intervention_no_go.json", no_go)
        raise RuntimeError("No validation layer passed the prespecified patch-control checks.")
    candidate_layers = [int(value) for value in eligible_layers["layer"].tolist()]
    selection = {
        "created_at": utc_now(),
        "selection_split": str(split),
        "selection_subset": "neutral_correct (never hidden_truth_flip)",
        "layer_objective": "mean(forward_patch_delta_margin - reverse_patch_delta_margin) / 2",
        "candidate_layers": candidate_layers,
        "top_k": int(top_k),
        "source_run_dir": coherence["source_run_dir"],
        "directions_manifest_sha256": coherence["directions_manifest_sha256"],
        "directions_npz_sha256": coherence["directions_npz_sha256"],
        "model_name": coherence["model_name"],
        "dataset_name": coherence["dataset_name"],
        "pilot": coherence.get("max_questions") is not None,
        "frozen_before_dose_tuning": True,
    }
    write_json(root / "selected_patch_layers.json", selection)
    return selection


def _steering_condition_did(frame: pd.DataFrame, intervention: str) -> pd.DataFrame:
    working = frame[frame["intervention"] == intervention].copy()
    metrics = (
        "delta_margin",
        "accuracy_change",
        "delta_p_correct",
        "delta_p_condition_suggested",
        "condition_suggestion_agreement_change",
        "baseline_margin_correct_minus_endorsed",
    )
    keys = ["layer", "alpha", "question_id", "draw_idx", "condition"]
    averaged = working.groupby(keys, dropna=False, as_index=False)[list(metrics)].mean()
    flag_columns = [
        "neutral_correct",
        "sycophantic_flip",
        "hidden_truth_flip",
        "probe_follows_user",
        "probe_other",
        "sycophantic_flip_probe_user",
        "sycophantic_flip_probe_other",
        "high_confidence_neutral_correct",
        "neutral_wrong_to_correct_suggestion_correct",
        "baseline_replay_matched",
        "probe_score_gap_correct_minus_selected",
        "neutral_p_correct_saved",
    ]
    flags = working.groupby(keys, dropna=False, as_index=False)[flag_columns].first()
    averaged = averaged.merge(flags, on=keys, validate="one_to_one")
    item_keys = ["layer", "alpha", "question_id", "draw_idx"]

    def condition_frame(condition: str, prefix: str) -> pd.DataFrame:
        subset = averaged[averaged["condition"] == condition].copy()
        rename = {metric: f"{prefix}_{metric}" for metric in metrics}
        return subset[item_keys + list(metrics) + flag_columns].rename(columns=rename)

    biased = condition_frame(PRIMARY_BIASED_CONDITION, "biased")
    neutral = condition_frame(PRIMARY_NEUTRAL_CONDITION, "neutral")
    correct = condition_frame(PRIMARY_CORRECT_SUGGESTION_CONDITION, "correct_suggestion")
    merged = biased.merge(
        neutral.drop(columns=flag_columns),
        on=item_keys,
        validate="one_to_one",
    ).merge(
        correct.drop(columns=flag_columns),
        on=item_keys,
        validate="one_to_one",
    )
    merged["mitigation_did"] = merged["biased_delta_margin"] - merged["neutral_delta_margin"]
    merged["genuine_agreement_relative_margin"] = (
        merged["correct_suggestion_delta_margin"] - merged["neutral_delta_margin"]
    )
    return merged


def select_validation_dose(
    *,
    output_root: Path,
    split: str = "val",
    max_neutral_accuracy_cost: float = 0.02,
    max_correct_suggestion_accuracy_cost: float = 0.02,
    max_correct_suggestion_probability_cost: float = 0.02,
    max_genuine_agreement_relative_margin_cost: float = 0.10,
    min_dose_response_spearman: float = 0.70,
) -> Dict[str, Any]:
    """Freeze layer and dose using a wrong-pressure-minus-neutral DiD."""

    if str(split) != "val":
        raise ValueError("Confirmatory dose selection is restricted to split='val'.")
    root = Path(output_root).expanduser().resolve()
    layer_selection_path = root / "selected_patch_layers.json"
    if not layer_selection_path.exists():
        raise FileNotFoundError(f"Missing frozen patch-layer candidates: {layer_selection_path}")
    if (root / "selected_intervention.json").exists() or _result_paths(
        root, split="test", protocol="confirm"
    ):
        raise FileExistsError("Refusing to reselect a dose after a frozen selection/test exists.")
    layer_selection = json.loads(layer_selection_path.read_text(encoding="utf-8"))
    candidate_layers = [int(value) for value in layer_selection["candidate_layers"]]
    paths = _result_paths(root, split=split, protocol="dose-tune")
    coherence = _validate_result_manifests(paths, expected_protocol="dose-tune")
    if coherence["validated_layers"] != sorted(candidate_layers):
        raise ValueError(
            "Dose-tuning shards do not exactly match frozen candidate layers: "
            f"expected={sorted(candidate_layers)}, found={coherence['validated_layers']}."
        )
    for field in ("source_run_dir", "directions_manifest_sha256", "directions_npz_sha256"):
        if coherence[field] != layer_selection[field]:
            raise ValueError(f"Dose-tuning artifacts do not match layer selection field {field!r}.")
    frame = _read_result_tree(
        root,
        split=split,
        protocol="dose-tune",
        interventions=(
            "steer_restoration_meandiff",
            "steer_rademacher_null",
            "steer_random_direction",
        ),
        layers=candidate_layers,
    )
    restoration = _steering_condition_did(frame, "steer_restoration_meandiff")
    null = _steering_condition_did(frame, "steer_rademacher_null")
    random = _steering_condition_did(frame, "steer_random_direction")
    # The random_all-defined hidden-truth subgroup is frozen for moderation on
    # held-out test and must not influence either validation selection stage.
    restoration = restoration[~restoration["hidden_truth_flip"].astype(bool)].copy()
    null = null[~null["hidden_truth_flip"].astype(bool)].copy()
    random = random[~random["hidden_truth_flip"].astype(bool)].copy()
    if restoration.empty:
        raise ValueError("No non-hidden-truth validation items remain for dose selection.")

    dose_response_spearman: Dict[int, float] = {}
    for layer, layer_group in restoration.groupby("layer"):
        profile = (
            layer_group.groupby("alpha", as_index=False)["mitigation_did"]
            .mean()
            .sort_values("alpha")
        )
        if len(profile) < 3:
            dose_response_spearman[int(layer)] = float("nan")
            continue
        alpha_ranks = profile["alpha"].rank(method="average")
        effect_ranks = profile["mitigation_did"].rank(method="average")
        dose_response_spearman[int(layer)] = float(alpha_ranks.corr(effect_ranks))

    rows = []
    for (layer, alpha), group in restoration.groupby(["layer", "alpha"]):
        if float(alpha) <= 0.0:
            continue
        negative_group = restoration[
            (restoration["layer"] == layer)
            & np.isclose(restoration["alpha"].astype(float), -float(alpha))
        ]
        null_group = null[(null["layer"] == layer) & (null["alpha"] == alpha)]
        random_group = random[(random["layer"] == layer) & (random["alpha"] == alpha)]
        neutral_cost = -float(group["neutral_accuracy_change"].mean())
        correct_cost = -float(group["correct_suggestion_accuracy_change"].mean())
        correct_probability_cost = -float(group["correct_suggestion_delta_p_correct"].mean())
        genuine_relative = float(group["genuine_agreement_relative_margin"].mean())
        beneficial = group[group["neutral_wrong_to_correct_suggestion_correct"].astype(bool)]
        beneficial_agreement_cost = (
            -float(beneficial["correct_suggestion_condition_suggestion_agreement_change"].mean())
            if len(beneficial)
            else float("nan")
        )
        mitigation_did = float(group["mitigation_did"].mean())
        negative_mitigation_did = (
            float(negative_group["mitigation_did"].mean())
            if len(negative_group)
            else float("nan")
        )
        monotonic_spearman = dose_response_spearman.get(int(layer), float("nan"))
        null_did = float(null_group["mitigation_did"].mean())
        random_did = float(random_group["mitigation_did"].mean())
        eligible = bool(
            mitigation_did > 0.0
            and np.isfinite(negative_mitigation_did)
            and negative_mitigation_did < 0.0
            and np.isfinite(monotonic_spearman)
            and monotonic_spearman >= float(min_dose_response_spearman)
            and mitigation_did > null_did
            and mitigation_did > random_did
            and neutral_cost <= float(max_neutral_accuracy_cost)
            and correct_cost <= float(max_correct_suggestion_accuracy_cost)
            and correct_probability_cost <= float(max_correct_suggestion_probability_cost)
            and genuine_relative >= -float(max_genuine_agreement_relative_margin_cost)
            and (
                not len(beneficial)
                or beneficial_agreement_cost <= float(max_correct_suggestion_accuracy_cost)
            )
        )
        rows.append(
            {
                "layer": int(layer),
                "alpha": float(alpha),
                "n_items": int(len(group)),
                "mitigation_did": mitigation_did,
                "negative_alpha_mitigation_did": negative_mitigation_did,
                "signed_dose_contrast": mitigation_did - negative_mitigation_did,
                "dose_response_spearman": monotonic_spearman,
                "min_dose_response_spearman": float(min_dose_response_spearman),
                "null_mitigation_did": null_did,
                "random_mitigation_did": random_did,
                "restoration_minus_null_did": mitigation_did - null_did,
                "restoration_minus_random_did": mitigation_did - random_did,
                "biased_delta_margin": float(group["biased_delta_margin"].mean()),
                "neutral_delta_margin": float(group["neutral_delta_margin"].mean()),
                "neutral_accuracy_cost": neutral_cost,
                "correct_suggestion_accuracy_cost": correct_cost,
                "correct_suggestion_probability_cost": correct_probability_cost,
                "genuine_agreement_relative_margin": genuine_relative,
                "beneficial_correction_n": int(len(beneficial)),
                "beneficial_correction_agreement_cost": beneficial_agreement_cost,
                "eligible": eligible,
            }
        )
    dose_table = pd.DataFrame(rows)
    dose_table.to_csv(root / "validation_dose_selection.csv", index=False)
    eligible = dose_table[dose_table.get("eligible", False).astype(bool)].sort_values(
        ["mitigation_did", "alpha", "layer"], ascending=[False, True, True]
    ) if not dose_table.empty else pd.DataFrame()
    if eligible.empty:
        no_go = {
            "created_at": utc_now(),
            "stage": "dose_selection",
            "reason": "no_positive_selective_signed_monotonic_did_beating_controls",
            "diagnostic_table": str(root / "validation_dose_selection.csv"),
            "frozen_test_blocked": True,
        }
        write_json(root / "intervention_no_go.json", no_go)
        raise RuntimeError(
            "No MeanDiff layer/dose passed the prespecified validation checks; test confirmation is blocked."
        )
    selected = eligible.iloc[0]
    chosen_layer = int(selected["layer"])
    chosen_alpha = float(selected["alpha"])
    selection = {
        "created_at": utc_now(),
        "selection_split": "val",
        "selection_subset": "all paired items except hidden_truth_flip",
        "primary_selection_estimand": "delta_margin(strong_wrong) - delta_margin(neutral)",
        "signed_dose_gate": (
            "matched negative alpha must have negative mitigation DiD; "
            f"layer-level Spearman(alpha, mitigation DiD) >= {float(min_dose_response_spearman):.2f}"
        ),
        "selected_layer": chosen_layer,
        "selected_alpha": chosen_alpha,
        "test_alphas": [0.0, chosen_alpha, -chosen_alpha],
        "candidate_layers": candidate_layers,
        "source_run_dir": coherence["source_run_dir"],
        "directions_manifest_sha256": coherence["directions_manifest_sha256"],
        "directions_npz_sha256": coherence["directions_npz_sha256"],
        "model_name": coherence["model_name"],
        "dataset_name": coherence["dataset_name"],
        "pilot": bool(
            layer_selection.get("pilot") or coherence.get("max_questions") is not None
        ),
        "selected_dose_row": selected.to_dict(),
        "frozen_before_test": True,
    }
    selection["test_confirmation_allowed"] = not bool(selection["pilot"])
    write_json(root / "selected_intervention.json", selection)
    return selection


def _primary_contrast_frame(
    frame: pd.DataFrame,
    *,
    selection: Mapping[str, Any],
) -> pd.DataFrame:
    """Build question-level causal contrasts at the frozen layer/dose."""

    rows: List[Dict[str, Any]] = []
    selected_alpha = float(selection["selected_alpha"])
    steering = frame[
        frame["intervention"].isin(
            [
                "steer_restoration_meandiff",
                "steer_rademacher_null",
                "steer_random_direction",
            ]
        )
    ]
    if not steering.empty:
        treatment = _steering_condition_did(steering, "steer_restoration_meandiff")
        null = _steering_condition_did(steering, "steer_rademacher_null")
        random = _steering_condition_did(steering, "steer_random_direction")
        item_keys = ["layer", "alpha", "question_id", "draw_idx"]
        for alpha in sorted(
            set(treatment["alpha"].astype(float))
            & {0.0, selected_alpha, -selected_alpha}
        ):
            treatment_alpha = treatment[treatment["alpha"].astype(float) == float(alpha)].copy()
            controls = {
                "restoration_mitigation_did": None,
                "restoration_minus_null_did": null[
                    null["alpha"].astype(float) == float(alpha)
                ],
                "restoration_minus_random_did": random[
                    random["alpha"].astype(float) == float(alpha)
                ],
            }
            for contrast, control in controls.items():
                working = treatment_alpha.copy()
                if control is None:
                    working["effect"] = working["mitigation_did"]
                else:
                    working = working.merge(
                        control[item_keys + ["mitigation_did"]].rename(
                            columns={"mitigation_did": "control_mitigation_did"}
                        ),
                        on=item_keys,
                        validate="one_to_one",
                    )
                    working["effect"] = (
                        working["mitigation_did"] - working["control_mitigation_did"]
                    )
                for row in working.to_dict(orient="records"):
                    rows.append(
                        {
                            **row,
                            "contrast": contrast,
                            "condition": "strong_wrong_minus_neutral",
                        }
                    )

    patch = frame[
        frame["intervention"].isin(
            [
                "patch_paired_full",
                "patch_reverse_full",
                "patch_wrong_question",
                "patch_random_matched",
            ]
        )
    ].copy()
    if not patch.empty:
        item_keys = ["layer", "question_id", "draw_idx"]
        full = patch[
            (patch["intervention"] == "patch_paired_full")
            & (patch["condition"] == PRIMARY_BIASED_CONDITION)
        ].copy()
        reverse = patch[
            (patch["intervention"] == "patch_reverse_full")
            & (patch["condition"] == PRIMARY_NEUTRAL_CONDITION)
        ][item_keys + ["delta_margin"]].rename(columns={"delta_margin": "reverse_delta"})
        wrong = patch[
            (patch["intervention"] == "patch_wrong_question")
            & (patch["condition"] == PRIMARY_BIASED_CONDITION)
        ][item_keys + ["delta_margin"]].rename(columns={"delta_margin": "wrong_delta"})
        random_patch = patch[
            (patch["intervention"] == "patch_random_matched")
            & (patch["condition"] == PRIMARY_BIASED_CONDITION)
        ].groupby(item_keys, as_index=False)["delta_margin"].mean().rename(
            columns={"delta_margin": "random_delta"}
        )
        full = full.merge(reverse, on=item_keys, how="left", validate="one_to_one")
        full = full.merge(wrong, on=item_keys, how="left", validate="one_to_one")
        full = full.merge(random_patch, on=item_keys, how="left", validate="one_to_one")
        patch_contrasts = {
            "paired_patch_delta_margin": full["delta_margin"],
            "paired_patch_bidirectional": (full["delta_margin"] - full["reverse_delta"]) / 2.0,
            "paired_patch_minus_wrong_question": full["delta_margin"] - full["wrong_delta"],
            "paired_patch_minus_matched_random": full["delta_margin"] - full["random_delta"],
        }
        for contrast, values in patch_contrasts.items():
            working = full.copy()
            working["effect"] = values
            working["alpha"] = 1.0
            working["condition"] = "paired_patch"
            for row in working.to_dict(orient="records"):
                rows.append({**row, "contrast": contrast})

    if not rows:
        return pd.DataFrame()
    output = pd.DataFrame(rows)
    output["model_name"] = str(frame["model_name"].iloc[0])
    output["dataset_name"] = str(frame["dataset_name"].iloc[0])
    output["split"] = str(frame["split"].iloc[0])
    output["protocol"] = str(frame["protocol"].iloc[0])
    return output


def _probe_moderator_summary(
    contrast_frame: pd.DataFrame,
    *,
    selection: Mapping[str, Any],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if contrast_frame.empty:
        return pd.DataFrame()
    selected = contrast_frame[
        (contrast_frame["contrast"] == "restoration_mitigation_did")
        & (contrast_frame["alpha"].astype(float) == float(selection["selected_alpha"]))
        & contrast_frame["sycophantic_flip"].astype(bool)
        & contrast_frame["baseline_replay_matched"].astype(bool)
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    hidden = selected[selected["hidden_truth_flip"].astype(bool)]["effect"].to_numpy(float)
    probe_user = selected[selected["sycophantic_flip_probe_user"].astype(bool)]["effect"].to_numpy(float)
    difference, ci_low, ci_high = bootstrap_difference_interval(
        hidden,
        probe_user,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
    )
    rows = [
        {
            "model_name": selected["model_name"].iloc[0] if len(selected) else "",
            "dataset_name": selected["dataset_name"].iloc[0] if len(selected) else "",
            "split": selected["split"].iloc[0] if len(selected) else "",
            "layer": int(selection["selected_layer"]),
            "alpha": float(selection["selected_alpha"]),
            "moderator_contrast": "hidden_truth_flip_minus_probe_follows_user",
            "effect_difference": difference,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_hidden_truth": int(len(hidden)),
            "n_probe_follows_user": int(len(probe_user)),
            "ci_method": "independent_group_bootstrap",
        }
    ]
    regression = selected.dropna(
        subset=[
            "effect",
            "probe_score_gap_correct_minus_selected",
            "biased_baseline_margin_correct_minus_endorsed",
            "neutral_p_correct_saved",
        ]
    )
    if len(regression) >= 20:
        predictors = regression[
            [
                "probe_score_gap_correct_minus_selected",
                "biased_baseline_margin_correct_minus_endorsed",
                "neutral_p_correct_saved",
            ]
        ].to_numpy(float)
        means = predictors.mean(axis=0)
        scales = predictors.std(axis=0, ddof=1)
        scales[scales == 0.0] = 1.0
        design = np.column_stack((np.ones(len(predictors)), (predictors - means) / scales))
        outcome = regression["effect"].to_numpy(float)
        coefficient = float(np.linalg.lstsq(design, outcome, rcond=None)[0][1])
        rng = np.random.default_rng(int(seed) + 101)
        boot = np.empty(int(n_bootstrap), dtype=float)
        for index in range(int(n_bootstrap)):
            sample = rng.integers(0, len(outcome), size=len(outcome))
            boot[index] = float(
                np.linalg.lstsq(design[sample], outcome[sample], rcond=None)[0][1]
            )
        rows.append(
            {
                "model_name": regression["model_name"].iloc[0],
                "dataset_name": regression["dataset_name"].iloc[0],
                "split": regression["split"].iloc[0],
                "layer": int(selection["selected_layer"]),
                "alpha": float(selection["selected_alpha"]),
                "moderator_contrast": "standardized_probe_margin_partial_slope",
                "effect_difference": coefficient,
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "n_hidden_truth": int(len(regression)),
                "n_probe_follows_user": None,
                "ci_method": "question_bootstrap_OLS_controlling_baseline_margin_and_confidence",
            }
        )
    return pd.DataFrame(rows)


def aggregate_intervention_results(
    *,
    output_root: Path,
    split: Optional[str] = None,
    n_bootstrap: int = 2000,
    seed: int = 5,
) -> Dict[str, Path]:
    root = Path(output_root).expanduser().resolve()
    paths = _result_paths(root, split=split)
    if not paths:
        raise FileNotFoundError(f"No intervention result shards below {root}.")
    for protocol in ("patch-localize", "dose-tune", "confirm"):
        protocol_paths = [
            path for path in paths if f"item_results_{protocol.replace('-', '_')}" in path.name
        ]
        if protocol_paths:
            # Validate separately by split because the selected layer legitimately
            # appears once in validation and once in held-out confirmation.
            for split_name in sorted({path.parent.name for path in protocol_paths}):
                _validate_result_manifests(
                    [path for path in protocol_paths if path.parent.name == split_name],
                    expected_protocol=protocol,
                )
    group_columns = (
        "model_name",
        "dataset_name",
        "split",
        "protocol",
        "layer",
        "condition",
        "intervention",
        "direction_family",
        "alpha",
        "confirmatory_status",
        "subset",
    )
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(split) if split else "all_splits"
    selection_path = root / "selected_intervention.json"
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.exists()
        else None
    )
    summary_frames: List[pd.DataFrame] = []
    primary_frames: List[pd.DataFrame] = []
    contrast_item_frames: List[pd.DataFrame] = []
    plot_frames: List[pd.DataFrame] = []
    catalog_rows: List[Dict[str, Any]] = []
    total_item_rows = 0
    total_expanded_rows = 0
    total_unit_rows = 0
    primary_subsets = {
        "all_replay_matched",
        "neutral_correct",
        "sycophantic_flip",
        "hidden_truth_flip_replay_matched",
        "sycophantic_flip_probe_user",
        "sycophantic_flip_probe_other",
        "neutral_wrong_to_correct_suggestion_correct",
    }
    primary_metrics = (
        "delta_margin",
        "accuracy_change",
        "delta_p_correct",
        "delta_p_condition_suggested",
        "condition_suggestion_agreement_change",
        "normalized_recovery",
    )
    for path in paths:
        rows = read_jsonl(path)
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        total_item_rows += len(frame)
        catalog_rows.append(
            {
                "path": str(path),
                "protocol": str(frame["protocol"].iloc[0]),
                "split": str(frame["split"].iloc[0]),
                "layer": int(frame["layer"].iloc[0]),
                "n_rows": int(len(frame)),
                "sha256": sha256_file(path),
            }
        )
        plot_mask = frame["intervention"].isin(
            [
                "patch_paired_full",
                "patch_reverse_full",
                "steer_restoration_meandiff",
                "steer_rademacher_null",
                "steer_random_direction",
            ]
        )
        if selection is not None:
            plot_mask &= (
                frame["protocol"].eq("patch-localize")
                | frame["layer"].astype(int).eq(int(selection["selected_layer"]))
            )
        plot_frames.append(frame.loc[plot_mask].copy())

        expanded = expand_result_subsets(frame.to_dict(orient="records"))
        total_expanded_rows += len(expanded)
        metric_columns = [
            column for column in DEFAULT_SUMMARY_METRICS if column in expanded.columns
        ]
        unit_columns = [*group_columns, "question_id", "draw_idx"]
        unit_frame = expanded.groupby(
            unit_columns, dropna=False, as_index=False
        )[metric_columns].mean()
        total_unit_rows += len(unit_frame)
        summary_frames.append(
            summarize_result_frame(
                unit_frame,
                group_columns=group_columns,
                metric_columns=metric_columns,
                ci_method="normal",
            )
        )

        if selection is not None and int(frame["layer"].iloc[0]) == int(
            selection["selected_layer"]
        ):
            contrast_items = _primary_contrast_frame(frame, selection=selection)
            if not contrast_items.empty:
                contrast_item_frames.append(contrast_items)
            selected_alpha = float(selection["selected_alpha"])
            primary = expanded[
                expanded["subset"].isin(primary_subsets)
                & (
                    expanded["intervention"].isin(
                        [
                            "patch_paired_full",
                            "patch_reverse_full",
                            "patch_wrong_question",
                            "patch_random_matched",
                            "steer_restoration_meandiff",
                        ]
                    )
                )
                & (
                    ~expanded["intervention"].eq("steer_restoration_meandiff")
                    | expanded["alpha"].astype(float).isin(
                        [0.0, selected_alpha, -selected_alpha]
                    )
                )
            ].copy()
            available_primary_metrics = [
                column for column in primary_metrics if column in primary.columns
            ]
            if not primary.empty:
                primary_unit = primary.groupby(
                    unit_columns, dropna=False, as_index=False
                )[available_primary_metrics].mean()
                primary_frames.append(
                    summarize_result_frame(
                        primary_unit,
                        group_columns=group_columns,
                        metric_columns=available_primary_metrics,
                        n_bootstrap=int(n_bootstrap),
                        seed=int(seed),
                        ci_method="bootstrap",
                    )
                )

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    primary_summary = (
        pd.concat(primary_frames, ignore_index=True) if primary_frames else pd.DataFrame()
    )
    contrast_items = (
        pd.concat(contrast_item_frames, ignore_index=True)
        if contrast_item_frames
        else pd.DataFrame()
    )
    if not contrast_items.empty:
        contrast_expanded = expand_result_subsets(contrast_items.to_dict(orient="records"))
        contrast_group_columns = (
            "model_name",
            "dataset_name",
            "split",
            "protocol",
            "layer",
            "alpha",
            "condition",
            "contrast",
            "subset",
        )
        contrast_summary = summarize_result_frame(
            contrast_expanded,
            group_columns=contrast_group_columns,
            metric_columns=("effect",),
            n_bootstrap=int(n_bootstrap),
            seed=int(seed) + 17,
            ci_method="bootstrap",
        )
        moderator_frames = [
            _probe_moderator_summary(
                group,
                selection=selection or {},
                n_bootstrap=int(n_bootstrap),
                seed=int(seed) + group_index,
            )
            for group_index, (_key, group) in enumerate(
                contrast_items.groupby(["model_name", "dataset_name", "split"])
            )
        ]
        moderator_frames = [frame for frame in moderator_frames if not frame.empty]
        moderator_summary = (
            pd.concat(moderator_frames, ignore_index=True)
            if moderator_frames
            else pd.DataFrame()
        )
    else:
        contrast_summary = pd.DataFrame()
        moderator_summary = pd.DataFrame()
    catalog = pd.DataFrame(catalog_rows)
    catalog_path = aggregate_dir / f"item_result_catalog_{suffix}.csv"
    summary_csv = aggregate_dir / f"summary_{suffix}.csv"
    primary_path = aggregate_dir / f"primary_bootstrap_{suffix}.csv"
    contrast_path = aggregate_dir / f"causal_contrasts_{suffix}.csv"
    moderator_path = aggregate_dir / f"probe_moderator_{suffix}.csv"
    catalog.to_csv(catalog_path, index=False)
    summary.to_csv(summary_csv, index=False)
    primary_summary.to_csv(primary_path, index=False)
    contrast_summary.to_csv(contrast_path, index=False)
    moderator_summary.to_csv(moderator_path, index=False)

    hidden_truth = primary_summary[
        (primary_summary["subset"] == "hidden_truth_flip_replay_matched")
        & (primary_summary["condition"] == PRIMARY_BIASED_CONDITION)
    ].copy() if not primary_summary.empty else pd.DataFrame()
    hidden_truth_path = aggregate_dir / f"hidden_truth_results_{suffix}.csv"
    hidden_truth.to_csv(hidden_truth_path, index=False)
    plot_paths: List[str] = []
    plot_frame = pd.concat(plot_frames, ignore_index=True) if plot_frames else pd.DataFrame()
    layer_plot = plot_validation_layer_profile(
        plot_frame,
        aggregate_dir / "plots" / "validation_bidirectional_layer_profile.png",
    )
    if layer_plot is not None:
        plot_paths.append(str(layer_plot))
    if selection is not None:
        dose_plot = plot_selected_dose_response(
            plot_frame,
            selection,
            aggregate_dir / "plots" / "validation_selected_dose_response.png",
            split="val",
        )
        if dose_plot is not None:
            plot_paths.append(str(dose_plot))
    manifest_path = aggregate_dir / f"manifest_{suffix}.json"
    write_json(
        manifest_path,
        {
            "created_at": utc_now(),
            "split_filter": split,
            "n_item_rows": int(total_item_rows),
            "n_expanded_rows": int(total_expanded_rows),
            "n_question_level_summary_units": int(total_unit_rows),
            "n_summary_rows": int(len(summary)),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "item_result_catalog": str(catalog_path),
            "summary": str(summary_csv),
            "primary_bootstrap_summary": str(primary_path),
            "causal_contrasts": str(contrast_path),
            "probe_moderator": str(moderator_path),
            "hidden_truth_summary": str(hidden_truth_path),
            "plots": plot_paths,
        },
    )
    return {
        "items": catalog_path,
        "summary": summary_csv,
        "primary": primary_path,
        "contrasts": contrast_path,
        "moderator": moderator_path,
        "hidden_truth": hidden_truth_path,
        "manifest": manifest_path,
    }


__all__ = [
    "DEFAULT_ALPHAS",
    "EXPERIMENT_CONDITIONS",
    "PRIMARY_BIASED_CONDITION",
    "PRIMARY_CORRECT_SUGGESTION_CONDITION",
    "PRIMARY_NEUTRAL_CONDITION",
    "RuntimeBundle",
    "aggregate_intervention_results",
    "fit_restoration_directions",
    "load_runtime",
    "resolve_device",
    "run_intervention_layer",
    "runtime_fingerprint",
    "select_validation_dose",
    "select_validation_layers",
    "sha256_file",
]
