from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "controlled_prompt_only_v1_20260725"
CANONICAL_OPTION_LABELS = "ABCDE"
DIRECTION_CONDITIONS = {
    "wn": ("incorrect_suggestion", "neutral"),
    "cn": ("suggest_correct", "neutral"),
    "wc": ("incorrect_suggestion", "suggest_correct"),
    "sw": ("incorrect_suggestion_strong", "incorrect_suggestion"),
}
REQUIRED_CONDITIONS = (
    "neutral",
    "incorrect_suggestion",
    "incorrect_suggestion_strong",
    "suggest_correct",
)
PRIMARY_ALPHA_GRID = (
    -128.0,
    -64.0,
    -32.0,
    -16.0,
    -8.0,
    -4.0,
    -2.0,
    -1.0,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
)
CONTROL_TYPES = ("isotropic", "coordinate_sign", "item_sign_native", "item_sign_matched")


def canonical_choice_map(source_choices: Sequence[Any]) -> Dict[str, str]:
    """Map model-facing option labels to stable positional A-E labels."""

    choices = [str(value).strip().upper() for value in source_choices]
    if not choices or len(choices) > len(CANONICAL_OPTION_LABELS):
        raise ValueError(f"Expected one to five option labels, got {choices!r}.")
    if any(not choice for choice in choices) or len(choices) != len(set(choices)):
        raise ValueError(f"Option labels must be non-empty and unique: {choices!r}.")
    canonical = list(CANONICAL_OPTION_LABELS[: len(choices)])
    if choices == canonical:
        return dict(zip(choices, canonical))
    numeric = [str(index) for index in range(1, len(choices) + 1)]
    if choices == numeric:
        return dict(zip(choices, canonical))
    raise ValueError(
        "Controlled option labels must be the positional prefix A-E or 1-5; "
        f"got {choices!r}."
    )


def canonicalize_choice_mapping(
    values: Mapping[str, Any],
    choice_map: Mapping[str, str],
) -> Dict[str, Any]:
    """Re-key a source-label mapping with stable positional labels."""

    normalized_values = {
        str(key).strip().upper(): value for key, value in values.items()
    }
    expected = [str(key).strip().upper() for key in choice_map]
    missing = [choice for choice in expected if choice not in normalized_values]
    if missing:
        raise KeyError(f"Choice mapping is missing source labels: {missing!r}.")
    return {
        str(canonical): normalized_values[str(source).strip().upper()]
        for source, canonical in choice_map.items()
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(str(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_value(value: Any, *, path: str = "root") -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strict_json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist(), path=path)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite value at {path}: {number!r}")
        return number
    if isinstance(value, Path):
        return str(value)
    return value


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    target.write_text(
        json.dumps(_strict_json_value(dict(payload)), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_strict_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    with target.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    _strict_json_value(dict(row), path=f"rows[{index}]"),
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            rows.append(row)
    return rows


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _strict_json_value(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_text(canonical)


def stable_question_key(row: Mapping[str, Any]) -> str:
    dataset = str(row.get("dataset", "") or "")
    source_example_id = str(row.get("source_example_id", "") or "")
    if not dataset or not source_example_id:
        raise ValueError(
            "Every controlled question needs non-empty dataset and source_example_id."
        )
    return f"{dataset}::{source_example_id}"


def validate_question_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_human_approval: bool,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Question manifest is empty.")
    keys: list[str] = []
    split_keys: Dict[str, set[str]] = {}
    by_dataset_split: Dict[str, int] = {}
    by_dataset_b: Dict[str, int] = {}
    for index, row in enumerate(rows):
        key = stable_question_key(row)
        keys.append(key)
        split = str(row.get("split", "") or "")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Row {index} has invalid split={split!r}.")
        correct = str(row.get("correct_choice", "") or "").strip().upper()
        endorsed = str(row.get("endorsed_choice", "") or "").strip().upper()
        if (
            correct not in CANONICAL_OPTION_LABELS
            or endorsed not in CANONICAL_OPTION_LABELS
            or correct == endorsed
        ):
            raise ValueError(
                f"Row {index} must have distinct canonical A-E correct/endorsed choices."
            )
        if row.get("source_choices") is not None:
            choice_map = canonical_choice_map(row["source_choices"])
            source_correct = str(row.get("source_correct_choice", "") or "").upper()
            source_endorsed = str(
                row.get("source_endorsed_choice", "") or ""
            ).upper()
            if (
                choice_map.get(source_correct) != correct
                or choice_map.get(source_endorsed) != endorsed
            ):
                raise ValueError(
                    f"Row {index} source/canonical choice mapping is inconsistent."
                )
            declared_map = row.get("choice_label_map")
            if declared_map is not None and dict(declared_map) != choice_map:
                raise ValueError(f"Row {index} has an invalid choice_label_map.")
        review_status = str(row.get("semantic_b_review_status", "") or "")
        if require_human_approval and review_status != "approved":
            raise ValueError(
                f"Row {index} ({key}) is not human-approved: status={review_status!r}."
            )
        if require_human_approval:
            reviewer = str(row.get("semantic_b_reviewer", "") or "").strip()
            reviewed_at = str(
                row.get("semantic_b_reviewed_at", "") or ""
            ).strip()
            if not reviewer or not reviewed_at:
                raise ValueError(
                    f"Row {index} ({key}) lacks reviewer identity or review timestamp."
                )
            try:
                datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    f"Row {index} ({key}) has a non-ISO review timestamp."
                ) from exc
        split_keys.setdefault(split, set()).add(key)
        dataset = str(row["dataset"])
        by_dataset_split[f"{dataset}::{split}"] = (
            by_dataset_split.get(f"{dataset}::{split}", 0) + 1
        )
        by_dataset_b[f"{dataset}::{endorsed}"] = (
            by_dataset_b.get(f"{dataset}::{endorsed}", 0) + 1
        )
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"Question manifest has duplicate stable keys: {duplicates[:10]}")
    split_names = sorted(split_keys)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = split_keys[left] & split_keys[right]
            if overlap:
                raise ValueError(
                    f"Question manifest splits {left!r}/{right!r} overlap: {sorted(overlap)[:10]}"
                )
    return {
        "n_questions": len(rows),
        "stable_keys_sha256": sha256_text("\n".join(sorted(keys))),
        "by_dataset_split": dict(sorted(by_dataset_split.items())),
        "by_dataset_endorsed_choice": dict(sorted(by_dataset_b.items())),
        "human_approval_required": bool(require_human_approval),
    }


def _assert_finite_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not np.isfinite(array).all():
        bad = np.argwhere(~np.isfinite(array))
        raise ValueError(f"{name} contains NaN/Inf at {bad[:5].tolist()}.")
    return array


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    values = _assert_finite_array(name, np.asarray(vector, dtype=np.float64))
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {values.shape}.")
    norm = float(np.linalg.norm(values))
    if norm <= 0:
        raise ValueError(f"{name} has zero norm.")
    return np.asarray(values / norm, dtype=np.float32)


def _balanced_signs(size: int, rng: np.random.Generator) -> np.ndarray:
    signs = np.ones(int(size), dtype=np.float32)
    indices = np.arange(int(size))
    rng.shuffle(indices)
    signs[indices[: size // 2]] = -1.0
    if size % 2:
        signs[indices[-1]] = float(rng.choice((-1.0, 1.0)))
    return signs


def fit_controlled_direction_arrays(
    states_by_condition: Mapping[str, np.ndarray],
    *,
    layers: Sequence[int],
    question_keys: Sequence[str],
    control_seeds: Sequence[int],
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Fit raw paired prompt-only directions and deterministic controls.

    Every state array has shape ``[question, layer, hidden]``. Paired
    differences and their means are computed and stored in float32.
    """

    missing = sorted(set(REQUIRED_CONDITIONS) - set(states_by_condition))
    if missing:
        raise ValueError(f"Missing direction conditions: {missing}")
    states = {
        condition: _assert_finite_array(
            condition,
            np.asarray(states_by_condition[condition], dtype=np.float32),
        )
        for condition in REQUIRED_CONDITIONS
    }
    shapes = {array.shape for array in states.values()}
    if len(shapes) != 1:
        raise ValueError(f"Condition state shapes differ: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 3:
        raise ValueError(f"Expected [question, layer, hidden] states, got {shape}.")
    n_questions, n_layers, hidden_dim = shape
    if n_questions < 2:
        raise ValueError("At least two paired questions are required.")
    if len(question_keys) != n_questions or len(set(question_keys)) != n_questions:
        raise ValueError("question_keys must uniquely identify every state row.")
    layer_values = np.asarray([int(layer) for layer in layers], dtype=np.int32)
    if len(layer_values) != n_layers or len(set(layer_values.tolist())) != n_layers:
        raise ValueError("Layer list does not match the state layer dimension.")
    seeds = np.asarray([int(seed) for seed in control_seeds], dtype=np.int64)
    if not len(seeds) or len(set(seeds.tolist())) != len(seeds):
        raise ValueError("control_seeds must be non-empty and unique.")

    arrays: Dict[str, np.ndarray] = {
        "layers": layer_values,
        "control_seeds": seeds,
    }
    centroids = {
        condition: states[condition].mean(axis=0, dtype=np.float32)
        for condition in REQUIRED_CONDITIONS
    }
    deltas: Dict[str, np.ndarray] = {}
    diagnostics: list[Dict[str, Any]] = []
    raw_by_name: Dict[str, np.ndarray] = {}
    for name, (positive, negative) in DIRECTION_CONDITIONS.items():
        item_deltas = np.subtract(
            states[positive],
            states[negative],
            dtype=np.float32,
        )
        raw = item_deltas.mean(axis=0, dtype=np.float32)
        raw = _assert_finite_array(f"{name}_raw", raw).astype(np.float32)
        arrays[f"{name}_raw"] = raw
        arrays[f"{name}_unit"] = np.stack(
            [_unit(raw[layer_index], name=f"{name} layer {layer}") for layer_index, layer in enumerate(layer_values)]
        )
        raw_by_name[name] = raw
        deltas[name] = item_deltas

    n_controls = len(seeds)
    isotropic = np.empty((n_layers, n_controls, hidden_dim), dtype=np.float32)
    coordinate_sign = np.empty_like(isotropic)
    item_sign_raw = np.empty_like(isotropic)
    item_sign_matched = np.empty_like(isotropic)
    item_deltas_wn = deltas["wn"]
    for layer_index, layer in enumerate(layer_values.tolist()):
        wn = raw_by_name["wn"][layer_index]
        wn_norm = float(np.linalg.norm(wn.astype(np.float64)))
        if wn_norm <= 0:
            raise ValueError(f"Layer {layer} has zero W-N direction.")
        for seed_index, seed in enumerate(seeds.tolist()):
            rng = np.random.default_rng(int(seed))
            isotropic[layer_index, seed_index] = (
                _unit(rng.normal(size=hidden_dim), name=f"isotropic seed={seed}")
                * wn_norm
            )
            signs = rng.choice((-1.0, 1.0), size=hidden_dim).astype(np.float32)
            coordinate_sign[layer_index, seed_index] = wn * signs
            item_signs = _balanced_signs(n_questions, rng)
            raw_null = (
                item_deltas_wn[:, layer_index, :] * item_signs[:, None]
            ).mean(axis=0, dtype=np.float32)
            raw_null = _assert_finite_array("item_sign_raw", raw_null).astype(np.float32)
            item_sign_raw[layer_index, seed_index] = raw_null
            item_sign_matched[layer_index, seed_index] = (
                _unit(raw_null, name=f"item-sign seed={seed} layer={layer}") * wn_norm
            )

        neutral = states["neutral"][:, layer_index, :].astype(np.float64)
        wrong = states["incorrect_suggestion"][:, layer_index, :].astype(np.float64)
        delta_norms = np.linalg.norm(wrong - neutral, axis=1)
        residual_norms = np.concatenate(
            (np.linalg.norm(neutral, axis=1), np.linalg.norm(wrong, axis=1))
        )
        mean_n = centroids["neutral"][layer_index]
        mean_w = centroids["incorrect_suggestion"][layer_index]
        distance_before = float(np.linalg.norm(mean_w - mean_n))
        distance_after = float(np.linalg.norm((mean_w - 0.1 * wn) - mean_n))
        orientation = float(np.dot(wn.astype(np.float64), mean_w - mean_n))
        if orientation <= 0:
            raise AssertionError(f"Layer {layer} W-N orientation is not positive.")
        if distance_after >= distance_before:
            raise AssertionError(
                f"Layer {layer} subtracting 0.1*W-N did not move W toward N."
            )
        diagnostics.append(
            {
                "layer": int(layer),
                "wn_raw_norm": wn_norm,
                "mean_residual_norm": float(residual_norms.mean()),
                "median_residual_norm": float(np.median(residual_norms)),
                "median_item_delta_norm": float(np.median(delta_norms)),
                "wn_orientation_dot": orientation,
                "centroid_distance_before_subtraction": distance_before,
                "centroid_distance_after_subtract_0_1": distance_after,
            }
        )

    arrays.update(
        {
            "isotropic_matched": isotropic,
            "coordinate_sign_matched": coordinate_sign,
            "item_sign_raw": item_sign_raw,
            "item_sign_matched": item_sign_matched,
        }
    )
    for condition, centroid in centroids.items():
        arrays[f"centroid_{condition}"] = centroid.astype(np.float32)
    for name, array in arrays.items():
        _assert_finite_array(name, np.asarray(array))
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "direction_definition": {
            name: f"unweighted paired mean({positive} - {negative})"
            for name, (positive, negative) in DIRECTION_CONDITIONS.items()
        },
        "primary_direction": "wn",
        "positive_alpha_meaning": "more_like_ordinary_wrong_pressure",
        "alpha_definition": "h_intervened = h + alpha * raw_direction",
        "alpha_one_meaning": "one raw paired mean activation shift",
        "intervention_site": "post_block_residual_final_rendered_prompt_token",
        "direction_construction_uses_answer_tokens": False,
        "n_questions": int(n_questions),
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden_dim),
        "question_keys_sha256": sha256_text("\n".join(question_keys)),
        "control_seeds": seeds.tolist(),
        "controls": {
            "isotropic": "same-norm isotropic Gaussian vector",
            "coordinate_sign": "independent coordinate signs applied to W-N",
            "item_sign_native": "balanced item-sign permutation mean in native units",
            "item_sign_matched": "item-sign permutation rescaled to W-N norm",
        },
        "diagnostics": diagnostics,
    }
    return arrays, metadata


@dataclass(frozen=True)
class ControlledDirectionArtifact:
    path: Path
    manifest_path: Path
    arrays: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

    @property
    def layers(self) -> np.ndarray:
        return np.asarray(self.arrays["layers"], dtype=int)

    def layer_index(self, layer: int) -> int:
        matches = np.flatnonzero(self.layers == int(layer))
        if len(matches) != 1:
            raise KeyError(f"Layer {layer} absent from {self.path}.")
        return int(matches[0])

    def raw_direction(self, name: str, layer: int) -> np.ndarray:
        if name not in DIRECTION_CONDITIONS:
            raise KeyError(f"Unknown learned direction {name!r}.")
        return np.asarray(
            self.arrays[f"{name}_raw"][self.layer_index(layer)],
            dtype=np.float32,
        )

    def control_direction(self, control_type: str, layer: int, seed: int) -> np.ndarray:
        array_name = {
            "isotropic": "isotropic_matched",
            "coordinate_sign": "coordinate_sign_matched",
            "item_sign_native": "item_sign_raw",
            "item_sign_matched": "item_sign_matched",
        }.get(str(control_type))
        if array_name is None:
            raise KeyError(f"Unknown control type {control_type!r}.")
        seeds = np.asarray(self.arrays["control_seeds"], dtype=int)
        matches = np.flatnonzero(seeds == int(seed))
        if len(matches) != 1:
            raise KeyError(f"Control seed {seed} absent from artifact.")
        return np.asarray(
            self.arrays[array_name][self.layer_index(layer), int(matches[0])],
            dtype=np.float32,
        )


def save_controlled_direction_artifact(
    output_dir: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> ControlledDirectionArtifact:
    target = Path(output_dir).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite direction directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.number):
            _assert_finite_array(name, array)
    artifact_path = target / "directions.npz"
    np.savez_compressed(
        artifact_path,
        **{name: np.asarray(value) for name, value in arrays.items()},
    )
    manifest = {
        **dict(metadata),
        "created_at": utc_now(),
        "artifact_sha256": sha256_file(artifact_path),
    }
    write_strict_json(target / "manifest.json", manifest)
    return load_controlled_direction_artifact(artifact_path)


def load_controlled_direction_artifact(path: Path) -> ControlledDirectionArtifact:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "directions.npz"
    manifest_path = resolved.with_name("manifest.json")
    if not resolved.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Incomplete controlled direction artifact: {resolved}")
    with np.load(resolved, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    for name, values in arrays.items():
        if np.issubdtype(values.dtype, np.number):
            _assert_finite_array(name, values)
    metadata = read_json(manifest_path)
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"Protocol mismatch: {metadata.get('protocol_version')!r} != {PROTOCOL_VERSION!r}"
        )
    if metadata.get("artifact_sha256") != sha256_file(resolved):
        raise ValueError(f"Direction artifact hash mismatch: {resolved}")
    return ControlledDirectionArtifact(
        path=resolved,
        manifest_path=manifest_path,
        arrays=arrays,
        metadata=metadata,
    )


def intervention_specs(
    artifact: ControlledDirectionArtifact,
    *,
    layer: int,
    alphas: Sequence[float],
    control_seeds: Sequence[int],
    learned_directions: Sequence[str] = ("wn", "cn", "wc", "sw"),
) -> list[Dict[str, Any]]:
    specs: list[Dict[str, Any]] = []
    wn_norm = float(np.linalg.norm(artifact.raw_direction("wn", layer)))
    for direction_name in learned_directions:
        raw = artifact.raw_direction(direction_name, layer)
        native_norm = float(np.linalg.norm(raw))
        scales = (("native", raw),)
        if direction_name != "wn":
            scales = (
                ("native", raw),
                ("wn_norm_matched", _unit(raw, name=direction_name) * wn_norm),
            )
        for scale_name, vector in scales:
            for alpha in alphas:
                specs.append(
                    {
                        "treatment_type": "learned",
                        "direction_name": direction_name,
                        "scale_convention": scale_name,
                        "control_seed": None,
                        "alpha": float(alpha),
                        "base_vector": np.asarray(vector, dtype=np.float32),
                        "addition_vector": np.asarray(vector * float(alpha), dtype=np.float32),
                        "raw_direction_norm": native_norm,
                        "applied_base_norm": float(np.linalg.norm(vector)),
                    }
                )
    for control_type in CONTROL_TYPES:
        for seed in control_seeds:
            vector = artifact.control_direction(control_type, layer, int(seed))
            for alpha in alphas:
                specs.append(
                    {
                        "treatment_type": "control",
                        "direction_name": control_type,
                        "scale_convention": (
                            "native" if control_type == "item_sign_native" else "wn_norm_matched"
                        ),
                        "control_seed": int(seed),
                        "alpha": float(alpha),
                        "base_vector": vector,
                        "addition_vector": np.asarray(vector * float(alpha), dtype=np.float32),
                        "raw_direction_norm": float(np.linalg.norm(vector)),
                        "applied_base_norm": float(np.linalg.norm(vector)),
                    }
                )
    return specs


def assert_noop_contract(
    disabled_probabilities: Sequence[Mapping[str, float]],
    zero_hook_probabilities: Sequence[Mapping[str, float]],
    *,
    exact: bool,
    max_probability_error: float,
) -> Dict[str, Any]:
    if len(disabled_probabilities) != len(zero_hook_probabilities):
        raise AssertionError("No-op comparison row counts differ.")
    maximum = 0.0
    top_agreement = 0
    for disabled, zero in zip(disabled_probabilities, zero_hook_probabilities):
        choices = sorted(set(disabled) | set(zero))
        if not choices:
            raise AssertionError("No-op comparison has no choices.")
        if max(disabled, key=disabled.get) == max(zero, key=zero.get):
            top_agreement += 1
        for choice in choices:
            left = float(disabled.get(choice, 0.0))
            right = float(zero.get(choice, 0.0))
            maximum = max(maximum, abs(left - right))
            if exact and left != right:
                raise AssertionError(
                    f"Same-shape alpha=0 differs for choice={choice}: {left} != {right}"
                )
    agreement = top_agreement / len(disabled_probabilities)
    if agreement != 1.0 or maximum > float(max_probability_error):
        raise AssertionError(
            "No-op contract failed: "
            f"top_agreement={agreement} max_probability_error={maximum}"
        )
    return {
        "top_choice_agreement": agreement,
        "max_abs_probability_error": maximum,
        "exact_required": bool(exact),
        "threshold": float(max_probability_error),
    }


def wide_choice_columns(
    probabilities: Mapping[str, float],
    log_scores: Mapping[str, float],
    *,
    letters: str = "ABCDE",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for choice in str(letters):
        probability = probabilities.get(choice)
        score = log_scores.get(choice)
        row[f"prob_{choice}"] = None if probability is None else float(probability)
        row[f"option_log_score_{choice}"] = None if score is None else float(score)
    return row


def make_controlled_result_row(
    *,
    metadata: Mapping[str, Any],
    probabilities: Mapping[str, float],
    log_scores: Mapping[str, float],
    baseline_probabilities: Mapping[str, float],
    baseline_log_scores: Mapping[str, float],
    correct_choice: str,
    endorsed_choice: str,
    median_residual_norm: float,
) -> Dict[str, Any]:
    choices = sorted(probabilities)
    predicted = max(choices, key=lambda choice: float(probabilities[choice]))
    p_correct = float(probabilities[correct_choice])
    p_endorsed = float(probabilities[endorsed_choice])
    base_p_correct = float(baseline_probabilities[correct_choice])
    base_p_endorsed = float(baseline_probabilities[endorsed_choice])
    score_margin = float(log_scores[correct_choice] - log_scores[endorsed_choice])
    base_score_margin = float(
        baseline_log_scores[correct_choice] - baseline_log_scores[endorsed_choice]
    )
    probability_values = np.asarray([probabilities[choice] for choice in choices], dtype=float)
    _assert_finite_array("result probabilities", probability_values)
    entropy = float(
        -np.sum(
            probability_values
            * np.log(np.clip(probability_values, np.finfo(float).tiny, None))
        )
    )
    addition_norm = float(metadata.get("injected_norm", 0.0))
    residual_norm = float(median_residual_norm)
    row = {
        **dict(metadata),
        "correct_choice": correct_choice,
        "endorsed_choice": endorsed_choice,
        "predicted_option": predicted,
        "is_correct": bool(predicted == correct_choice),
        "equals_endorsed": bool(predicted == endorsed_choice),
        "p_correct": p_correct,
        "p_endorsed": p_endorsed,
        "delta_p_correct": p_correct - base_p_correct,
        "delta_p_endorsed": p_endorsed - base_p_endorsed,
        "log_score_margin_correct_minus_endorsed": score_margin,
        "baseline_log_score_margin_correct_minus_endorsed": base_score_margin,
        "delta_log_score_margin": score_margin - base_score_margin,
        "probability_margin_correct_minus_endorsed": p_correct - p_endorsed,
        "entropy": entropy,
        "valid_answer": True,
        "answer_format_failure": False,
        "nonfinite_failure": False,
        "injected_norm_ratio": (
            addition_norm / residual_norm if residual_norm > 0 else 0.0
        ),
        **wide_choice_columns(probabilities, log_scores),
    }
    _strict_json_value(row)
    return row


def deterministic_derangement(size: int, *, seed: int) -> np.ndarray:
    if int(size) < 2:
        raise ValueError("A derangement requires at least two items.")
    rng = np.random.default_rng(int(seed))
    base = np.arange(int(size))
    for _ in range(10_000):
        candidate = rng.permutation(base)
        if bool(np.all(candidate != base)):
            return candidate
    # Deterministic fallback that is always a derangement.
    shift = 1 + (abs(int(seed)) % (int(size) - 1))
    return np.roll(base, shift)


def _rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, np.finfo(float).tiny, None)


def geometry_pair_rows(
    heldout_states: Mapping[str, np.ndarray],
    *,
    training_mean: np.ndarray,
    median_residual_norm: float,
    permutation_seeds: Sequence[int],
) -> pd.DataFrame:
    """Build the requested A-F geometry comparisons for one model/layer."""

    states = {
        condition: _assert_finite_array(
            condition,
            np.asarray(heldout_states[condition], dtype=np.float64),
        )
        for condition in REQUIRED_CONDITIONS
    }
    shape = states["neutral"].shape
    if len(shape) != 2 or any(value.shape != shape for value in states.values()):
        raise ValueError("Geometry states must share shape [question, hidden].")
    mean = np.asarray(training_mean, dtype=np.float64)
    if mean.shape != (shape[1],):
        raise ValueError("Training centering mean has the wrong hidden dimension.")
    if not np.isfinite(mean).all() or not np.isfinite(float(median_residual_norm)):
        raise ValueError("Geometry normalization statistics must be finite.")
    if float(median_residual_norm) <= 0:
        raise ValueError("median_residual_norm must be positive.")

    pair_specs = {
        "A_same_question_N_W": ("neutral", "incorrect_suggestion", False),
        "B_same_question_W_S": (
            "incorrect_suggestion",
            "incorrect_suggestion_strong",
            False,
        ),
        "C_same_question_N_C": ("neutral", "suggest_correct", False),
        "D_different_questions_N_N": ("neutral", "neutral", True),
        "E_different_questions_W_W": (
            "incorrect_suggestion",
            "incorrect_suggestion",
            True,
        ),
        "F_different_questions_N_W": ("neutral", "incorrect_suggestion", True),
    }
    rows: list[Dict[str, Any]] = []
    for group, (left_name, right_name, unmatched) in pair_specs.items():
        seeds = permutation_seeds if unmatched else (None,)
        for seed in seeds:
            right_indices = (
                deterministic_derangement(shape[0], seed=int(seed))
                if unmatched
                else np.arange(shape[0])
            )
            left = states[left_name]
            right = states[right_name][right_indices]
            raw_cosine = _rowwise_cosine(left, right)
            centered_cosine = _rowwise_cosine(left - mean, right - mean)
            normalized_distance = (
                np.linalg.norm(left - right, axis=1) / float(median_residual_norm)
            )
            for item_index in range(shape[0]):
                rows.append(
                    {
                        "group": group,
                        "left_condition": left_name,
                        "right_condition": right_name,
                        "left_index": int(item_index),
                        "right_index": int(right_indices[item_index]),
                        "permutation_seed": None if seed is None else int(seed),
                        "raw_cosine": float(raw_cosine[item_index]),
                        "centered_cosine": float(centered_cosine[item_index]),
                        "normalized_euclidean_distance": float(
                            normalized_distance[item_index]
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    _assert_finite_array(
        "geometry metrics",
        frame[
            ["raw_cosine", "centered_cosine", "normalized_euclidean_distance"]
        ].to_numpy(),
    )
    return frame


def identity_framing_ratio(pair_rows: pd.DataFrame) -> float:
    same = pair_rows.loc[
        pair_rows["group"].eq("A_same_question_N_W"),
        "normalized_euclidean_distance",
    ]
    framing = pair_rows.loc[
        pair_rows["group"].eq("E_different_questions_W_W"),
        "normalized_euclidean_distance",
    ]
    denominator = float(np.median(framing))
    if denominator <= 0:
        raise ValueError("Different-question same-framing median distance is non-positive.")
    return float(np.median(same) / denominator)


def framing_classification_and_retrieval(
    training_states: Mapping[str, np.ndarray],
    heldout_states: Mapping[str, np.ndarray],
    *,
    seed: int = 5,
) -> Dict[str, Any]:
    """Question-held-out framing classification and paired cross-framing retrieval."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    condition_names = list(REQUIRED_CONDITIONS)
    training = {
        condition: _assert_finite_array(
            f"training {condition}",
            np.asarray(training_states[condition], dtype=np.float64),
        )
        for condition in condition_names
    }
    heldout = {
        condition: _assert_finite_array(
            f"heldout {condition}",
            np.asarray(heldout_states[condition], dtype=np.float64),
        )
        for condition in condition_names
    }
    train_shapes = {value.shape for value in training.values()}
    test_shapes = {value.shape for value in heldout.values()}
    if len(train_shapes) != 1 or len(test_shapes) != 1:
        raise ValueError("Framing classifier condition arrays must share shapes.")
    if next(iter(train_shapes))[1] != next(iter(test_shapes))[1]:
        raise ValueError("Training/test framing hidden dimensions differ.")

    x_train = np.concatenate([training[condition] for condition in condition_names], axis=0)
    y_train = np.concatenate(
        [
            np.full(training[condition].shape[0], index, dtype=np.int32)
            for index, condition in enumerate(condition_names)
        ]
    )
    x_test = np.concatenate([heldout[condition] for condition in condition_names], axis=0)
    y_test = np.concatenate(
        [
            np.full(heldout[condition].shape[0], index, dtype=np.int32)
            for index, condition in enumerate(condition_names)
        ]
    )
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=int(seed),
        solver="liblinear",
    ).fit(scaler.transform(x_train), y_train)
    predicted = classifier.predict(scaler.transform(x_test))

    binary_train = np.concatenate(
        (training["neutral"], training["incorrect_suggestion"]),
        axis=0,
    )
    binary_y_train = np.concatenate(
        (
            np.zeros(training["neutral"].shape[0], dtype=np.int32),
            np.ones(training["incorrect_suggestion"].shape[0], dtype=np.int32),
        )
    )
    binary_test = np.concatenate(
        (heldout["neutral"], heldout["incorrect_suggestion"]),
        axis=0,
    )
    binary_y_test = np.concatenate(
        (
            np.zeros(heldout["neutral"].shape[0], dtype=np.int32),
            np.ones(heldout["incorrect_suggestion"].shape[0], dtype=np.int32),
        )
    )
    binary_scaler = StandardScaler().fit(binary_train)
    binary_classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=int(seed),
        solver="liblinear",
    ).fit(binary_scaler.transform(binary_train), binary_y_train)
    binary_predicted = binary_classifier.predict(binary_scaler.transform(binary_test))

    center = x_train.mean(axis=0)
    neutral = heldout["neutral"] - center
    wrong = heldout["incorrect_suggestion"] - center
    distances = np.linalg.norm(neutral[:, None, :] - wrong[None, :, :], axis=2)
    ranks_n_to_w = np.empty(neutral.shape[0], dtype=np.int32)
    ranks_w_to_n = np.empty(wrong.shape[0], dtype=np.int32)
    for index in range(neutral.shape[0]):
        ordering = np.argsort(distances[index], kind="stable")
        ranks_n_to_w[index] = int(np.flatnonzero(ordering == index)[0]) + 1
    for index in range(wrong.shape[0]):
        ordering = np.argsort(distances[:, index], kind="stable")
        ranks_w_to_n[index] = int(np.flatnonzero(ordering == index)[0]) + 1
    ranks = np.concatenate((ranks_n_to_w, ranks_w_to_n))
    return {
        "four_way_accuracy": float(np.mean(predicted == y_test)),
        "four_way_balanced_accuracy": float(
            balanced_accuracy_score(y_test, predicted)
        ),
        "neutral_vs_wrong_accuracy": float(np.mean(binary_predicted == binary_y_test)),
        "neutral_vs_wrong_balanced_accuracy": float(
            balanced_accuracy_score(binary_y_test, binary_predicted)
        ),
        "cross_framing_retrieval_top1": float(np.mean(ranks == 1)),
        "cross_framing_retrieval_mean_reciprocal_rank": float(
            np.mean(1.0 / ranks.astype(np.float64))
        ),
        "cross_framing_retrieval_median_rank": float(np.median(ranks)),
        "cross_framing_N_to_W_top1": float(np.mean(ranks_n_to_w == 1)),
        "cross_framing_N_to_W_mean_reciprocal_rank": float(
            np.mean(1.0 / ranks_n_to_w.astype(np.float64))
        ),
        "cross_framing_W_to_N_top1": float(np.mean(ranks_w_to_n == 1)),
        "cross_framing_W_to_N_mean_reciprocal_rank": float(
            np.mean(1.0 / ranks_w_to_n.astype(np.float64))
        ),
        "classifier_C": 1.0,
        "classifier_seed": int(seed),
        "classifier_standardization": "training_only",
    }


def git_fingerprint(repo_dir: Path) -> Dict[str, Any]:
    root = Path(repo_dir).expanduser().resolve()

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    commit = run("rev-parse", "HEAD")
    diff = run("diff", "--binary", "HEAD")
    untracked = run("ls-files", "--others", "--exclude-standard")
    untracked_entries = []
    for relative_path in [line for line in untracked.splitlines() if line.strip()]:
        path = root / relative_path
        if path.is_file():
            untracked_entries.append(f"{relative_path}\0{sha256_file(path)}")
    return {
        "git_commit": commit,
        "dirty": bool(diff or untracked),
        "tracked_diff_sha256": sha256_text(diff),
        "untracked_path_manifest_sha256": sha256_text(untracked),
        "untracked_content_manifest_sha256": sha256_text(
            "\n".join(untracked_entries)
        ),
    }


def runtime_provenance(
    *,
    repo_dir: Path,
    config_path: Path,
    question_manifest_path: Path,
    argv: Sequence[str],
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    config = getattr(model, "config", None)
    init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    identity_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": sha256_file(config_path),
        "question_manifest_sha256": sha256_file(question_manifest_path),
        "argv": list(argv),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        **git_fingerprint(repo_dir),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "question_manifest_path": str(Path(question_manifest_path).resolve()),
        "question_manifest_sha256": sha256_file(question_manifest_path),
        "run_identity_sha256": canonical_json_hash(identity_payload),
        "model_identifier": str(getattr(config, "_name_or_path", "") or ""),
        "model_revision": str(getattr(config, "_commit_hash", "") or ""),
        "tokenizer_identifier": str(getattr(tokenizer, "name_or_path", "") or ""),
        "tokenizer_revision": str(init_kwargs.get("_commit_hash", "") or ""),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(argv),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }


__all__ = [
    "CANONICAL_OPTION_LABELS",
    "CONTROL_TYPES",
    "ControlledDirectionArtifact",
    "DIRECTION_CONDITIONS",
    "PRIMARY_ALPHA_GRID",
    "PROTOCOL_VERSION",
    "REQUIRED_CONDITIONS",
    "assert_noop_contract",
    "canonical_choice_map",
    "canonicalize_choice_mapping",
    "canonical_json_hash",
    "deterministic_derangement",
    "fit_controlled_direction_arrays",
    "framing_classification_and_retrieval",
    "geometry_pair_rows",
    "identity_framing_ratio",
    "intervention_specs",
    "load_controlled_direction_artifact",
    "make_controlled_result_row",
    "read_json",
    "read_jsonl",
    "runtime_provenance",
    "save_controlled_direction_artifact",
    "stable_question_key",
    "validate_question_manifest",
    "wide_choice_columns",
    "write_strict_json",
    "write_strict_jsonl",
]
