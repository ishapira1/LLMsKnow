from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .data import DEFAULT_PROBE_NAME, load_json, resolve_layer_probe_dir


@dataclass(frozen=True)
class DirectionArtifact:
    path: Path
    metadata_path: Path
    arrays: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

    @property
    def layers(self) -> np.ndarray:
        return np.asarray(self.arrays["layers"], dtype=int)

    def layer_index(self, layer: int) -> int:
        matches = np.flatnonzero(self.layers == int(layer))
        if len(matches) != 1:
            raise KeyError(f"Layer {layer} is not present in {self.path}.")
        return int(matches[0])

    def vector(self, name: str, layer: int) -> np.ndarray:
        return np.asarray(self.arrays[name][self.layer_index(layer)], dtype=np.float32)

    def scalar(self, name: str, layer: int) -> float:
        return float(np.asarray(self.arrays[name])[self.layer_index(layer)])

    def control_vector(self, name: str, layer: int, control_seed: int) -> np.ndarray:
        values = np.asarray(self.arrays[name][self.layer_index(layer)], dtype=np.float32)
        seeds = np.asarray(self.arrays["control_seeds"], dtype=int)
        matches = np.flatnonzero(seeds == int(control_seed))
        if len(matches) != 1:
            raise KeyError(
                f"Control seed {control_seed} is not present; available={seeds.tolist()}."
            )
        return np.asarray(values[int(matches[0])], dtype=np.float32)

    def control_scalar(self, name: str, layer: int, control_seed: int) -> float:
        values = np.asarray(self.arrays[name][self.layer_index(layer)])
        seeds = np.asarray(self.arrays["control_seeds"], dtype=int)
        matches = np.flatnonzero(seeds == int(control_seed))
        if len(matches) != 1:
            raise KeyError(
                f"Control seed {control_seed} is not present; available={seeds.tolist()}."
            )
        return float(values[int(matches[0])])


def unit_vector(vector: np.ndarray, *, name: str = "direction") -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape={values.shape}.")
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} has invalid zero/non-finite norm={norm}.")
    return np.asarray(values / norm, dtype=np.float32)


def parallel_component(vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    unit = unit_vector(np.asarray(direction), name="projection direction").astype(np.float64)
    return np.asarray(float(np.dot(values, unit)) * unit, dtype=np.float32)


def orthogonal_component(vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    return np.asarray(values - parallel_component(values, direction), dtype=np.float32)


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    try:
        unit_a = unit_vector(vector_a, name="vector_a")
        unit_b = unit_vector(vector_b, name="vector_b")
    except ValueError:
        return float("nan")
    return float(np.dot(unit_a.astype(np.float64), unit_b.astype(np.float64)))


def _stratified_mean(values: np.ndarray, strata: Sequence[str]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected [items, hidden] values, got shape={array.shape}.")
    if len(strata) != array.shape[0]:
        raise ValueError("Stratum count must match the number of item rows.")
    unique_strata = sorted(set(str(value) for value in strata))
    if not unique_strata:
        raise ValueError("At least one option-position stratum is required.")
    means = []
    strata_array = np.asarray([str(value) for value in strata], dtype=object)
    for stratum in unique_strata:
        subset = array[strata_array == stratum]
        if len(subset):
            means.append(subset.mean(axis=0))
    return np.asarray(np.stack(means, axis=0).mean(axis=0), dtype=np.float32)


def _balanced_rademacher_signs(
    strata: Sequence[str],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    signs = np.ones(len(strata), dtype=np.float32)
    strata_array = np.asarray([str(value) for value in strata], dtype=object)
    for stratum in sorted(set(strata_array.tolist())):
        indices = np.flatnonzero(strata_array == stratum)
        rng.shuffle(indices)
        half = len(indices) // 2
        signs[indices[:half]] = -1.0
        signs[indices[half:]] = 1.0
        if len(indices) % 2:
            signs[indices[-1]] = float(rng.choice([-1.0, 1.0]))
    return signs


def load_probe_vector(
    run_dir: Path,
    *,
    layer: int,
    probe_name: str = DEFAULT_PROBE_NAME,
) -> tuple[np.ndarray, Dict[str, Any], Path]:
    probe_dir = resolve_layer_probe_dir(run_dir, layer=int(layer), probe_name=probe_name)
    metadata_path = probe_dir / "metadata.json"
    model_path = probe_dir / "model.pkl"
    metadata = load_json(metadata_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with model_path.open("rb") as handle:
            classifier = pickle.load(handle)
    coefficients = getattr(classifier, "coef_", None)
    if coefficients is None:
        raise ValueError(f"Saved probe has no coef_: {model_path}")
    vector = np.asarray(coefficients[0], dtype=np.float32)
    expected_dim = int(dict(metadata.get("model", {}) or {}).get("input_dim", vector.shape[0]))
    if vector.ndim != 1 or vector.shape[0] != expected_dim:
        raise ValueError(
            f"Invalid probe coefficient shape={vector.shape}; expected hidden dimension={expected_dim}."
        )
    classes = np.asarray(getattr(classifier, "classes_", []))
    if classes.size and classes.tolist() != [0, 1]:
        raise ValueError(f"Unexpected probe classes {classes.tolist()} in {model_path}.")
    return vector, metadata, probe_dir


def fit_direction_arrays(
    neutral_states: np.ndarray,
    biased_states: np.ndarray,
    *,
    layers: Sequence[int],
    option_position_strata: Sequence[str],
    seed: int,
    probe_vectors_by_layer: Optional[Mapping[int, np.ndarray]] = None,
    n_control_directions: int = 20,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Fit pre-answer restoration MeanDiff and matched null controls.

    ``neutral_states`` and ``biased_states`` have shape
    ``[questions, layers, hidden]``. The primary direction is N-B, so positive
    alpha restores the mean pressure-induced displacement.
    """

    neutral = np.asarray(neutral_states)
    biased = np.asarray(biased_states)
    if neutral.shape != biased.shape or neutral.ndim != 3:
        raise ValueError(
            "Neutral and biased states must share shape [questions, layers, hidden], "
            f"got {neutral.shape} and {biased.shape}."
        )
    layer_values = np.asarray([int(layer) for layer in layers], dtype=np.int32)
    if neutral.shape[1] != len(layer_values):
        raise ValueError("State layer dimension does not match the supplied layer list.")
    if neutral.shape[0] != len(option_position_strata):
        raise ValueError("State question count does not match option-position strata.")
    if neutral.shape[0] < 2:
        raise ValueError("At least two paired questions are required to fit directions.")

    rng = np.random.default_rng(int(seed))
    n_controls = int(n_control_directions)
    if n_controls < 1:
        raise ValueError("n_control_directions must be at least 1.")
    restoration = neutral.astype(np.float32) - biased.astype(np.float32)
    null_signs = np.stack(
        [
            _balanced_rademacher_signs(option_position_strata, rng=rng)
            for _ in range(n_controls)
        ],
        axis=0,
    )
    n_layers = len(layer_values)
    hidden_dim = int(neutral.shape[-1])

    restoration_raw = np.empty((n_layers, hidden_dim), dtype=np.float32)
    restoration_unit = np.empty_like(restoration_raw)
    restoration_scale = np.empty(n_layers, dtype=np.float32)
    null_raw = np.empty((n_layers, n_controls, hidden_dim), dtype=np.float32)
    null_unit = np.empty_like(null_raw)
    null_scale = np.empty((n_layers, n_controls), dtype=np.float32)
    random_unit = np.empty_like(null_raw)
    random_scale = np.empty((n_layers, n_controls), dtype=np.float32)
    probe_unit = np.full_like(restoration_raw, np.nan)
    probe_scale = np.full(n_layers, np.nan, dtype=np.float32)
    probe_available = np.zeros(n_layers, dtype=np.int8)
    diagnostic_rows = []

    for layer_idx, layer in enumerate(layer_values.tolist()):
        layer_restoration = restoration[:, layer_idx, :]
        raw = _stratified_mean(layer_restoration, option_position_strata)
        unit = unit_vector(raw, name=f"layer-{layer} restoration MeanDiff")
        states_n = neutral[:, layer_idx, :].astype(np.float32)
        states_b = biased[:, layer_idx, :].astype(np.float32)

        def projection_scale(direction: np.ndarray) -> float:
            projections = np.concatenate((states_n @ direction, states_b @ direction))
            value = float(np.std(projections.astype(np.float64), ddof=1))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Layer {layer} has invalid projection SD={value}.")
            return value

        restoration_raw[layer_idx] = raw
        restoration_unit[layer_idx] = unit
        restoration_scale[layer_idx] = projection_scale(unit)
        for control_index in range(n_controls):
            signed = layer_restoration * null_signs[control_index, :, None]
            raw_null = _stratified_mean(signed, option_position_strata)
            try:
                unit_null = unit_vector(
                    raw_null,
                    name=f"layer-{layer} Rademacher-null MeanDiff seed-{control_index}",
                )
            except ValueError:
                raw_null = rng.normal(size=hidden_dim).astype(np.float32)
                unit_null = unit_vector(
                    raw_null, name=f"layer-{layer} null fallback seed-{control_index}"
                )
            unit_random = unit_vector(
                rng.normal(size=hidden_dim).astype(np.float32),
                name=f"layer-{layer} random control seed-{control_index}",
            )
            null_raw[layer_idx, control_index] = raw_null
            null_unit[layer_idx, control_index] = unit_null
            null_scale[layer_idx, control_index] = projection_scale(unit_null)
            random_unit[layer_idx, control_index] = unit_random
            random_scale[layer_idx, control_index] = projection_scale(unit_random)

        probe_vector = None if probe_vectors_by_layer is None else probe_vectors_by_layer.get(int(layer))
        if probe_vector is not None:
            probe_direction = unit_vector(probe_vector, name=f"layer-{layer} transported probe")
            if probe_direction.shape[0] != hidden_dim:
                raise ValueError(
                    f"Layer-{layer} probe dim={probe_direction.shape[0]} does not match hidden dim={hidden_dim}."
                )
            probe_unit[layer_idx] = probe_direction
            probe_scale[layer_idx] = projection_scale(probe_direction)
            probe_available[layer_idx] = 1

        diagnostic_rows.append(
            {
                "layer": int(layer),
                "restoration_raw_norm": float(np.linalg.norm(raw)),
                "restoration_projection_sd": float(restoration_scale[layer_idx]),
                "null_raw_norm_mean": float(
                    np.linalg.norm(null_raw[layer_idx], axis=1).mean()
                ),
                "null_projection_sd_mean": float(null_scale[layer_idx].mean()),
                "random_projection_sd_mean": float(random_scale[layer_idx].mean()),
                "probe_available": bool(probe_available[layer_idx]),
                "cosine_restoration_probe": (
                    cosine_similarity(unit, probe_unit[layer_idx])
                    if probe_available[layer_idx]
                    else float("nan")
                ),
                "cosine_restoration_null_mean": float(
                    np.nanmean(
                        [
                            cosine_similarity(unit, null_unit[layer_idx, index])
                            for index in range(n_controls)
                        ]
                    )
                ),
                "cosine_restoration_random_mean": float(
                    np.nanmean(
                        [
                            cosine_similarity(unit, random_unit[layer_idx, index])
                            for index in range(n_controls)
                        ]
                    )
                ),
            }
        )

    arrays = {
        "layers": layer_values,
        "restoration_raw": restoration_raw,
        "restoration_unit": restoration_unit,
        "restoration_scale": restoration_scale,
        "null_raw": null_raw,
        "null_unit": null_unit,
        "null_scale": null_scale,
        "random_unit": random_unit,
        "random_scale": random_scale,
        "probe_unit": probe_unit,
        "probe_scale": probe_scale,
        "probe_available": probe_available,
        "control_seeds": np.arange(n_controls, dtype=np.int32),
    }
    metadata = {
        "protocol_version": "legacy_restoration_v0",
        "direction_definition": "balanced-stratum mean(final_prompt_neutral - final_prompt_strong_incorrect)",
        "intervention_site": "final_generation_prompt_token",
        "probe_source_site": "last_token_of_teacher_forced_candidate_answer",
        "probe_transport_status": "exploratory_cross_token_position",
        "alpha_definition": "h_intervened = h + alpha * projection_sd * unit_direction",
        "positive_restoration_alpha_meaning": "move the biased prompt state toward neutral",
        "n_pairs": int(neutral.shape[0]),
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden_dim),
        "seed": int(seed),
        "n_option_position_strata": int(len(set(str(value) for value in option_position_strata))),
        "n_control_directions": int(n_controls),
        "null_control": "balanced within-stratum Rademacher sign flip of paired restoration deltas",
        "diagnostics": diagnostic_rows,
    }
    return arrays, metadata


def save_direction_artifact(
    output_dir: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> DirectionArtifact:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "directions.npz"
    metadata_path = target_dir / "manifest.json"
    np.savez_compressed(path, **{name: np.asarray(value) for name, value in arrays.items()})
    def strict_json(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): strict_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [strict_json(item) for item in value]
        if isinstance(value, np.ndarray):
            return strict_json(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return number if np.isfinite(number) else None
        if isinstance(value, Path):
            return str(value)
        return value

    metadata_path.write_text(
        json.dumps(strict_json(dict(metadata)), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return load_direction_artifact(path)


def load_direction_artifact(path: Path) -> DirectionArtifact:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "directions.npz"
    if not resolved.exists():
        raise FileNotFoundError(f"Missing direction artifact: {resolved}")
    metadata_path = resolved.with_name("manifest.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing direction manifest: {metadata_path}")
    with np.load(resolved, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return DirectionArtifact(
        path=resolved,
        metadata_path=metadata_path,
        arrays=arrays,
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
    )


__all__ = [
    "DirectionArtifact",
    "cosine_similarity",
    "fit_direction_arrays",
    "load_direction_artifact",
    "load_probe_vector",
    "orthogonal_component",
    "parallel_component",
    "save_direction_artifact",
    "unit_vector",
]
