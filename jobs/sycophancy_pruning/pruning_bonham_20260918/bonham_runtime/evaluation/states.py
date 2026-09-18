from __future__ import annotations

from contextlib import contextmanager
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from .artifacts import sha256_file
from .schemas import StateSpec


class StateRuntimeError(RuntimeError):
    """Raised when an intervention state cannot be applied and restored exactly."""


def _message_role(message: Mapping[str, Any]) -> str:
    value = str(message.get("role", message.get("type", "")) or "").strip().lower()
    return {"human": "user", "ai": "assistant"}.get(value, value)


def messages_for_state(
    messages: Sequence[Mapping[str, Any]],
    state: StateSpec,
) -> list[Dict[str, Any]]:
    """Apply a prompt baseline without changing any non-text transcript fields."""

    result = [dict(message) for message in messages]
    if not result:
        raise StateRuntimeError("Evaluation messages cannot be empty")
    if state.intervention_kind != "system_prompt":
        return result
    prompt = str(state.system_prompt or "").strip()
    if not prompt:
        raise StateRuntimeError("System-prompt state has no prompt")
    if _message_role(result[0]) == "system":
        existing = str(result[0].get("content", "") or "").strip()
        result[0]["content"] = prompt if not existing else f"{prompt}\n\n{existing}"
    else:
        style = "type" if "type" in result[0] and "role" not in result[0] else "role"
        result.insert(
            0,
            {style: "system", "content": prompt},
        )
    return result


def _load_indices_payload(path: Path) -> Mapping[str, Any]:
    import torch

    raw = Path(path).read_bytes()
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch compatibility
        payload = torch.load(io.BytesIO(raw), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise StateRuntimeError("Mask indices artifact must contain a mapping")
    if "indices" in payload and isinstance(payload["indices"], Mapping):
        payload = payload["indices"]
    return payload


def _mask_paths(state: StateSpec) -> Tuple[Path, Path]:
    indices_value = state.metadata.get("indices_path")
    metadata_value = state.metadata.get("metadata_path")
    if not indices_value or not metadata_value:
        raise StateRuntimeError("Mask state metadata requires indices_path and metadata_path")
    indices_path = Path(str(indices_value)).expanduser().resolve()
    metadata_path = Path(str(metadata_value)).expanduser().resolve()
    if not indices_path.is_file() or not metadata_path.is_file():
        raise StateRuntimeError("Mask indices/metadata artifact is absent")
    expected_indices = state.artifact_sha256.get("indices")
    expected_metadata = state.artifact_sha256.get("metadata")
    if not expected_indices or not expected_metadata:
        raise StateRuntimeError("Mask state must authenticate indices and metadata hashes")
    if sha256_file(indices_path) != expected_indices or sha256_file(metadata_path) != expected_metadata:
        raise StateRuntimeError("Mask artifact hash mismatch")
    return indices_path, metadata_path


def _backup_mask_values(model: Any, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    import torch

    modules = dict(model.named_modules())
    backups: Dict[str, Any] = {}
    for name, raw_indices in payload.items():
        module = modules.get(str(name))
        if not isinstance(module, torch.nn.Linear):
            raise StateRuntimeError(f"Mask references missing/non-linear module {name!r}")
        if not isinstance(raw_indices, torch.Tensor) or raw_indices.ndim != 1:
            raise StateRuntimeError(f"Mask indices for {name!r} must be a one-dimensional tensor")
        indices = raw_indices.detach().cpu().to(dtype=torch.long)
        if indices.numel() == 0:
            backups[str(name)] = (indices, torch.empty(0, dtype=module.weight.dtype))
            continue
        if int(indices.min()) < 0 or int(indices.max()) >= int(module.weight.numel()):
            raise StateRuntimeError(f"Mask indices for {name!r} are out of bounds")
        device_indices = indices.to(module.weight.device)
        values = module.weight.detach().reshape(-1).index_select(0, device_indices).clone()
        backups[str(name)] = (indices, values)
    return backups


def _restore_mask_values(model: Any, backups: Mapping[str, Any]) -> None:
    import torch

    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, (indices, values) in backups.items():
            if indices.numel() == 0:
                continue
            module = modules[name]
            device_indices = indices.to(device=module.weight.device, dtype=torch.long)
            flat = module.weight.data.reshape(-1)
            flat[device_indices] = values.to(device=flat.device, dtype=flat.dtype)
        for name, (indices, values) in backups.items():
            if indices.numel() == 0:
                continue
            module = modules[name]
            device_indices = indices.to(device=module.weight.device, dtype=torch.long)
            observed = module.weight.detach().reshape(-1).index_select(0, device_indices)
            expected = values.to(device=observed.device, dtype=observed.dtype)
            if not torch.equal(observed, expected):
                raise StateRuntimeError(f"Mask restoration parity failed for {name!r}")


@contextmanager
def activated_state(model: Any, state: StateSpec) -> Iterator[Mapping[str, Any]]:
    """Activate a state and restore every changed scalar on context exit."""

    if state.intervention_kind not in {"mask", "random_mask"}:
        yield {"intervention_kind": state.intervention_kind, "parameters_set_to_zero": 0}
        return
    indices_path, metadata_path = _mask_paths(state)
    payload = _load_indices_payload(indices_path)
    backups = _backup_mask_values(model, payload)
    audit: Optional[Mapping[str, Any]] = None
    try:
        from ..pruning.live_inference import load_and_apply_strict_harm_mask

        audit = load_and_apply_strict_harm_mask(
            model,
            indices_path,
            metadata_path=metadata_path,
            expected_count=state.parameters_set_to_zero,
            expected_model=state.model_id,
            expected_revision=state.model_revision,
            expected_indices_sha256=state.artifact_sha256["indices"],
            expected_metadata_sha256=state.artifact_sha256["metadata"],
        )
        if int(audit["actual_mask_count"]) != state.parameters_set_to_zero:
            raise StateRuntimeError("Applied mask count differs from state registry")
        yield audit
    finally:
        _restore_mask_values(model, backups)


def load_steering_addition(state: StateSpec) -> Any:
    """Load and authenticate the frozen direction, then apply the frozen alpha."""

    if state.intervention_kind != "activation_steering":
        raise StateRuntimeError("Only activation-steering states have a direction")
    direction_value = state.metadata.get("direction_path")
    expected_hash = state.artifact_sha256.get("direction")
    if not direction_value or not expected_hash:
        raise StateRuntimeError("Activation state lacks direction path/hash")
    path = Path(str(direction_value)).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise StateRuntimeError("Activation direction is absent or changed")
    import numpy as np

    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        key = str(state.metadata.get("direction_key", "direction"))
        if key not in loaded.files:
            raise StateRuntimeError(f"Direction archive lacks key {key!r}")
        vector = np.asarray(loaded[key], dtype=np.float32)
        loaded.close()
    else:
        vector = np.asarray(loaded, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise StateRuntimeError("Activation direction must be a finite one-dimensional vector")
    return vector * float(state.steering_alpha)


__all__ = [
    "StateRuntimeError",
    "activated_state",
    "load_steering_addition",
    "messages_for_state",
]
