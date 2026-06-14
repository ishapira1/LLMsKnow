from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional


def _import_torch():
    import torch

    return torch


@dataclass(frozen=True)
class MaskSelectionResult:
    masks: Dict[str, Any]
    requested_sparsity: float
    requested_count: int
    selected_count: int
    total_prunable_weights: int
    preserve_exclude_fraction: float


def _total_numel(tensors: Mapping[str, Any]) -> int:
    return int(sum(int(tensor.numel()) for tensor in tensors.values()))


def _eligible_mask_for_tensor(preserve_scores: Any, preserve_exclude_fraction: float):
    torch = _import_torch()
    if preserve_scores is None or preserve_exclude_fraction <= 0.0:
        return None
    flat = preserve_scores.detach().abs().flatten()
    n = int(flat.numel())
    if n <= 0:
        return torch.ones_like(preserve_scores, dtype=torch.bool, device="cpu")
    exclude_count = int(math.floor(n * float(preserve_exclude_fraction)))
    if exclude_count <= 0:
        return torch.ones_like(preserve_scores, dtype=torch.bool, device="cpu")
    exclude_count = min(exclude_count, n)
    eligible = torch.ones(n, dtype=torch.bool, device="cpu")
    excluded = torch.topk(flat.cpu(), k=exclude_count, largest=True).indices
    eligible[excluded] = False
    return eligible.reshape(tuple(preserve_scores.shape))


def _empty_index_masks_like(scores: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _import_torch()
    return {name: torch.empty(0, dtype=torch.long, device="cpu") for name in scores}


def _mask_indices(mask: Any, *, parameter: Any = None):
    torch = _import_torch()
    mask_cpu = mask.detach().cpu()
    if mask_cpu.dtype == torch.bool:
        return torch.nonzero(mask_cpu.flatten(), as_tuple=False).flatten()
    indices = mask_cpu.to(dtype=torch.long).flatten()
    if parameter is not None:
        n = int(parameter.numel())
        indices = indices[(indices >= 0) & (indices < n)]
    return indices


def select_pruning_mask(
    syc_scores: Mapping[str, Any],
    preserve_scores: Optional[Mapping[str, Any]],
    *,
    sparsity: float,
    preserve_exclude_fraction: float,
) -> MaskSelectionResult:
    torch = _import_torch()
    total = _total_numel(syc_scores)
    requested_count = int(math.floor(total * float(sparsity)))
    masks = _empty_index_masks_like(syc_scores)
    if requested_count <= 0 or total <= 0:
        return MaskSelectionResult(masks, float(sparsity), 0, 0, total, float(preserve_exclude_fraction))

    names = list(syc_scores.keys())
    best_scores = None
    best_tensor_indices = None
    best_flat_indices = None
    total_eligible = 0
    for tensor_index, name in enumerate(names):
        score = syc_scores[name]
        eligible = _eligible_mask_for_tensor(
            None if preserve_scores is None else preserve_scores.get(name),
            preserve_exclude_fraction,
        )
        if eligible is None:
            eligible = torch.ones_like(score, dtype=torch.bool, device="cpu")
        eligible = eligible & torch.isfinite(score.detach().cpu())
        eligible = eligible.flatten()
        eligible_count = int(eligible.sum().item())
        if eligible_count <= 0:
            continue
        total_eligible += eligible_count
        local_count = min(requested_count, eligible_count)
        flat_score = score.detach().cpu().flatten()
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        eligible_scores = flat_score.index_select(0, eligible_indices)
        chosen_local = torch.topk(-eligible_scores, k=local_count, largest=True).indices
        local_scores = eligible_scores.index_select(0, chosen_local)
        local_flat_indices = eligible_indices.index_select(0, chosen_local)
        local_tensor_indices = torch.full((local_count,), tensor_index, dtype=torch.long)

        if best_scores is None:
            merged_scores = local_scores
            merged_tensor_indices = local_tensor_indices
            merged_flat_indices = local_flat_indices
        else:
            merged_scores = torch.cat([best_scores, local_scores])
            merged_tensor_indices = torch.cat([best_tensor_indices, local_tensor_indices])
            merged_flat_indices = torch.cat([best_flat_indices, local_flat_indices])
        keep_count = min(requested_count, int(merged_scores.numel()))
        keep = torch.topk(-merged_scores, k=keep_count, largest=True).indices
        best_scores = merged_scores.index_select(0, keep)
        best_tensor_indices = merged_tensor_indices.index_select(0, keep)
        best_flat_indices = merged_flat_indices.index_select(0, keep)

    selected_count = 0
    if best_scores is not None and best_tensor_indices is not None and best_flat_indices is not None:
        for tensor_index, name in enumerate(names):
            local = best_flat_indices[best_tensor_indices == tensor_index]
            if local.numel():
                masks[name] = local.cpu().to(dtype=torch.long)
                selected_count += int(local.numel())

    if total_eligible <= 0:
        selected_count = 0
    return MaskSelectionResult(masks, float(sparsity), requested_count, selected_count, total, float(preserve_exclude_fraction))


def count_masked_weights(masks: Mapping[str, Any]) -> int:
    torch = _import_torch()
    count = 0
    for mask in masks.values():
        if mask.detach().cpu().dtype == torch.bool:
            count += int(mask.sum().item())
        else:
            count += int(mask.numel())
    return count


def apply_mask(model: Any, masks: Mapping[str, Any]) -> Dict[str, Any]:
    named_params = dict(model.named_parameters())
    originals: Dict[str, Any] = {}
    for name, mask in masks.items():
        parameter = named_params.get(name)
        if parameter is None:
            continue
        indices = _mask_indices(mask, parameter=parameter)
        if indices.numel() <= 0:
            continue
        device_indices = indices.to(device=parameter.device, dtype=_import_torch().long)
        flat_parameter = parameter.view(-1)
        originals[name] = flat_parameter.detach().index_select(0, device_indices).clone()
        with _import_torch().no_grad():
            flat_parameter[device_indices] = 0
    return originals


def restore_masked_values(model: Any, masks: Mapping[str, Any], originals: Mapping[str, Any]) -> None:
    named_params = dict(model.named_parameters())
    for name, values in originals.items():
        parameter = named_params.get(name)
        mask = masks.get(name)
        if parameter is None or mask is None:
            continue
        indices = _mask_indices(mask, parameter=parameter)
        if indices.numel() <= 0:
            continue
        device_indices = indices.to(device=parameter.device, dtype=_import_torch().long)
        flat_parameter = parameter.view(-1)
        with _import_torch().no_grad():
            flat_parameter[device_indices] = values.to(device=parameter.device, dtype=parameter.dtype)


def build_random_mask(prunable_params: Mapping[str, Any], *, count: int, seed: int) -> Dict[str, Any]:
    torch = _import_torch()
    masks = {name: torch.empty(0, dtype=torch.long, device="cpu") for name in prunable_params}
    total = _total_numel(prunable_params)
    count = max(0, min(int(count), total))
    if count <= 0:
        return masks
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    best_values = None
    best_tensor_indices = None
    best_flat_indices = None
    names = list(prunable_params.keys())
    for tensor_index, name in enumerate(names):
        n = int(prunable_params[name].numel())
        if n <= 0:
            continue
        local_count = min(count, n)
        values = torch.rand(n, generator=generator, dtype=torch.float32)
        chosen = torch.topk(values, k=local_count, largest=False).indices
        local_values = values.index_select(0, chosen)
        local_tensor_indices = torch.full((local_count,), tensor_index, dtype=torch.long)

        if best_values is None:
            merged_values = local_values
            merged_tensor_indices = local_tensor_indices
            merged_flat_indices = chosen
        else:
            merged_values = torch.cat([best_values, local_values])
            merged_tensor_indices = torch.cat([best_tensor_indices, local_tensor_indices])
            merged_flat_indices = torch.cat([best_flat_indices, chosen])
        keep_count = min(count, int(merged_values.numel()))
        keep = torch.topk(merged_values, k=keep_count, largest=False).indices
        best_values = merged_values.index_select(0, keep)
        best_tensor_indices = merged_tensor_indices.index_select(0, keep)
        best_flat_indices = merged_flat_indices.index_select(0, keep)

    if best_tensor_indices is not None and best_flat_indices is not None:
        for tensor_index, name in enumerate(names):
            local = best_flat_indices[best_tensor_indices == tensor_index]
            if local.numel():
                masks[name] = local.cpu().to(dtype=torch.long)
    return masks


def build_magnitude_mask(prunable_params: Mapping[str, Any], *, count: int) -> Dict[str, Any]:
    torch = _import_torch()
    masks = {name: torch.empty(0, dtype=torch.long, device="cpu") for name in prunable_params}
    total = _total_numel(prunable_params)
    count = max(0, min(int(count), total))
    if count <= 0:
        return masks
    best_values = None
    best_tensor_indices = None
    best_flat_indices = None
    names = list(prunable_params.keys())
    for tensor_index, name in enumerate(names):
        param = prunable_params[name]
        n = int(param.numel())
        if n <= 0:
            continue
        local_count = min(count, n)
        values = param.detach().abs().float().cpu().flatten()
        chosen = torch.topk(values, k=local_count, largest=False).indices
        local_values = values.index_select(0, chosen)
        local_tensor_indices = torch.full((local_count,), tensor_index, dtype=torch.long)

        if best_values is None:
            merged_values = local_values
            merged_tensor_indices = local_tensor_indices
            merged_flat_indices = chosen
        else:
            merged_values = torch.cat([best_values, local_values])
            merged_tensor_indices = torch.cat([best_tensor_indices, local_tensor_indices])
            merged_flat_indices = torch.cat([best_flat_indices, chosen])
        keep_count = min(count, int(merged_values.numel()))
        keep = torch.topk(merged_values, k=keep_count, largest=False).indices
        best_values = merged_values.index_select(0, keep)
        best_tensor_indices = merged_tensor_indices.index_select(0, keep)
        best_flat_indices = merged_flat_indices.index_select(0, keep)

    if best_tensor_indices is not None and best_flat_indices is not None:
        for tensor_index, name in enumerate(names):
            local = best_flat_indices[best_tensor_indices == tensor_index]
            if local.numel():
                masks[name] = local.cpu().to(dtype=torch.long)
    return masks


__all__ = [
    "MaskSelectionResult",
    "apply_mask",
    "build_magnitude_mask",
    "build_random_mask",
    "count_masked_weights",
    "restore_masked_values",
    "select_pruning_mask",
]
