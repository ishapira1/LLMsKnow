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


def _empty_masks_like(scores: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _import_torch()
    return {name: torch.zeros_like(score, dtype=torch.bool, device="cpu") for name, score in scores.items()}


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
    masks = _empty_masks_like(syc_scores)
    if requested_count <= 0 or total <= 0:
        return MaskSelectionResult(masks, float(sparsity), 0, 0, total, float(preserve_exclude_fraction))

    eligible_counts: Dict[str, int] = {}
    total_eligible = 0
    eligible_masks: Dict[str, Any] = {}
    for name, score in syc_scores.items():
        eligible = _eligible_mask_for_tensor(
            None if preserve_scores is None else preserve_scores.get(name),
            preserve_exclude_fraction,
        )
        if eligible is None:
            eligible = torch.ones_like(score, dtype=torch.bool, device="cpu")
        eligible = eligible & torch.isfinite(score.detach().cpu())
        count = int(eligible.sum().item())
        eligible_masks[name] = eligible
        eligible_counts[name] = count
        total_eligible += count

    if total_eligible <= 0:
        return MaskSelectionResult(masks, float(sparsity), requested_count, 0, total, float(preserve_exclude_fraction))

    best_scores = None
    best_tensor_indices = None
    best_flat_indices = None
    names = list(syc_scores.keys())
    for tensor_index, name in enumerate(names):
        score = syc_scores[name]
        eligible = eligible_masks[name].flatten()
        eligible_count = eligible_counts[name]
        if eligible_count <= 0:
            continue
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
                masks[name].flatten()[local] = True
                selected_count += int(local.numel())

    return MaskSelectionResult(masks, float(sparsity), requested_count, selected_count, total, float(preserve_exclude_fraction))


def count_masked_weights(masks: Mapping[str, Any]) -> int:
    return int(sum(int(mask.sum().item()) for mask in masks.values()))


def apply_mask(model: Any, masks: Mapping[str, Any]) -> Dict[str, Any]:
    named_params = dict(model.named_parameters())
    originals: Dict[str, Any] = {}
    for name, mask in masks.items():
        parameter = named_params.get(name)
        if parameter is None:
            continue
        device_mask = mask.to(device=parameter.device, dtype=bool)
        originals[name] = parameter.detach()[device_mask].clone()
        with _import_torch().no_grad():
            parameter[device_mask] = 0
    return originals


def restore_masked_values(model: Any, masks: Mapping[str, Any], originals: Mapping[str, Any]) -> None:
    named_params = dict(model.named_parameters())
    for name, values in originals.items():
        parameter = named_params.get(name)
        mask = masks.get(name)
        if parameter is None or mask is None:
            continue
        device_mask = mask.to(device=parameter.device, dtype=bool)
        with _import_torch().no_grad():
            parameter[device_mask] = values.to(device=parameter.device, dtype=parameter.dtype)


def build_random_mask(prunable_params: Mapping[str, Any], *, count: int, seed: int) -> Dict[str, Any]:
    torch = _import_torch()
    masks = {name: torch.zeros_like(param.detach(), dtype=torch.bool, device="cpu") for name, param in prunable_params.items()}
    total = _total_numel(prunable_params)
    count = max(0, min(int(count), total))
    if count <= 0:
        return masks
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    chosen = torch.randperm(total, generator=generator)[:count]
    offset = 0
    for name, param in prunable_params.items():
        n = int(param.numel())
        local = chosen[(chosen >= offset) & (chosen < offset + n)] - offset
        if local.numel():
            masks[name].flatten()[local] = True
        offset += n
    return masks


def build_magnitude_mask(prunable_params: Mapping[str, Any], *, count: int) -> Dict[str, Any]:
    torch = _import_torch()
    masks = {name: torch.zeros_like(param.detach(), dtype=torch.bool, device="cpu") for name, param in prunable_params.items()}
    total = _total_numel(prunable_params)
    count = max(0, min(int(count), total))
    if count <= 0:
        return masks
    values = torch.cat([param.detach().abs().float().cpu().flatten() for param in prunable_params.values()])
    chosen = torch.topk(-values, k=count, largest=True).indices
    offset = 0
    for name, param in prunable_params.items():
        n = int(param.numel())
        local = chosen[(chosen >= offset) & (chosen < offset + n)] - offset
        if local.numel():
            masks[name].flatten()[local] = True
        offset += n
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
