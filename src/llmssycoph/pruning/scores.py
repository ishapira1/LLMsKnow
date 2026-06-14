from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from tqdm.auto import tqdm

from ..logging_utils import log_status, tqdm_desc
from .losses import loss_for_example


def _import_torch():
    import torch

    return torch


def _excluded_parameter_name(name: str) -> bool:
    lowered = str(name).lower()
    excluded_fragments = ("embed", "embedding", "lm_head", "norm", "layernorm", "layer_norm")
    return any(fragment in lowered for fragment in excluded_fragments)


def collect_prunable_linear_weights(model: Any) -> Dict[str, Any]:
    torch = _import_torch()
    by_parameter_id: Dict[int, str] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = getattr(module, "weight", None)
            if weight is not None:
                by_parameter_id[id(weight)] = f"{module_name}.weight" if module_name else "weight"

    params: Dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if id(parameter) not in by_parameter_id:
            continue
        if _excluded_parameter_name(name):
            continue
        if not getattr(parameter, "requires_grad", False):
            continue
        if len(tuple(parameter.shape)) != 2:
            continue
        params[name] = parameter
    return params


def _zero_model_gradients(model: Any) -> None:
    zero_grad = getattr(model, "zero_grad", None)
    if callable(zero_grad):
        zero_grad(set_to_none=True)
        return
    for parameter in model.parameters():
        parameter.grad = None


def empty_score_like(prunable_params: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _import_torch()
    return {
        name: torch.zeros_like(parameter.detach(), dtype=torch.float32, device="cpu")
        for name, parameter in prunable_params.items()
    }


def score_weight_importance(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    *,
    desc: str,
) -> Dict[str, Any]:
    prunable_params = collect_prunable_linear_weights(model)
    scores = empty_score_like(prunable_params)
    if not examples:
        return scores

    log_status("pruning/scores.py", f"scoring {len(examples)} examples for {desc}")
    for example in tqdm(examples, desc=tqdm_desc("pruning/scores.py", desc), unit="example"):
        _zero_model_gradients(model)
        loss = loss_for_example(model, tokenizer, example)
        loss.backward()
        for name, parameter in prunable_params.items():
            grad = parameter.grad
            if grad is None:
                continue
            scores[name] += (parameter.detach().float().cpu() * grad.detach().float().cpu()) / float(len(examples))
    _zero_model_gradients(model)
    return scores


__all__ = ["collect_prunable_linear_weights", "empty_score_like", "score_weight_importance"]
