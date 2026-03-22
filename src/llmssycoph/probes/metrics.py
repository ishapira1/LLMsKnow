from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from ..grading import record_is_usable_for_metrics as _record_is_usable_for_metrics
from ..logging_utils import log_status, tqdm_desc, warn_status
from .finite import filter_non_finite_feature_rows
from .features import get_hidden_feature_all_layers_for_completion
from .records import _probe_completion_text


_LOG_SOURCE = "probes/metrics.py"
PROBE_METRIC_SCHEMA_VERSION = 1
PROBE_METADATA_SCHEMA_VERSION = 1
PROBE_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_PROBE_THRESHOLD = 0.5
PROBE_METRIC_NAMES = [
    "accuracy",
    "accuracy_label_1",
    "accuracy_label_0",
    "true_label_accuracy",
    "false_label_accuracy",
    "balanced_accuracy",
    "auc",
    "positive_rate",
    "predicted_positive_rate",
    "tp",
    "tn",
    "fp",
    "fn",
    "n_total",
    "n_label_1",
    "n_label_0",
]


def filter_usable_probe_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if _record_is_usable_for_metrics(record)]


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _dataset_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    totals: Counter[str] = Counter()
    label_true: Counter[str] = Counter()
    label_false: Counter[str] = Counter()
    for record in records:
        dataset_name = str(record.get("dataset", "") or "")
        totals[dataset_name] += 1
        correctness = _safe_int(record.get("correctness"))
        if correctness == 1:
            label_true[dataset_name] += 1
        elif correctness == 0:
            label_false[dataset_name] += 1

    return {
        dataset_name: {
            "n_records": int(totals[dataset_name]),
            "n_label_1": int(label_true[dataset_name]),
            "n_label_0": int(label_false[dataset_name]),
        }
        for dataset_name in sorted(totals)
    }


def summarize_probe_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    all_records = list(records)
    usable_records = filter_usable_probe_records(all_records)
    labels = [_safe_int(record.get("correctness")) for record in usable_records]
    label_ones = sum(1 for label in labels if label == 1)
    label_zeros = sum(1 for label in labels if label == 0)
    question_ids = {
        str(record.get("question_id", "") or "")
        for record in all_records
        if str(record.get("question_id", "") or "")
    }
    usable_question_ids = {
        str(record.get("question_id", "") or "")
        for record in usable_records
        if str(record.get("question_id", "") or "")
    }
    usable_weights = [
        weight
        for record in usable_records
        for weight in [_safe_float(record.get("probe_sample_weight", 1.0))]
        if weight is not None
    ]
    return {
        "n_records": len(all_records),
        "n_usable_records": len(usable_records),
        "n_label_1": int(label_ones),
        "n_label_0": int(label_zeros),
        "n_questions": len(question_ids),
        "n_usable_questions": len(usable_question_ids),
        "sum_sample_weight": float(np.sum(usable_weights)) if usable_weights else 0.0,
        "mean_sample_weight": float(np.mean(usable_weights)) if usable_weights else 0.0,
        "datasets": _dataset_summary(usable_records),
    }


def build_split_data_summary(split_records: Mapping[str, Sequence[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    return {
        split_name: summarize_probe_records(split_records.get(split_name, []))
        for split_name in ("train", "val", "test")
    }


def _empty_metric_block() -> Dict[str, Optional[float] | int]:
    return {
        "accuracy": None,
        "accuracy_label_1": None,
        "accuracy_label_0": None,
        "true_label_accuracy": None,
        "false_label_accuracy": None,
        "balanced_accuracy": None,
        "auc": None,
        "positive_rate": None,
        "predicted_positive_rate": None,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "n_total": 0,
        "n_label_1": 0,
        "n_label_0": 0,
    }


def compute_binary_probe_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = DEFAULT_PROBE_THRESHOLD,
) -> Dict[str, Optional[float] | int]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if labels.size == 0:
        return _empty_metric_block()

    preds = (probs >= threshold).astype(int)
    label_one_mask = labels == 1
    label_zero_mask = labels == 0

    tp = int(np.sum((preds == 1) & label_one_mask))
    tn = int(np.sum((preds == 0) & label_zero_mask))
    fp = int(np.sum((preds == 1) & label_zero_mask))
    fn = int(np.sum((preds == 0) & label_one_mask))

    accuracy = float(np.mean(preds == labels))
    positive_acc = float(np.mean(preds[label_one_mask] == 1)) if np.any(label_one_mask) else None
    negative_acc = float(np.mean(preds[label_zero_mask] == 0)) if np.any(label_zero_mask) else None
    balanced_accuracy = None
    if positive_acc is not None and negative_acc is not None:
        balanced_accuracy = float((positive_acc + negative_acc) / 2.0)

    auc = None
    if len(np.unique(labels)) >= 2:
        auc = float(roc_auc_score(labels, probs))

    return {
        "accuracy": accuracy,
        "accuracy_label_1": positive_acc,
        "accuracy_label_0": negative_acc,
        "true_label_accuracy": positive_acc,
        "false_label_accuracy": negative_acc,
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
        "positive_rate": float(np.mean(labels)),
        "predicted_positive_rate": float(np.mean(preds)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_total": int(labels.size),
        "n_label_1": int(np.sum(label_one_mask)),
        "n_label_0": int(np.sum(label_zero_mask)),
    }


def prepare_probe_eval_cache(
    model,
    tokenizer,
    split_records: Mapping[str, Sequence[Dict[str, Any]]],
    layer_grid: Sequence[int],
    desc: str,
) -> Dict[str, Any]:
    ordered_layers = [int(layer) for layer in layer_grid]
    cache: Dict[str, Any] = {
        "layer_grid": ordered_layers,
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        usable_records = filter_usable_probe_records(split_records.get(split_name, []))
        labels = np.array([int(record["correctness"]) for record in usable_records], dtype=int)
        if usable_records:
            features = []
            for record in tqdm(
                usable_records,
                desc=tqdm_desc(_LOG_SOURCE, f"{desc} {split_name} all-layer eval features"),
                unit="record",
            ):
                features.append(
                    get_hidden_feature_all_layers_for_completion(
                        model,
                        tokenizer,
                        record["prompt_messages"],
                        _probe_completion_text(record),
                        layer_grid=ordered_layers,
                    )
                )
            feature_tensor = np.stack(features)
        else:
            feature_tensor = None

        log_status(
            _LOG_SOURCE,
            f"prepared probe eval cache for {desc}: split={split_name} usable_records={len(usable_records)}",
        )
        cache["splits"][split_name] = {
            "records": usable_records,
            "labels": labels,
            "features": feature_tensor,
        }
    return cache


def evaluate_probe_from_cache(
    cache: Mapping[str, Any],
    clf,
    layer: Optional[int],
    threshold: float = DEFAULT_PROBE_THRESHOLD,
) -> Dict[str, Any]:
    layer_grid = [int(layer_id) for layer_id in cache.get("layer_grid", [])]
    layer_to_index = {layer_id: idx for idx, layer_id in enumerate(layer_grid)}
    metrics: Dict[str, Any] = {
        "metric_schema_version": PROBE_METRIC_SCHEMA_VERSION,
        "metric_names": list(PROBE_METRIC_NAMES),
        "threshold": float(threshold),
        "evaluated_layer": layer,
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        split_payload = cache.get("splits", {}).get(split_name, {})
        labels = np.asarray(split_payload.get("labels", np.array([], dtype=int)), dtype=int)
        features = split_payload.get("features")
        if clf is None or layer is None or layer not in layer_to_index or features is None or labels.size == 0:
            metrics["splits"][split_name] = _empty_metric_block()
            continue

        layer_idx = layer_to_index[int(layer)]
        split_features, keep_mask, split_labels = filter_non_finite_feature_rows(
            features[:, layer_idx, :],
            labels,
        )
        dropped = int((~keep_mask).sum())
        if dropped:
            warn_status(
                _LOG_SOURCE,
                "probe_eval_non_finite_features",
                f"probe evaluation for layer={layer} split={split_name} dropped non-finite feature rows: "
                f"dropped={dropped}/{len(keep_mask)}",
            )
        if split_labels.size == 0:
            metrics["splits"][split_name] = _empty_metric_block()
            continue
        probs = clf.predict_proba(split_features)[:, 1]
        metrics["splits"][split_name] = compute_binary_probe_metrics(split_labels, probs, threshold=threshold)

    return metrics


def probe_model_metadata(clf) -> Dict[str, Any]:
    if clf is None:
        return {
            "model_type": None,
            "input_dim": None,
            "coef_shape": None,
            "classes": [],
            "intercept_shape": None,
        }

    coef = getattr(clf, "coef_", None)
    intercept = getattr(clf, "intercept_", None)
    classes = getattr(clf, "classes_", [])
    coef_shape = list(coef.shape) if hasattr(coef, "shape") else None
    intercept_shape = list(intercept.shape) if hasattr(intercept, "shape") else None
    return {
        "model_type": type(clf).__name__,
        "input_dim": _safe_int(getattr(clf, "n_features_in_", None)),
        "coef_shape": coef_shape,
        "classes": [int(cls) for cls in classes] if classes is not None else [],
        "intercept_shape": intercept_shape,
    }


def add_path_fields(payload: MutableMapping[str, Any], **paths: Path) -> MutableMapping[str, Any]:
    path_payload = payload.setdefault("paths", {})
    for key, value in paths.items():
        path_payload[key] = str(value)
    return payload


__all__ = [
    "DEFAULT_PROBE_THRESHOLD",
    "PROBE_ARTIFACT_SCHEMA_VERSION",
    "PROBE_METADATA_SCHEMA_VERSION",
    "PROBE_METRIC_NAMES",
    "PROBE_METRIC_SCHEMA_VERSION",
    "add_path_fields",
    "build_split_data_summary",
    "compute_binary_probe_metrics",
    "evaluate_probe_from_cache",
    "filter_usable_probe_records",
    "prepare_probe_eval_cache",
    "probe_model_metadata",
    "summarize_probe_records",
]
