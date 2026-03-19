from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from ..correctness import record_is_usable_for_metrics as _record_is_usable_for_metrics
from ..logging_utils import log_status, tqdm_desc
from .features import get_hidden_feature_all_layers_for_completion
from .records import _probe_completion_text, maybe_subsample


_LOG_SOURCE = 'probes/select_layer.py'


def select_best_layer_by_auc(
    model,
    tokenizer,
    train_records: List[Dict],
    val_records: List[Dict],
    layer_grid: Sequence[int],
    seed: int,
    max_selection_samples: Optional[int],
    desc: str,
) -> Tuple[
    Optional[int],
    Optional[float],
    Dict[int, Optional[float]],
    Dict[int, Optional[LogisticRegression]],
]:
    train_records = [record for record in train_records if _record_is_usable_for_metrics(record)]
    val_records = [record for record in val_records if _record_is_usable_for_metrics(record)]
    train_records = maybe_subsample(train_records, max_selection_samples, seed)
    val_records = maybe_subsample(val_records, max_selection_samples, seed + 1)
    log_status(
        _LOG_SOURCE,
        f'layer selection for {desc}: train_records={len(train_records)} '
        f'val_records={len(val_records)} layers={len(layer_grid)}',
    )

    if len(train_records) < 2 or len(val_records) < 2:
        log_status(
            _LOG_SOURCE,
            f'skipping layer selection for {desc}: too few samples '
            f'train={len(train_records)} val={len(val_records)}',
        )
        return (
            None,
            None,
            {layer: None for layer in layer_grid},
            {layer: None for layer in layer_grid},
        )

    y_train = np.array([int(record['correctness']) for record in train_records], dtype=int)
    y_val = np.array([int(record['correctness']) for record in val_records], dtype=int)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        log_status(
            _LOG_SOURCE,
            f'skipping layer selection for {desc}: train or val split has only one class',
        )
        return (
            None,
            None,
            {layer: None for layer in layer_grid},
            {layer: None for layer in layer_grid},
        )

    train_features: List[np.ndarray] = []
    for record in tqdm(
        train_records,
        desc=tqdm_desc(_LOG_SOURCE, f'{desc} train all-layer features'),
        unit='record',
    ):
        train_features.append(
            get_hidden_feature_all_layers_for_completion(
                model,
                tokenizer,
                record['prompt_messages'],
                _probe_completion_text(record),
                layer_grid=layer_grid,
            )
        )

    val_features: List[np.ndarray] = []
    for record in tqdm(
        val_records,
        desc=tqdm_desc(_LOG_SOURCE, f'{desc} val all-layer features'),
        unit='record',
    ):
        val_features.append(
            get_hidden_feature_all_layers_for_completion(
                model,
                tokenizer,
                record['prompt_messages'],
                _probe_completion_text(record),
                layer_grid=layer_grid,
            )
        )

    auc_per_layer: Dict[int, Optional[float]] = {}
    clf_per_layer: Dict[int, Optional[LogisticRegression]] = {}
    best_layer = None
    best_auc = -1.0

    for li, layer in enumerate(
        tqdm(layer_grid, desc=tqdm_desc(_LOG_SOURCE, f'{desc} layer selection'), unit='layer')
    ):
        X_train = np.stack([mat[li] for mat in train_features])
        X_val = np.stack([mat[li] for mat in val_features])
        try:
            clf = LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed, solver='liblinear')
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
            clf_per_layer[layer] = clf
        except Exception:
            auc = None
            clf_per_layer[layer] = None
        auc_per_layer[layer] = auc
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_layer = layer

    if best_layer is None:
        log_status(_LOG_SOURCE, f'no valid layer selected for {desc}')
        return None, None, auc_per_layer, clf_per_layer

    log_status(_LOG_SOURCE, f'selected best layer for {desc}: layer={best_layer} dev_auc={best_auc:.4f}')
    return best_layer, best_auc, auc_per_layer, clf_per_layer


__all__ = ['select_best_layer_by_auc']
