from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from ..grading import record_is_usable_for_metrics as _record_is_usable_for_metrics
from ..logging_utils import log_status, tqdm_desc, warn_status
from .features import get_hidden_feature_for_completion as _get_hidden_feature_for_completion
from .records import _probe_completion_text, maybe_subsample


_LOG_SOURCE = 'probes/train.py'


def _record_weight(record: Dict) -> float:
    try:
        weight = float(record.get("probe_sample_weight", 1.0) or 1.0)
    except Exception:
        return 1.0
    if not math.isfinite(weight):
        return 1.0
    return max(weight, 1e-6)


def train_probe_for_layer(
    model,
    tokenizer,
    records: List[Dict],
    layer: int,
    seed: int,
    max_train_samples: Optional[int],
    desc: str,
) -> Optional[LogisticRegression]:
    records = [record for record in records if _record_is_usable_for_metrics(record)]
    records = maybe_subsample(records, max_train_samples, seed)
    log_status(
        _LOG_SOURCE,
        f'training probe for {desc}: records={len(records)} layer={layer}',
    )
    if len(records) < 2:
        warn_status(
            _LOG_SOURCE,
            "probe_training_too_few_samples",
            f'skipping training for {desc}: too few train samples={len(records)}',
        )
        return None

    y = np.array([int(record['correctness']) for record in records], dtype=int)
    sample_weight = np.array([_record_weight(record) for record in records], dtype=float)
    if len(np.unique(y)) < 2:
        warn_status(
            _LOG_SOURCE,
            "probe_training_single_class",
            f'skipping training for {desc}: only one class in training data',
        )
        return None

    X = []
    for record in tqdm(
        records,
        desc=tqdm_desc(_LOG_SOURCE, f'{desc} layer-{layer} features'),
        unit='record',
    ):
        X.append(
            _get_hidden_feature_for_completion(
                model,
                tokenizer,
                record['prompt_messages'],
                _probe_completion_text(record),
                layer=layer,
            )
        )
    X = np.stack(X)

    clf = LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed, solver='liblinear')
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


__all__ = ['train_probe_for_layer']
