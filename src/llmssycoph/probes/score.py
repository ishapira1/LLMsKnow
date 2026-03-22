from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm

from ..logging_utils import log_status, tqdm_desc, warn_status
from .features import get_hidden_feature_for_completion as _get_hidden_feature_for_completion
from .records import _probe_completion_text


_LOG_SOURCE = 'probes/score.py'


def score_records_with_probe(
    model,
    tokenizer,
    records: List[Dict],
    clf,
    layer: Optional[int],
    score_key: str,
    desc: str,
) -> None:
    if clf is None or layer is None:
        for record in records:
            record[score_key] = np.nan
        return

    log_status(
        _LOG_SOURCE,
        f'scoring records for {desc}: records={len(records)} layer={layer} score_key={score_key}',
    )
    dropped_non_finite = 0
    for record in tqdm(
        records,
        desc=tqdm_desc(_LOG_SOURCE, f'{desc} scoring'),
        unit='record',
    ):
        x = _get_hidden_feature_for_completion(
            model,
            tokenizer,
            record['prompt_messages'],
            _probe_completion_text(record),
            layer=layer,
        )
        if not np.isfinite(np.asarray(x, dtype=float)).all():
            record[score_key] = np.nan
            dropped_non_finite += 1
            continue
        record[score_key] = float(clf.predict_proba(x.reshape(1, -1))[0, 1])
    if dropped_non_finite:
        warn_status(
            _LOG_SOURCE,
            "probe_scoring_non_finite_features",
            f"scoring for {desc} skipped non-finite feature rows at layer={layer}: "
            f"dropped={dropped_non_finite}/{len(records)} score_key={score_key}",
        )


__all__ = ['score_records_with_probe']
