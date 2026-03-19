from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm

from ..logging_utils import log_status, tqdm_desc
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
        record[score_key] = float(clf.predict_proba(x.reshape(1, -1))[0, 1])


__all__ = ['score_records_with_probe']
