from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


def _probe_completion_text(record: Dict[str, Any]) -> str:
    response_raw = record.get('response_raw')
    if isinstance(response_raw, str):
        return response_raw
    response = record.get('response')
    if isinstance(response, str):
        return response
    return ''


def maybe_subsample(records: List[Dict[str, Any]], max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(records) <= max_samples:
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, max_samples)


__all__ = ['maybe_subsample']
