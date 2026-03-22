from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np


def filter_non_finite_feature_rows(
    features: Any,
    *aligned_arrays: Iterable[Any],
) -> Tuple[np.ndarray, ...]:
    feature_array = np.asarray(features, dtype=float)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(1, -1)
    if feature_array.ndim < 2:
        raise ValueError("feature arrays must have at least 2 dimensions after normalization")

    reduce_axes = tuple(range(1, feature_array.ndim))
    keep_mask = np.isfinite(feature_array).all(axis=reduce_axes)

    filtered = [feature_array[keep_mask], keep_mask]
    for array in aligned_arrays:
        filtered.append(np.asarray(array)[keep_mask])
    return tuple(filtered)


__all__ = ["filter_non_finite_feature_rows"]
