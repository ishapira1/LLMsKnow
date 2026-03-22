from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


MC_LETTERS = list("ABCDE")
MC_OPTION_COLUMNS = [f"P({letter})" for letter in MC_LETTERS]


def row_probability_values(frame: pd.DataFrame) -> np.ndarray:
    return frame[MC_OPTION_COLUMNS].to_numpy(dtype=float, copy=True)


def top1_top2_from_probs(prob_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(prob_values, axis=1)
    top1 = ordered[:, -1]
    top2 = ordered[:, -2]
    return top1, top2


def entropy_and_effective_responses(prob_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe_probs = np.clip(prob_values, 1e-12, 1.0)
    ent = -(safe_probs * np.log(safe_probs)).sum(axis=1)
    eff = np.exp(ent)
    return ent, eff


def correct_in_top_k(frame: pd.DataFrame, k: int) -> pd.Series:
    ranks = frame[MC_OPTION_COLUMNS].rank(axis=1, method="min", ascending=False)
    outputs = []
    for idx, correct_letter in frame["correct_letter"].astype(str).str.strip().str.upper().items():
        column = f"P({correct_letter})"
        if column not in ranks.columns:
            outputs.append(False)
            continue
        outputs.append(bool(ranks.loc[idx, column] <= k))
    return pd.Series(outputs, index=frame.index)


def probability_of_option(frame: pd.DataFrame, option_series: pd.Series) -> pd.Series:
    values = []
    option_series = option_series.astype(str).str.strip().str.upper()
    for idx, option in option_series.items():
        column = f"P({option})"
        values.append(frame.loc[idx, column] if column in frame.columns else np.nan)
    return pd.Series(values, index=frame.index, dtype=float)


def probability_rank(frame: pd.DataFrame, option_series: pd.Series) -> pd.Series:
    ranks = frame[MC_OPTION_COLUMNS].rank(axis=1, method="min", ascending=False)
    outputs = []
    option_series = option_series.astype(str).str.strip().str.upper()
    for idx, option in option_series.items():
        column = f"P({option})"
        outputs.append(ranks.loc[idx, column] if column in ranks.columns else np.nan)
    return pd.Series(outputs, index=frame.index, dtype=float)


def js_divergence(prob_p: np.ndarray, prob_q: np.ndarray) -> np.ndarray:
    safe_p = np.clip(prob_p, 1e-12, 1.0)
    safe_q = np.clip(prob_q, 1e-12, 1.0)
    midpoint = 0.5 * (safe_p + safe_q)
    kl_pm = np.sum(safe_p * (np.log(safe_p) - np.log(midpoint)), axis=1)
    kl_qm = np.sum(safe_q * (np.log(safe_q) - np.log(midpoint)), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def total_variation(prob_p: np.ndarray, prob_q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(prob_p - prob_q).sum(axis=1)


def reliability_curve(
    y_true: Sequence[float],
    y_score: Sequence[float],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["bin", "bin_left", "bin_right", "mean_score", "empirical_accuracy", "n"])
    frame["bin"] = pd.cut(frame["y_score"], bins=np.linspace(0.0, 1.0, n_bins + 1), include_lowest=True)
    summary = (
        frame.groupby("bin", observed=False)
        .agg(mean_score=("y_score", "mean"), empirical_accuracy=("y_true", "mean"), n=("y_true", "size"))
        .reset_index()
    )
    summary["bin_left"] = summary["bin"].map(lambda interval: float(interval.left) if pd.notna(interval) else np.nan)
    summary["bin_right"] = summary["bin"].map(lambda interval: float(interval.right) if pd.notna(interval) else np.nan)
    return summary


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic=np.mean,
    n_boot: int = 500,
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = rng.choice(array, size=len(array), replace=True)
        draws.append(float(statistic(sample)))
    alpha = (100.0 - ci) / 2.0
    return (float(np.percentile(draws, alpha)), float(np.percentile(draws, 100.0 - alpha)))


def quantile_bucket_labels(values: pd.Series, n_bins: int = 10) -> pd.Series:
    ranked = values.rank(method="first")
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    buckets = pd.qcut(ranked, q=n_bins, labels=labels)
    return pd.Series(pd.Categorical(buckets, categories=labels, ordered=True), index=values.index)


def score_rank(score_frame: pd.DataFrame, option_series: pd.Series, *, prefix: str = "score_") -> pd.Series:
    score_columns = [f"{prefix}{letter}" for letter in MC_LETTERS if f"{prefix}{letter}" in score_frame.columns]
    ranks = score_frame[score_columns].rank(axis=1, method="min", ascending=False)
    outputs = []
    option_series = option_series.astype(str).str.strip().str.upper()
    for idx, option in option_series.items():
        column = f"{prefix}{option}"
        outputs.append(ranks.loc[idx, column] if column in ranks.columns else np.nan)
    return pd.Series(outputs, index=score_frame.index, dtype=float)


def score_value(score_frame: pd.DataFrame, option_series: pd.Series, *, prefix: str = "score_") -> pd.Series:
    outputs = []
    option_series = option_series.astype(str).str.strip().str.upper()
    for idx, option in option_series.items():
        column = f"{prefix}{option}"
        outputs.append(score_frame.loc[idx, column] if column in score_frame.columns else np.nan)
    return pd.Series(outputs, index=score_frame.index, dtype=float)
