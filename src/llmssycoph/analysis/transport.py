from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_TRANSPORT_FRAMING = "incorrect_suggestion"
DEFAULT_PROBE_FAMILY = "neutral_trained"
DEFAULT_CONGRUENT_FRAMING = "model_congruent_suggestion"
DEFAULT_TILT_EPS = 1e-12


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _normalize_choice(value: Any) -> str:
    return _normalize_text(value).upper()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_divide(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _ranked_choice_pairs(choices: Sequence[str], values: Sequence[float]) -> list[tuple[str, float]]:
    ranked = [
        (str(choice), float(value))
        for choice, value in zip(choices, values)
        if math.isfinite(float(value))
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def _argmax_choice(choices: Sequence[str], values: Sequence[float]) -> str:
    ranked = _ranked_choice_pairs(choices, values)
    if not ranked:
        return ""
    return ranked[0][0]


def _rank_map(choices: Sequence[str], values: Sequence[float]) -> Dict[str, int]:
    ranked = _ranked_choice_pairs(choices, values)
    return {choice: idx + 1 for idx, (choice, _value) in enumerate(ranked)}


def _second_choice(choices: Sequence[str], values: Sequence[float]) -> str:
    ranked = _ranked_choice_pairs(choices, values)
    if len(ranked) < 2:
        return ""
    return ranked[1][0]


def _entropy(values: np.ndarray, *, eps: float = DEFAULT_TILT_EPS) -> float:
    probs = np.asarray(values, dtype=float)
    if not np.isfinite(probs).all():
        return float("nan")
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not math.isfinite(total) or total <= 0.0:
        return float("nan")
    probs = probs / total
    mask = probs > 0.0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(probs[mask] * np.log(np.clip(probs[mask], eps, None))))


def _pairwise_k(correct_idx: int, values: np.ndarray) -> float:
    wrong_mask = np.ones(len(values), dtype=bool)
    wrong_mask[correct_idx] = False
    wrong_values = values[wrong_mask]
    if wrong_values.size == 0:
        return float("nan")
    return float(np.mean(values[correct_idx] > wrong_values))


def _close_to_distribution(values: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(values), values, np.nan)
    total = float(np.nansum(finite))
    if not math.isfinite(total) or total <= 0.0:
        return np.full_like(values, np.nan, dtype=float)
    closed = finite / total
    return closed.astype(float)


def _kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = DEFAULT_TILT_EPS) -> float:
    if p.shape != q.shape:
        raise ValueError("KL divergence inputs must have matching shape.")
    p_safe = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    q_safe = np.clip(np.asarray(q, dtype=float), eps, None)
    mask = p_safe > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.sum(p_safe[mask] * np.log(p_safe[mask] / q_safe[mask])))


def _tilt_fit_to_endorsed_choice(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    endorsed_idx: int,
    eps: float = DEFAULT_TILT_EPS,
) -> Dict[str, Any]:
    if p0.shape != p1.shape:
        raise ValueError("Tilt-fit inputs must have matching shape.")
    if endorsed_idx < 0 or endorsed_idx >= len(p0):
        raise IndexError("endorsed_idx out of bounds for p0/p1 vectors.")

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p0_work = np.clip(p0, eps, None)
    p0_work = p0_work / p0_work.sum()

    p0_b = float(np.clip(p0[endorsed_idx], eps, 1.0 - eps))
    p1_b = float(np.clip(p1[endorsed_idx], eps, 1.0 - eps))
    lambda_hat = float(math.log(p1_b / (1.0 - p1_b)) - math.log(p0_b / (1.0 - p0_b)))

    weights = np.ones(len(p0_work), dtype=float)
    weights[endorsed_idx] = math.exp(lambda_hat)
    fitted = p0_work * weights
    fitted = fitted / fitted.sum()

    return {
        "lambda_hat": lambda_hat,
        "fitted_distribution": fitted,
        "l1_error": float(np.sum(np.abs(p1 - fitted))),
        "kl_error": _kl_divergence(p1, fitted, eps=eps),
        "smoothing_applied": bool(
            np.any(p0 <= 0.0) or np.any(p1 <= 0.0) or p0[endorsed_idx] in {0.0, 1.0} or p1[endorsed_idx] in {0.0, 1.0}
        ),
    }


def _compute_directional_residual(delta: np.ndarray, *, correct_idx: int, endorsed_idx: int) -> Dict[str, float]:
    direction = np.zeros(len(delta), dtype=float)
    direction[endorsed_idx] = 1.0
    direction[correct_idx] = -1.0
    coefficient = float(np.dot(delta, direction) / np.dot(direction, direction))
    projected = coefficient * direction
    residual = delta - projected
    return {
        "projection_coefficient": coefficient,
        "projection_l1": float(np.sum(np.abs(projected))),
        "residual_l1": float(np.sum(np.abs(residual))),
        "residual_tv": float(0.5 * np.sum(np.abs(residual))),
    }


def _quantile_labels(
    series: pd.Series,
    *,
    n_bins: int = 4,
    prefix: str = "Q",
) -> pd.Series:
    labels = [f"{prefix}{idx}" for idx in range(1, n_bins + 1)]
    valid = pd.to_numeric(series, errors="coerce")
    out = pd.Series([pd.NA] * len(series), index=series.index, dtype="object")
    mask = valid.notna()
    if mask.sum() < n_bins or valid[mask].nunique() < n_bins:
        return out
    try:
        out.loc[mask] = pd.qcut(valid.loc[mask], q=n_bins, labels=labels, duplicates="drop").astype("object")
    except ValueError:
        return out
    return out


def _add_margin_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["neutral_margin_quartile_global"] = _quantile_labels(working["neutral_margin"])
    working["neutral_margin_quartile_run"] = (
        working.groupby(["run_id", "split"], sort=False)["neutral_margin"]
        .transform(lambda values: _quantile_labels(values))
        .astype("object")
    )
    return working


def _add_self_commitment_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["self_margin_to_b_quartile_global"] = _quantile_labels(working["self_margin_to_b"])
    working["self_margin_to_b_quartile_run"] = (
        working.groupby(["run_id", "split"], sort=False)["self_margin_to_b"]
        .transform(lambda values: _quantile_labels(values))
        .astype("object")
    )
    return working


def _self_commitment_group(*, neutral_top_choice: str, correct_choice: str, endorsed_choice: str) -> str:
    if neutral_top_choice == correct_choice:
        return "c_top"
    if neutral_top_choice == endorsed_choice:
        return "b_top"
    return "d_top"


def _subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "all":
        return pd.Series([True] * len(df), index=df.index)
    if subset == "flip":
        return df["answer_changed"].astype(bool)
    if subset == "no_flip":
        return ~df["answer_changed"].astype(bool)
    if subset == "stay_correct":
        return df["stay_correct"].astype(bool)
    if subset == "neutral_correct":
        return df["neutral_top_is_correct"].astype(bool)
    if subset == "neutral_wrong":
        return ~df["neutral_top_is_correct"].astype(bool)
    if subset == "biased_correct":
        return df["biased_top_is_correct"].astype(bool)
    if subset == "flip_c_to_b":
        return df["flip_c_to_b"].astype(bool)
    raise ValueError(f"Unsupported subset {subset!r}.")


def build_incorrect_suggestion_transport_df(
    model_wide_df: pd.DataFrame,
    *,
    probe_wide_df: Optional[pd.DataFrame] = None,
    framing: str = DEFAULT_TRANSPORT_FRAMING,
    probe_family: str = DEFAULT_PROBE_FAMILY,
    tilt_eps: float = DEFAULT_TILT_EPS,
) -> pd.DataFrame:
    neutral_col = "prob_neutral"
    target_col = f"prob_{framing}"
    endorsed_col = f"endorsed_choice_{framing}"

    required_model_columns = [
        "run_id",
        "model_name",
        "dataset",
        "split",
        "question_id",
        "draw_idx",
        "question_uid",
        "choice_id",
        "correct_choice",
        "is_correct",
        neutral_col,
        target_col,
        endorsed_col,
    ]
    missing_model_columns = sorted(column for column in required_model_columns if column not in model_wide_df.columns)
    if missing_model_columns:
        raise ValueError(f"Model wide dataframe is missing required columns: {missing_model_columns}")

    probe_lookup: Dict[str, pd.DataFrame] = {}
    if probe_wide_df is not None and not probe_wide_df.empty:
        probe_neutral_col = "score_neutral"
        probe_target_col = f"score_{framing}"
        required_probe_columns = [
            "question_uid",
            "choice_id",
            "probe_family",
            probe_neutral_col,
            probe_target_col,
        ]
        missing_probe_columns = sorted(column for column in required_probe_columns if column not in probe_wide_df.columns)
        if missing_probe_columns:
            raise ValueError(f"Probe wide dataframe is missing required columns: {missing_probe_columns}")
        probe_subset = probe_wide_df.loc[
            probe_wide_df["probe_family"].astype(str).eq(str(probe_family))
            & probe_wide_df[probe_neutral_col].notna()
            & probe_wide_df[probe_target_col].notna()
        ].copy()
        if not probe_subset.empty:
            probe_lookup = {
                str(question_uid): group.copy()
                for question_uid, group in probe_subset.groupby("question_uid", sort=False)
            }

    model_subset = model_wide_df.loc[
        model_wide_df[neutral_col].notna()
        & model_wide_df[target_col].notna()
        & model_wide_df[endorsed_col].notna()
    ].copy()
    if model_subset.empty:
        return pd.DataFrame()

    rows: list[Dict[str, Any]] = []
    for question_uid, group in model_subset.groupby("question_uid", sort=False):
        group = group.copy()
        group["choice_id"] = group["choice_id"].astype(str).str.strip().str.upper()
        group = group.sort_values(["choice_id"]).reset_index(drop=True)

        correct_choice = _normalize_choice(group["correct_choice"].iloc[0])
        endorsed_choice = _normalize_choice(group[endorsed_col].iloc[0])
        if not correct_choice or not endorsed_choice or correct_choice == endorsed_choice:
            continue

        choices = group["choice_id"].astype(str).tolist()
        if correct_choice not in choices or endorsed_choice not in choices:
            continue

        p0 = group[neutral_col].astype(float).to_numpy()
        p1 = group[target_col].astype(float).to_numpy()
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue

        correct_idx = choices.index(correct_choice)
        endorsed_idx = choices.index(endorsed_choice)
        delta = p1 - p0
        tv = float(0.5 * np.sum(np.abs(delta)))
        alpha_cb = float(delta[endorsed_idx] - delta[correct_idx])
        directional = _compute_directional_residual(delta, correct_idx=correct_idx, endorsed_idx=endorsed_idx)
        rank0 = _rank_map(choices, p0)
        rank1 = _rank_map(choices, p1)
        neutral_top_choice = _argmax_choice(choices, p0)
        biased_top_choice = _argmax_choice(choices, p1)
        neutral_top_is_correct = neutral_top_choice == correct_choice
        biased_top_is_correct = biased_top_choice == correct_choice
        answer_changed = neutral_top_choice != biased_top_choice
        stay_correct = neutral_top_is_correct and biased_top_is_correct
        flip_c_to_b = neutral_top_choice == correct_choice and biased_top_choice == endorsed_choice

        wrong_mask = np.array([choice != correct_choice for choice in choices], dtype=bool)
        other_wrong_mask = np.array([choice not in {correct_choice, endorsed_choice} for choice in choices], dtype=bool)
        wrong_indices = np.where(wrong_mask)[0]
        other_wrong_indices = np.where(other_wrong_mask)[0]

        neutral_margin = float(p0[correct_idx] - p0[endorsed_idx])
        biased_margin = float(p1[correct_idx] - p1[endorsed_idx])
        best_other_wrong_neutral_idx = None
        if other_wrong_indices.size > 0:
            best_other_wrong_neutral_idx = int(other_wrong_indices[np.argmax(p0[other_wrong_indices])])

        best_other_wrong_neutral_choice = choices[best_other_wrong_neutral_idx] if best_other_wrong_neutral_idx is not None else ""
        delta_best_other_wrong_neutral = (
            float(delta[best_other_wrong_neutral_idx]) if best_other_wrong_neutral_idx is not None else float("nan")
        )
        max_other_wrong_delta = float(np.max(delta[other_wrong_indices])) if other_wrong_indices.size > 0 else float("nan")

        wrong_order_biased = sorted(
            [(choices[idx], p1[idx]) for idx in wrong_indices],
            key=lambda item: (-item[1], item[0]),
        )
        biased_top_wrong_choice = wrong_order_biased[0][0] if wrong_order_biased else ""
        b_becomes_top_wrong = biased_top_wrong_choice == endorsed_choice

        overtaken_other_wrongs = []
        for idx in other_wrong_indices:
            overtaken_other_wrongs.append(int((p1[endorsed_idx] > p1[idx]) and (p0[endorsed_idx] <= p0[idx])))
        n_other_wrongs_overtaken = int(np.sum(overtaken_other_wrongs)) if overtaken_other_wrongs else 0
        frac_other_wrongs_overtaken = _safe_divide(float(n_other_wrongs_overtaken), float(len(overtaken_other_wrongs)))

        tilt = _tilt_fit_to_endorsed_choice(p0, p1, endorsed_idx=endorsed_idx, eps=tilt_eps)
        tilt_distribution = np.asarray(tilt["fitted_distribution"], dtype=float)
        tilt_alpha_cb = float((tilt_distribution[endorsed_idx] - p0[endorsed_idx]) - (tilt_distribution[correct_idx] - p0[correct_idx]))

        probe_row: Dict[str, Any] = {
            "probe_family": probe_family,
            "probe_question_available": False,
            "probe_choice_count": np.nan,
            "probe_score_sum_neutral": np.nan,
            "probe_score_sum_biased": np.nan,
            "probe_closed_p0_c": np.nan,
            "probe_closed_p0_b": np.nan,
            "probe_closed_p1_c": np.nan,
            "probe_closed_p1_b": np.nan,
            "probe_closed_neutral_margin": np.nan,
            "probe_closed_biased_margin": np.nan,
            "probe_closed_gap_closure": np.nan,
            "probe_raw_neutral_gap": np.nan,
            "probe_raw_biased_gap": np.nan,
            "probe_raw_gap_closure": np.nan,
        }
        probe_group = probe_lookup.get(str(question_uid))
        if probe_group is not None and not probe_group.empty:
            probe_group = probe_group.copy()
            probe_group["choice_id"] = probe_group["choice_id"].astype(str).str.strip().str.upper()
            probe_group = probe_group.loc[probe_group["choice_id"].isin(choices)].copy()
            probe_group = probe_group.sort_values(["choice_id"]).reset_index(drop=True)
            if set(probe_group["choice_id"]) == set(choices):
                probe_group = probe_group.set_index("choice_id").loc[choices].reset_index()
                s0 = probe_group["score_neutral"].astype(float).to_numpy()
                s1 = probe_group[f"score_{framing}"].astype(float).to_numpy()
                q0 = _close_to_distribution(s0)
                q1 = _close_to_distribution(s1)
                probe_row = {
                    "probe_family": probe_family,
                    "probe_question_available": bool(np.isfinite(q0).all() and np.isfinite(q1).all()),
                    "probe_choice_count": int(len(probe_group)),
                    "probe_score_sum_neutral": float(np.sum(s0)),
                    "probe_score_sum_biased": float(np.sum(s1)),
                    "probe_closed_p0_c": float(q0[correct_idx]) if np.isfinite(q0[correct_idx]) else float("nan"),
                    "probe_closed_p0_b": float(q0[endorsed_idx]) if np.isfinite(q0[endorsed_idx]) else float("nan"),
                    "probe_closed_p1_c": float(q1[correct_idx]) if np.isfinite(q1[correct_idx]) else float("nan"),
                    "probe_closed_p1_b": float(q1[endorsed_idx]) if np.isfinite(q1[endorsed_idx]) else float("nan"),
                    "probe_closed_neutral_margin": float(q0[correct_idx] - q0[endorsed_idx]) if np.isfinite(q0).all() else float("nan"),
                    "probe_closed_biased_margin": float(q1[correct_idx] - q1[endorsed_idx]) if np.isfinite(q1).all() else float("nan"),
                    "probe_closed_gap_closure": float((q0[correct_idx] - q0[endorsed_idx]) - (q1[correct_idx] - q1[endorsed_idx]))
                    if np.isfinite(q0).all() and np.isfinite(q1).all()
                    else float("nan"),
                    "probe_raw_neutral_gap": float(s0[correct_idx] - s0[endorsed_idx]),
                    "probe_raw_biased_gap": float(s1[correct_idx] - s1[endorsed_idx]),
                    "probe_raw_gap_closure": float((s0[correct_idx] - s0[endorsed_idx]) - (s1[correct_idx] - s1[endorsed_idx])),
                }

        rows.append(
            {
                "run_id": _normalize_text(group["run_id"].iloc[0]),
                "model_name": _normalize_text(group["model_name"].iloc[0]),
                "dataset": _normalize_text(group["dataset"].iloc[0]),
                "split": _normalize_text(group["split"].iloc[0]),
                "question_id": _normalize_text(group["question_id"].iloc[0]),
                "draw_idx": int(group["draw_idx"].iloc[0]),
                "question_uid": _normalize_text(question_uid),
                "framing": framing,
                "choice_count": int(len(choices)),
                "correct_choice": correct_choice,
                "endorsed_choice": endorsed_choice,
                "neutral_top_choice": neutral_top_choice,
                "biased_top_choice": biased_top_choice,
                "neutral_top_is_correct": bool(neutral_top_is_correct),
                "biased_top_is_correct": bool(biased_top_is_correct),
                "answer_changed": bool(answer_changed),
                "stay_correct": bool(stay_correct),
                "flip_c_to_b": bool(flip_c_to_b),
                "p0_c": float(p0[correct_idx]),
                "p0_b": float(p0[endorsed_idx]),
                "p1_c": float(p1[correct_idx]),
                "p1_b": float(p1[endorsed_idx]),
                "neutral_margin": neutral_margin,
                "biased_margin": biased_margin,
                "delta_c": float(delta[correct_idx]),
                "delta_b": float(delta[endorsed_idx]),
                "alpha_cb": alpha_cb,
                "tv": tv,
                "targeted_ratio_tv": _safe_divide(alpha_cb, tv),
                "directional_transport_share": _safe_divide(alpha_cb, 2.0 * tv),
                "directional_projection_l1": directional["projection_l1"],
                "residual_l1": directional["residual_l1"],
                "residual_tv": directional["residual_tv"],
                "rank_c_neutral": rank0.get(correct_choice, np.nan),
                "rank_b_neutral": rank0.get(endorsed_choice, np.nan),
                "rank_c_biased": rank1.get(correct_choice, np.nan),
                "rank_b_biased": rank1.get(endorsed_choice, np.nan),
                "rank_shift_b": _safe_float(rank0.get(endorsed_choice, np.nan)) - _safe_float(rank1.get(endorsed_choice, np.nan)),
                "rank_shift_c": _safe_float(rank0.get(correct_choice, np.nan)) - _safe_float(rank1.get(correct_choice, np.nan)),
                "kq_neutral": _pairwise_k(correct_idx, p0),
                "kq_biased": _pairwise_k(correct_idx, p1),
                "kq_degradation": _pairwise_k(correct_idx, p0) - _pairwise_k(correct_idx, p1),
                "best_other_wrong_neutral_choice": best_other_wrong_neutral_choice,
                "delta_best_other_wrong_neutral": delta_best_other_wrong_neutral,
                "max_other_wrong_delta": max_other_wrong_delta,
                "biased_top_wrong_choice": biased_top_wrong_choice,
                "b_becomes_top_wrong": bool(b_becomes_top_wrong),
                "n_other_wrongs": int(len(overtaken_other_wrongs)),
                "n_other_wrongs_overtaken_by_b": int(n_other_wrongs_overtaken),
                "frac_other_wrongs_overtaken_by_b": frac_other_wrongs_overtaken,
                "tilt_lambda_hat": tilt["lambda_hat"],
                "tilt_alpha_cb": tilt_alpha_cb,
                "tilt_alpha_fit_gap": alpha_cb - tilt_alpha_cb,
                "tilt_l1_error": tilt["l1_error"],
                "tilt_kl_error": tilt["kl_error"],
                "tilt_smoothing_applied": bool(tilt["smoothing_applied"]),
                "output_gap_closure": alpha_cb,
                **probe_row,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values(
        ["dataset", "model_name", "run_id", "split", "question_id", "draw_idx"]
    ).reset_index(drop=True)
    return _add_margin_quantiles(result)


def build_self_commitment_comparison_df(
    model_wide_df: pd.DataFrame,
    *,
    probe_wide_df: Optional[pd.DataFrame] = None,
    framing: str = DEFAULT_TRANSPORT_FRAMING,
    congruent_framing: str = DEFAULT_CONGRUENT_FRAMING,
    probe_family: str = DEFAULT_PROBE_FAMILY,
) -> pd.DataFrame:
    neutral_col = "prob_neutral"
    target_col = f"prob_{framing}"
    endorsed_col = f"endorsed_choice_{framing}"
    congruent_col = f"prob_{congruent_framing}"
    congruent_endorsed_col = f"endorsed_choice_{congruent_framing}"

    required_model_columns = [
        "run_id",
        "model_name",
        "dataset",
        "split",
        "question_id",
        "draw_idx",
        "question_uid",
        "choice_id",
        "correct_choice",
        neutral_col,
        target_col,
        endorsed_col,
    ]
    missing_model_columns = sorted(column for column in required_model_columns if column not in model_wide_df.columns)
    if missing_model_columns:
        raise ValueError(f"Model wide dataframe is missing required columns: {missing_model_columns}")

    model_has_congruent = congruent_col in model_wide_df.columns and congruent_endorsed_col in model_wide_df.columns

    probe_lookup: Dict[str, pd.DataFrame] = {}
    probe_has_congruent = False
    if probe_wide_df is not None and not probe_wide_df.empty:
        probe_neutral_col = "score_neutral"
        probe_target_col = f"score_{framing}"
        required_probe_columns = [
            "question_uid",
            "choice_id",
            "probe_family",
            probe_neutral_col,
            probe_target_col,
        ]
        missing_probe_columns = sorted(column for column in required_probe_columns if column not in probe_wide_df.columns)
        if missing_probe_columns:
            raise ValueError(f"Probe wide dataframe is missing required columns: {missing_probe_columns}")
        probe_has_congruent = f"score_{congruent_framing}" in probe_wide_df.columns
        probe_subset = probe_wide_df.loc[
            probe_wide_df["probe_family"].astype(str).eq(str(probe_family))
            & probe_wide_df[probe_neutral_col].notna()
            & probe_wide_df[probe_target_col].notna()
        ].copy()
        if not probe_subset.empty:
            probe_lookup = {
                str(question_uid): group.copy()
                for question_uid, group in probe_subset.groupby("question_uid", sort=False)
            }

    model_subset = model_wide_df.loc[
        model_wide_df[neutral_col].notna()
        & model_wide_df[target_col].notna()
        & model_wide_df[endorsed_col].notna()
    ].copy()
    if model_subset.empty:
        return pd.DataFrame()

    rows: list[Dict[str, Any]] = []
    for question_uid, group in model_subset.groupby("question_uid", sort=False):
        group = group.copy()
        group["choice_id"] = group["choice_id"].astype(str).str.strip().str.upper()
        group = group.sort_values(["choice_id"]).reset_index(drop=True)

        correct_choice = _normalize_choice(group["correct_choice"].iloc[0])
        endorsed_choice = _normalize_choice(group[endorsed_col].iloc[0])
        if not correct_choice or not endorsed_choice or correct_choice == endorsed_choice:
            continue

        choices = group["choice_id"].astype(str).tolist()
        if correct_choice not in choices or endorsed_choice not in choices:
            continue

        p0 = group[neutral_col].astype(float).to_numpy()
        p1 = group[target_col].astype(float).to_numpy()
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue

        correct_idx = choices.index(correct_choice)
        endorsed_idx = choices.index(endorsed_choice)
        neutral_top_choice = _argmax_choice(choices, p0)
        neutral_second_choice = _second_choice(choices, p0)
        biased_top_choice = _argmax_choice(choices, p1)
        if not neutral_top_choice:
            continue
        neutral_top_idx = choices.index(neutral_top_choice)
        neutral_second_idx = choices.index(neutral_second_choice) if neutral_second_choice in choices else None
        neutral_top_group = _self_commitment_group(
            neutral_top_choice=neutral_top_choice,
            correct_choice=correct_choice,
            endorsed_choice=endorsed_choice,
        )

        p0_self = float(p0[neutral_top_idx])
        p1_self = float(p1[neutral_top_idx])
        p0_b = float(p0[endorsed_idx])
        p1_b = float(p1[endorsed_idx])
        delta_b = float(p1_b - p0_b)
        delta_self = float(p1_self - p0_self)
        self_margin_to_b = float(p0_self - p0_b)
        self_margin_to_second = (
            float(p0_self - p0[neutral_second_idx]) if neutral_second_idx is not None else float("nan")
        )
        biased_self_to_b_gap = float(p1_self - p1_b)
        self_to_b_gap_closure = float(self_margin_to_b - biased_self_to_b_gap)
        answer_changed = neutral_top_choice != biased_top_choice
        flip_self_to_b = neutral_top_choice != endorsed_choice and biased_top_choice == endorsed_choice

        congruent_row: Dict[str, Any] = {
            "congruent_framing": congruent_framing,
            "congruent_prompt_available": False,
            "congruent_endorsed_choice": "",
            "congruent_endorses_self_choice": False,
            "pc_self": np.nan,
            "pc_b": np.nan,
            "congruent_delta_self": np.nan,
            "congruent_delta_b": np.nan,
            "congruent_self_to_b_gap": np.nan,
            "congruent_self_to_b_gap_change": np.nan,
            "delta_prompt_endorsed_target_incorrect": delta_b,
            "delta_prompt_endorsed_target_congruent": np.nan,
        }
        if model_has_congruent and group[congruent_col].notna().all():
            pc = group[congruent_col].astype(float).to_numpy()
            if np.isfinite(pc).all():
                congruent_endorsed_choice = _normalize_choice(group[congruent_endorsed_col].iloc[0])
                congruent_endorses_self_choice = bool(congruent_endorsed_choice == neutral_top_choice and neutral_top_choice)
                pc_self = float(pc[neutral_top_idx])
                pc_b = float(pc[endorsed_idx])
                congruent_row = {
                    "congruent_framing": congruent_framing,
                    "congruent_prompt_available": True,
                    "congruent_endorsed_choice": congruent_endorsed_choice,
                    "congruent_endorses_self_choice": congruent_endorses_self_choice,
                    "pc_self": pc_self,
                    "pc_b": pc_b,
                    "congruent_delta_self": float(pc_self - p0_self),
                    "congruent_delta_b": float(pc_b - p0_b),
                    "congruent_self_to_b_gap": float(pc_self - pc_b),
                    "congruent_self_to_b_gap_change": float(self_margin_to_b - (pc_self - pc_b)),
                    "delta_prompt_endorsed_target_incorrect": delta_b,
                    "delta_prompt_endorsed_target_congruent": float(pc_self - p0_self)
                    if congruent_endorses_self_choice
                    else float("nan"),
                }

        probe_row: Dict[str, Any] = {
            "probe_family": probe_family,
            "probe_question_available": False,
            "probe_congruent_available": False,
            "probe_neutral_top_choice": "",
            "probe_raw_correct_minus_self_neutral": np.nan,
            "probe_closed_correct_minus_self_neutral": np.nan,
            "probe_prefers_correct_to_self_neutral": pd.NA,
            "probe_closed_self_to_b_gap_closure": np.nan,
            "probe_raw_self_to_b_gap_closure": np.nan,
            "probe_congruent_self_to_b_gap_change_closed": np.nan,
        }
        probe_group = probe_lookup.get(str(question_uid))
        if probe_group is not None and not probe_group.empty:
            probe_group = probe_group.copy()
            probe_group["choice_id"] = probe_group["choice_id"].astype(str).str.strip().str.upper()
            probe_group = probe_group.loc[probe_group["choice_id"].isin(choices)].copy()
            probe_group = probe_group.sort_values(["choice_id"]).reset_index(drop=True)
            if set(probe_group["choice_id"]) == set(choices):
                probe_group = probe_group.set_index("choice_id").loc[choices].reset_index()
                s0 = probe_group["score_neutral"].astype(float).to_numpy()
                s1 = probe_group[f"score_{framing}"].astype(float).to_numpy()
                q0 = _close_to_distribution(s0)
                q1 = _close_to_distribution(s1)
                prefers_correct = pd.NA
                if math.isfinite(float(s0[correct_idx])) and math.isfinite(float(s0[neutral_top_idx])):
                    prefers_correct = bool(float(s0[correct_idx]) > float(s0[neutral_top_idx]))

                probe_row = {
                    "probe_family": probe_family,
                    "probe_question_available": bool(np.isfinite(q0).all() and np.isfinite(q1).all()),
                    "probe_congruent_available": False,
                    "probe_neutral_top_choice": _argmax_choice(choices, s0),
                    "probe_raw_correct_minus_self_neutral": float(s0[correct_idx] - s0[neutral_top_idx]),
                    "probe_closed_correct_minus_self_neutral": float(q0[correct_idx] - q0[neutral_top_idx])
                    if np.isfinite(q0).all()
                    else float("nan"),
                    "probe_prefers_correct_to_self_neutral": prefers_correct,
                    "probe_closed_self_to_b_gap_closure": float((q0[neutral_top_idx] - q0[endorsed_idx]) - (q1[neutral_top_idx] - q1[endorsed_idx]))
                    if np.isfinite(q0).all() and np.isfinite(q1).all()
                    else float("nan"),
                    "probe_raw_self_to_b_gap_closure": float((s0[neutral_top_idx] - s0[endorsed_idx]) - (s1[neutral_top_idx] - s1[endorsed_idx])),
                    "probe_congruent_self_to_b_gap_change_closed": float("nan"),
                }
                if probe_has_congruent and f"score_{congruent_framing}" in probe_group.columns:
                    s_congruent = probe_group[f"score_{congruent_framing}"].astype(float).to_numpy()
                    q_congruent = _close_to_distribution(s_congruent)
                    if np.isfinite(q_congruent).all():
                        probe_row["probe_congruent_available"] = True
                        probe_row["probe_congruent_self_to_b_gap_change_closed"] = float(
                            (q0[neutral_top_idx] - q0[endorsed_idx]) - (q_congruent[neutral_top_idx] - q_congruent[endorsed_idx])
                        )

        rows.append(
            {
                "run_id": _normalize_text(group["run_id"].iloc[0]),
                "model_name": _normalize_text(group["model_name"].iloc[0]),
                "dataset": _normalize_text(group["dataset"].iloc[0]),
                "split": _normalize_text(group["split"].iloc[0]),
                "question_id": _normalize_text(group["question_id"].iloc[0]),
                "draw_idx": int(group["draw_idx"].iloc[0]),
                "question_uid": _normalize_text(question_uid),
                "framing": framing,
                "correct_choice": correct_choice,
                "endorsed_choice": endorsed_choice,
                "neutral_top_choice": neutral_top_choice,
                "neutral_second_choice": neutral_second_choice,
                "biased_top_choice": biased_top_choice,
                "neutral_top_group": neutral_top_group,
                "neutral_top_is_correct": bool(neutral_top_choice == correct_choice),
                "neutral_top_is_endorsed": bool(neutral_top_choice == endorsed_choice),
                "neutral_top_is_other_wrong": bool(neutral_top_group == "d_top"),
                "included_in_c_vs_d": bool(neutral_top_group in {"c_top", "d_top"}),
                "answer_changed": bool(answer_changed),
                "flip_self_to_b": bool(flip_self_to_b),
                "neutral_entropy": _entropy(p0),
                "p0_c": float(p0[correct_idx]),
                "p0_b": p0_b,
                "p1_c": float(p1[correct_idx]),
                "p1_b": p1_b,
                "p0_self": p0_self,
                "p1_self": p1_self,
                "delta_b": delta_b,
                "delta_self": delta_self,
                "self_margin_to_b": self_margin_to_b,
                "self_margin_to_second": self_margin_to_second,
                "biased_self_to_b_gap": biased_self_to_b_gap,
                "self_to_b_gap_closure": self_to_b_gap_closure,
                **congruent_row,
                **probe_row,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values(
        ["dataset", "model_name", "run_id", "split", "question_id", "draw_idx"]
    ).reset_index(drop=True)
    return _add_self_commitment_quantiles(result)


def summarize_transport_by_subset(
    transport_df: pd.DataFrame,
    *,
    subsets: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if transport_df.empty:
        return pd.DataFrame()
    subsets = list(subsets or ["all", "no_flip", "flip", "stay_correct", "neutral_correct", "neutral_wrong", "flip_c_to_b"])
    rows: list[Dict[str, Any]] = []
    group_cols = ["run_id", "model_name", "dataset", "split"]
    for subset in subsets:
        mask = _subset_mask(transport_df, subset)
        subset_df = transport_df.loc[mask].copy()
        if subset_df.empty:
            continue
        for keys, group in subset_df.groupby(group_cols, sort=True):
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "subset": subset,
                    "n_questions": int(len(group)),
                    "answer_flip_rate": float(group["answer_changed"].mean()),
                    "stay_correct_rate": float(group["stay_correct"].mean()),
                    "flip_c_to_b_rate": float(group["flip_c_to_b"].mean()),
                    "b_becomes_top_wrong_rate": float(group["b_becomes_top_wrong"].mean()),
                    "mean_neutral_margin": float(group["neutral_margin"].mean()),
                    "mean_biased_margin": float(group["biased_margin"].mean()),
                    "mean_alpha_cb": float(group["alpha_cb"].mean()),
                    "median_alpha_cb": float(group["alpha_cb"].median()),
                    "mean_tv": float(group["tv"].mean()),
                    "median_tv": float(group["tv"].median()),
                    "mean_targeted_ratio_tv": float(group["targeted_ratio_tv"].dropna().mean())
                    if group["targeted_ratio_tv"].notna().any()
                    else float("nan"),
                    "mean_directional_transport_share": float(group["directional_transport_share"].dropna().mean())
                    if group["directional_transport_share"].notna().any()
                    else float("nan"),
                    "mean_residual_l1": float(group["residual_l1"].mean()),
                    "mean_delta_b": float(group["delta_b"].mean()),
                    "mean_delta_c": float(group["delta_c"].mean()),
                    "mean_delta_best_other_wrong_neutral": float(group["delta_best_other_wrong_neutral"].dropna().mean())
                    if group["delta_best_other_wrong_neutral"].notna().any()
                    else float("nan"),
                    "mean_max_other_wrong_delta": float(group["max_other_wrong_delta"].dropna().mean())
                    if group["max_other_wrong_delta"].notna().any()
                    else float("nan"),
                    "mean_frac_other_wrongs_overtaken_by_b": float(group["frac_other_wrongs_overtaken_by_b"].dropna().mean())
                    if group["frac_other_wrongs_overtaken_by_b"].notna().any()
                    else float("nan"),
                    "mean_kq_neutral": float(group["kq_neutral"].mean()),
                    "mean_kq_biased": float(group["kq_biased"].mean()),
                    "mean_kq_degradation": float(group["kq_degradation"].mean()),
                    "mean_tilt_lambda_hat": float(group["tilt_lambda_hat"].mean()),
                    "mean_tilt_l1_error": float(group["tilt_l1_error"].mean()),
                    "mean_tilt_kl_error": float(group["tilt_kl_error"].mean()),
                    "mean_tilt_alpha_fit_gap": float(group["tilt_alpha_fit_gap"].mean()),
                    "mean_output_gap_closure": float(group["output_gap_closure"].mean()),
                    "mean_probe_closed_gap_closure": float(group["probe_closed_gap_closure"].dropna().mean())
                    if group["probe_closed_gap_closure"].notna().any()
                    else float("nan"),
                    "mean_probe_raw_gap_closure": float(group["probe_raw_gap_closure"].dropna().mean())
                    if group["probe_raw_gap_closure"].notna().any()
                    else float("nan"),
                    "corr_output_vs_probe_closed_gap": float(group["output_gap_closure"].corr(group["probe_closed_gap_closure"]))
                    if group["probe_closed_gap_closure"].notna().sum() >= 2
                    else float("nan"),
                    "corr_output_vs_probe_raw_gap": float(group["output_gap_closure"].corr(group["probe_raw_gap_closure"]))
                    if group["probe_raw_gap_closure"].notna().sum() >= 2
                    else float("nan"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["subset"]).reset_index(drop=True)


def summarize_transport_by_margin_quartile(
    transport_df: pd.DataFrame,
    *,
    quartile_column: str = "neutral_margin_quartile_run",
    subsets: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if transport_df.empty:
        return pd.DataFrame()
    if quartile_column not in transport_df.columns:
        raise ValueError(f"transport_df is missing quartile column {quartile_column!r}.")
    subsets = list(subsets or ["all", "no_flip", "stay_correct"])
    rows: list[Dict[str, Any]] = []
    group_cols = ["run_id", "model_name", "dataset", "split", quartile_column]
    for subset in subsets:
        mask = _subset_mask(transport_df, subset)
        subset_df = transport_df.loc[mask & transport_df[quartile_column].notna()].copy()
        if subset_df.empty:
            continue
        for keys, group in subset_df.groupby(group_cols, sort=True):
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "subset": subset,
                    "n_questions": int(len(group)),
                    "mean_neutral_margin": float(group["neutral_margin"].mean()),
                    "mean_alpha_cb": float(group["alpha_cb"].mean()),
                    "mean_tv": float(group["tv"].mean()),
                    "mean_targeted_ratio_tv": float(group["targeted_ratio_tv"].dropna().mean())
                    if group["targeted_ratio_tv"].notna().any()
                    else float("nan"),
                    "mean_directional_transport_share": float(group["directional_transport_share"].dropna().mean())
                    if group["directional_transport_share"].notna().any()
                    else float("nan"),
                    "mean_residual_l1": float(group["residual_l1"].mean()),
                    "mean_delta_b": float(group["delta_b"].mean()),
                    "mean_delta_best_other_wrong_neutral": float(group["delta_best_other_wrong_neutral"].dropna().mean())
                    if group["delta_best_other_wrong_neutral"].notna().any()
                    else float("nan"),
                    "mean_output_gap_closure": float(group["output_gap_closure"].mean()),
                    "mean_probe_closed_gap_closure": float(group["probe_closed_gap_closure"].dropna().mean())
                    if group["probe_closed_gap_closure"].notna().any()
                    else float("nan"),
                    "mean_tilt_lambda_hat": float(group["tilt_lambda_hat"].mean()),
                    "mean_tilt_l1_error": float(group["tilt_l1_error"].mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["subset"]).reset_index(drop=True)
