#!/usr/bin/env python3
"""Paired analysis of matched single- and multi-turn source-family experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SINGLE = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "mmlupro_epistemic_source_families_20260915"
)
DEFAULT_MULTI = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "mmlupro_epistemic_source_families_multiturn_20260917"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "analysis"
    / "single_multi_source_family_connection_20260917"
)

MODEL_ORDER = ("terra", "luna", "opus5", "sonnet5")
MODEL_LABELS = {
    "terra": "GPT-5.6 Terra",
    "luna": "GPT-5.6 Luna",
    "opus5": "Claude Opus 5",
    "sonnet5": "Claude Sonnet 5",
}
FAMILY_ORDER = (
    "unsupported_user",
    "individual_expert",
    "authoritative_reference",
    "independent_corroboration",
)
FAMILY_LABELS = {
    "unsupported_user": "Unsupported user",
    "individual_expert": "Individual expert",
    "authoritative_reference": "Authoritative reference",
    "independent_corroboration": "Independent corroboration",
}
CLASS_ORDER = ("endorsed_wrong", "correct", "other_wrong", "invalid")
CLASS_LABELS = {
    "endorsed_wrong": "Endorsed wrong",
    "correct": "Correct",
    "other_wrong": "Other wrong",
    "invalid": "Invalid",
}
POSITIVE = "#d4651a"
NEGATIVE = "#73b3ab"
BLUE = "#4D87C1"
PURPLE = "#906DA1"
TEXT = "#20242C"
GRID = "#D7DCE2"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _prepare_placement(root: Path, placement: str) -> pd.DataFrame:
    manifest = pd.DataFrame(_read_jsonl(root / "requests" / "manifest.jsonl"))
    records = pd.DataFrame(_read_jsonl(root / "responses" / "records.jsonl"))
    manifest_fields = [
        "custom_id",
        "model_key",
        "question_id",
        "category",
        "family",
        "condition",
        "repetition",
        "question",
        "correct_letter",
        "incorrect_letter",
        "incorrect_text",
        "template_id",
        "treatment_text",
    ]
    missing = sorted(set(manifest_fields) - set(manifest.columns))
    if missing:
        raise ValueError(f"{placement} manifest is missing fields: {missing}")
    keep_records = [
        "custom_id",
        "parsed_letter",
        "raw_response",
        "valid_output",
        "chose_x",
        "correct",
        "resolved_model",
    ]
    missing = sorted(set(keep_records) - set(records.columns))
    if missing:
        raise ValueError(f"{placement} records are missing fields: {missing}")
    frame = manifest[manifest_fields].merge(
        records[keep_records], on="custom_id", how="left", validate="one_to_one"
    )
    if frame["raw_response"].isna().any():
        raise ValueError(f"{placement} has manifest rows without responses")
    frame["family"] = frame["family"].fillna(frame["condition"])
    frame["placement"] = placement
    frame["valid_output"] = frame["valid_output"].astype(bool)
    frame["chose_x"] = frame["chose_x"].astype(bool)
    frame["correct"] = frame["correct"].astype(bool)
    frame["response_class"] = np.select(
        [
            ~frame["valid_output"],
            frame["parsed_letter"] == frame["incorrect_letter"],
            frame["parsed_letter"] == frame["correct_letter"],
        ],
        ["invalid", "endorsed_wrong", "correct"],
        default="other_wrong",
    )
    frame["answer_for_agreement"] = frame["parsed_letter"].where(
        frame["valid_output"], "INVALID"
    )
    return frame


def _paired_cells(single: pd.DataFrame, multi: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_key", "question_id", "family", "repetition"]
    if single.duplicated(keys).any() or multi.duplicated(keys).any():
        raise ValueError("Placement data are not unique on the exact pairing key")
    single_keys = set(map(tuple, single[keys].itertuples(index=False, name=None)))
    multi_keys = set(map(tuple, multi[keys].itertuples(index=False, name=None)))
    if single_keys != multi_keys:
        raise ValueError(
            f"Pairing-key mismatch: single-only={len(single_keys-multi_keys)}, "
            f"multi-only={len(multi_keys-single_keys)}"
        )
    pair = single.merge(
        multi,
        on=keys,
        how="inner",
        suffixes=("_single", "_multi"),
        validate="one_to_one",
    )
    invariants = (
        "category",
        "question",
        "correct_letter",
        "incorrect_letter",
        "incorrect_text",
        "template_id",
        "treatment_text",
    )
    for field in invariants:
        left = pair[f"{field}_single"].fillna("").astype(str)
        right = pair[f"{field}_multi"].fillna("").astype(str)
        if not left.equals(right):
            raise ValueError(f"Matched cells disagree on {field}")
    pair["single_adopt"] = pair["chose_x_single"].astype(int)
    pair["multi_adopt"] = pair["chose_x_multi"].astype(int)
    pair["delta_adopt"] = pair["multi_adopt"] - pair["single_adopt"]
    pair["exact_answer_match"] = (
        pair["answer_for_agreement_single"]
        == pair["answer_for_agreement_multi"]
    )
    pair["response_class_match"] = (
        pair["response_class_single"] == pair["response_class_multi"]
    )
    return pair


def _paired_groups(pair: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_key", "question_id", "family"]
    grouped = (
        pair.groupby(keys, observed=True)
        .agg(
            category=("category_single", "first"),
            question=("question_single", "first"),
            correct_letter=("correct_letter_single", "first"),
            incorrect_letter=("incorrect_letter_single", "first"),
            incorrect_text=("incorrect_text_single", "first"),
            template_id=("template_id_single", "first"),
            treatment_text=("treatment_text_single", "first"),
            repetitions=("repetition", "size"),
            single_count=("single_adopt", "sum"),
            multi_count=("multi_adopt", "sum"),
            single_valid_n=("valid_output_single", "sum"),
            multi_valid_n=("valid_output_multi", "sum"),
            single_valid_rate=("valid_output_single", "mean"),
            multi_valid_rate=("valid_output_multi", "mean"),
            single_correct_rate=("correct_single", "mean"),
            multi_correct_rate=("correct_multi", "mean"),
        )
        .reset_index()
    )
    if set(grouped["repetitions"]) != {3}:
        raise ValueError("Each matched question/suggestion cell must have three repetitions")
    if (grouped[["single_valid_n", "multi_valid_n"]] == 0).any().any():
        raise ValueError("Every placement must have at least one valid response per cell")
    # Match the paper estimand: within each question-suggestion cell, estimate
    # adoption among valid option-letter responses, then weight questions equally.
    grouped["single_rate"] = grouped["single_count"] / grouped["single_valid_n"]
    grouped["multi_rate"] = grouped["multi_count"] / grouped["multi_valid_n"]
    grouped["delta"] = grouped["multi_rate"] - grouped["single_rate"]
    grouped["stratum"] = grouped["model_key"] + "|" + grouped["family"]
    grouped["single_state"] = np.select(
        [grouped["single_rate"] == 0, grouped["single_rate"] == 1],
        ["none", "all"],
        default="mixed",
    )
    grouped["multi_state"] = np.select(
        [grouped["multi_rate"] == 0, grouped["multi_rate"] == 1],
        ["none", "all"],
        default="mixed",
    )
    return grouped


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.pearsonr(x, y).statistic)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _bootstrap_cluster_mean(
    frame: pd.DataFrame, column: str, *, replicates: int, seed: int
) -> np.ndarray:
    by_question = frame.groupby("question_id", observed=True)[column].mean().to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(by_question), size=(replicates, len(by_question)))
    return by_question[indices].mean(axis=1)


def _rowwise_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(x), np.nan, dtype=float),
        where=denominator > 0,
    )


def _bootstrap_cluster_correlations(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    ordering = [
        column
        for column in ("question_id", "model_key", "family")
        if column in frame.columns
    ]
    ordered = frame.sort_values(ordering).reset_index(drop=True)
    sizes = ordered.groupby("question_id", observed=True).size().to_numpy()
    if len(set(sizes)) != 1:
        raise ValueError("Clustered correlation bootstrap requires balanced question blocks")
    n_questions = len(sizes)
    per_question = int(sizes[0])
    x = ordered["single_rate"].to_numpy(dtype=float).reshape(n_questions, per_question)
    y = ordered["multi_rate"].to_numpy(dtype=float).reshape(n_questions, per_question)
    rng = np.random.default_rng(seed)
    pearson = np.empty(replicates, dtype=float)
    spearman = np.empty(replicates, dtype=float)
    batch_size = 250
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, n_questions, size=(stop - start, n_questions))
        x_sample = x[indices].reshape(stop - start, -1)
        y_sample = y[indices].reshape(stop - start, -1)
        pearson[start:stop] = _rowwise_correlation(x_sample, y_sample)
        x_rank = stats.rankdata(x_sample, axis=1)
        y_rank = stats.rankdata(y_sample, axis=1)
        spearman[start:stop] = _rowwise_correlation(x_rank, y_rank)
    return pearson, spearman


def _ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def _sign_flip_pvalue(values: np.ndarray, *, replicates: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    batch = 5000
    while completed < replicates:
        size = min(batch, replicates - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(values)))
        permuted = np.abs(np.mean(signs * values[None, :], axis=1))
        extreme += int(np.sum(permuted >= observed - 1e-15))
        completed += size
    return (extreme + 1.0) / (replicates + 1.0)


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _effect_table(
    groups: pd.DataFrame, *, bootstrap_replicates: int, permutation_replicates: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for family in FAMILY_ORDER:
            subset = groups.loc[
                (groups["model_key"] == model) & (groups["family"] == family)
            ].copy()
            if len(subset) != 81:
                raise ValueError(f"Expected 81 paired questions for {model}/{family}")
            effect = float(subset["delta"].mean())
            boot = _bootstrap_cluster_mean(
                subset,
                "delta",
                replicates=bootstrap_replicates,
                seed=17_000 + 101 * MODEL_ORDER.index(model) + FAMILY_ORDER.index(family),
            )
            low, high = _ci(boot)
            rows.append(
                {
                    "model_key": model,
                    "model": MODEL_LABELS[model],
                    "family": family,
                    "family_label": FAMILY_LABELS[family],
                    "n_questions": len(subset),
                    "single_rate": float(subset["single_rate"].mean()),
                    "multi_rate": float(subset["multi_rate"].mean()),
                    "delta": effect,
                    "ci_low": low,
                    "ci_high": high,
                    "p_sign_flip": _sign_flip_pvalue(
                        subset["delta"].to_numpy(),
                        replicates=permutation_replicates,
                        seed=29_000
                        + 101 * MODEL_ORDER.index(model)
                        + FAMILY_ORDER.index(family),
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["q_bh"] = _benjamini_hochberg(result["p_sign_flip"].to_numpy())
    return result


def _correlation_row(
    frame: pd.DataFrame,
    *,
    analysis: str,
    model_key: str = "all",
    family: str = "all",
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    x = frame["single_rate"].to_numpy(dtype=float)
    y = frame["multi_rate"].to_numpy(dtype=float)
    pearson = _safe_pearson(x, y)
    spearman = _safe_spearman(x, y)
    boot_p, boot_s = _bootstrap_cluster_correlations(
        frame, replicates=bootstrap_replicates, seed=seed
    )
    pearson_low, pearson_high = _ci(boot_p)
    spearman_low, spearman_high = _ci(boot_s)
    return {
        "analysis": analysis,
        "model_key": model_key,
        "family": family,
        "n": len(frame),
        "pearson_r": pearson,
        "pearson_ci_low": pearson_low,
        "pearson_ci_high": pearson_high,
        "spearman_rho": spearman,
        "spearman_ci_low": spearman_low,
        "spearman_ci_high": spearman_high,
    }


def _correlation_table(
    groups: pd.DataFrame, *, bootstrap_replicates: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    rows.append(
        _correlation_row(
            groups,
            analysis="group_raw",
            bootstrap_replicates=bootstrap_replicates,
            seed=41_000,
        )
    )

    residual = groups.copy()
    residual["single_rate"] -= residual.groupby("stratum")["single_rate"].transform(
        "mean"
    )
    residual["multi_rate"] -= residual.groupby("stratum")["multi_rate"].transform(
        "mean"
    )
    rows.append(
        _correlation_row(
            residual,
            analysis="group_within_model_family",
            bootstrap_replicates=bootstrap_replicates,
            seed=42_000,
        )
    )

    for model in MODEL_ORDER:
        subset = groups.loc[groups["model_key"] == model]
        rows.append(
            _correlation_row(
                subset,
                analysis="group_by_model",
                model_key=model,
                bootstrap_replicates=bootstrap_replicates,
                seed=43_000 + MODEL_ORDER.index(model),
            )
        )
    for family in FAMILY_ORDER:
        subset = groups.loc[groups["family"] == family]
        rows.append(
            _correlation_row(
                subset,
                analysis="group_by_family",
                family=family,
                bootstrap_replicates=bootstrap_replicates,
                seed=44_000 + FAMILY_ORDER.index(family),
            )
        )
    for model in MODEL_ORDER:
        for family in FAMILY_ORDER:
            subset = groups.loc[
                (groups["model_key"] == model) & (groups["family"] == family)
            ]
            rows.append(
                _correlation_row(
                    subset,
                    analysis="group_by_model_family",
                    model_key=model,
                    family=family,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=45_000
                    + 101 * MODEL_ORDER.index(model)
                    + FAMILY_ORDER.index(family),
                )
            )

    question = (
        groups.groupby(["model_key", "question_id"], observed=True)
        .agg(
            category=("category", "first"),
            question=("question", "first"),
            single_rate=("single_rate", "mean"),
            multi_rate=("multi_rate", "mean"),
        )
        .reset_index()
    )
    question["delta"] = question["multi_rate"] - question["single_rate"]
    for model in MODEL_ORDER:
        subset = question.loc[question["model_key"] == model]
        rows.append(
            _correlation_row(
                subset,
                analysis="question_across_families_by_model",
                model_key=model,
                bootstrap_replicates=bootstrap_replicates,
                seed=46_000 + MODEL_ORDER.index(model),
            )
        )
    return pd.DataFrame(rows), question


def _cronbach_alpha(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    k = matrix.shape[1]
    total_variance = np.var(matrix.sum(axis=1), ddof=1)
    if total_variance == 0:
        return float("nan")
    return float(
        k
        / (k - 1)
        * (1.0 - np.var(matrix, axis=0, ddof=1).sum() / total_variance)
    )


def _reliability(pair: pd.DataFrame, groups: pd.DataFrame) -> dict[str, float]:
    index = ["model_key", "question_id", "family"]
    single = pair.pivot(index=index, columns="repetition", values="single_adopt")
    multi = pair.pivot(index=index, columns="repetition", values="multi_adopt")
    alpha_single = _cronbach_alpha(single.to_numpy())
    alpha_multi = _cronbach_alpha(multi.to_numpy())
    raw_r = _safe_pearson(
        groups["single_rate"].to_numpy(), groups["multi_rate"].to_numpy()
    )
    denominator = math.sqrt(max(alpha_single * alpha_multi, 0.0))
    corrected = raw_r / denominator if denominator > 0 else float("nan")
    return {
        "cronbach_alpha_single": alpha_single,
        "cronbach_alpha_multi": alpha_multi,
        "raw_group_rate_correlation": raw_r,
        "attenuation_corrected_correlation": float(np.clip(corrected, -1, 1)),
    }


def _class_transition(pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(
        pd.Categorical(pair["response_class_single"], categories=CLASS_ORDER),
        pd.Categorical(pair["response_class_multi"], categories=CLASS_ORDER),
        dropna=False,
    )
    counts.index = [CLASS_LABELS[value] for value in CLASS_ORDER]
    counts.columns = [CLASS_LABELS[value] for value in CLASS_ORDER]
    row_rates = counts.div(counts.sum(axis=1), axis=0)
    return counts, row_rates


def _distribution_similarity(pair: pd.DataFrame) -> pd.DataFrame:
    answer_categories = [chr(code) for code in range(ord("A"), ord("J") + 1)] + [
        "INVALID"
    ]
    rows: list[dict[str, object]] = []
    for keys, subset in pair.groupby(
        ["model_key", "question_id", "family"], observed=True
    ):
        p = np.asarray(
            [
                np.mean(subset["answer_for_agreement_single"] == category)
                for category in answer_categories
            ]
        )
        q = np.asarray(
            [
                np.mean(subset["answer_for_agreement_multi"] == category)
                for category in answer_categories
            ]
        )
        js_distance = float(jensenshannon(p, q, base=2.0))
        rows.append(
            {
                "model_key": keys[0],
                "question_id": keys[1],
                "family": keys[2],
                "expected_cross_placement_exact_agreement": float(np.dot(p, q)),
                "jensen_shannon_similarity": 1.0 - js_distance**2,
            }
        )
    return pd.DataFrame(rows)


def _paired_descriptives(pair: pd.DataFrame, groups: pd.DataFrame) -> dict[str, object]:
    valid_both = pair["valid_output_single"] & pair["valid_output_multi"]
    adoption_kappa = float(
        cohen_kappa_score(pair["single_adopt"], pair["multi_adopt"])
    )
    answer_kappa = float(
        cohen_kappa_score(
            pair.loc[valid_both, "answer_for_agreement_single"],
            pair.loc[valid_both, "answer_for_agreement_multi"],
        )
    )
    n00 = int(((pair["single_adopt"] == 0) & (pair["multi_adopt"] == 0)).sum())
    n01 = int(((pair["single_adopt"] == 0) & (pair["multi_adopt"] == 1)).sum())
    n10 = int(((pair["single_adopt"] == 1) & (pair["multi_adopt"] == 0)).sum())
    n11 = int(((pair["single_adopt"] == 1) & (pair["multi_adopt"] == 1)).sum())
    mcnemar = stats.binomtest(min(n01, n10), n=n01 + n10, p=0.5)
    count_kappa = float(
        cohen_kappa_score(
            groups["single_count"], groups["multi_count"], weights="quadratic"
        )
    )
    single_any = groups["single_count"] > 0
    single_all = groups["single_count"] == 3
    return {
        "note": (
            "Repetition indices are exchangeable, unseeded API draws. Exact-pair "
            "agreement and McNemar results are secondary; grouped three-repetition "
            "rates are the primary estimand."
        ),
        "exact_pair_binary_table": {
            "neither_adopts": n00,
            "multi_only": n01,
            "single_only": n10,
            "both_adopt": n11,
        },
        "exact_pair_adoption_phi": _safe_pearson(
            pair["single_adopt"].to_numpy(), pair["multi_adopt"].to_numpy()
        ),
        "exact_pair_adoption_kappa": adoption_kappa,
        "exact_pair_mcnemar_p": float(mcnemar.pvalue),
        "exact_letter_agreement_given_both_valid": float(
            pair.loc[valid_both, "exact_answer_match"].mean()
        ),
        "exact_letter_kappa_given_both_valid": answer_kappa,
        "response_class_agreement": float(pair["response_class_match"].mean()),
        "group_count_quadratic_kappa": count_kappa,
        "p_multi_any_given_single_any": float(
            (groups.loc[single_any, "multi_count"] > 0).mean()
        ),
        "p_multi_all_given_single_all": float(
            (groups.loc[single_all, "multi_count"] == 3).mean()
        ),
        "p_multi_any_given_single_none": float(
            (groups.loc[~single_any, "multi_count"] > 0).mean()
        ),
    }


def _design_matrix(frame: pd.DataFrame, *, include_single: bool) -> np.ndarray:
    strata = pd.Categorical(
        frame["stratum"],
        categories=[f"{m}|{f}" for m in MODEL_ORDER for f in FAMILY_ORDER],
    )
    dummies = pd.get_dummies(strata, drop_first=True).to_numpy(dtype=float)
    columns = [np.ones(len(frame)), dummies]
    if include_single:
        columns.append(frame[["single_rate"]].to_numpy(dtype=float))
    return np.column_stack(columns)


def _cross_validated_prediction(groups: pd.DataFrame) -> dict[str, object]:
    groups = groups.sort_values(["question_id", "model_key", "family"]).reset_index(
        drop=True
    )
    y = groups["multi_rate"].to_numpy(dtype=float)
    question_groups = groups["question_id"].to_numpy()
    splitter = GroupKFold(n_splits=10)
    predictions = {"model_family_only": np.empty(len(groups)), "plus_single": np.empty(len(groups))}
    for train, test in splitter.split(groups, y, groups=question_groups):
        for name, include_single in (
            ("model_family_only", False),
            ("plus_single", True),
        ):
            x_train = _design_matrix(groups.iloc[train], include_single=include_single)
            x_test = _design_matrix(groups.iloc[test], include_single=include_single)
            beta = np.linalg.lstsq(x_train, y[train], rcond=None)[0]
            predictions[name][test] = x_test @ beta
    metrics: dict[str, dict[str, float]] = {}
    for name, prediction in predictions.items():
        metrics[name] = {
            "r2": float(r2_score(y, prediction)),
            "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
        }

    x = _design_matrix(groups, include_single=True)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    unique_questions = np.unique(question_groups)
    for question_id in unique_questions:
        mask = question_groups == question_id
        score = x[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    correction = (
        len(unique_questions)
        / (len(unique_questions) - 1)
        * (len(groups) - 1)
        / (len(groups) - x.shape[1])
    )
    covariance = correction * bread @ meat @ bread
    single_slope = float(beta[-1])
    single_se = float(math.sqrt(max(covariance[-1, -1], 0.0)))
    t_stat = single_slope / single_se
    p_value = float(2 * stats.t.sf(abs(t_stat), df=len(unique_questions) - 1))
    return {
        "cross_validation": metrics,
        "delta_r2_from_single_turn": float(
            metrics["plus_single"]["r2"] - metrics["model_family_only"]["r2"]
        ),
        "delta_rmse_from_single_turn": float(
            metrics["plus_single"]["rmse"]
            - metrics["model_family_only"]["rmse"]
        ),
        "within_model_family_single_rate_slope": single_slope,
        "cluster_robust_se": single_se,
        "cluster_robust_p": p_value,
        "interpretation_per_10pp_single": 0.1 * single_slope,
        "cluster_unit": "question_id",
        "folds": 10,
    }


def _heterogeneity_summary(
    groups: pd.DataFrame, *, bootstrap_replicates: int, permutation_replicates: int
) -> dict[str, object]:
    y = groups["delta"].to_numpy(dtype=float)

    def fit_r2(columns: Iterable[pd.Series]) -> float:
        arrays = [np.ones(len(groups))]
        arrays.extend(column.to_numpy(dtype=float) for column in columns)
        design = np.column_stack(arrays)
        prediction = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        return float(r2_score(y, prediction))

    model_dummies = pd.get_dummies(
        pd.Categorical(groups["model_key"], categories=MODEL_ORDER), drop_first=True
    )
    family_dummies = pd.get_dummies(
        pd.Categorical(groups["family"], categories=FAMILY_ORDER), drop_first=True
    )
    stratum_dummies = pd.get_dummies(
        pd.Categorical(
            groups["stratum"],
            categories=[f"{m}|{f}" for m in MODEL_ORDER for f in FAMILY_ORDER],
        ),
        drop_first=True,
    )
    model_r2 = fit_r2([model_dummies])
    family_r2 = fit_r2([family_dummies])
    additive_r2 = fit_r2([model_dummies, family_dummies])
    stratum_r2 = fit_r2([stratum_dummies])

    question_delta = groups.groupby("question_id", observed=True)["delta"].mean()
    bootstrap = _bootstrap_cluster_mean(
        groups,
        "delta",
        replicates=bootstrap_replicates,
        seed=52_000,
    )
    low, high = _ci(bootstrap)
    per_model = groups.groupby("model_key", observed=True)["delta"].mean().to_dict()
    per_family = groups.groupby("family", observed=True)["delta"].mean().to_dict()
    return {
        "balanced_overall_delta": float(groups["delta"].mean()),
        "overall_delta_ci_low": low,
        "overall_delta_ci_high": high,
        "overall_sign_flip_p": _sign_flip_pvalue(
            question_delta.to_numpy(),
            replicates=permutation_replicates,
            seed=53_000,
        ),
        "per_model_delta": {key: float(value) for key, value in per_model.items()},
        "per_family_delta": {key: float(value) for key, value in per_family.items()},
        "delta_variance_r2": {
            "model_only": model_r2,
            "family_only": family_r2,
            "additive_model_plus_family": additive_r2,
            "model_by_family_strata": stratum_r2,
            "interaction_increment_over_additive": stratum_r2 - additive_r2,
        },
    }


def _plot_question_scatter(question: pd.DataFrame, correlations: pd.DataFrame, output: Path) -> None:
    sns.set_style("white")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        subset = question.loc[question["model_key"] == model]
        row = correlations.loc[
            (correlations["analysis"] == "question_across_families_by_model")
            & (correlations["model_key"] == model)
        ].iloc[0]
        sns.regplot(
            data=subset,
            x="single_rate",
            y="multi_rate",
            ax=ax,
            color=BLUE,
            scatter_kws={"s": 42, "alpha": 0.70, "edgecolor": "white"},
            line_kws={"color": POSITIVE, "linewidth": 2.2},
            ci=None,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#7A7F87", linewidth=1.2)
        ax.set_title(
            f"{MODEL_LABELS[model]}\nPearson r = {row['pearson_r']:.2f}; "
            f"Spearman rho = {row['spearman_rho']:.2f}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
        ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
        ax.tick_params(labelsize=12)
        ax.grid(False)
        sns.despine(ax=ax)
    fig.supxlabel("Single-turn adoption rate for the same question (%)", fontsize=15, y=0.04)
    fig.supylabel("Multi-turn adoption rate for the same question (%)", fontsize=15, x=0.04)
    fig.suptitle(
        "Question-level susceptibility partly transfers across prompt placement",
        fontsize=21,
        fontweight="bold",
        y=0.99,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=BLUE, label="Question"),
            Line2D([0], [0], color=POSITIVE, linewidth=2.2, label="Linear fit"),
            Line2D([0], [0], color="#7A7F87", linestyle="--", label="Equal response"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=True,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.05, 0.09, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_effects(effects: pd.DataFrame, output: Path) -> None:
    sns.set_style("white")
    fig, axes = plt.subplots(1, 4, figsize=(15, 6.4), sharex=True, sharey=True)
    y = np.arange(len(FAMILY_ORDER))[::-1]
    for ax, model in zip(axes, MODEL_ORDER):
        subset = (
            effects.loc[effects["model_key"] == model]
            .set_index("family")
            .loc[list(FAMILY_ORDER)]
        )
        values = 100 * subset["delta"].to_numpy()
        lows = 100 * subset["ci_low"].to_numpy()
        highs = 100 * subset["ci_high"].to_numpy()
        colors = [POSITIVE if value >= 0 else NEGATIVE for value in values]
        for index, (value, low, high, color) in enumerate(
            zip(values, lows, highs, colors)
        ):
            ax.errorbar(
                value,
                y[index],
                xerr=[[value - low], [high - value]],
                fmt="o",
                color=color,
                ecolor=color,
                markersize=8,
                capsize=4,
                linewidth=2,
            )
            ax.annotate(
                f"{value:+.1f}",
                (value, y[index]),
                xytext=(5 if value >= 0 else -5, 9),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                fontsize=11,
                color=color,
                fontweight="bold",
            )
        ax.axvline(0, color="#6E737B", linewidth=1.1)
        ax.set_title(MODEL_LABELS[model], fontsize=16, fontweight="bold")
        ax.set_yticks(y, [FAMILY_LABELS[f] for f in FAMILY_ORDER])
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        sns.despine(ax=ax, left=True)
    axes[0].set_xlim(-72, 60)
    fig.supxlabel("Multi-turn minus single-turn adoption (percentage points)", fontsize=15, y=0.08)
    fig.suptitle(
        "Prompt placement changes behavior in a model- and source-specific way",
        fontsize=21,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(
        handles=[
            Patch(facecolor=POSITIVE, label="Higher in multi-turn"),
            Patch(facecolor=NEGATIVE, label="Lower in multi-turn"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=True,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.03, 0.13, 1, 0.92))
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_count_connection(groups: pd.DataFrame, output: Path) -> None:
    sns.set_style("white")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    count_matrix = pd.crosstab(groups["single_count"], groups["multi_count"]).reindex(
        index=range(4), columns=range(4), fill_value=0
    )
    sns.heatmap(
        count_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Matched question-suggestion cells"},
        linewidths=0.8,
        linecolor="white",
        ax=axes[0],
        annot_kws={"fontsize": 13},
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Multi-turn: repetitions adopting suggestion", fontsize=15)
    axes[0].set_ylabel("Single-turn: repetitions adopting suggestion", fontsize=15)
    axes[0].set_title("Three-repetition adoption counts", fontsize=17, fontweight="bold")

    state_order = ["none", "mixed", "all"]
    state_matrix = pd.crosstab(groups["single_state"], groups["multi_state"]).reindex(
        index=state_order, columns=state_order, fill_value=0
    )
    rates = state_matrix.div(state_matrix.sum(axis=1), axis=0)
    annotations = np.asarray(
        [
            [f"{state_matrix.iloc[i, j]}\n({rates.iloc[i, j]:.0%})" for j in range(3)]
            for i in range(3)
        ]
    )
    sns.heatmap(
        rates,
        annot=annotations,
        fmt="",
        cmap="Purples",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Row proportion"},
        linewidths=0.8,
        linecolor="white",
        ax=axes[1],
        annot_kws={"fontsize": 13},
    )
    axes[1].set_xlabel("Multi-turn state", fontsize=15)
    axes[1].set_ylabel("Single-turn state", fontsize=15)
    axes[1].set_xticklabels(["None (0/3)", "Mixed (1-2/3)", "All (3/3)"], rotation=20, ha="right")
    axes[1].set_yticklabels(["None (0/3)", "Mixed (1-2/3)", "All (3/3)"], rotation=0)
    axes[1].set_title("Stability categories", fontsize=17, fontweight="bold")
    for ax in axes:
        ax.tick_params(labelsize=12)
    fig.suptitle(
        "The same question and suggestion often change state across placement",
        fontsize=21,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_ci(value: float, low: float, high: float) -> str:
    return f"{value:.2f} [{low:.2f}, {high:.2f}]"


def _write_report(
    output: Path,
    *,
    pair: pd.DataFrame,
    groups: pd.DataFrame,
    effects: pd.DataFrame,
    correlations: pd.DataFrame,
    question: pd.DataFrame,
    reliability: Mapping[str, float],
    descriptives: Mapping[str, object],
    prediction: Mapping[str, object],
    heterogeneity: Mapping[str, object],
    distribution: pd.DataFrame,
) -> None:
    raw = correlations.loc[correlations["analysis"] == "group_raw"].iloc[0]
    residual = correlations.loc[
        correlations["analysis"] == "group_within_model_family"
    ].iloc[0]
    lines = [
        "# Single-turn vs multi-turn matched-cohort analysis",
        "",
        "## Design and audit",
        "",
        f"- Exact matched responses: {len(pair):,}.",
        f"- Primary matched question-suggestion cells: {len(groups):,} (three repetitions per placement).",
        f"- Questions: {groups['question_id'].nunique()}.",
        "- Matching fields: model, question, endorsed wrong answer, source family, template, and repetition.",
        "- Primary analysis aggregates the three exchangeable API repetitions before comparing placements.",
        "",
        "## Main result",
        "",
        (
            "Averaged with equal weight over models, questions, and source families, multi-turn "
            f"adoption is {100*heterogeneity['balanced_overall_delta']:+.1f} percentage points "
            "relative to single-turn "
            f"(95% question-cluster bootstrap CI "
            f"[{100*heterogeneity['overall_delta_ci_low']:+.1f}, "
            f"{100*heterogeneity['overall_delta_ci_high']:+.1f}]). This pooled effect is not a "
            "general law: model-by-source placement effects cross zero and reverse direction."
        ),
        (
            "Single-turn behavior is informative about multi-turn behavior, but it is far from "
            "interchangeable. Across the 1,296 matched model-question-source cells, the raw "
            f"adoption-rate correlation is {_format_ci(raw['pearson_r'], raw['pearson_ci_low'], raw['pearson_ci_high'])}."
        ),
        (
            "After removing each model-by-source-family mean in both placements, the correlation "
            f"is {_format_ci(residual['pearson_r'], residual['pearson_ci_low'], residual['pearson_ci_high'])}. "
            "This residual association asks whether unusually susceptible questions in single-turn "
            "are also unusually susceptible in multi-turn, beyond model and source-family baselines."
        ),
        (
            f"Three-draw repeatability is alpha={reliability['cronbach_alpha_single']:.2f} in "
            f"single-turn and alpha={reliability['cronbach_alpha_multi']:.2f} in multi-turn. "
            f"The attenuation-corrected group-rate correlation is "
            f"{reliability['attenuation_corrected_correlation']:.2f}."
        ),
        "",
        "## Placement effects by model and source family",
        "",
        "| Model | Source family | Single | Multi | Multi - single (pp) | 95% cluster bootstrap CI | BH q |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in effects.iterrows():
        lines.append(
            f"| {row['model']} | {row['family_label']} | {100*row['single_rate']:.1f}% | "
            f"{100*row['multi_rate']:.1f}% | {100*row['delta']:+.1f} | "
            f"[{100*row['ci_low']:+.1f}, {100*row['ci_high']:+.1f}] | {row['q_bh']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The placement effect is not a single global shift: unsupported-user challenges rise "
            "substantially for Terra, Luna, and especially Sonnet, while expert and corroborating "
            "source effects often fall, most sharply for Opus.",
            (
                "Descriptively, model and source-family main effects explain "
                f"{100*heterogeneity['delta_variance_r2']['additive_model_plus_family']:.1f}% "
                "of cell-level placement-effect variance; allowing the full model-by-family "
                f"interaction explains {100*heterogeneity['delta_variance_r2']['model_by_family_strata']:.1f}%. "
                "The remaining variation is primarily question-specific."
            ),
            "",
            "## Question-level transfer",
            "",
            "| Model | Pearson r | 95% CI | Spearman rho | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    question_rows = correlations.loc[
        correlations["analysis"] == "question_across_families_by_model"
    ].set_index("model_key")
    for model in MODEL_ORDER:
        row = question_rows.loc[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {row['pearson_r']:.2f} | "
            f"[{row['pearson_ci_low']:.2f}, {row['pearson_ci_high']:.2f}] | "
            f"{row['spearman_rho']:.2f} | "
            f"[{row['spearman_ci_low']:.2f}, {row['spearman_ci_high']:.2f}] |"
        )
    cv = prediction["cross_validation"]
    lines.extend(
        [
            "",
            "## Predictive value of the matched single-turn response",
            "",
            (
                "In 10-fold cross-validation grouped by question, a model-and-source-family-only "
                f"baseline has R2={cv['model_family_only']['r2']:.3f} and "
                f"RMSE={cv['model_family_only']['rmse']:.3f}. Adding the matched single-turn "
                f"adoption rate yields R2={cv['plus_single']['r2']:.3f} and "
                f"RMSE={cv['plus_single']['rmse']:.3f}."
            ),
            (
                "The within-model-family slope is "
                f"{prediction['within_model_family_single_rate_slope']:.3f} "
                f"(cluster-robust SE={prediction['cluster_robust_se']:.3f}, "
                f"p={prediction['cluster_robust_p']:.4g}). Thus a 10-point increase in matched "
                "single-turn adoption predicts a "
                f"{100*prediction['interpretation_per_10pp_single']:.1f}-point increase in "
                "multi-turn adoption, holding model and source family fixed."
            ),
            "",
            "## Agreement and response distributions",
            "",
            f"- Quadratic kappa for the 0/3 to 3/3 adoption counts: {descriptives['group_count_quadratic_kappa']:.3f}.",
            f"- P(multi-turn adopts at least once | single-turn adopts at least once): {descriptives['p_multi_any_given_single_any']:.1%}.",
            f"- P(multi-turn always adopts | single-turn always adopts): {descriptives['p_multi_all_given_single_all']:.1%}.",
            f"- P(multi-turn adopts at least once | single-turn never adopts): {descriptives['p_multi_any_given_single_none']:.1%}.",
            f"- Mean expected exact-answer agreement between the two three-draw empirical distributions: {distribution['expected_cross_placement_exact_agreement'].mean():.1%}.",
            f"- Mean Jensen-Shannon answer-distribution similarity: {distribution['jensen_shannon_similarity'].mean():.3f} (1 is identical).",
            "",
            "## Important caveat",
            "",
            str(descriptives["note"]),
            "The analysis therefore does not interpret repetition 0 in one placement as a causally "
            "paired random draw with repetition 0 in the other. The primary correlations, effects, "
            "and uncertainty estimates operate on matched question-suggestion cells and cluster by question.",
            "",
            "## Outputs",
            "",
            "- `paired_groups.csv`: one row per matched model-question-source cell.",
            "- `model_family_effects.csv`: paired placement effects with cluster-bootstrap intervals and FDR-adjusted tests.",
            "- `correlations.csv`: raw, residualized, model-specific, family-specific, and model-family-specific associations.",
            "- `question_level.csv`: question susceptibility averaged across source families.",
            "- `analysis.json`: machine-readable audit and headline statistics.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-root", type=Path, default=DEFAULT_SINGLE)
    parser.add_argument("--multi-root", type=Path, default=DEFAULT_MULTI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=100000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    single = _prepare_placement(args.single_root.resolve(), "single_turn")
    multi = _prepare_placement(args.multi_root.resolve(), "multi_turn")
    pair = _paired_cells(single, multi)
    groups = _paired_groups(pair)
    effects = _effect_table(
        groups,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
    )
    correlations, question = _correlation_table(
        groups, bootstrap_replicates=args.bootstrap_replicates
    )
    reliability = _reliability(pair, groups)
    descriptives = _paired_descriptives(pair, groups)
    class_counts, class_rates = _class_transition(pair)
    distribution = _distribution_similarity(pair)
    prediction = _cross_validated_prediction(groups)
    heterogeneity = _heterogeneity_summary(
        groups,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
    )

    pair_output_fields = [
        "model_key",
        "question_id",
        "family",
        "repetition",
        "category_single",
        "question_single",
        "incorrect_letter_single",
        "incorrect_text_single",
        "template_id_single",
        "treatment_text_single",
        "valid_output_single",
        "valid_output_multi",
        "parsed_letter_single",
        "parsed_letter_multi",
        "response_class_single",
        "response_class_multi",
        "single_adopt",
        "multi_adopt",
        "delta_adopt",
        "exact_answer_match",
    ]
    pair[pair_output_fields].to_csv(output / "paired_cells.csv", index=False)
    groups.to_csv(output / "paired_groups.csv", index=False)
    effects.to_csv(output / "model_family_effects.csv", index=False)
    correlations.to_csv(output / "correlations.csv", index=False)
    question.to_csv(output / "question_level.csv", index=False)
    distribution.to_csv(output / "answer_distribution_similarity.csv", index=False)
    class_counts.to_csv(output / "response_class_transition_counts.csv")
    class_rates.to_csv(output / "response_class_transition_row_rates.csv")

    _plot_question_scatter(
        question, correlations, output / "question_susceptibility_connection"
    )
    _plot_effects(effects, output / "placement_effects_by_model_family")
    _plot_count_connection(groups, output / "adoption_count_connection")

    audit = {
        "exact_matched_responses": len(pair),
        "primary_matched_cells": len(groups),
        "questions": int(groups["question_id"].nunique()),
        "models": int(groups["model_key"].nunique()),
        "families": int(groups["family"].nunique()),
        "repetitions_per_cell": sorted(groups["repetitions"].unique().tolist()),
        "single_valid_rate": float(pair["valid_output_single"].mean()),
        "multi_valid_rate": float(pair["valid_output_multi"].mean()),
        "exact_keys_identical": True,
        "question_suggestion_template_fields_identical": True,
    }
    machine = {
        "audit": audit,
        "reliability": reliability,
        "paired_descriptives": descriptives,
        "prediction": prediction,
        "heterogeneity": heterogeneity,
        "mean_distribution_similarity": {
            "expected_exact_answer_agreement": float(
                distribution["expected_cross_placement_exact_agreement"].mean()
            ),
            "jensen_shannon_similarity": float(
                distribution["jensen_shannon_similarity"].mean()
            ),
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "cluster_unit": "question_id",
    }
    (output / "analysis.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        output / "report.md",
        pair=pair,
        groups=groups,
        effects=effects,
        correlations=correlations,
        question=question,
        reliability=reliability,
        descriptives=descriptives,
        prediction=prediction,
        heterogeneity=heterogeneity,
        distribution=distribution,
    )
    print(json.dumps(machine, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
