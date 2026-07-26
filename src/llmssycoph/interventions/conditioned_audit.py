from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd

from .controlled import (
    DIRECTION_CONDITIONS,
    PROTOCOL_VERSION,
    REQUIRED_CONDITIONS,
    canonical_json_hash,
    load_controlled_direction_artifact,
    read_jsonl,
    save_controlled_direction_artifact,
    sha256_file,
    sha256_text,
    stable_question_key,
    write_strict_json,
)
from .controlled_runtime import _load_sources_and_pairs, _semantic_approval_required


AUDIT_PROTOCOL_VERSION = "mean_cancellation_audit_v1_20260726"
CONDITIONED_ARTIFACT_SCHEMA_VERSION = 2
CANONICAL_LABELS = ("A", "B", "C", "D", "E")
DEFAULT_PCA_RANKS = (1, 2, 4, 8, 16)
DEFAULT_RIDGE_GRID = (1e-4, 1e-2, 1.0, 100.0)


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        return np.zeros_like(values)
    return values / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = _unit(left)
    right_unit = _unit(right)
    if not np.any(left_unit) or not np.any(right_unit):
        return 0.0
    return float(left_unit @ right_unit)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(s).all():
        raise FloatingPointError("Nonfinite scores are forbidden in the conditioned audit.")
    if np.unique(y).size != 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _question_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    return _safe_auc(
        np.concatenate(
            [np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)]
        ),
        np.concatenate([positive, negative]),
    )


def deterministic_stratified_folds(
    datasets: Sequence[str],
    endorsed_labels: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 5,
) -> np.ndarray:
    """Assign whole questions to deterministic dataset×endorsed-label folds."""

    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least two.")
    dataset_values = np.asarray(datasets, dtype=str)
    label_values = np.asarray(endorsed_labels, dtype=str)
    if dataset_values.shape != label_values.shape:
        raise ValueError("datasets and endorsed_labels must have equal shape.")
    folds = np.full(len(dataset_values), -1, dtype=int)
    rng = np.random.default_rng(int(seed))
    strata = sorted(set(zip(dataset_values.tolist(), label_values.tolist())))
    for dataset, label in strata:
        indices = np.flatnonzero(
            (dataset_values == str(dataset)) & (label_values == str(label))
        )
        shuffled = rng.permutation(indices)
        offset = int(rng.integers(0, int(n_folds)))
        folds[shuffled] = (np.arange(len(shuffled), dtype=int) + offset) % int(n_folds)
    if np.any(folds < 0):
        raise AssertionError("Every question must be assigned to exactly one fold.")
    return folds


def _bank_from_labels(
    deltas: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    delta_values = np.asarray(deltas, dtype=np.float64)
    label_values = np.asarray(labels, dtype=str)
    global_mean = delta_values[train_indices].mean(axis=0, dtype=np.float64)
    bank = np.empty((len(CANONICAL_LABELS), delta_values.shape[1]), dtype=np.float64)
    for label_index, label in enumerate(CANONICAL_LABELS):
        members = train_indices[label_values[train_indices] == label]
        bank[label_index] = (
            delta_values[members].mean(axis=0, dtype=np.float64)
            if len(members)
            else global_mean
        )
    return bank


def _vectors_for_labels(bank: np.ndarray, labels: np.ndarray) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(CANONICAL_LABELS)}
    try:
        indices = np.asarray([lookup[str(label)] for label in labels], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Noncanonical label in conditioned bank: {exc}") from exc
    return np.asarray(bank, dtype=np.float64)[indices]


def _paired_projection_scores(
    positive_states: np.ndarray,
    negative_states: np.ndarray,
    vectors: np.ndarray,
    *,
    training_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(positive_states, dtype=np.float64)
    negative = np.asarray(negative_states, dtype=np.float64)
    directions = np.asarray(vectors, dtype=np.float64)
    if directions.ndim == 1:
        directions = np.broadcast_to(directions, positive.shape)
    if directions.shape != positive.shape or negative.shape != positive.shape:
        raise ValueError("State and per-question direction shapes must match.")
    centers = np.asarray(training_centers, dtype=np.float64)
    if centers.ndim == 1:
        centers = np.broadcast_to(centers, positive.shape)
    if centers.shape != positive.shape:
        raise ValueError("Training centers must be [hidden] or [question, hidden].")
    return (
        np.einsum("ij,ij->i", positive - centers, directions),
        np.einsum("ij,ij->i", negative - centers, directions),
    )


def _centers_by_label(
    positive_states: np.ndarray,
    negative_states: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    positive = np.asarray(positive_states, dtype=np.float64)
    negative = np.asarray(negative_states, dtype=np.float64)
    label_values = np.asarray(labels, dtype=str)
    global_center = 0.5 * (
        positive[train_indices].mean(axis=0, dtype=np.float64)
        + negative[train_indices].mean(axis=0, dtype=np.float64)
    )
    centers = np.empty((len(CANONICAL_LABELS), positive.shape[1]), dtype=np.float64)
    for label_index, label in enumerate(CANONICAL_LABELS):
        members = train_indices[label_values[train_indices] == label]
        centers[label_index] = (
            0.5
            * (
                positive[members].mean(axis=0, dtype=np.float64)
                + negative[members].mean(axis=0, dtype=np.float64)
            )
            if len(members)
            else global_center
        )
    return centers


def _bootstrap_auc_interval(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError("Bootstrap inputs must be equal-length question score vectors.")
    observed = _question_auc(positive, negative)
    if len(positive) <= 1 or int(n_bootstrap) <= 0:
        return observed, observed, observed
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_bootstrap), dtype=np.float64)
    for index in range(int(n_bootstrap)):
        sampled = rng.integers(0, len(positive), size=len(positive))
        draws[index] = _question_auc(positive[sampled], negative[sampled])
    return (
        observed,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _subspace_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_rank = int(np.linalg.matrix_rank(left_values))
    right_rank = int(np.linalg.matrix_rank(right_values))
    rank = min(left_rank, right_rank)
    if rank <= 0:
        return 0.0
    q_left, _ = np.linalg.qr(left_values.T, mode="reduced")
    q_right, _ = np.linalg.qr(right_values.T, mode="reduced")
    singular = np.linalg.svd(q_left[:, :left_rank].T @ q_right[:, :right_rank], compute_uv=False)
    return float(np.mean(np.square(singular[:rank])))


def _split_half_stability(
    deltas: np.ndarray,
    labels: np.ndarray,
    datasets: np.ndarray,
    *,
    family: str,
    belief_classes: np.ndarray,
    n_repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(int(seed))
    similarities: list[float] = []
    strata = sorted(set(zip(datasets.tolist(), labels.tolist())))
    for _ in range(int(n_repeats)):
        halves: list[list[int]] = [[], []]
        for dataset, label in strata:
            members = np.flatnonzero((datasets == dataset) & (labels == label))
            shuffled = rng.permutation(members)
            halves[0].extend(shuffled[::2].tolist())
            halves[1].extend(shuffled[1::2].tolist())
        first = np.asarray(sorted(halves[0]), dtype=int)
        second = np.asarray(sorted(halves[1]), dtype=int)
        if family == "global_wc":
            similarities.append(
                abs(
                    _cosine(
                        deltas[first].mean(axis=0, dtype=np.float64),
                        deltas[second].mean(axis=0, dtype=np.float64),
                    )
                )
            )
        elif family == "b_conditioned_wc":
            left = _bank_from_labels(deltas, labels, first)
            right = _bank_from_labels(deltas, labels, second)
            similarities.append(_subspace_similarity(left, right))
        elif family == "belief_conflict":
            oriented = np.where(
                (belief_classes == "neutral_is_c")[:, None],
                deltas,
                -deltas,
            )
            first = first[belief_classes[first] != "neutral_is_other"]
            second = second[belief_classes[second] != "neutral_is_other"]
            if len(first) and len(second):
                similarities.append(abs(_cosine(oriented[first].mean(0), oriented[second].mean(0))))
        else:
            raise KeyError(f"Unsupported split-half family={family!r}.")
    if not similarities:
        return float("nan"), float("nan"), float("nan")
    values = np.asarray(similarities, dtype=np.float64)
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _sum_to_zero_label_binding(
    deltas: np.ndarray,
    endorsed: np.ndarray,
    correct: np.ndarray,
    train_indices: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(CANONICAL_LABELS)}
    design = np.zeros((len(train_indices), len(CANONICAL_LABELS)), dtype=np.float64)
    for row_index, question_index in enumerate(train_indices):
        design[row_index, lookup[str(endorsed[question_index])]] += 1.0
        design[row_index, lookup[str(correct[question_index])]] -= 1.0
    # An orthonormal contrast basis enforces sum_b u_b = 0 exactly.
    contrast = np.linalg.qr(
        np.column_stack(
            [
                np.eye(len(CANONICAL_LABELS) - 1),
                -np.ones(len(CANONICAL_LABELS) - 1),
            ]
        ).T
    )[0][:, : len(CANONICAL_LABELS) - 1]
    reduced = design @ contrast
    gram = reduced.T @ reduced + float(ridge) * np.eye(reduced.shape[1])
    coefficients = np.linalg.solve(
        gram,
        reduced.T @ np.asarray(deltas, dtype=np.float64)[train_indices],
    )
    bank = contrast @ coefficients
    if not np.allclose(bank.sum(axis=0), 0.0, atol=1e-8):
        raise AssertionError("Label-binding bank violated its sum-to-zero constraint.")
    return bank


def _binding_vectors(
    bank: np.ndarray,
    endorsed: np.ndarray,
    correct: np.ndarray,
) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(CANONICAL_LABELS)}
    return np.stack(
        [
            bank[lookup[str(b)]] - bank[lookup[str(c)]]
            for b, c in zip(endorsed, correct)
        ],
        axis=0,
    )


def _select_ridge(
    deltas: np.ndarray,
    endorsed: np.ndarray,
    correct: np.ndarray,
    train_indices: np.ndarray,
    folds: np.ndarray,
    *,
    ridge_grid: Sequence[float],
) -> float:
    candidates: list[tuple[float, float]] = []
    train_folds = sorted(set(folds[train_indices].tolist()))
    for ridge in ridge_grid:
        errors: list[float] = []
        for held_fold in train_folds:
            inner_test = train_indices[folds[train_indices] == held_fold]
            inner_train = train_indices[folds[train_indices] != held_fold]
            if not len(inner_test) or not len(inner_train):
                continue
            bank = _sum_to_zero_label_binding(
                deltas,
                endorsed,
                correct,
                inner_train,
                ridge=float(ridge),
            )
            predicted = _binding_vectors(bank, endorsed[inner_test], correct[inner_test])
            errors.append(
                float(np.mean(np.square(deltas[inner_test] - predicted), dtype=np.float64))
            )
        candidates.append((float(np.mean(errors)) if errors else float("inf"), float(ridge)))
    return min(candidates)[1]


def _fit_pca(delta_train: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(delta_train, dtype=np.float64)
    mean = values.mean(axis=0, dtype=np.float64)
    selected = min(int(rank), len(values) - 1, values.shape[1])
    if selected <= 0:
        return mean, np.empty((0, values.shape[1]), dtype=np.float64)
    _, _, vh = randomized_svd(
        values - mean,
        n_components=selected,
        n_iter=5,
        random_state=5,
        flip_sign=True,
    )
    return mean, vh


def _captured_energy(
    values: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
) -> float:
    centered = np.asarray(values, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    total = float(np.square(centered).sum())
    if total <= np.finfo(np.float64).tiny:
        return 0.0
    projected = centered @ np.asarray(basis, dtype=np.float64).T
    return float(np.square(projected).sum() / total)


def _select_pca_rank(
    deltas: np.ndarray,
    train_indices: np.ndarray,
    folds: np.ndarray,
    ranks: Sequence[int],
) -> int:
    fold_captured: Dict[int, list[float]] = {int(rank): [] for rank in ranks}
    maximum_rank = max(int(rank) for rank in ranks)
    for inner_fold in sorted(set(folds[train_indices].tolist())):
        inner_test = train_indices[folds[train_indices] == inner_fold]
        inner_train = train_indices[folds[train_indices] != inner_fold]
        if not len(inner_test) or len(inner_train) < 2:
            continue
        mean, maximum_basis = _fit_pca(deltas[inner_train], maximum_rank)
        for rank in ranks:
            basis = maximum_basis[: min(int(rank), len(maximum_basis))]
            fold_captured[int(rank)].append(
                _captured_energy(deltas[inner_test], mean, basis)
            )
    losses = [
        (
            1.0 - float(np.mean(fold_captured[int(rank)]))
            if fold_captured[int(rank)]
            else float("inf"),
            int(rank),
        )
        for rank in ranks
    ]
    return min(losses)[1]


@dataclass(frozen=True)
class CompactLeaceEraser:
    mean: np.ndarray
    proj_left: np.ndarray
    proj_right: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        centered = array - self.mean
        return array - (centered @ self.proj_right)[:, None] * self.proj_left[None, :]


def fit_weighted_binary_leace(
    states: Mapping[str, np.ndarray],
) -> CompactLeaceEraser:
    """Fit rank-one weighted LEACE in the exact nonzero sample span."""

    neutral = np.asarray(states["neutral"], dtype=np.float64)
    suggested = np.concatenate(
        [
            np.asarray(states["incorrect_suggestion"], dtype=np.float64),
            np.asarray(states["suggest_correct"], dtype=np.float64),
            np.asarray(states["incorrect_suggestion_strong"], dtype=np.float64),
        ],
        axis=0,
    )
    x = np.concatenate([neutral, suggested], axis=0)
    z = np.concatenate(
        [np.zeros(len(neutral), dtype=np.float64), np.ones(len(suggested), dtype=np.float64)]
    )
    # Give neutral and suggestion-present classes equal total weight.
    weights = np.concatenate(
        [
            np.full(len(neutral), 0.5 / len(neutral), dtype=np.float64),
            np.full(len(suggested), 0.5 / len(suggested), dtype=np.float64),
        ]
    )
    mean_x = np.einsum("i,ij->j", weights, x)
    mean_z = float(weights @ z)
    centered = x - mean_x
    weighted = centered * np.sqrt(weights)[:, None]
    _, singular, vh = np.linalg.svd(weighted, full_matrices=False)
    eigenvalues = np.square(singular)
    positive = eigenvalues > (
        eigenvalues.max(initial=0.0)
        * max(weighted.shape)
        * np.finfo(np.float64).eps
    )
    basis = vh[positive].T
    eigenvalues = eigenvalues[positive]
    cross = np.einsum("i,ij,i->j", weights, centered, z - mean_z)
    if not len(eigenvalues) or np.linalg.norm(cross) <= np.finfo(np.float64).tiny:
        return CompactLeaceEraser(
            mean=mean_x,
            proj_left=np.zeros(x.shape[1], dtype=np.float64),
            proj_right=np.zeros(x.shape[1], dtype=np.float64),
        )
    # OAS-style shrinkage toward the mean variance, computed without d×d storage.
    dimension = x.shape[1]
    n_effective = 1.0 / float(np.square(weights).sum())
    trace = float(eigenvalues.sum())
    trace_square = float(np.square(eigenvalues).sum())
    denominator = (n_effective + 1.0 - 2.0 / dimension) * (
        trace_square - trace * trace / dimension
    )
    numerator = (1.0 - 2.0 / dimension) * trace_square + trace * trace
    shrinkage = 1.0 if denominator <= 0 else min(1.0, numerator / denominator)
    target = trace / dimension
    shrunk = (1.0 - shrinkage) * eigenvalues + shrinkage * target
    cross_coordinates = basis.T @ cross
    whitened = basis @ (cross_coordinates / np.sqrt(shrunk))
    whitened_unit = _unit(whitened)
    if not np.any(whitened_unit):
        return CompactLeaceEraser(
            mean=mean_x,
            proj_left=np.zeros(x.shape[1], dtype=np.float64),
            proj_right=np.zeros(x.shape[1], dtype=np.float64),
        )
    unit_coordinates = basis.T @ whitened_unit
    proj_left = basis @ (np.sqrt(shrunk) * unit_coordinates)
    proj_right = basis @ (unit_coordinates / np.sqrt(shrunk))
    normalization = float(proj_right @ proj_left)
    if abs(normalization) <= np.finfo(np.float64).tiny:
        raise FloatingPointError("Degenerate LEACE projection.")
    proj_right = proj_right / normalization
    return CompactLeaceEraser(
        mean=mean_x,
        proj_left=proj_left,
        proj_right=proj_right,
    )


def _fold_geometry(
    states: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Dict[str, float]:
    train_pooled = np.concatenate(
        [np.asarray(states[condition], dtype=np.float64)[train_indices] for condition in REQUIRED_CONDITIONS],
        axis=0,
    )
    train_mean = train_pooled.mean(axis=0, dtype=np.float64)
    centered = train_pooled - train_mean
    # Work in the sample Gram matrix.  This preserves the exact nonzero
    # sample-span spectrum without materializing a hidden×hidden covariance.
    gram = centered @ centered.T
    eigenvalues_raw, left_vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues_raw = np.clip(eigenvalues_raw[order], 0.0, None)
    left_vectors = left_vectors[:, order]
    singular = np.sqrt(eigenvalues_raw)
    eigenvalues = eigenvalues_raw / max(1, len(centered) - 1)
    positive = eigenvalues > (
        eigenvalues.max(initial=0.0)
        * max(centered.shape)
        * np.finfo(np.float64).eps
    )
    singular = singular[positive]
    left_vectors = left_vectors[:, positive]
    eigenvalues = eigenvalues[positive]
    # Transform statistics are fitted on train_pooled above, but every reported
    # direction below is a centroid difference from the held-out questions.
    wn = (
        states["incorrect_suggestion"][test_indices]
        - states["neutral"][test_indices]
    ).mean(axis=0, dtype=np.float64)
    cn = (
        states["suggest_correct"][test_indices]
        - states["neutral"][test_indices]
    ).mean(axis=0, dtype=np.float64)
    wc = (
        states["incorrect_suggestion"][test_indices]
        - states["suggest_correct"][test_indices]
    ).mean(axis=0, dtype=np.float64)
    wn_norm = max(float(np.linalg.norm(wn)), np.finfo(np.float64).tiny)
    result: Dict[str, float] = {
        "raw_wn_cn_cosine": _cosine(wn, cn),
        "raw_wc_over_wn_norm": float(np.linalg.norm(wc) / wn_norm),
        # Common affine mean-centering cancels algebraically from all differences.
        "common_centering_max_abs_difference": float(
            max(
                np.max(np.abs(((states["incorrect_suggestion"][test_indices] - train_mean)
                    - (states["neutral"][test_indices] - train_mean))
                    - (states["incorrect_suggestion"][test_indices] - states["neutral"][test_indices]))),
                np.max(np.abs(((states["suggest_correct"][test_indices] - train_mean)
                    - (states["neutral"][test_indices] - train_mean))
                    - (states["suggest_correct"][test_indices] - states["neutral"][test_indices]))),
            )
        ),
    }
    scale = train_pooled.std(axis=0, ddof=1)
    scale = np.where(scale > np.finfo(np.float64).tiny, scale, 1.0)
    result["standardized_wn_cn_cosine"] = _cosine(wn / scale, cn / scale)
    result["standardized_wc_over_wn_norm"] = float(
        np.linalg.norm(wc / scale)
        / max(float(np.linalg.norm(wn / scale)), np.finfo(np.float64).tiny)
    )
    for count in range(1, 6):
        selected_count = min(count, len(eigenvalues))
        if selected_count:
            # V^T v = U^T X v / s, evaluated only for the requested PCs.
            wn_coordinates = (
                left_vectors[:, :selected_count].T @ (centered @ wn)
            ) / singular[:selected_count]
            cn_coordinates = (
                left_vectors[:, :selected_count].T @ (centered @ cn)
            ) / singular[:selected_count]
            wc_coordinates = (
                left_vectors[:, :selected_count].T @ (centered @ wc)
            ) / singular[:selected_count]
        else:
            wn_coordinates = cn_coordinates = wc_coordinates = np.empty(0)
        wn_removed_sq = max(
            0.0, float(wn @ wn) - float(wn_coordinates @ wn_coordinates)
        )
        cn_removed_sq = max(
            0.0, float(cn @ cn) - float(cn_coordinates @ cn_coordinates)
        )
        wc_removed_sq = max(
            0.0, float(wc @ wc) - float(wc_coordinates @ wc_coordinates)
        )
        removed_dot = float(wn @ cn) - float(wn_coordinates @ cn_coordinates)
        result[f"remove_pc_{count}_wn_cn_cosine"] = float(
            removed_dot
            / max(
                math.sqrt(wn_removed_sq * cn_removed_sq),
                np.finfo(np.float64).tiny,
            )
        )
        result[f"remove_pc_{count}_wc_over_wn_norm"] = float(
            math.sqrt(wc_removed_sq)
            / max(math.sqrt(wn_removed_sq), np.finfo(np.float64).tiny)
        )
    if len(eigenvalues):
        dimension = centered.shape[1]
        trace = float(eigenvalues.sum())
        trace_square = float(np.square(eigenvalues).sum())
        denominator = (len(centered) + 1.0 - 2.0 / dimension) * (
            trace_square - trace * trace / dimension
        )
        numerator = (1.0 - 2.0 / dimension) * trace_square + trace * trace
        shrinkage = 1.0 if denominator <= 0 else min(1.0, numerator / denominator)
        shrunk = (1.0 - shrinkage) * eigenvalues + shrinkage * trace / dimension

        def whiten(vector: np.ndarray) -> np.ndarray:
            span_coordinates = (left_vectors.T @ (centered @ vector)) / singular
            return span_coordinates / np.sqrt(shrunk)

        wn_white = whiten(wn)
        cn_white = whiten(cn)
        wc_white = whiten(wc)
        result["whitened_wn_cn_cosine"] = _cosine(wn_white, cn_white)
        result["whitened_wc_over_wn_norm"] = float(
            np.linalg.norm(wc_white)
            / max(float(np.linalg.norm(wn_white)), np.finfo(np.float64).tiny)
        )
        result["whitening_span_rank"] = int(len(eigenvalues))
        result["whitening_shrinkage"] = float(shrinkage)
    else:
        result["whitened_wn_cn_cosine"] = float("nan")
        result["whitened_wc_over_wn_norm"] = float("nan")
        result["whitening_span_rank"] = 0
        result["whitening_shrinkage"] = float("nan")
    return result


def _nested_logistic_scores(
    positive: np.ndarray,
    negative: np.ndarray,
    folds: np.ndarray,
    *,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
) -> tuple[np.ndarray, np.ndarray]:
    n_questions = len(folds)
    positive_oof = np.full(n_questions, np.nan, dtype=np.float64)
    negative_oof = np.full(n_questions, np.nan, dtype=np.float64)
    for outer_fold in sorted(set(folds.tolist())):
        train = np.flatnonzero(folds != outer_fold)
        test = np.flatnonzero(folds == outer_fold)
        best: tuple[float, float] = (-float("inf"), float(c_grid[0]))
        for c_value in c_grid:
            inner_scores: list[float] = []
            for inner_fold in sorted(set(folds[train].tolist())):
                inner_train = train[folds[train] != inner_fold]
                inner_test = train[folds[train] == inner_fold]
                if not len(inner_train) or not len(inner_test):
                    continue
                x_train = np.concatenate([positive[inner_train], negative[inner_train]], axis=0)
                y_train = np.concatenate(
                    [np.ones(len(inner_train), dtype=int), np.zeros(len(inner_train), dtype=int)]
                )
                classifier = LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    solver="liblinear",
                    dual=True,
                    max_iter=2000,
                    random_state=5,
                ).fit(x_train, y_train)
                inner_scores.append(
                    _question_auc(
                        classifier.decision_function(positive[inner_test]),
                        classifier.decision_function(negative[inner_test]),
                    )
                )
            candidate = (float(np.mean(inner_scores)), -float(c_value))
            if candidate > (best[0], -best[1]):
                best = (candidate[0], float(c_value))
        x_train = np.concatenate([positive[train], negative[train]], axis=0)
        y_train = np.concatenate(
            [np.ones(len(train), dtype=int), np.zeros(len(train), dtype=int)]
        )
        classifier = LogisticRegression(
            C=best[1],
            penalty="l2",
            solver="liblinear",
            dual=True,
            max_iter=2000,
            random_state=5,
        ).fit(x_train, y_train)
        positive_oof[test] = classifier.decision_function(positive[test])
        negative_oof[test] = classifier.decision_function(negative[test])
    if not np.isfinite(positive_oof).all() or not np.isfinite(negative_oof).all():
        raise FloatingPointError("Nested logistic regression left nonfinite OOF scores.")
    return positive_oof, negative_oof


def _permuted_bank_aucs(
    deltas: np.ndarray,
    wrong_states: np.ndarray,
    correct_states: np.ndarray,
    labels: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Efficient label-placebo CV using precomputed state×delta Gram matrices."""

    n = len(labels)
    results = np.empty(int(n_permutations), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    fold_cache: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for fold in sorted(set(folds.tolist())):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        train_deltas = np.asarray(deltas[train], dtype=np.float64)
        wrong_gram = np.asarray(wrong_states[test], dtype=np.float64) @ train_deltas.T
        correct_gram = (
            np.asarray(correct_states[test], dtype=np.float64) @ train_deltas.T
        )
        train_midpoints = 0.5 * np.asarray(
            wrong_states[train] + correct_states[train], dtype=np.float64
        )
        center_gram = train_midpoints @ train_deltas.T
        fold_cache.append((train, test, wrong_gram, correct_gram, center_gram))
    for permutation_index in range(int(n_permutations)):
        permuted = labels.copy()
        for dataset in sorted(set(datasets.tolist())):
            members = np.flatnonzero(datasets == dataset)
            permuted[members] = rng.permutation(permuted[members])
        positive_oof = np.full(n, np.nan, dtype=np.float64)
        negative_oof = np.full(n, np.nan, dtype=np.float64)
        for train, test, wrong_gram, correct_gram, center_gram in fold_cache:
            positive_scores = np.empty(len(test), dtype=np.float64)
            negative_scores = np.empty(len(test), dtype=np.float64)
            offsets: Dict[str, float] = {}
            for label in CANONICAL_LABELS:
                members = permuted[train] == label
                if not np.any(members):
                    members = np.ones(len(train), dtype=bool)
                member_indices = np.flatnonzero(members)
                offsets[label] = float(
                    center_gram[np.ix_(member_indices, member_indices)].mean()
                )
            for test_row, question_index in enumerate(test):
                members = permuted[train] == permuted[question_index]
                if not np.any(members):
                    members = np.ones(len(train), dtype=bool)
                offset = offsets[str(permuted[question_index])]
                positive_scores[test_row] = (
                    float(wrong_gram[test_row, members].mean()) - offset
                )
                negative_scores[test_row] = (
                    float(correct_gram[test_row, members].mean()) - offset
                )
            positive_oof[test] = positive_scores
            negative_oof[test] = negative_scores
        results[permutation_index] = _question_auc(positive_oof, negative_oof)
    return results


def _permuted_belief_aucs(
    deltas: np.ndarray,
    wrong_states: np.ndarray,
    correct_states: np.ndarray,
    belief_classes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    n = len(belief_classes)
    rng = np.random.default_rng(int(seed))
    fold_cache = []
    for fold in sorted(set(folds.tolist())):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        train_deltas = np.asarray(deltas[train], dtype=np.float64)
        wrong_gram = np.asarray(wrong_states[test], dtype=np.float64) @ train_deltas.T
        correct_gram = (
            np.asarray(correct_states[test], dtype=np.float64) @ train_deltas.T
        )
        center_gram = (
            0.5
            * np.asarray(
                wrong_states[train] + correct_states[train], dtype=np.float64
            )
            @ train_deltas.T
        )
        fold_cache.append((train, test, wrong_gram, correct_gram, center_gram))
    results = np.empty(int(n_permutations), dtype=np.float64)
    for permutation_index in range(int(n_permutations)):
        permuted = belief_classes.copy()
        for dataset in sorted(set(datasets.tolist())):
            members = np.flatnonzero(datasets == dataset)
            permuted[members] = rng.permutation(permuted[members])
        positives: list[float] = []
        negatives: list[float] = []
        for train, test, wrong_gram, correct_gram, center_gram in fold_cache:
            train_eligible = permuted[train] != "neutral_is_other"
            signs = np.where(
                permuted[train] == "neutral_is_c",
                1.0,
                np.where(permuted[train] == "neutral_is_b", -1.0, 0.0),
            )
            count = int(np.count_nonzero(train_eligible))
            if count <= 0:
                continue
            signed_weights = signs / count
            center_rows = np.flatnonzero(train_eligible)
            offset = float(
                np.mean(center_gram[center_rows] @ signed_weights)
            )
            positive_wrong = wrong_gram @ signed_weights - offset
            positive_correct = correct_gram @ signed_weights - offset
            for row_index, question_index in enumerate(test):
                if permuted[question_index] == "neutral_is_c":
                    positives.append(float(positive_wrong[row_index]))
                    negatives.append(float(positive_correct[row_index]))
                elif permuted[question_index] == "neutral_is_b":
                    positives.append(float(positive_correct[row_index]))
                    negatives.append(float(positive_wrong[row_index]))
        results[permutation_index] = _question_auc(
            np.asarray(positives), np.asarray(negatives)
        )
    return results


def _metadata_from_pairs(
    pairs: Sequence[Mapping[str, Any]],
    artifact: Any,
) -> pd.DataFrame:
    ordered_keys = [str(pair["stable_question_key"]) for pair in pairs]
    expected_hash = str(artifact.metadata.get("question_keys_sha256", ""))
    actual_hash = sha256_text("\n".join(ordered_keys))
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            "Reconstructed ordered question-key hash does not match the direction artifact: "
            f"{actual_hash} != {expected_hash}."
        )
    rows = []
    for pair in pairs:
        manifest = dict(pair["manifest_row"])
        neutral_source = str(pair["neutral_selected_choice"])
        choice_map = dict(pair["choice_label_map"])
        neutral = choice_map.get(neutral_source, "")
        correct = str(pair["canonical_correct_choice"])
        endorsed = str(pair["canonical_endorsed_choice"])
        belief_class = (
            "neutral_is_c"
            if neutral == correct
            else "neutral_is_b"
            if neutral == endorsed
            else "neutral_is_other"
        )
        condition_probabilities: Dict[str, float] = {}
        for condition in REQUIRED_CONDITIONS:
            probabilities = dict(
                pair["records"][condition].get("choice_probabilities", {}) or {}
            )
            for semantic_name, canonical_label in (
                ("correct", correct),
                ("endorsed", endorsed),
            ):
                source_labels = [
                    source_label
                    for source_label, mapped in choice_map.items()
                    if mapped == canonical_label
                ]
                value = (
                    float(probabilities[source_labels[0]])
                    if len(source_labels) == 1
                    and source_labels[0] in probabilities
                    else float("nan")
                )
                condition_probabilities[
                    f"source_p_{semantic_name}_{condition}"
                ] = value
        rows.append(
            {
                "stable_question_key": pair["stable_question_key"],
                "dataset": pair["dataset"],
                "split": pair["split"],
                "endorsed_choice": endorsed,
                "correct_choice": correct,
                "source_neutral_choice": neutral,
                "belief_class": belief_class,
                "question_id": manifest.get("question_id", pair.get("question_id", "")),
                **condition_probabilities,
            }
        )
    return pd.DataFrame(rows)


def _artifact_training_states(artifact: Any, layer_index: int) -> Dict[str, np.ndarray]:
    states = {
        condition: np.asarray(
            artifact.arrays[f"training_states_{condition}"][:, layer_index, :],
            dtype=np.float64,
        )
        for condition in REQUIRED_CONDITIONS
    }
    shapes = {values.shape for values in states.values()}
    if len(shapes) != 1:
        raise ValueError(f"Training-state shapes differ: {sorted(shapes)}")
    if not all(np.isfinite(values).all() for values in states.values()):
        raise FloatingPointError("Saved training activations contain NaN or Inf.")
    return states


def _full_conditioned_arrays(
    artifact: Any,
    metadata: pd.DataFrame,
    *,
    selected_ranks: Mapping[int, int],
) -> Dict[str, np.ndarray]:
    layers = artifact.layers
    labels = metadata["endorsed_choice"].to_numpy(dtype=str)
    correct = metadata["correct_choice"].to_numpy(dtype=str)
    belief = metadata["belief_class"].to_numpy(dtype=str)
    datasets = metadata["dataset"].to_numpy(dtype=str)
    all_indices = np.arange(len(metadata), dtype=int)
    arc_indices = np.flatnonzero(datasets == "arc_challenge")
    if not len(arc_indices):
        raise ValueError("Conditioned Stage-B artifacts require ARC training rows.")
    hidden = int(artifact.arrays["training_states_neutral"].shape[-1])
    label_banks = np.empty((len(layers), len(CANONICAL_LABELS), hidden), dtype=np.float32)
    binding_banks = np.empty_like(label_banks)
    belief_directions = np.empty((len(layers), hidden), dtype=np.float32)
    arc_label_banks = np.empty_like(label_banks)
    arc_belief_directions = np.empty_like(belief_directions)
    arc_wc = np.empty_like(belief_directions)
    arc_wn = np.empty_like(belief_directions)
    pca_bases = np.zeros(
        (len(layers), max(DEFAULT_PCA_RANKS), hidden),
        dtype=np.float32,
    )
    pca_means = np.empty((len(layers), hidden), dtype=np.float32)
    pca_rank = np.empty(len(layers), dtype=np.int16)
    for layer_index, layer in enumerate(layers):
        states = _artifact_training_states(artifact, layer_index)
        deltas = states["incorrect_suggestion"] - states["suggest_correct"]
        label_banks[layer_index] = _bank_from_labels(
            deltas, labels, all_indices
        ).astype(np.float32)
        binding_banks[layer_index] = _sum_to_zero_label_binding(
            deltas,
            labels,
            correct,
            all_indices,
            ridge=1.0,
        ).astype(np.float32)
        oriented = np.where(
            (belief == "neutral_is_c")[:, None],
            deltas,
            -deltas,
        )
        fit = belief != "neutral_is_other"
        belief_directions[layer_index] = oriented[fit].mean(axis=0).astype(np.float32)
        arc_label_banks[layer_index] = _bank_from_labels(
            deltas, labels, arc_indices
        ).astype(np.float32)
        arc_fit = arc_indices[belief[arc_indices] != "neutral_is_other"]
        arc_belief_directions[layer_index] = oriented[arc_fit].mean(axis=0).astype(
            np.float32
        )
        arc_wc[layer_index] = deltas[arc_indices].mean(axis=0).astype(np.float32)
        arc_wn[layer_index] = (
            states["incorrect_suggestion"][arc_indices]
            - states["neutral"][arc_indices]
        ).mean(axis=0).astype(np.float32)
        rank = int(selected_ranks[int(layer)])
        mean, basis = _fit_pca(deltas, rank)
        pca_means[layer_index] = mean.astype(np.float32)
        pca_bases[layer_index, : len(basis)] = basis.astype(np.float32)
        pca_rank[layer_index] = int(len(basis))
    arrays = {
        name: np.asarray(values)
        for name, values in artifact.arrays.items()
        if name in {"layers", "control_seeds"}
        or name.endswith("_raw")
        or name.startswith("centroid_")
    }
    arrays.update(
        {
            "conditioned_labels": np.asarray(CANONICAL_LABELS, dtype="U1"),
            "training_dataset": datasets.astype("U32"),
            "training_endorsed_choice": labels.astype("U1"),
            "training_correct_choice": correct.astype("U1"),
            "training_belief_class": belief.astype("U32"),
            "b_conditioned_wc_bank": label_banks,
            "label_binding_wc_bank": binding_banks,
            "belief_conflict_direction": belief_directions,
            "arc_b_conditioned_wc_bank": arc_label_banks,
            "arc_belief_conflict_direction": arc_belief_directions,
            "arc_wc_raw": arc_wc,
            "arc_wn_raw": arc_wn,
            "wc_low_rank_basis": pca_bases,
            "wc_low_rank_mean": pca_means,
            "wc_low_rank_selected_rank": pca_rank,
        }
    )
    return arrays


def run_mean_cancellation_audit(
    *,
    config_path: Path,
    question_manifest_path: Path,
    cells: Sequence[tuple[Path, Sequence[Path]]],
    output_dir: Path,
    n_folds: int = 5,
    n_permutations: int = 1000,
    n_bootstrap: int = 2000,
    n_split_half: int = 200,
    seed: int = 5,
) -> Path:
    """Audit saved training activations and emit an immutable two-model decision."""

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    manifest_rows = read_jsonl(question_manifest_path)
    manifest_hash = sha256_file(question_manifest_path)
    if not cells:
        raise ValueError("At least one directions/source cell is required.")

    layer_rows: list[Dict[str, Any]] = []
    permutation_rows: list[Dict[str, Any]] = []
    geometry_rows: list[Dict[str, Any]] = []
    pca_rows: list[Dict[str, Any]] = []
    leace_rows: list[Dict[str, Any]] = []
    behavioral_rows: list[Dict[str, Any]] = []
    model_summaries: list[Dict[str, Any]] = []
    audit_artifacts: list[Dict[str, Any]] = []

    for cell_index, (directions_path, source_run_dirs) in enumerate(cells):
        artifact = load_controlled_direction_artifact(directions_path)
        sources, pairs, _ = _load_sources_and_pairs(
            source_run_dirs,
            manifest_path=question_manifest_path,
            splits=("train",),
            require_human_approval=_semantic_approval_required(config),
            require_probe=False,
        )
        model_name = str(artifact.metadata.get("model_name", ""))
        if not model_name or {source.model_name for source in sources} != {model_name}:
            raise ValueError("Direction/source model mismatch in audit cell.")
        metadata = _metadata_from_pairs(pairs, artifact)
        labels = metadata["endorsed_choice"].to_numpy(dtype=str)
        correct = metadata["correct_choice"].to_numpy(dtype=str)
        datasets = metadata["dataset"].to_numpy(dtype=str)
        belief = metadata["belief_class"].to_numpy(dtype=str)
        folds = deterministic_stratified_folds(
            datasets,
            labels,
            n_folds=int(n_folds),
            seed=int(seed),
        )
        metadata["fold"] = folds
        metadata_path = target / f"question_metadata_model_{cell_index}.csv"
        metadata.to_csv(metadata_path, index=False)
        for dataset_scope, scope_frame in [
            ("pooled", metadata),
            *[
                (str(dataset), group)
                for dataset, group in metadata.groupby("dataset")
            ],
        ]:
            for belief_scope, behavior_frame in [
                ("all", scope_frame),
                *[
                    (str(value), group)
                    for value, group in scope_frame.groupby("belief_class")
                ],
            ]:
                for contrast, positive, negative in (
                    (
                        "wrong_minus_neutral",
                        "incorrect_suggestion",
                        "neutral",
                    ),
                    (
                        "wrong_minus_correct_suggestion",
                        "incorrect_suggestion",
                        "suggest_correct",
                    ),
                ):
                    p_b_gap = (
                        behavior_frame[f"source_p_endorsed_{positive}"]
                        - behavior_frame[f"source_p_endorsed_{negative}"]
                    )
                    p_c_gap = (
                        behavior_frame[f"source_p_correct_{positive}"]
                        - behavior_frame[f"source_p_correct_{negative}"]
                    )
                    behavioral_rows.append(
                        {
                            "model_name": model_name,
                            "dataset": dataset_scope,
                            "belief_class": belief_scope,
                            "contrast": contrast,
                            "n_questions": len(behavior_frame),
                            "mean_source_delta_p_endorsed": float(
                                p_b_gap.mean()
                            ),
                            "mean_source_delta_p_correct": float(p_c_gap.mean()),
                            "finite_p_endorsed_questions": int(
                                np.isfinite(p_b_gap.to_numpy(dtype=float)).sum()
                            ),
                        }
                    )

        probe_coefficient: Optional[np.ndarray] = None
        for source in sources:
            model_path = source.chosen_probe_dir / "model.pkl"
            if model_path.exists():
                with model_path.open("rb") as handle:
                    classifier = pickle.load(handle)
                coefficient = np.asarray(classifier.coef_, dtype=np.float64).reshape(-1)
                if coefficient.shape[0] == artifact.arrays["training_states_neutral"].shape[-1]:
                    probe_coefficient = coefficient
                    break

        selected_ranks: Dict[int, int] = {}
        for layer_index, layer_value in enumerate(artifact.layers):
            layer = int(layer_value)
            states = _artifact_training_states(artifact, layer_index)
            wrong = states["incorrect_suggestion"]
            neutral = states["neutral"]
            correct_states = states["suggest_correct"]
            wc_delta = wrong - correct_states
            wn_delta = wrong - neutral
            global_wc = wc_delta.mean(axis=0, dtype=np.float64)
            global_wn = wn_delta.mean(axis=0, dtype=np.float64)
            bank_full = _bank_from_labels(
                wc_delta, labels, np.arange(len(labels), dtype=int)
            )
            bank_norms = np.linalg.norm(bank_full, axis=1)
            present = np.asarray(
                [np.any(labels == label) for label in CANONICAL_LABELS], dtype=bool
            )
            rms_bank_norm = float(np.sqrt(np.mean(np.square(bank_norms[present]))))
            cancellation = float(
                np.linalg.norm(global_wc)
                / max(rms_bank_norm, np.finfo(np.float64).tiny)
            )

            oof: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
            binding_explained: list[float] = []
            for family in ("global_wc", "b_conditioned_wc", "belief_conflict"):
                positive_oof = np.full(len(labels), np.nan, dtype=np.float64)
                negative_oof = np.full(len(labels), np.nan, dtype=np.float64)
                explained_energy: list[float] = []
                for fold in range(int(n_folds)):
                    train = np.flatnonzero(folds != fold)
                    test = np.flatnonzero(folds == fold)
                    if family == "global_wc":
                        vector = wc_delta[train].mean(axis=0, dtype=np.float64)
                        center = 0.5 * (
                            wrong[train].mean(axis=0, dtype=np.float64)
                            + correct_states[train].mean(axis=0, dtype=np.float64)
                        )
                        positive_oof[test], negative_oof[test] = _paired_projection_scores(
                            wrong[test],
                            correct_states[test],
                            vector,
                            training_centers=center,
                        )
                        predicted_delta = np.broadcast_to(
                            vector, wc_delta[test].shape
                        )
                        target_delta = wc_delta[test]
                    elif family == "b_conditioned_wc":
                        bank = _bank_from_labels(wc_delta, labels, train)
                        vectors = _vectors_for_labels(bank, labels[test])
                        center_bank = _centers_by_label(
                            wrong, correct_states, labels, train
                        )
                        centers = _vectors_for_labels(center_bank, labels[test])
                        positive_oof[test], negative_oof[test] = _paired_projection_scores(
                            wrong[test],
                            correct_states[test],
                            vectors,
                            training_centers=centers,
                        )
                        predicted_delta = vectors
                        target_delta = wc_delta[test]
                    else:
                        fit_train = train[belief[train] != "neutral_is_other"]
                        oriented_train = np.where(
                            (belief[fit_train] == "neutral_is_c")[:, None],
                            wc_delta[fit_train],
                            -wc_delta[fit_train],
                        )
                        vector = oriented_train.mean(axis=0, dtype=np.float64)
                        train_conflict = np.where(
                            (belief[fit_train] == "neutral_is_c")[:, None],
                            wrong[fit_train],
                            correct_states[fit_train],
                        )
                        train_congruent = np.where(
                            (belief[fit_train] == "neutral_is_c")[:, None],
                            correct_states[fit_train],
                            wrong[fit_train],
                        )
                        center = 0.5 * (
                            train_conflict.mean(axis=0, dtype=np.float64)
                            + train_congruent.mean(axis=0, dtype=np.float64)
                        )
                        eligible = test[belief[test] != "neutral_is_other"]
                        conflict = np.where(
                            (belief[eligible] == "neutral_is_c")[:, None],
                            wrong[eligible],
                            correct_states[eligible],
                        )
                        congruent = np.where(
                            (belief[eligible] == "neutral_is_c")[:, None],
                            correct_states[eligible],
                            wrong[eligible],
                        )
                        positive_oof[eligible], negative_oof[eligible] = _paired_projection_scores(
                            conflict,
                            congruent,
                            vector,
                            training_centers=center,
                        )
                        predicted_delta = np.broadcast_to(
                            vector, (len(eligible), vector.shape[0])
                        )
                        target_delta = np.where(
                            (belief[eligible] == "neutral_is_c")[:, None],
                            wc_delta[eligible],
                            -wc_delta[eligible],
                        )
                    denominator = float(np.square(target_delta).sum())
                    explained_energy.append(
                        1.0
                        - float(
                            np.square(target_delta - predicted_delta).sum()
                        )
                        / max(denominator, np.finfo(np.float64).tiny)
                    )
                eligible = np.isfinite(positive_oof) & np.isfinite(negative_oof)
                observed, ci_low, ci_high = _bootstrap_auc_interval(
                    positive_oof[eligible],
                    negative_oof[eligible],
                    n_bootstrap=int(n_bootstrap),
                    seed=int(seed) + 1000 * cell_index + 10 * layer + len(layer_rows),
                )
                stability, stability_low, stability_high = _split_half_stability(
                    wc_delta,
                    labels,
                    datasets,
                    family=family,
                    belief_classes=belief,
                    n_repeats=int(n_split_half),
                    seed=int(seed) + 2000 * cell_index + 10 * layer,
                )
                layer_rows.append(
                    {
                        "model_name": model_name,
                        "layer": layer,
                        "family": family,
                        "n_questions": int(eligible.sum()),
                        "diffmean_auroc": observed,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "split_half_similarity_median": stability,
                        "split_half_similarity_ci_low": stability_low,
                        "split_half_similarity_ci_high": stability_high,
                        "heldout_explained_energy": float(
                            np.mean(explained_energy)
                        ),
                        "global_wc_norm": float(np.linalg.norm(global_wc)),
                        "global_wn_norm": float(np.linalg.norm(global_wn)),
                        "median_item_wc_norm": float(
                            np.median(np.linalg.norm(wc_delta, axis=1))
                        ),
                        "rms_b_conditioned_wc_norm": rms_bank_norm,
                        "rms_item_wc_norm": float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        np.linalg.norm(wc_delta, axis=1)
                                    )
                                )
                            )
                        ),
                        "cancellation_factor": cancellation,
                        **{
                            f"b_conditioned_wc_norm_{label}": float(
                                bank_norms[label_index]
                            )
                            for label_index, label in enumerate(
                                CANONICAL_LABELS
                            )
                        },
                    }
                )
                oof[family] = (positive_oof, negative_oof)

            binding_positive = np.full(len(labels), np.nan, dtype=np.float64)
            binding_negative = np.full(len(labels), np.nan, dtype=np.float64)
            for fold in range(int(n_folds)):
                train = np.flatnonzero(folds != fold)
                test = np.flatnonzero(folds == fold)
                ridge = _select_ridge(
                    wc_delta,
                    labels,
                    correct,
                    train,
                    folds,
                    ridge_grid=DEFAULT_RIDGE_GRID,
                )
                binding_bank = _sum_to_zero_label_binding(
                    wc_delta,
                    labels,
                    correct,
                    train,
                    ridge=ridge,
                )
                predicted = _binding_vectors(
                    binding_bank, labels[test], correct[test]
                )
                center_bank = _centers_by_label(
                    wrong, correct_states, labels, train
                )
                centers = _vectors_for_labels(center_bank, labels[test])
                binding_positive[test], binding_negative[test] = _paired_projection_scores(
                    wrong[test],
                    correct_states[test],
                    predicted,
                    training_centers=centers,
                )
                denominator = float(np.square(wc_delta[test]).sum())
                binding_explained.append(
                    1.0
                    - float(np.square(wc_delta[test] - predicted).sum())
                    / max(denominator, np.finfo(np.float64).tiny)
                )
            binding_auc, binding_low, binding_high = _bootstrap_auc_interval(
                binding_positive,
                binding_negative,
                n_bootstrap=int(n_bootstrap),
                seed=int(seed) + 3000 * cell_index + layer,
            )
            layer_rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "family": "label_binding_wc",
                    "n_questions": len(labels),
                    "diffmean_auroc": binding_auc,
                    "bootstrap_ci_low": binding_low,
                    "bootstrap_ci_high": binding_high,
                    "split_half_similarity_median": float("nan"),
                    "split_half_similarity_ci_low": float("nan"),
                    "split_half_similarity_ci_high": float("nan"),
                    "heldout_explained_energy": float(np.mean(binding_explained)),
                    "global_wc_norm": float(np.linalg.norm(global_wc)),
                    "global_wn_norm": float(np.linalg.norm(global_wn)),
                    "median_item_wc_norm": float(
                        np.median(np.linalg.norm(wc_delta, axis=1))
                    ),
                    "rms_b_conditioned_wc_norm": rms_bank_norm,
                    "cancellation_factor": cancellation,
                }
            )

            logistic_positive, logistic_negative = _nested_logistic_scores(
                wrong, correct_states, folds
            )
            logistic_auc = _question_auc(logistic_positive, logistic_negative)
            for row in reversed(layer_rows):
                if row["model_name"] == model_name and row["layer"] == layer:
                    row["nested_logistic_wc_auroc"] = logistic_auc

            null_aucs = _permuted_bank_aucs(
                wc_delta,
                wrong,
                correct_states,
                labels,
                datasets,
                folds,
                n_permutations=int(n_permutations),
                seed=int(seed) + 4000 * cell_index + layer,
            )
            for permutation_index, auc in enumerate(null_aucs):
                permutation_rows.append(
                    {
                        "model_name": model_name,
                        "layer": layer,
                        "family": "b_conditioned_wc",
                        "permutation_index": permutation_index,
                        "placebo_auroc": float(auc),
                    }
                )
            null_p95 = float(np.quantile(null_aucs, 0.95))
            for row in layer_rows:
                if (
                    row["model_name"] == model_name
                    and row["layer"] == layer
                    and row["family"] == "b_conditioned_wc"
                ):
                    row["permutation_null_p95"] = null_p95
                    row["permutation_exceedance_p"] = float(
                        (1 + np.sum(null_aucs >= row["diffmean_auroc"]))
                        / (1 + len(null_aucs))
                    )
            belief_null_aucs = _permuted_belief_aucs(
                wc_delta,
                wrong,
                correct_states,
                belief,
                datasets,
                folds,
                n_permutations=int(n_permutations),
                seed=int(seed) + 4500 * cell_index + layer,
            )
            for permutation_index, auc in enumerate(belief_null_aucs):
                permutation_rows.append(
                    {
                        "model_name": model_name,
                        "layer": layer,
                        "family": "belief_conflict",
                        "permutation_index": permutation_index,
                        "placebo_auroc": float(auc),
                    }
                )
            belief_p95 = float(np.quantile(belief_null_aucs, 0.95))
            for row in layer_rows:
                if (
                    row["model_name"] == model_name
                    and row["layer"] == layer
                    and row["family"] == "belief_conflict"
                ):
                    row["permutation_null_p95"] = belief_p95
                    row["permutation_exceedance_p"] = float(
                        (
                            1
                            + np.sum(
                                belief_null_aucs >= row["diffmean_auroc"]
                            )
                        )
                        / (1 + len(belief_null_aucs))
                    )

            for fold in range(int(n_folds)):
                train = np.flatnonzero(folds != fold)
                test = np.flatnonzero(folds == fold)
                geometry = _fold_geometry(states, train, test)
                geometry_rows.append(
                    {
                        "model_name": model_name,
                        "layer": layer,
                        "fold": fold,
                        **geometry,
                    }
                )

            selected_rank_by_fold: list[int] = []
            captured_by_fold: list[float] = []
            random_by_fold: list[float] = []
            for fold in range(int(n_folds)):
                train = np.flatnonzero(folds != fold)
                test = np.flatnonzero(folds == fold)
                rank = _select_pca_rank(
                    wc_delta, train, folds, DEFAULT_PCA_RANKS
                )
                selected_rank_by_fold.append(rank)
                mean, basis = _fit_pca(wc_delta[train], rank)
                captured_by_fold.append(_captured_energy(wc_delta[test], mean, basis))
                rng = np.random.default_rng(
                    int(seed) + 5000 * cell_index + 100 * layer + fold
                )
                random_matrix = rng.standard_normal((rank, wc_delta.shape[1]))
                random_basis, _ = np.linalg.qr(random_matrix.T, mode="reduced")
                random_by_fold.append(
                    _captured_energy(wc_delta[test], mean, random_basis.T)
                )
            selected_rank = int(
                min(
                    DEFAULT_PCA_RANKS,
                    key=lambda value: (
                        -selected_rank_by_fold.count(value),
                        value,
                    ),
                )
            )
            selected_ranks[layer] = selected_rank
            pca_rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "selected_rank": selected_rank,
                    "heldout_captured_energy": float(np.mean(captured_by_fold)),
                    "matched_random_subspace_energy": float(np.mean(random_by_fold)),
                }
            )

            leace_aucs: Dict[str, list[float]] = {
                "global_wc": [],
                "b_conditioned_wc": [],
                "belief_conflict": [],
            }
            leace_distortion: list[float] = []
            probe_overlap: list[float] = []
            for fold in range(int(n_folds)):
                train = np.flatnonzero(folds != fold)
                test = np.flatnonzero(folds == fold)
                eraser = fit_weighted_binary_leace(
                    {condition: values[train] for condition, values in states.items()}
                )
                transformed = {
                    condition: eraser.transform(values[test])
                    for condition, values in states.items()
                }
                train_transformed = {
                    condition: eraser.transform(values[train])
                    for condition, values in states.items()
                }
                train_delta = (
                    train_transformed["incorrect_suggestion"]
                    - train_transformed["suggest_correct"]
                )
                test_delta = (
                    transformed["incorrect_suggestion"]
                    - transformed["suggest_correct"]
                )
                vector = train_delta.mean(axis=0, dtype=np.float64)
                global_center = 0.5 * (
                    train_transformed["incorrect_suggestion"].mean(
                        axis=0, dtype=np.float64
                    )
                    + train_transformed["suggest_correct"].mean(
                        axis=0, dtype=np.float64
                    )
                )
                pos, neg = _paired_projection_scores(
                    transformed["incorrect_suggestion"],
                    transformed["suggest_correct"],
                    vector,
                    training_centers=global_center,
                )
                leace_aucs["global_wc"].append(_question_auc(pos, neg))
                bank = _bank_from_labels(
                    train_delta,
                    labels[train],
                    np.arange(len(train), dtype=int),
                )
                vectors = _vectors_for_labels(bank, labels[test])
                leace_center_bank = _centers_by_label(
                    train_transformed["incorrect_suggestion"],
                    train_transformed["suggest_correct"],
                    labels[train],
                    np.arange(len(train), dtype=int),
                )
                leace_centers = _vectors_for_labels(
                    leace_center_bank, labels[test]
                )
                pos, neg = _paired_projection_scores(
                    transformed["incorrect_suggestion"],
                    transformed["suggest_correct"],
                    vectors,
                    training_centers=leace_centers,
                )
                leace_aucs["b_conditioned_wc"].append(_question_auc(pos, neg))
                belief_train = belief[train]
                eligible_train = belief_train != "neutral_is_other"
                oriented = np.where(
                    (belief_train[eligible_train] == "neutral_is_c")[:, None],
                    train_delta[eligible_train],
                    -train_delta[eligible_train],
                )
                belief_vector = oriented.mean(axis=0, dtype=np.float64)
                belief_train_conflict = np.where(
                    (belief_train[eligible_train] == "neutral_is_c")[:, None],
                    train_transformed["incorrect_suggestion"][eligible_train],
                    train_transformed["suggest_correct"][eligible_train],
                )
                belief_train_congruent = np.where(
                    (belief_train[eligible_train] == "neutral_is_c")[:, None],
                    train_transformed["suggest_correct"][eligible_train],
                    train_transformed["incorrect_suggestion"][eligible_train],
                )
                belief_center = 0.5 * (
                    belief_train_conflict.mean(axis=0, dtype=np.float64)
                    + belief_train_congruent.mean(axis=0, dtype=np.float64)
                )
                eligible_test = belief[test] != "neutral_is_other"
                conflict = np.where(
                    (belief[test][eligible_test] == "neutral_is_c")[:, None],
                    transformed["incorrect_suggestion"][eligible_test],
                    transformed["suggest_correct"][eligible_test],
                )
                congruent = np.where(
                    (belief[test][eligible_test] == "neutral_is_c")[:, None],
                    transformed["suggest_correct"][eligible_test],
                    transformed["incorrect_suggestion"][eligible_test],
                )
                pos, neg = _paired_projection_scores(
                    conflict,
                    congruent,
                    belief_vector,
                    training_centers=belief_center,
                )
                leace_aucs["belief_conflict"].append(_question_auc(pos, neg))
                raw_pooled = np.concatenate(
                    [states[condition][test] for condition in REQUIRED_CONDITIONS], axis=0
                )
                erased_pooled = np.concatenate(
                    [transformed[condition] for condition in REQUIRED_CONDITIONS], axis=0
                )
                leace_distortion.append(
                    float(
                        np.sqrt(np.square(erased_pooled - raw_pooled).sum())
                        / max(
                            float(np.sqrt(np.square(raw_pooled).sum())),
                            np.finfo(np.float64).tiny,
                        )
                    )
                )
                if probe_coefficient is not None:
                    probe_overlap.append(abs(_cosine(eraser.proj_right, probe_coefficient)))
            leace_rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "global_wc_auroc": float(np.mean(leace_aucs["global_wc"])),
                    "b_conditioned_wc_auroc": float(
                        np.mean(leace_aucs["b_conditioned_wc"])
                    ),
                    "belief_conflict_auroc": float(
                        np.mean(leace_aucs["belief_conflict"])
                    ),
                    "relative_erasure_distortion": float(np.mean(leace_distortion)),
                    "probe_coefficient_subspace_overlap": (
                        float(np.mean(probe_overlap)) if probe_overlap else float("nan")
                    ),
                    "probe_preservation_claim_authorized": False,
                }
            )

        conditioned_arrays = _full_conditioned_arrays(
            artifact,
            metadata,
            selected_ranks=selected_ranks,
        )
        conditioned_metadata = {
            **{
                key: value
                for key, value in artifact.metadata.items()
                if key not in {"artifact_sha256", "created_at"}
            },
            "protocol_version": PROTOCOL_VERSION,
            "artifact_schema_version": CONDITIONED_ARTIFACT_SCHEMA_VERSION,
            "conditioned_audit_protocol_version": AUDIT_PROTOCOL_VERSION,
            "conditioning_families": {
                "b_conditioned_wc_bank": "mean(W-C | endorsed label b)",
                "label_binding_wc_bank": "sum-to-zero ridge fit delta_WC ~= u_b-u_c",
                "belief_conflict_direction": (
                    "mean(W-C) when neutral top1=c and mean(C-W) when neutral top1=b"
                ),
                "arc_b_conditioned_wc_bank": (
                    "mean(W-C | endorsed label b) on saved ARC training rows"
                ),
                "arc_belief_conflict_direction": (
                    "belief-oriented mean on saved ARC training rows"
                ),
                "arc_wc_raw": "mean(W-C) on saved ARC training rows",
                "arc_wn_raw": "mean(W-N) on saved ARC training rows",
                "wc_low_rank_basis": "PCA of centered item-level W-C deltas",
            },
            "conditioned_labels": list(CANONICAL_LABELS),
            "question_metadata_sha256": sha256_file(metadata_path),
            "source_direction_artifact": str(artifact.path),
            "source_direction_sha256": sha256_file(artifact.path),
            "question_manifest_sha256": manifest_hash,
        }
        conditioned_dir = target / f"conditioned_directions_model_{cell_index}"
        conditioned_artifact = save_controlled_direction_artifact(
            conditioned_dir,
            arrays=conditioned_arrays,
            metadata=conditioned_metadata,
        )
        audit_artifacts.append(
            {
                "model_name": model_name,
                "path": str(conditioned_artifact.path),
                "sha256": sha256_file(conditioned_artifact.path),
                "question_metadata": str(metadata_path),
            }
        )
        model_summaries.append(
            {
                "model_name": model_name,
                "n_questions": len(metadata),
                "datasets": metadata["dataset"].value_counts().sort_index().to_dict(),
                "belief_classes": metadata["belief_class"].value_counts().sort_index().to_dict(),
                "ordered_question_keys_sha256": str(
                    artifact.metadata["question_keys_sha256"]
                ),
            }
        )

    layer_table = pd.DataFrame(layer_rows)
    permutation_table = pd.DataFrame(permutation_rows)
    geometry_table = pd.DataFrame(geometry_rows)
    pca_table = pd.DataFrame(pca_rows)
    leace_table = pd.DataFrame(leace_rows)
    behavioral_table = pd.DataFrame(behavioral_rows)
    for filename, frame in (
        ("layer_table.csv", layer_table),
        ("permutation_table.csv", permutation_table),
        ("anisotropy_table.csv", geometry_table),
        ("low_rank_table.csv", pca_table),
        ("leace_table.csv", leace_table),
        ("behavioral_gap_table.csv", behavioral_table),
    ):
        frame.to_csv(target / filename, index=False)

    decisions: Dict[str, Any] = {}
    for model_name, model_frame in layer_table.groupby("model_name"):
        global_by_layer = (
            model_frame[model_frame["family"] == "global_wc"]
            .set_index("layer")["diffmean_auroc"]
            .to_dict()
        )
        family_decisions: Dict[str, Any] = {}
        for family in ("b_conditioned_wc", "belief_conflict"):
            family_frame = model_frame[model_frame["family"] == family].copy()
            family_frame["improvement_over_global"] = family_frame.apply(
                lambda row: float(row["diffmean_auroc"])
                - float(global_by_layer[int(row["layer"])]),
                axis=1,
            )
            family_frame["above_placebo"] = (
                family_frame["diffmean_auroc"]
                > family_frame["permutation_null_p95"]
            )
            family_frame["layer_pass"] = (
                (family_frame["diffmean_auroc"] >= 0.65)
                & (family_frame["improvement_over_global"] >= 0.05)
                & family_frame["above_placebo"]
                & (family_frame["bootstrap_ci_low"] > 0.50)
                & (family_frame["split_half_similarity_median"] >= 0.70)
            )
            passing_layers = sorted(
                family_frame.loc[family_frame["layer_pass"], "layer"].astype(int).tolist()
            )
            adjacent = sorted(
                {
                    layer
                    for layer in passing_layers
                    if layer - 1 in passing_layers or layer + 1 in passing_layers
                }
            )
            family_pass = len(adjacent) >= 2
            nominated: Optional[int] = None
            if family_pass:
                eligible = family_frame[family_frame["layer"].isin(adjacent)].copy()
                eligible["selection_score"] = eligible["bootstrap_ci_low"]
                nominated = int(
                    eligible.sort_values(
                        ["selection_score", "diffmean_auroc", "layer"],
                        ascending=[False, False, True],
                    ).iloc[0]["layer"]
                )
            family_decisions[family] = {
                "passes": bool(family_pass),
                "passing_adjacent_layers": adjacent,
                "nominated_layer": nominated,
            }
        primary_family = (
            "b_conditioned_wc"
            if family_decisions["b_conditioned_wc"]["passes"]
            else "belief_conflict"
            if family_decisions["belief_conflict"]["passes"]
            else None
        )
        nominated_layer = (
            family_decisions[primary_family]["nominated_layer"]
            if primary_family is not None
            else None
        )
        decisions[model_name] = {
            "families": family_decisions,
            "primary_family": primary_family,
            "nominated_layer": nominated_layer,
            "nominated_layers_with_neighbors": (
                [nominated_layer - 1, nominated_layer, nominated_layer + 1]
                if nominated_layer is not None
                else []
            ),
        }
    both_models_pass = len(decisions) == 2 and all(
        value["primary_family"] is not None for value in decisions.values()
    )
    decision = {
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "gpu_stage_authorized": bool(both_models_pass),
        "authorization_rule": (
            "both models must pass the preregistered adjacent-layer conditioned-additive gate"
        ),
        "models": decisions,
        "low_rank_or_leace_success_does_not_authorize_gpu": True,
        "n_permutations": int(n_permutations),
        "n_bootstrap": int(n_bootstrap),
        "n_split_half": int(n_split_half),
        "seed": int(seed),
    }
    write_strict_json(target / "decision.json", decision)
    manifest = {
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "question_manifest_path": str(Path(question_manifest_path).resolve()),
        "question_manifest_sha256": manifest_hash,
        "input_manifest_canonical_hash": canonical_json_hash(
            {"rows": manifest_rows}
        ),
        "models": model_summaries,
        "conditioned_artifacts": audit_artifacts,
        "outputs": {
            filename: sha256_file(target / filename)
            for filename in (
                "layer_table.csv",
                "permutation_table.csv",
                "anisotropy_table.csv",
                "low_rank_table.csv",
                "leace_table.csv",
                "behavioral_gap_table.csv",
                "decision.json",
            )
        },
    }
    write_strict_json(target / "manifest.json", manifest)
    _write_audit_report(target, layer_table, pca_table, leace_table, decision)
    return target / "decision.json"


def _write_audit_report(
    output_dir: Path,
    layer_table: pd.DataFrame,
    pca_table: pd.DataFrame,
    leace_table: pd.DataFrame,
    decision: Mapping[str, Any],
) -> None:
    lines = [
        "# Mean-cancellation audit",
        "",
        f"Protocol: `{AUDIT_PROTOCOL_VERSION}`.",
        "",
        "This report is computed only from the saved prompt-boundary training activations. "
        "All question-level conditions remained in one deterministic fold, and fitted "
        "statistics were restricted to training folds.",
        "",
        "## Decision",
        "",
        (
            "**Stage B authorized.**"
            if decision["gpu_stage_authorized"]
            else "**Stage B not authorized by the preregistered CPU gate.**"
        ),
        "",
    ]
    for model_name, model_decision in decision["models"].items():
        lines.extend(
            [
                f"- `{model_name}`: primary family "
                f"`{model_decision['primary_family']}`; nominated layer "
                f"`{model_decision['nominated_layer']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Layerwise evidence",
            "",
            "| Model | Family | Best layer | AUROC | 95% CI | Global improvement | Null p95 | Stability |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    globals_by_model = {
        model: frame[frame["family"] == "global_wc"].set_index("layer")[
            "diffmean_auroc"
        ].to_dict()
        for model, frame in layer_table.groupby("model_name")
    }
    for (model, family), frame in layer_table[
        layer_table["family"].isin(
            ["global_wc", "b_conditioned_wc", "belief_conflict", "label_binding_wc"]
        )
    ].groupby(["model_name", "family"]):
        best = frame.sort_values(
            ["bootstrap_ci_low", "diffmean_auroc"], ascending=False
        ).iloc[0]
        improvement = float(best["diffmean_auroc"]) - float(
            globals_by_model[model][int(best["layer"])]
        )
        null_p95 = best.get("permutation_null_p95", float("nan"))
        stability = best.get("split_half_similarity_median", float("nan"))
        lines.append(
            f"| {model} | {family} | {int(best['layer'])} | "
            f"{float(best['diffmean_auroc']):.3f} | "
            f"[{float(best['bootstrap_ci_low']):.3f}, {float(best['bootstrap_ci_high']):.3f}] | "
            f"{improvement:.3f} | "
            f"{float(null_p95):.3f} | {float(stability):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The earlier experiment showed that the global unaligned `W-N` mean is dominated "
            "by a shared suggestion axis. A small global `W-C` centroid difference is not "
            "evidence that item-, label-, or belief-conditioned information is absent. "
            "This audit tests those narrower alternatives; it does not itself establish a "
            "causal correction.",
            "",
            "Low-rank and LEACE results are descriptive CPU analyses and cannot authorize "
            "the additive Stage-B intervention by themselves. Probe overlap is reported, "
            "but probe preservation is not claimed because candidate-answer-token states "
            "were not saved.",
            "",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "AUDIT_PROTOCOL_VERSION",
    "CANONICAL_LABELS",
    "CONDITIONED_ARTIFACT_SCHEMA_VERSION",
    "CompactLeaceEraser",
    "deterministic_stratified_folds",
    "fit_weighted_binary_leace",
    "run_mean_cancellation_audit",
]
