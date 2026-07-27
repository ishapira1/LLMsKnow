from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from llmssycoph.interventions.conditioned_audit import (
    _binding_vectors,
    _common_primary_family,
    _fold_geometry,
    _nested_logistic_scores,
    _permuted_bank_aucs,
    _question_auc,
    _sum_to_zero_label_binding,
    deterministic_stratified_folds,
    fit_weighted_binary_leace,
)
from llmssycoph.interventions.controlled import (
    PROTOCOL_VERSION,
    load_controlled_direction_artifact,
    save_controlled_direction_artifact,
)


class ConditionedAuditContractTests(unittest.TestCase):
    def test_primary_family_must_pass_in_both_models(self):
        decisions = {
            "llama": {
                "families": {
                    "b_conditioned_wc": {"passes": True},
                    "belief_conflict": {"passes": True},
                }
            },
            "qwen": {
                "families": {
                    "b_conditioned_wc": {"passes": False},
                    "belief_conflict": {"passes": True},
                }
            },
        }
        self.assertEqual(
            _common_primary_family(decisions),
            "belief_conflict",
        )
        decisions["qwen"]["families"]["belief_conflict"]["passes"] = False
        self.assertIsNone(_common_primary_family(decisions))

    def test_nested_logistic_row_span_matches_full_width_predictions(self):
        rng = np.random.default_rng(29)
        n_questions = 25
        latent = rng.normal(size=(n_questions, 4))
        decoder = rng.normal(size=(4, 40))
        direction = rng.normal(size=(1, 40))
        positive = latent @ decoder + direction
        negative = latent @ decoder - direction
        folds = np.arange(n_questions, dtype=int) % 5

        (
            positive_scores,
            negative_scores,
            convergence_warnings,
        ) = _nested_logistic_scores(
            positive,
            negative,
            folds,
            c_grid=(0.1, 1.0),
        )
        reference_positive = np.full(n_questions, np.nan)
        reference_negative = np.full(n_questions, np.nan)
        for outer_fold in range(5):
            train = np.flatnonzero(folds != outer_fold)
            test = np.flatnonzero(folds == outer_fold)
            best_score = -np.inf
            best_c = 0.1
            for c_value in (0.1, 1.0):
                inner_scores = []
                for inner_fold in sorted(set(folds[train].tolist())):
                    inner_train = train[folds[train] != inner_fold]
                    inner_test = train[folds[train] == inner_fold]
                    classifier = LogisticRegression(
                        C=c_value,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=5000,
                        tol=1e-6,
                        random_state=5,
                    ).fit(
                        np.concatenate(
                            [positive[inner_train], negative[inner_train]]
                        ),
                        np.concatenate(
                            [
                                np.ones(len(inner_train), dtype=int),
                                np.zeros(len(inner_train), dtype=int),
                            ]
                        ),
                    )
                    inner_scores.append(
                        _question_auc(
                            classifier.decision_function(
                                positive[inner_test]
                            ),
                            classifier.decision_function(
                                negative[inner_test]
                            ),
                        )
                    )
                candidate = float(np.mean(inner_scores))
                if candidate > best_score:
                    best_score = candidate
                    best_c = c_value
            classifier = LogisticRegression(
                C=best_c,
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
                tol=1e-6,
                random_state=5,
            ).fit(
                np.concatenate([positive[train], negative[train]]),
                np.concatenate(
                    [
                        np.ones(len(train), dtype=int),
                        np.zeros(len(train), dtype=int),
                    ]
                ),
            )
            reference_positive[test] = classifier.decision_function(
                positive[test]
            )
            reference_negative[test] = classifier.decision_function(
                negative[test]
            )

        self.assertTrue(np.isfinite(positive_scores).all())
        self.assertTrue(np.isfinite(negative_scores).all())
        self.assertEqual(convergence_warnings, 0)
        np.testing.assert_allclose(
            positive_scores, reference_positive, rtol=2e-4, atol=2e-4
        )
        np.testing.assert_allclose(
            negative_scores, reference_negative, rtol=2e-4, atol=2e-4
        )
        self.assertGreater(
            np.mean(positive_scores > negative_scores),
            0.95,
        )

    def test_anisotropy_geometry_is_reported_on_heldout_questions(self):
        neutral = np.zeros((4, 2), dtype=np.float64)
        wrong = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        correct = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        geometry = _fold_geometry(
            {
                "neutral": neutral,
                "incorrect_suggestion": wrong,
                "suggest_correct": correct,
                "incorrect_suggestion_strong": 2.0 * wrong,
            },
            np.asarray([0, 1], dtype=int),
            np.asarray([2, 3], dtype=int),
        )
        self.assertAlmostEqual(geometry["raw_wn_cn_cosine"], 0.0)
        self.assertEqual(geometry["common_centering_max_abs_difference"], 0.0)

    def test_question_disjoint_folds_are_deterministic_and_stratified(self):
        datasets = np.asarray(["arc"] * 20 + ["csqa"] * 20)
        labels = np.asarray(list("ABCD") * 10)
        first = deterministic_stratified_folds(datasets, labels, n_folds=5, seed=17)
        second = deterministic_stratified_folds(datasets, labels, n_folds=5, seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first.tolist()), set(range(5)))
        for dataset in ("arc", "csqa"):
            for label in "ABCD":
                members = first[(datasets == dataset) & (labels == label)]
                self.assertEqual(len(members), 5)
                self.assertEqual(set(members.tolist()), set(range(5)))

    def test_label_binding_enforces_sum_to_zero_and_recovers_pair_deltas(self):
        rng = np.random.default_rng(3)
        true_bank = rng.normal(size=(5, 12))
        true_bank -= true_bank.mean(axis=0)
        endorsed = np.asarray(list("ABCDE") * 8)
        correct = np.roll(endorsed, 1)
        deltas = _binding_vectors(true_bank, endorsed, correct)
        fitted = _sum_to_zero_label_binding(
            deltas,
            endorsed,
            correct,
            np.arange(len(deltas)),
            ridge=1e-10,
        )
        np.testing.assert_allclose(fitted.sum(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(
            _binding_vectors(fitted, endorsed, correct),
            deltas,
            atol=1e-8,
        )

    def test_label_permutation_placebo_is_deterministic(self):
        rng = np.random.default_rng(9)
        n, hidden = 30, 16
        labels = np.asarray(list("ABC") * 10)
        datasets = np.asarray(["arc"] * 15 + ["csqa"] * 15)
        folds = deterministic_stratified_folds(datasets, labels, seed=5)
        deltas = rng.normal(size=(n, hidden))
        wrong = rng.normal(size=(n, hidden)) + 0.5 * deltas
        correct = wrong - deltas
        first = _permuted_bank_aucs(
            deltas,
            wrong,
            correct,
            labels,
            datasets,
            folds,
            n_permutations=20,
            seed=11,
        )
        second = _permuted_bank_aucs(
            deltas,
            wrong,
            correct,
            labels,
            datasets,
            folds,
            n_permutations=20,
            seed=11,
        )
        np.testing.assert_array_equal(first, second)

    def test_weighted_leace_erases_balanced_suggestion_mean_difference(self):
        rng = np.random.default_rng(21)
        direction = rng.normal(size=24)
        neutral = rng.normal(size=(80, 24))
        states = {
            "neutral": neutral,
            "incorrect_suggestion": rng.normal(size=(80, 24)) + direction,
            "suggest_correct": rng.normal(size=(80, 24)) + direction,
            "incorrect_suggestion_strong": rng.normal(size=(80, 24)) + direction,
        }
        eraser = fit_weighted_binary_leace(states)
        erased_neutral = eraser.transform(states["neutral"])
        erased_suggested = eraser.transform(
            np.concatenate(
                [
                    states["incorrect_suggestion"],
                    states["suggest_correct"],
                    states["incorrect_suggestion_strong"],
                ],
                axis=0,
            )
        )
        raw_gap = np.linalg.norm(
            np.concatenate(
                [
                    states["incorrect_suggestion"],
                    states["suggest_correct"],
                    states["incorrect_suggestion_strong"],
                ],
                axis=0,
            ).mean(axis=0)
            - states["neutral"].mean(axis=0)
        )
        erased_gap = np.linalg.norm(
            erased_suggested.mean(axis=0) - erased_neutral.mean(axis=0)
        )
        self.assertLess(erased_gap, raw_gap * 1e-8)

    def test_schema_v2_conditioned_vectors_keep_legacy_loader_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = {
                "layers": np.asarray([1, 2]),
                "wn_raw": np.ones((2, 4), dtype=np.float32),
                "conditioned_labels": np.asarray(list("ABCDE"), dtype="U1"),
                "b_conditioned_wc_bank": np.arange(
                    2 * 5 * 4, dtype=np.float32
                ).reshape(2, 5, 4),
                "belief_conflict_direction": np.full(
                    (2, 4), 3.0, dtype=np.float32
                ),
            }
            artifact = save_controlled_direction_artifact(
                root / "directions",
                arrays=arrays,
                metadata={
                    "protocol_version": PROTOCOL_VERSION,
                    "artifact_schema_version": 2,
                },
            )
            loaded = load_controlled_direction_artifact(artifact.path)
            np.testing.assert_array_equal(
                loaded.raw_direction("wn", 1), np.ones(4, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                loaded.conditioned_direction(
                    "b_conditioned_wc", 2, conditioning_key="C"
                ),
                arrays["b_conditioned_wc_bank"][1, 2],
            )
            np.testing.assert_array_equal(
                loaded.conditioned_direction("belief_conflict", 1),
                np.full(4, 3.0, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
