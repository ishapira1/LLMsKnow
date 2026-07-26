from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from llmssycoph.interventions.conditioned_audit import (
    _binding_vectors,
    _permuted_bank_aucs,
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
