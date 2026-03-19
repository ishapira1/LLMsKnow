from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from llmssycoph.runtime import preferred_run_artifact_path
from llmssycoph.probes import (
    evaluate_probe_from_cache,
    find_sublist,
    maybe_subsample,
    save_probe_family_artifacts,
    score_records_with_probe,
    select_best_layer_by_auc,
    train_probe_for_layer,
)
from llmssycoph import build_probe_record_sets


def make_records(n: int = 20, offset: int = 0):
    records = []
    for idx in range(offset, offset + n):
        correctness = idx % 2
        records.append(
            {
                'record_id': idx,
                'prompt_messages': [{'type': 'human', 'content': f'question {idx}'}],
                'response': f'answer {idx}',
                'correctness': correctness,
            }
        )
    return records


class FakeLogisticRegression:
    def __init__(self, *args, **kwargs):
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float)
        pos = np.average(X[y == 1], axis=0, weights=sample_weight[y == 1])
        neg = np.average(X[y == 0], axis=0, weights=sample_weight[y == 0])
        self.weights = pos - neg
        self.bias = -0.5 * float(np.dot(self.weights, pos + neg))
        self.n_features_in_ = int(X.shape[1])
        self.coef_ = self.weights.reshape(1, -1)
        self.intercept_ = np.array([self.bias], dtype=float)
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        logits = X @ self.weights + self.bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - probs, probs], axis=1)


class ProbeContractTests(unittest.TestCase):
    def test_find_sublist_and_maybe_subsample_contract(self):
        self.assertEqual(find_sublist([1, 2, 3, 4], [3, 4]), 2)
        self.assertIsNone(find_sublist([1, 2], [3]))

        records = make_records(10)
        subsampled_a = maybe_subsample(records, max_samples=4, seed=7)
        subsampled_b = maybe_subsample(records, max_samples=4, seed=7)
        self.assertEqual(subsampled_a, subsampled_b)
        self.assertEqual(len(subsampled_a), 4)
        self.assertEqual(maybe_subsample(records, max_samples=None, seed=7), records)

    def test_select_best_layer_by_auc_contract(self):
        train_records = make_records(10)
        val_records = make_records(10, offset=100)

        def fake_all_layer_features(model, tokenizer, messages, answer, layer_grid):
            idx = int(answer.split()[-1])
            label = idx % 2
            layer_one = np.array([3.0, 0.0]) if label == 1 else np.array([0.0, 3.0])
            layer_two = np.array([1.0, 1.0])
            return np.stack([layer_one, layer_two], axis=0)

        with patch(
            'llmssycoph.probes.select_layer.get_hidden_feature_all_layers_for_completion',
            side_effect=fake_all_layer_features,
        ), patch(
            'llmssycoph.probes.select_layer.LogisticRegression',
            FakeLogisticRegression,
        ):
            best_layer, best_auc, auc_per_layer, clf_per_layer = select_best_layer_by_auc(
                model=None,
                tokenizer=None,
                train_records=train_records,
                val_records=val_records,
                layer_grid=[1, 2],
                seed=0,
                max_selection_samples=None,
                desc='test',
            )

        self.assertEqual(best_layer, 1)
        self.assertIsNotNone(best_auc)
        self.assertAlmostEqual(best_auc, 1.0)
        self.assertAlmostEqual(auc_per_layer[1], 1.0)
        self.assertAlmostEqual(auc_per_layer[2], 0.5)
        self.assertIsNotNone(clf_per_layer[1])

    def test_train_probe_and_score_records_contract(self):
        records = make_records(20)

        def fake_single_layer_feature(model, tokenizer, messages, answer, layer):
            idx = int(answer.split()[-1])
            label = idx % 2
            return np.array([float(label), float(1 - label)])

        with patch(
            'llmssycoph.probes.train._get_hidden_feature_for_completion',
            side_effect=fake_single_layer_feature,
        ), patch(
            'llmssycoph.probes.score._get_hidden_feature_for_completion',
            side_effect=fake_single_layer_feature,
        ), patch(
            'llmssycoph.probes.train.LogisticRegression',
            FakeLogisticRegression,
        ):
            clf = train_probe_for_layer(
                model=None,
                tokenizer=None,
                records=records,
                layer=3,
                seed=0,
                max_train_samples=None,
                desc='test',
            )
            self.assertIsNotNone(clf)

            score_records_with_probe(
                model=None,
                tokenizer=None,
                records=records,
                clf=clf,
                layer=3,
                score_key='probe_score',
                desc='test',
            )

        positive_scores = [record['probe_score'] for record in records if record['correctness'] == 1]
        negative_scores = [record['probe_score'] for record in records if record['correctness'] == 0]
        self.assertGreater(min(positive_scores), max(negative_scores))

    def test_build_probe_record_sets_uses_choice_candidates_for_strict_mc(self):
        sampled_records = [
            {
                "record_id": 10,
                "split": "train",
                "question_id": "q_1",
                "prompt_id": "q_1__neutral",
                "dataset": "aqua_mc",
                "template_type": "neutral",
                "draw_idx": 0,
                "question": "Question 1",
                "prompt_text": "Question 1\n\nAnswer:",
                "prompt_messages": [{"type": "human", "content": "Question 1\n\nAnswer:"}],
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "letters": "ABCD",
                "correct_letter": "C",
                "response_raw": "C",
                "sampling_mode": "choice_probabilities",
                "choice_probabilities": {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.1},
            }
        ]

        bundles = build_probe_record_sets(
            train_records=sampled_records,
            val_records=[],
            test_records=[],
            all_records=sampled_records,
            bias_types=[],
            probe_construction="auto",
            probe_example_weighting="model_probability",
        )

        neutral = bundles["neutral"]
        self.assertEqual(neutral["probe_construction"], "choice_candidates")
        self.assertEqual(len(neutral["train_records"]), 4)
        self.assertEqual(len(neutral["score_records"]), 1)
        self.assertEqual(len(neutral["candidate_score_records"]), 4)
        candidate_rows = {row["candidate_choice"]: row for row in neutral["train_records"]}
        self.assertAlmostEqual(candidate_rows["C"]["probe_sample_weight"], 0.6)
        self.assertEqual(candidate_rows["C"]["correctness"], 1)
        self.assertEqual(candidate_rows["A"]["correctness"], 0)

    def test_score_records_with_probe_none_contract(self):
        records = make_records(4)
        score_records_with_probe(
            model=None,
            tokenizer=None,
            records=records,
            clf=None,
            layer=None,
            score_key='probe_score',
            desc='none',
        )
        for record in records:
            self.assertTrue(math.isnan(record['probe_score']))

    def test_probe_training_ignores_unusable_records(self):
        records = make_records(20) + [
            {
                'record_id': 99,
                'prompt_messages': [{'type': 'human', 'content': 'question ambiguous'}],
                'response': 'answer ambiguous',
                'correctness': None,
                'usable_for_metrics': False,
            }
        ]

        def fake_single_layer_feature(model, tokenizer, messages, answer, layer):
            idx = int(answer.split()[-1])
            label = idx % 2
            return np.array([float(label), float(1 - label)])

        with patch(
            'llmssycoph.probes.train._get_hidden_feature_for_completion',
            side_effect=fake_single_layer_feature,
        ), patch(
            'llmssycoph.probes.train.LogisticRegression',
            FakeLogisticRegression,
        ):
            clf = train_probe_for_layer(
                model=None,
                tokenizer=None,
                records=records,
                layer=3,
                seed=0,
                max_train_samples=None,
                desc='test',
            )

        self.assertIsNotNone(clf)

    def test_evaluate_probe_from_cache_reports_split_metrics(self):
        labels_train = np.array([0, 1, 0, 1], dtype=int)
        labels_val = np.array([0, 1], dtype=int)
        labels_test = np.array([0, 1], dtype=int)
        layer_one_train = np.array([[0.0, 2.0], [2.0, 0.0], [0.0, 2.5], [2.5, 0.0]])
        layer_two_train = np.ones((4, 2), dtype=float)
        train_features = np.stack([layer_one_train, layer_two_train], axis=1)
        val_features = np.stack(
            [
                np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float),
                np.ones((2, 2), dtype=float),
            ],
            axis=1,
        )
        test_features = np.stack(
            [
                np.array([[0.0, 3.0], [3.0, 0.0]], dtype=float),
                np.ones((2, 2), dtype=float),
            ],
            axis=1,
        )

        clf = FakeLogisticRegression().fit(layer_one_train, labels_train)
        metrics = evaluate_probe_from_cache(
            {
                "layer_grid": [1, 2],
                "splits": {
                    "train": {"labels": labels_train, "features": train_features},
                    "val": {"labels": labels_val, "features": val_features},
                    "test": {"labels": labels_test, "features": test_features},
                },
            },
            clf,
            1,
        )

        self.assertEqual(metrics["evaluated_layer"], 1)
        self.assertAlmostEqual(metrics["splits"]["train"]["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["splits"]["val"]["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["splits"]["test"]["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["splits"]["train"]["true_label_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["splits"]["train"]["false_label_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["splits"]["train"]["auc"], 1.0)

    def test_save_probe_family_artifacts_writes_all_and_chosen_layout(self):
        train_records = [
            {
                "record_id": 1,
                "split": "train",
                "question_id": "q_1",
                "template_type": "neutral",
                "draw_idx": 0,
                "dataset": "toy",
                "prompt_messages": [{"type": "human", "content": "q1"}],
                "response": "answer 1",
                "correctness": 1,
                "usable_for_metrics": True,
            },
            {
                "record_id": 2,
                "split": "train",
                "question_id": "q_2",
                "template_type": "neutral",
                "draw_idx": 0,
                "dataset": "toy",
                "prompt_messages": [{"type": "human", "content": "q2"}],
                "response": "answer 2",
                "correctness": 0,
                "usable_for_metrics": True,
            },
        ]
        val_records = [
            {
                "record_id": 3,
                "split": "val",
                "question_id": "q_3",
                "template_type": "neutral",
                "draw_idx": 0,
                "dataset": "toy",
                "prompt_messages": [{"type": "human", "content": "q3"}],
                "response": "answer 3",
                "correctness": 1,
                "usable_for_metrics": True,
            },
            {
                "record_id": 4,
                "split": "val",
                "question_id": "q_4",
                "template_type": "neutral",
                "draw_idx": 0,
                "dataset": "toy",
                "prompt_messages": [{"type": "human", "content": "q4"}],
                "response": "answer 4",
                "correctness": 0,
                "usable_for_metrics": True,
            },
        ]
        test_records = [
            {
                "record_id": 5,
                "split": "test",
                "question_id": "q_5",
                "template_type": "neutral",
                "draw_idx": 0,
                "dataset": "toy",
                "prompt_messages": [{"type": "human", "content": "q5"}],
                "response": "answer 5",
                "correctness": 1,
                "usable_for_metrics": True,
            },
            {
                "record_id": 6,
                "split": "test",
                "question_id": "q_6",
                "template_type": "neutral",
                "draw_idx": 0,
                "dataset": "toy",
                "prompt_messages": [{"type": "human", "content": "q6"}],
                "response": "answer 6",
                "correctness": 0,
                "usable_for_metrics": True,
            },
        ]

        clf = FakeLogisticRegression().fit(
            np.array([[2.0, 0.0], [0.0, 2.0]], dtype=float),
            np.array([1, 0], dtype=int),
        )
        metrics = {
            "metric_schema_version": 1,
            "metric_names": ["accuracy", "auc"],
            "threshold": 0.5,
            "evaluated_layer": 1,
            "splits": {
                "train": {"accuracy": 1.0, "auc": 1.0, "n_total": 2, "n_label_1": 1, "n_label_0": 1},
                "val": {"accuracy": 1.0, "auc": 1.0, "n_total": 2, "n_label_1": 1, "n_label_0": 1},
                "test": {"accuracy": 1.0, "auc": 1.0, "n_total": 2, "n_label_1": 1, "n_label_0": 1},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            all_probes_dir = preferred_run_artifact_path(run_dir, "all_probes_dir")
            chosen_probe_dir = preferred_run_artifact_path(run_dir, "chosen_probe_dir")
            summary = save_probe_family_artifacts(
                run_dir=run_dir,
                probe_name="probe_no_bias",
                template_type="neutral",
                desc="no_bias",
                feature_source={"probe_feature_mode": "response_raw_final_token"},
                split_records={"train": train_records, "val": val_records, "test": test_records},
                selection_models={1: clf},
                selection_metrics_by_layer={1: metrics},
                auc_per_layer={1: 1.0},
                best_layer=1,
                best_dev_auc=1.0,
                chosen_model=clf,
                chosen_metrics=metrics,
                selection_fit_records=train_records,
                selection_val_records=val_records,
                chosen_fit_records=train_records + val_records,
                selection_fit_max_samples=100,
                chosen_fit_max_samples=200,
                probe_seed=7,
                probe_construction="sampled_completions",
                probe_example_weighting="uniform",
            )

            self.assertTrue((all_probes_dir / "probe_no_bias" / "layer_001" / "model.pkl").exists())
            self.assertTrue((chosen_probe_dir / "probe_no_bias" / "model.pkl").exists())
            self.assertTrue((all_probes_dir / "probe_no_bias" / "manifest.json").exists())
            self.assertTrue((chosen_probe_dir / "probe_no_bias" / "manifest.json").exists())

            metadata = json.loads(
                (chosen_probe_dir / "probe_no_bias" / "metadata.json").read_text(encoding="utf-8")
            )
            membership_lines = (
                chosen_probe_dir / "probe_no_bias" / "record_membership.jsonl"
            ).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(metadata["training"]["fit_splits"], ["train", "val"])
            self.assertEqual(metadata["model"]["input_dim"], 2)
            self.assertEqual(len(membership_lines), 6)
            self.assertEqual(summary["best_layer"], 1)
            self.assertIsNotNone(summary["chosen_probe_metrics_path"])


if __name__ == '__main__':
    unittest.main()
