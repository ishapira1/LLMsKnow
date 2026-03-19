from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from llmssycoph.probes import (
    find_sublist,
    maybe_subsample,
    score_records_with_probe,
    select_best_layer_by_auc,
    train_probe_for_layer,
)


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

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        pos = X[y == 1].mean(axis=0)
        neg = X[y == 0].mean(axis=0)
        self.weights = pos - neg
        self.bias = -0.5 * float(np.dot(self.weights, pos + neg))
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


if __name__ == '__main__':
    unittest.main()
