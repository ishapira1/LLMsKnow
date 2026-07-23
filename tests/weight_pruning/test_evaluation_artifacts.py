import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools" / "weight_pruning")
)

import cohere_support  # noqa: E402
from paper_pruning import (  # noqa: E402
    copy_mask_artifacts,
    summarize_alpaca_scores,
    summarize_zero_shot_results,
    update_evaluation_metadata,
    write_evaluation_artifact,
    write_evaluation_items,
    write_evaluation_metadata,
)


class UtilitySummaryTests(unittest.TestCase):
    def test_alpaca_summary_reports_total_and_valid_counts(self):
        summary = summarize_alpaca_scores([8, None, float("nan"), 6])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["valid_count"], 2)
        self.assertEqual(summary["mean_score"], 7.0)

    def test_zero_shot_summary_matches_released_aggregation(self):
        summary = summarize_zero_shot_results(
            {
                "task_a": {"acc,none": 0.5, "acc_stderr,none": 0.1},
                "task_b": {"acc,none": 0.75, "acc_stderr,none": 0.2},
            }
        )
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["mean_accuracy"], 0.625)
        self.assertAlmostEqual(
            summary["mean_accuracy_stderr"],
            ((0.1 ** 2 + 0.2 ** 2) / 4) ** 0.5,
        )


class EvaluationPersistenceTests(unittest.TestCase):
    def test_utility_sections_and_items_persist_beside_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask_dir = Path(temporary) / "mask"
            result = {"mask_dir": str(mask_dir)}
            args = SimpleNamespace(
                model="org/model",
                revision="abc123",
                score_format="raw",
                loss_mode="completion_nll",
                p=1e-5,
                q=5e-5,
                seed=5,
                control="none",
            )
            evaluation_path = write_evaluation_metadata(
                args,
                result,
                preservation_loss=0.25,
                wikitext_perplexity=7.5,
                sparsity=5e-5,
            )
            alpaca = {"mean_score": 8.0, "count": 2, "valid_count": 2}
            metrics_path = write_evaluation_artifact(
                result, "alpaca_metrics.json", alpaca
            )
            items_path = write_evaluation_items(
                result,
                "alpaca_items.jsonl",
                [
                    {"prompt": "one", "output": "first", "judge_score": 7},
                    {"prompt": "two", "output": "second", "judge_score": 9},
                ],
            )
            update_evaluation_metadata(
                result,
                {
                    "alpaca": {
                        **alpaca,
                        "metrics_path": metrics_path.name,
                        "items_path": items_path.name,
                    },
                    "zero_shot": {"mean_accuracy": 0.6, "task_count": 6},
                },
            )

            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            self.assertEqual(evaluation["preservation_loss"], 0.25)
            self.assertEqual(evaluation["alpaca"]["mean_score"], 8.0)
            self.assertEqual(evaluation["zero_shot"]["task_count"], 6)
            self.assertTrue(metrics_path.exists())
            rows = [
                json.loads(line)
                for line in items_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["judge_score"] for row in rows], [7, 9])

    def test_saved_checkpoint_copies_available_sparse_mask_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_dir = root / "mask"
            checkpoint_dir = root / "checkpoint"
            mask_dir.mkdir()
            (mask_dir / "metadata.json").write_text(
                '{"surviving_count": 2}\n', encoding="utf-8"
            )
            (mask_dir / "indices.pt").write_bytes(b"sparse-mask")

            copied = copy_mask_artifacts(mask_dir, checkpoint_dir)

            self.assertEqual(copied, ["metadata.json", "indices.pt"])
            self.assertEqual(
                (checkpoint_dir / "metadata.json").read_text(encoding="utf-8"),
                '{"surviving_count": 2}\n',
            )
            self.assertEqual(
                (checkpoint_dir / "indices.pt").read_bytes(), b"sparse-mask"
            )


class CohereCredentialTests(unittest.TestCase):
    def test_missing_credentials_fail_only_when_client_is_requested(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError, "COHERE_API_KEY or COHERE_KEY"
            ):
                cohere_support.cohere_client("Alpaca judge")

    def test_legacy_cohere_key_is_supported(self):
        client = object()
        fake_module = SimpleNamespace(ClientV2=mock.Mock(return_value=client))
        with mock.patch.object(cohere_support, "cohere", fake_module):
            with mock.patch.dict(
                os.environ, {"COHERE_KEY": "legacy-key"}, clear=True
            ):
                self.assertIs(cohere_support.cohere_client("Alpaca judge"), client)
        fake_module.ClientV2.assert_called_once_with(api_key="legacy-key")


if __name__ == "__main__":
    unittest.main()
