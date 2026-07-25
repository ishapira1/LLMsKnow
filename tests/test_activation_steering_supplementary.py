from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.aggregate_activation_steering_supplementary import (
    aggregate_alpaca,
    aggregate_fixed_probe,
    aggregate_geometry,
)
from llmssycoph.interventions.controlled import PROTOCOL_VERSION, sha256_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class SupplementaryAggregationTests(unittest.TestCase):
    def test_fixed_probe_alpaca_and_geometry_are_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe_path = root / "probe.jsonl"
            probe_rows = []
            for question_index in range(2):
                for alpha in (0.0, 1.0):
                    probe_rows.append(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "stage": "score_fixed_probe",
                            "scoring_mode": "strict_choice",
                            "stable_question_key": f"csqa::{question_index}",
                            "model_name": "model",
                            "dataset": "commonsense_qa",
                            "condition": "incorrect_suggestion",
                            "layer": 17,
                            "direction_name": "wn",
                            "scale_convention": "native",
                            "control_seed": None,
                            "alpha": alpha,
                            "probe_structurally_informative": True,
                            "probe_correct_top1": alpha == 0.0,
                            "probe_correct_rank": 1 if alpha == 0.0 else 2,
                            "probe_margin_correct_minus_endorsed": 1.0 - alpha,
                            "external_probe_top1_agreement": True,
                            "external_probe_correctness_agreement": True,
                            "external_probe_margin_sign_agreement": True,
                        }
                    )
            _write_jsonl(probe_path, probe_rows)
            probe_summary = aggregate_fixed_probe(
                [probe_path],
                n_bootstrap=20,
                seed=5,
            )
            self.assertEqual(len(probe_summary), 2)
            self.assertEqual(set(probe_summary["n_units"]), {2})

            alpaca_path = root / "alpaca.jsonl"
            alpaca_rows = []
            for example_index in range(2):
                for alpha in (0.0, 2.0):
                    alpaca_rows.append(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "stage": "alpaca_guardrail",
                            "example_id": f"alpaca-{example_index}",
                            "model_name": "model",
                            "layer": 17,
                            "alpha": alpha,
                            "target_mean_nll": 1.0 + 0.1 * alpha,
                            "target_perplexity": 2.7 + 0.2 * alpha,
                        }
                    )
            _write_jsonl(alpaca_path, alpaca_rows)
            alpaca_summary = aggregate_alpaca(
                [alpaca_path],
                n_bootstrap=20,
                seed=5,
            )
            alpha_two = alpaca_summary[alpaca_summary["alpha"].eq(2.0)].iloc[0]
            self.assertAlmostEqual(
                float(alpha_two["delta_target_mean_nll_mean"]),
                0.2,
            )

            geometry_dir = root / "geometry"
            geometry_dir.mkdir()
            pair_path = geometry_dir / "geometry_pairs.csv"
            pd.DataFrame(
                [
                    {
                        "layer": 17,
                        "group": "A_same_question_N_W",
                        "raw_cosine": 0.8,
                        "centered_cosine": 0.7,
                        "normalized_euclidean_distance": 0.4,
                    },
                    {
                        "layer": 17,
                        "group": "A_same_question_N_W",
                        "raw_cosine": 0.6,
                        "centered_cosine": 0.5,
                        "normalized_euclidean_distance": 0.8,
                    },
                ]
            ).to_csv(pair_path, index=False)
            geometry_path = geometry_dir / "geometry_summary.json"
            geometry_path.write_text(
                json.dumps(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "stage": "run_geometry",
                        "model_name": "model",
                        "dataset": "commonsense_qa",
                        "split": "test",
                        "n_questions": 2,
                        "summary": [
                            {
                                "layer": 17,
                                "identity_framing_ratio": 2.0,
                                "classifier_accuracy": 0.8,
                            }
                        ],
                        "pairs_csv_sha256": sha256_file(pair_path),
                    }
                ),
                encoding="utf-8",
            )
            geometry, pair_summary = aggregate_geometry([geometry_path])
            self.assertEqual(len(geometry), 1)
            self.assertEqual(len(pair_summary), 1)
            self.assertAlmostEqual(
                float(pair_summary.iloc[0]["raw_cosine_mean"]),
                0.7,
            )


if __name__ == "__main__":
    unittest.main()
