#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
SPEC = importlib.util.spec_from_file_location("dense_campaign_tested", HERE / "campaign.py")
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class DenseCampaignTests(unittest.TestCase):
    def test_pressure_registry_covers_all_families(self) -> None:
        registry = campaign.pressure_templates()
        self.assertEqual(set(registry), set(campaign.PRESSURE_FAMILIES))
        self.assertTrue(all(registry[family] for family in campaign.PRESSURE_FAMILIES))

    def test_family_weight_profiles_are_complete(self) -> None:
        for unit in campaign.UNITS:
            prune = campaign.family_weights(unit, "prune")
            preserve = campaign.family_weights(unit, "preserve")
            self.assertEqual(len(prune), 1 + len(campaign.PRESSURE_FAMILIES))
            self.assertEqual(len(preserve), 5 + len(campaign.PRESSURE_FAMILIES))
            self.assertTrue(all(value > 0 for value in (*prune.values(), *preserve.values())))
        self.assertEqual(campaign.family_weights("dense_equal", "prune")["bad:doubt_correct_answer"], 1.0)
        self.assertEqual(campaign.family_weights("dense_core2_correct2", "prune")["bad:doubt_correct_answer"], 2.0)
        self.assertEqual(campaign.family_weights("dense_core2_correct4", "preserve")["good:correct_update"], 4.0)

    def test_matrix_grid_and_counts(self) -> None:
        for unit in campaign.UNITS:
            rows = campaign.matrix_for(unit)
            self.assertEqual(len(rows), 9)
            self.assertEqual({row["q"] for row in rows}, set(campaign.Q_VALUES))
            self.assertEqual({row["p_over_q"] for row in rows}, set(campaign.PQ_RATIOS))
            self.assertTrue(all(row["pruning_count"] == 3484 for row in rows))
            self.assertTrue(all(row["preservation_count"] == 6874 for row in rows))

    def test_dataset_take_is_deterministic_and_unique(self) -> None:
        rows = [
            {"dataset": "arc_challenge", "question_id": f"a{i}", "draw_idx": 0}
            for i in range(20)
        ] + [
            {"dataset": "commonsense_qa", "question_id": f"c{i}", "draw_idx": 0}
            for i in range(100)
        ]
        first = campaign._dataset_take(rows, count=50, namespace="test", arc_target=5)
        second = campaign._dataset_take(list(reversed(rows)), count=50, namespace="test", arc_target=5)
        self.assertEqual(first, second)
        self.assertEqual(len({campaign.row_key(row) for row in first}), 50)
        self.assertEqual(sum(row["dataset"] == "arc_challenge" for row in first), 5)

    def test_matrix_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            campaign.atomic_json(path, {"configurations": campaign.matrix_for("dense_equal")})
            self.assertEqual(campaign.load_matrix(path), campaign.matrix_for("dense_equal"))


if __name__ == "__main__":
    unittest.main()
