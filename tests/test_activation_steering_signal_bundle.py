from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from llmssycoph.interventions.controlled import validate_question_manifest
from llmssycoph.interventions.controlled_runtime import (
    _semantic_approval_required,
)
from llmssycoph.interventions.data import load_source_bundle


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/experiments/activation_steering_signal_20260726.json"
MANIFEST = REPO / "configs/experiments/activation_steering_signal_300_20260726.jsonl"
BUNDLE = (
    REPO
    / "jobs/sycophancy_bias_probe/activation_steering_signal_sharded_20260726"
)


class ExploratorySignalContractTests(unittest.TestCase):
    def test_sampling_only_source_is_allowed_when_probe_is_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            (root / "sampling/raw").mkdir(parents=True)
            (root / "meta/run_config.json").write_text(
                json.dumps({"model": "example/model", "dataset_name": "example"}),
                encoding="utf-8",
            )
            (root / "sampling/raw/sampling_records.jsonl").write_text(
                "",
                encoding="utf-8",
            )
            source = load_source_bundle(root, require_probe=False)
            self.assertEqual(source.model_name, "example/model")
            self.assertTrue(source.probe_scores.empty)
            self.assertEqual(source.probe_metadata, {})
            with self.assertRaises(FileNotFoundError):
                load_source_bundle(root, require_probe=True)

    def test_signal_scope_is_explicit_and_does_not_forge_human_approval(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertFalse(_semantic_approval_required(config))
        rows = [
            json.loads(line)
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 300)
        self.assertTrue(
            all(
                row["semantic_b_review_status"]
                == "not_requested_exploratory"
                for row in rows
            )
        )
        summary = validate_question_manifest(
            rows,
            require_human_approval=False,
        )
        self.assertFalse(summary["human_approval_required"])
        self.assertEqual(
            Counter((row["dataset"], row["split"]) for row in rows),
            Counter(
                {
                    ("arc_challenge", "train"): 90,
                    ("arc_challenge", "val"): 30,
                    ("arc_challenge", "test"): 30,
                    ("commonsense_qa", "train"): 90,
                    ("commonsense_qa", "val"): 30,
                    ("commonsense_qa", "test"): 30,
                }
            ),
        )

    def test_signal_bundle_is_lean_and_mail_enabled(self) -> None:
        scripts = sorted(BUNDLE.glob("*.sbatch"))
        self.assertEqual(len(scripts), 7)
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            self.assertIn("#SBATCH --mail-type=END,FAIL", text)
            self.assertIn(
                "#SBATCH --mail-user=itaishapira@g.harvard.edu",
                text,
            )
        screen = (BUNDLE / "screen_layers_array.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("--learned-directions wn", screen)
        self.assertIn('--control-seeds ""', screen)
        self.assertIn("--alphas=-8,-2,0,2,8", screen)
        test = (BUNDLE / "test_selected_array.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("--control-seeds 0,1,2", test)
        self.assertIn("--learned-directions wn", test)
        probe = (BUNDLE / "fixed_probe_array.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --array=0-1", probe)


if __name__ == "__main__":
    unittest.main()
