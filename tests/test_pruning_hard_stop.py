from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_pruning_hard_stop.py"
SPEC = importlib.util.spec_from_file_location("check_pruning_hard_stop", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PruningHardStopTests(unittest.TestCase):
    def _run(self, candidate: dict[str, float]) -> dict:
        baseline = {
            "p": 0.0,
            "q": 0.0,
            "neutral_accuracy": 0.80,
            "neutral_correct_probability": 0.70,
            "correction_accuracy": 0.75,
            "agreement_accuracy": 0.85,
            "preservation_loss": 1.0,
            "wikitext_perplexity": 10.0,
            "other_wrong_invalid_rate": 0.05,
        }
        row = {**baseline, "p": 1e-5, "q": 5e-5, **candidate}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "summary.csv"
            output = root / "stop.json"
            pd.DataFrame([baseline, row]).to_csv(source, index=False)
            argv = [
                str(SCRIPT),
                "--selection-summary", str(source),
                "--output", str(output),
                "--max-neutral-accuracy-drop", "0.02",
                "--max-neutral-probability-drop", "0.02",
                "--max-correction-accuracy-drop", "0.02",
                "--max-agreement-accuracy-drop", "0.02",
                "--max-preservation-loss-increase", "0.05",
                "--max-wikitext-perplexity-increase", "0.05",
                "--max-other-wrong-invalid-increase", "0.02",
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(MODULE.main(), 0)
            return json.loads(output.read_text(encoding="utf-8"))

    def test_all_declared_utility_limits_can_trigger_stop(self):
        payload = self._run(
            {
                "neutral_accuracy": 0.77,
                "neutral_correct_probability": 0.67,
                "correction_accuracy": 0.72,
                "agreement_accuracy": 0.82,
                "preservation_loss": 1.06,
                "wikitext_perplexity": 10.6,
                "other_wrong_invalid_rate": 0.08,
            }
        )
        self.assertTrue(payload["stop"])
        joined = " ".join(payload["reasons"])
        for metric in (
            "neutral_accuracy_drop",
            "neutral_correct_probability_drop",
            "correction_accuracy_drop",
            "agreement_accuracy_drop",
            "preservation_loss_relative_increase",
            "wikitext_perplexity_relative_increase",
            "other_wrong_invalid_increase",
        ):
            self.assertIn(metric, joined)

    def test_equal_to_limits_does_not_stop(self):
        payload = self._run(
            {
                "neutral_accuracy": 0.78,
                "neutral_correct_probability": 0.68,
                "correction_accuracy": 0.73,
                "agreement_accuracy": 0.83,
                "preservation_loss": 1.05,
                "wikitext_perplexity": 10.5,
                "other_wrong_invalid_rate": 0.07,
            }
        )
        self.assertFalse(payload["stop"])


if __name__ == "__main__":
    unittest.main()
