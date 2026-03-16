from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from sycophancy_bias_probe.answer_utils import (
    extract_gold_answers_from_base,
    extract_short_answer_from_generation,
    is_correct_short_answer,
    normalize_answer,
)
from sycophancy_bias_probe.model_utils import to_hf_chat


ROOT = Path(__file__).resolve().parents[1]


class ScriptCompatibilityTests(unittest.TestCase):
    def test_answer_and_chat_helper_contract(self):
        base = {
            "answer": {
                "value": "Paris",
                "aliases": ["Paris", "City of Paris"],
                "normalized_aliases": ["paris"],
            }
        }

        self.assertEqual(
            extract_gold_answers_from_base(base),
            ["Paris", "City of Paris", "paris"],
        )
        self.assertEqual(normalize_answer("  New-York!! "), "newyork")
        self.assertTrue(is_correct_short_answer("The answer is Paris", ["Paris"]))
        self.assertEqual(
            extract_short_answer_from_generation("Final answer: Paris"),
            "Paris",
        )
        self.assertEqual(
            to_hf_chat(
                [
                    {"type": "system", "content": "S"},
                    {"type": "human", "content": "Q"},
                    {"type": "assistant", "content": "A"},
                ]
            ),
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
        )

    def test_script_and_module_import_commands(self):
        commands = [
            "from script import generate_many, get_hidden_feature_for_answer; print('ok')",
            "from sycophancy_bias_probe.model_utils import generate_many; print('ok')",
            "from sycophancy_bias_probe.feature_utils import get_hidden_feature_for_answer; print('ok')",
        ]

        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-c", command],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
