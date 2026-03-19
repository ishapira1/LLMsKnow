from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from llmssycoph.data import read_jsonl
from llmssycoph.grading import (
    extract_gold_answers_from_base,
    extract_short_answer_from_generation,
    grade_short_answer,
    is_correct_short_answer,
    normalize_answer,
)
from llmssycoph.llm import to_hf_chat


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(SRC_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


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
        truthful_base = {
            "dataset": "truthful_qa",
            "correct_letter": "B",
            "letters": "AB",
            "answers": "(A) London\n(B) Paris is the capital of France.",
            "answers_list": ["London", "Paris is the capital of France."],
            "long_correct_answer": "Paris is the capital of France.",
            "correct_answer": "Paris",
        }
        self.assertEqual(
            extract_gold_answers_from_base(truthful_base),
            ["Paris", "Paris is the capital of France."],
        )
        self.assertEqual(normalize_answer("  New-York!! "), "newyork")
        self.assertTrue(is_correct_short_answer("The answer is Paris", ["Paris"]))
        self.assertEqual(grade_short_answer("Final answer: Paris", ["Paris"])["status"], "correct")
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
            "from llmssycoph.llm import generate_many; print('ok')",
            "from llmssycoph.probes import get_hidden_feature_for_answer; print('ok')",
            "from llmssycoph.grading import grade_short_answer; print(callable(grade_short_answer))",
            "from llmssycoph.data import build_question_groups; print(callable(build_question_groups))",
            "from llmssycoph.data import read_jsonl; print(callable(read_jsonl))",
        ]

        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-c", command],
                    cwd=ROOT,
                    env=_subprocess_env(),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn(result.stdout.strip(), {"ok", "True"})


if __name__ == "__main__":
    unittest.main()
