from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from sycophancy_bias_probe.pipeline import _next_record_id


ROOT = Path(__file__).resolve().parents[1]


class PipelineContractTests(unittest.TestCase):
    def test_next_record_id_contract(self):
        self.assertEqual(_next_record_id(), 0)
        self.assertEqual(_next_record_id([{"record_id": 2}], [{"record_id": "7"}]), 8)
        self.assertEqual(_next_record_id([{"record_id": "bad"}], [{"other": 3}]), 0)

    def test_runner_and_pipeline_import_commands(self):
        commands = [
            "import run_sycophancy_bias_probe; print(callable(run_sycophancy_bias_probe.main))",
            "from sycophancy_bias_probe.pipeline import run_pipeline; print(callable(run_pipeline))",
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
                self.assertEqual(result.stdout.strip(), "True")


if __name__ == "__main__":
    unittest.main()
