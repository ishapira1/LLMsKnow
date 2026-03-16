from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sycophancy_bias_probe.logging_utils import (
    clear_run_logging,
    configure_run_logging,
    log_status,
    tqdm_desc,
)


class LoggingContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_run_logging()

    def test_tqdm_desc_and_log_status_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            configure_run_logging(log_path)

            desc = tqdm_desc("pipeline.py", "sampling stage")
            self.assertEqual(desc, "[pipeline.py]: sampling stage")

            line = log_status("sampling.py", "sampling train split")
            self.assertEqual(line, "[sampling.py]: sampling train split")
            self.assertTrue(log_path.exists())
            self.assertEqual(
                log_path.read_text(encoding="utf-8").splitlines(),
                ["[sampling.py]: sampling train split"],
            )


if __name__ == "__main__":
    unittest.main()
