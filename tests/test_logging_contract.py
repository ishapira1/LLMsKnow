from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmssycoph.logging_utils import (
    clear_run_logging,
    configure_run_logging,
    log_status,
    tqdm_desc,
    warn_status,
)


class LoggingContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_run_logging()

    def test_tqdm_desc_and_log_status_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            warning_log_path = Path(tmpdir) / "warnings.log"
            configure_run_logging(log_path, warning_log_path=warning_log_path)

            desc = tqdm_desc("pipeline.py", "sampling stage")
            self.assertEqual(desc, "[pipeline.py]: sampling stage")

            line = log_status("sampling.py", "sampling train split")
            self.assertEqual(line, "[sampling.py]: sampling train split")
            self.assertTrue(log_path.exists())
            self.assertFalse(warning_log_path.exists())
            self.assertEqual(
                log_path.read_text(encoding="utf-8").splitlines(),
                ["[sampling.py]: sampling train split"],
            )

            warning_line = warn_status(
                "pipeline.py",
                "mixed_strict_mc_modes",
                "strict-MC choice scoring and text generation are mixed in this run.",
            )
            self.assertEqual(
                warning_line,
                "[warning][pipeline.py][mixed_strict_mc_modes]: strict-MC choice scoring and text generation are mixed in this run.",
            )
            self.assertTrue(warning_log_path.exists())
            self.assertEqual(
                warning_log_path.read_text(encoding="utf-8").splitlines(),
                [
                    "[warning][pipeline.py][mixed_strict_mc_modes]: strict-MC choice scoring and text generation are mixed in this run."
                ],
            )
            self.assertEqual(
                log_path.read_text(encoding="utf-8").splitlines(),
                [
                    "[sampling.py]: sampling train split",
                    "[warning][pipeline.py][mixed_strict_mc_modes]: strict-MC choice scoring and text generation are mixed in this run.",
                ],
            )


if __name__ == "__main__":
    unittest.main()
