from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmssycoph.logging_utils import (
    build_warning_summary_payload,
    clear_run_logging,
    configure_run_logging,
    get_run_warnings,
    log_status,
    ok_status,
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

            warnings = get_run_warnings()
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0]["warning_index"], 1)
            self.assertEqual(warnings[0]["source"], "pipeline.py")
            self.assertEqual(warnings[0]["warning_code"], "mixed_strict_mc_modes")

            warning_summary = build_warning_summary_payload()
            self.assertEqual(warning_summary["total_warnings"], 1)
            self.assertEqual(warning_summary["unique_warning_codes"], 1)
            self.assertEqual(warning_summary["unique_sources"], 1)
            self.assertEqual(
                warning_summary["by_code"],
                [
                    {
                        "warning_code": "mixed_strict_mc_modes",
                        "count": 1,
                        "sources": ["pipeline.py"],
                        "latest_message": "strict-MC choice scoring and text generation are mixed in this run.",
                    }
                ],
            )

    def test_warn_status_colors_console_output_when_stdout_supports_it(self):
        with patch("llmssycoph.logging_utils.sys.stdout.isatty", return_value=True), patch.dict(
            "llmssycoph.logging_utils.os.environ",
            {"TERM": "xterm-256color"},
            clear=True,
        ), patch("llmssycoph.logging_utils.tqdm.write") as mock_write:
            warning_line = warn_status(
                "pipeline.py",
                "mixed_strict_mc_modes",
                "strict-MC choice scoring and text generation are mixed in this run.",
            )

        self.assertEqual(
            warning_line,
            "[warning][pipeline.py][mixed_strict_mc_modes]: strict-MC choice scoring and text generation are mixed in this run.",
        )
        mock_write.assert_called_once_with(
            "\033[33m[warning][pipeline.py][mixed_strict_mc_modes]: "
            "strict-MC choice scoring and text generation are mixed in this run.\033[0m"
        )

    def test_ok_status_colors_console_output_when_stdout_supports_it(self):
        with patch("llmssycoph.logging_utils.sys.stdout.isatty", return_value=True), patch.dict(
            "llmssycoph.logging_utils.os.environ",
            {"TERM": "xterm-256color"},
            clear=True,
        ), patch("llmssycoph.logging_utils.tqdm.write") as mock_write:
            ok_line = ok_status(
                "sampling_integrity.py",
                "sampling integrity mode=choice_probabilities: Exact compliance: 100.00% | Integrity failure: 0.00%",
            )

        self.assertEqual(
            ok_line,
            "[sampling_integrity.py]: sampling integrity mode=choice_probabilities: Exact compliance: 100.00% | Integrity failure: 0.00%",
        )
        mock_write.assert_called_once_with(
            "\033[32m[sampling_integrity.py]: sampling integrity mode=choice_probabilities: "
            "Exact compliance: 100.00% | Integrity failure: 0.00%\033[0m"
        )

    def test_configure_run_logging_resets_warning_collector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_log_path = Path(tmpdir) / "first.log"
            second_log_path = Path(tmpdir) / "second.log"
            configure_run_logging(first_log_path)
            warn_status("pipeline.py", "first_warning", "first message")
            self.assertEqual(build_warning_summary_payload()["total_warnings"], 1)

            configure_run_logging(second_log_path)
            self.assertEqual(build_warning_summary_payload()["total_warnings"], 0)
            self.assertEqual(get_run_warnings(), [])


if __name__ == "__main__":
    unittest.main()
