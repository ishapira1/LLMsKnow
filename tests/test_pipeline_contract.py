from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from llmssycoph.cli import parse_args
from llmssycoph.pipeline import (
    _format_parsed_argument_lines,
    _log_strict_mc_quality_summary,
    _strict_mc_neutral_choice_distribution_collapse_summary,
    _strict_mc_neutral_choice_distribution_collapse_warning,
    _next_record_id,
    _strict_mc_neutral_below_chance_warning,
    _strict_mc_neutral_selected_label_skew_summary,
    _strict_mc_neutral_selected_label_skew_warning,
    _warn_strict_mc_temperature_bookkeeping,
)


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(SRC_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


class PipelineContractTests(unittest.TestCase):
    def _local_import_graph(self, entry: Path) -> list[str]:
        root = ROOT.resolve()
        src_root = SRC_ROOT.resolve()
        visited: set[Path] = set()
        stack = [entry.resolve()]
        local_files: list[str] = []

        module_map: dict[str, Path] = {}
        for package_file in (src_root / "llmssycoph").rglob("*.py"):
            rel = package_file.relative_to(src_root).with_suffix("")
            module_map[".".join(rel.parts)] = package_file.resolve()
        module_map["script"] = (root / "script.py").resolve()
        module_map["run_sycophancy_bias_probe"] = (root / "run_sycophancy_bias_probe.py").resolve()

        while stack:
            path = stack.pop()
            if path in visited or not path.exists():
                continue
            visited.add(path)
            local_files.append(path.relative_to(root).as_posix())

            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if path.is_relative_to(src_root):
                package_name = ".".join(path.relative_to(src_root).with_suffix("").parts[:-1])
            else:
                package_name = ".".join(path.relative_to(root).with_suffix("").parts[:-1])

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = module_map.get(alias.name)
                        if target is not None:
                            stack.append(target)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        package_parts = package_name.split(".") if package_name else []
                        if node.level == 1:
                            base_parts = package_parts
                        else:
                            base_parts = package_parts[: -(node.level - 1)]
                        base = ".".join(part for part in [*base_parts, node.module or ""] if part)
                    else:
                        base = node.module or ""

                    candidates = [base] if base else []
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        candidates.append(f"{base}.{alias.name}" if base else alias.name)

                    for candidate in candidates:
                        target = module_map.get(candidate)
                        if target is not None:
                            stack.append(target)

        return sorted(local_files)

    def test_next_record_id_contract(self):
        self.assertEqual(_next_record_id(), 0)
        self.assertEqual(_next_record_id([{"record_id": 2}], [{"record_id": "7"}]), 8)
        self.assertEqual(_next_record_id([{"record_id": "bad"}], [{"other": 3}]), 0)

    def test_parsed_argument_lines_are_human_readable(self):
        lines = _format_parsed_argument_lines(parse_args([]))
        joined = "\n".join(lines)

        self.assertEqual(lines[0], "parsed arguments:")
        self.assertIn("  model", joined)
        self.assertIn("= mistralai/Mistral-7B-Instruct-v0.2", joined)
        self.assertIn("  hf_cache_dir", joined)
        self.assertIn("= <unset>", joined)
        self.assertIn("derived settings:", joined)
        self.assertIn("mc_mode", joined)
        self.assertNotIn('{"', joined)

    def test_strict_mc_temperature_bookkeeping_warning_mentions_normalized_value(self):
        args = parse_args(
            [
                "--benchmark_source",
                "ays_mc_single_turn",
                "--input_jsonl",
                "are_you_sure.jsonl",
                "--mc_mode",
                "strict_mc",
                "--temperature",
                "1",
            ]
        )

        with patch("llmssycoph.pipeline.warn_status") as mock_warn:
            _warn_strict_mc_temperature_bookkeeping(args)

        mock_warn.assert_called_once_with(
            "pipeline.py",
            "strict_mc_temperature_bookkeeping",
            "strict MC mode records temperature=1.0 for bookkeeping. First-token choice scoring ignores "
            "temperature, but if any prompt later falls back to text generation this value will apply there.",
        )

    def test_strict_mc_quality_summary_uses_ok_status_when_gate_passes(self):
        summary = {
            "commitment_rate": 1.0,
            "starts_with_answer_rate": 1.0,
            "cap_hit_rate": 0.0,
            "explicit_parse_failures": 0,
            "exact_format_rate": 1.0,
            "multiple_answer_marker_rows": 0,
            "max_neutral_bias_answer_gap": 0.0,
            "by_template": {
                "neutral": {
                    "total": 4,
                    "committed_rate": 1.0,
                    "starts_with_answer_rate": 1.0,
                    "cap_hit_rate": 0.0,
                    "exact_format_rate": 1.0,
                    "multiple_answer_marker_rows": 0,
                }
            },
        }

        with patch("llmssycoph.pipeline.ok_status") as mock_ok, patch("llmssycoph.pipeline.log_status") as mock_log:
            _log_strict_mc_quality_summary(summary, issues=[])

        mock_ok.assert_called_once_with(
            "pipeline.py",
            "strict MC quality: commitment_rate=100.0% starts_with_answer_rate=100.0% "
            "cap_hit_rate=0.0% explicit_parse_failures=0 exact_format_rate=100.0% "
            "multiple_answer_marker_rows=0 max_neutral_bias_answer_gap=0.0%",
        )
        mock_log.assert_called_once_with(
            "pipeline.py",
            "strict MC quality template=neutral: total=4 commitment_rate=100.0% "
            "starts_with_answer_rate=100.0% cap_hit_rate=0.0% exact_format_rate=100.0% "
            "multiple_answer_marker_rows=0",
        )

    def test_strict_mc_neutral_below_chance_warning_mentions_random_baseline(self):
        records = [
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "answers_list": ["A", "B", "C", "D", "E"],
                "correctness": 0,
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "answers_list": ["A", "B", "C", "D", "E"],
                "correctness": 0,
            },
        ]

        warning = _strict_mc_neutral_below_chance_warning(records)

        self.assertEqual(
            warning,
            "neutral strict-MC accuracy=0.0% is below the random-choice baseline "
            "1/5 = 20.0% across 2 usable neutral rows.",
        )

    def test_strict_mc_neutral_below_chance_warning_skips_at_chance_level(self):
        records = [
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "answers_list": ["A", "B", "C", "D", "E"],
                "correctness": 1,
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "answers_list": ["A", "B", "C", "D", "E"],
                "correctness": 0,
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "answers_list": ["A", "B", "C", "D", "E"],
                "correctness": 0,
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "answers_list": ["A", "B", "C", "D", "E"],
                "correctness": 0,
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "answers_list": ["A", "B", "C", "D", "E"],
                "correctness": 0,
            },
        ]

        self.assertIsNone(_strict_mc_neutral_below_chance_warning(records))

    def test_dominant_selected_label_skew_warning_uses_clear_nameable_condition(self):
        records = [
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "response": "A",
                "correct_letter": "A",
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "response": "A",
                "correct_letter": "B",
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "response": "A",
                "correct_letter": "C",
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "usable_for_metrics": True,
                "response": "B",
                "correct_letter": "D",
            },
        ]

        summary = _strict_mc_neutral_selected_label_skew_summary(records)
        warning = _strict_mc_neutral_selected_label_skew_warning(summary)

        self.assertTrue(summary["warning_triggered"])
        self.assertEqual(summary["dominant_selected_label"], "A")
        self.assertAlmostEqual(summary["dominant_selected_label_rate"], 0.75)
        self.assertAlmostEqual(summary["correct_label_distribution"]["A"], 0.25)
        self.assertAlmostEqual(summary["dominant_selected_label_excess"], 0.5)
        self.assertIsNotNone(warning)
        self.assertIn("selected-label distribution is skewed toward A", warning)
        self.assertIn("q(A)=75.0% vs answer-key r(A)=25.0%", warning)
        self.assertIn("excess=50.0%", warning)
        self.assertIn("TV=50.0%", warning)

    def test_choice_distribution_collapse_warning_uses_entropy_and_confidence_checks(self):
        records = [
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "sampling_mode": "choice_probabilities",
                "usable_for_metrics": True,
                "response": "A",
                "choice_probability_selected": 0.99,
                "choice_probabilities": {"A": 0.99, "B": 0.01},
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "sampling_mode": "choice_probabilities",
                "usable_for_metrics": True,
                "response": "A",
                "choice_probability_selected": 0.98,
                "choice_probabilities": {"A": 0.98, "B": 0.02},
            },
            {
                "template_type": "neutral",
                "task_format": "multiple_choice",
                "mc_mode": "strict_mc",
                "sampling_mode": "choice_probabilities",
                "usable_for_metrics": True,
                "response": "A",
                "choice_probability_selected": 0.97,
                "choice_probabilities": {"A": 0.97, "B": 0.03},
            },
        ]

        summary = _strict_mc_neutral_choice_distribution_collapse_summary(records)
        warning = _strict_mc_neutral_choice_distribution_collapse_warning(summary)

        self.assertTrue(summary["warning_triggered"])
        self.assertLess(summary["median_effective_options"], 1.2)
        self.assertAlmostEqual(summary["high_confidence_selected_rate"], 1.0)
        self.assertIsNotNone(warning)
        self.assertIn("choice distribution appears collapsed", warning)
        self.assertIn("median(N_eff)=", warning)
        self.assertIn("mean(P(selected)>=0.95)=100.0%", warning)

    def test_runner_and_pipeline_import_commands(self):
        commands = [
            "import run_sycophancy_bias_probe; print(callable(run_sycophancy_bias_probe.main))",
            "from llmssycoph.pipeline import run_pipeline; print(callable(run_pipeline))",
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
                self.assertEqual(result.stdout.strip(), "True")

    def test_main_entrypoint_has_no_legacy_runtime_dependencies(self):
        local_files = self._local_import_graph(ROOT / "run_sycophancy_bias_probe.py")
        self.assertFalse(
            any(path.startswith("legacy/") for path in local_files),
            msg=f"main entrypoint should not depend on legacy/: {local_files}",
        )
        self.assertNotIn(
            "script.py",
            local_files,
            msg=f"main entrypoint should not depend on script.py: {local_files}",
        )


if __name__ == "__main__":
    unittest.main()
