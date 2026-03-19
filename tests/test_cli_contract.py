from __future__ import annotations

import contextlib
import io
import unittest

from llmssycoph.cli import build_parser, parse_args


class CliContractTests(unittest.TestCase):
    def test_instruction_policy_argument_uses_canonical_names(self):
        args = parse_args(
            [
                "--benchmark_source",
                "ays_mc_single_turn",
                "--input_jsonl",
                "are_you_sure.jsonl",
                "--instruction_policy",
                "answer_with_reasoning",
            ]
        )

        self.assertEqual(args.instruction_policy, "answer_with_reasoning")
        self.assertEqual(args.mc_mode, "mc_with_rationale")
        self.assertEqual(args.probe_construction, "auto")
        self.assertEqual(args.probe_example_weighting, "model_probability")

    def test_legacy_mc_mode_alias_still_normalizes(self):
        args = parse_args(
            [
                "--benchmark_source",
                "ays_mc_single_turn",
                "--input_jsonl",
                "are_you_sure.jsonl",
                "--mc_mode",
                "strict_mc",
            ]
        )

        self.assertEqual(args.instruction_policy, "answer_only")
        self.assertEqual(args.mc_mode, "strict_mc")
        self.assertEqual(args.n_draws, 1)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.requested_temperature, 0.1)

    def test_strict_mc_overrides_sampling_knobs_even_when_user_sets_them(self):
        args = parse_args(
            [
                "--benchmark_source",
                "ays_mc_single_turn",
                "--input_jsonl",
                "are_you_sure.jsonl",
                "--instruction_policy",
                "answer_only",
                "--n_draws",
                "8",
                "--temperature",
                "0.9",
            ]
        )

        self.assertEqual(args.mc_mode, "strict_mc")
        self.assertEqual(args.n_draws, 1)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.requested_temperature, 0.9)

    def test_parser_rejects_invalid_benchmark_input_pair(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--benchmark_source",
                        "answer_json",
                        "--input_jsonl",
                        "are_you_sure.jsonl",
                    ]
                )

    def test_help_mentions_instruction_policy_and_legacy_alias(self):
        help_text = build_parser().format_help()

        self.assertIn("--instruction_policy", help_text)
        self.assertIn("--mc_mode", help_text)
        self.assertIn("--probe_construction", help_text)
        self.assertIn("--probe_example_weighting", help_text)
        self.assertIn("--override_sampling_cache", help_text)
        self.assertIn("answer_with_reasoning", help_text)
        self.assertIn("legacy --mc_mode", help_text)

    def test_override_sampling_cache_alias_disables_sampling_reuse(self):
        args = parse_args(["--override_sampling_cache"])

        self.assertTrue(args.no_reuse_sampling_cache)


if __name__ == "__main__":
    unittest.main()
