from __future__ import annotations

import contextlib
import io
import unittest

from llmssycoph.cli import DEFAULT_PARAPHRASE_ARTIFACT_PATH, build_parser, parse_args
from llmssycoph.data import trainable_prompt_families


class CliContractTests(unittest.TestCase):
    def test_defaults_include_all_trainable_probe_families_and_paraphrases(self):
        args = parse_args([])

        expected_bias_types = ",".join(trainable_prompt_families(include_neutral=False))
        self.assertEqual(args.bias_types, expected_bias_types)
        self.assertIn("doubt_random", args.bias_types.split(","))
        self.assertIn("doubt_random_strong", args.bias_types.split(","))
        self.assertIn("random_all", args.bias_types.split(","))
        self.assertEqual(args.probe_families, ",".join(trainable_prompt_families(include_neutral=True)))
        self.assertEqual(args.paraphrase_artifact_path, DEFAULT_PARAPHRASE_ARTIFACT_PATH)

    def test_probe_families_can_select_one_sampled_family_without_changing_bias_types(self):
        args = parse_args(["--probe_families", "suggest_random"])

        expected_bias_types = ",".join(trainable_prompt_families(include_neutral=False))
        self.assertEqual(args.bias_types, expected_bias_types)
        self.assertEqual(args.probe_families, "suggest_random")

    def test_probe_families_default_tracks_reduced_bias_types(self):
        args = parse_args(["--bias_types", "incorrect_suggestion"])

        self.assertEqual(args.bias_types, "incorrect_suggestion")
        self.assertEqual(args.probe_families, "neutral,incorrect_suggestion")

    def test_probe_families_rejects_unknown_family(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--probe_families", "not_a_family"])

    def test_probe_families_rejects_family_not_in_sampled_bias_types(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--bias_types",
                        "incorrect_suggestion",
                        "--probe_families",
                        "suggest_random",
                    ]
                )

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
        self.assertIn("--sampling_only", help_text)
        self.assertIn("--probe_families", help_text)
        self.assertIn("--probe_construction", help_text)
        self.assertIn("--probe_example_weighting", help_text)
        self.assertIn("--override_sampling_cache", help_text)
        self.assertIn("--fresh_run", help_text)
        self.assertIn("answer_with_reasoning", help_text)
        self.assertIn("legacy --mc_mode", help_text)

    def test_override_sampling_cache_alias_disables_sampling_reuse(self):
        args = parse_args(["--override_sampling_cache"])

        self.assertTrue(args.no_reuse_sampling_cache)

    def test_fresh_run_flag_disables_sampling_reuse(self):
        args = parse_args(["--fresh_run"])

        self.assertTrue(args.fresh_run)
        self.assertTrue(args.no_reuse_sampling_cache)

    def test_sampling_only_flag_defaults_false_and_parses_true(self):
        self.assertFalse(parse_args([]).sampling_only)
        self.assertTrue(parse_args(["--sampling_only"]).sampling_only)

    def test_external_paraphrase_eval_flag_defaults_false_and_parses_true(self):
        self.assertFalse(parse_args([]).evaluate_external_paraphrases)
        self.assertTrue(parse_args(["--evaluate_external_paraphrases"]).evaluate_external_paraphrases)


if __name__ == "__main__":
    unittest.main()
