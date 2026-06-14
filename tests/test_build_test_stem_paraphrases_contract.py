from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from llmssycoph.data import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_test_stem_paraphrases.py"


def _load_script_module():
    module_name = "build_test_stem_paraphrases_contract"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class BuildTestStemParaphrasesContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_script_module()
        cls.dataset_names = list(cls.module.DEFAULT_DATASETS)
        cls.full_items, cls.full_stats = cls.module.build_prepared_items(
            cls.dataset_names,
            max_items=None,
        )

    def test_canonical_counts_match_expected_test_splits(self):
        self.assertEqual(
            self.full_stats["canonical_test_counts"],
            {
                "commonsense_qa": 2192,
                "arc_challenge": 1172,
            },
        )
        self.assertEqual(len(self.full_items), 3364)
        self.assertEqual(
            self.full_stats["prepared_counts_by_dataset"],
            {
                "commonsense_qa": 2192,
                "arc_challenge": 1172,
            },
        )

    def test_prepare_writes_smoke_outputs(self):
        with tempfile.TemporaryDirectory() as scratch_tmp, tempfile.TemporaryDirectory() as tracked_tmp:
            rc = self.module.main(
                [
                    "--scratch-output-dir",
                    scratch_tmp,
                    "--tracked-output-dir",
                    tracked_tmp,
                    "--max-items",
                    "5",
                    "--force",
                    "prepare",
                ]
            )

            self.assertEqual(rc, 0)
            scratch_dir = Path(scratch_tmp)
            tracked_dir = Path(tracked_tmp)
            self.assertTrue((scratch_dir / self.module.SCRATCH_PREPARED_ITEMS_NAME).exists())
            self.assertTrue((scratch_dir / self.module.SCRATCH_BATCH_REQUESTS_NAME).exists())
            self.assertTrue((scratch_dir / self.module.SCRATCH_RUN_STATE_NAME).exists())
            self.assertTrue((tracked_dir / self.module.TRACKED_MANIFEST_NAME).exists())
            self.assertTrue((tracked_dir / self.module.TRACKED_SUMMARY_NAME).exists())

            state = json.loads((scratch_dir / self.module.SCRATCH_RUN_STATE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(state["stage"], "prepared")
            self.assertEqual(state["prepared_item_count"], 5)
            self.assertGreater(state["cost_estimate"]["estimated_input_tokens"], 0)
            self.assertGreater(state["cost_estimate"]["estimated_output_tokens"], 0)
            self.assertIn("pricing_table_used", state)

    def test_prepared_custom_ids_are_deterministic(self):
        second_items, _ = self.module.build_prepared_items(self.dataset_names, max_items=None)
        first_ids = [row["custom_id"] for row in self.full_items[:50]]
        second_ids = [row["custom_id"] for row in second_items[:50]]
        self.assertEqual(first_ids, second_ids)

    def test_budget_guard_blocks_submit_before_api_use_and_persists_failure(self):
        with tempfile.TemporaryDirectory() as scratch_tmp, tempfile.TemporaryDirectory() as tracked_tmp:
            rc = self.module.main(
                [
                    "--scratch-output-dir",
                    scratch_tmp,
                    "--tracked-output-dir",
                    tracked_tmp,
                    "--max-items",
                    "5",
                    "--force",
                    "prepare",
                ]
            )
            self.assertEqual(rc, 0)

            with self.assertRaisesRegex(RuntimeError, "Estimated cost exceeds the configured cap"):
                self.module.main(
                    [
                        "--scratch-output-dir",
                        scratch_tmp,
                        "--tracked-output-dir",
                        tracked_tmp,
                        "--max-estimated-cost-usd",
                        "0.000001",
                        "submit",
                    ]
                )

            state = json.loads((Path(scratch_tmp) / self.module.SCRATCH_RUN_STATE_NAME).read_text(encoding="utf-8"))
            self.assertTrue(state["submission_blocked_by_cost_guard"])
            self.assertAlmostEqual(state["budget_cap_usd"], 0.000001)

    def test_pricing_table_rejects_unknown_models_and_estimates_known_models(self):
        with self.assertRaisesRegex(ValueError, "No batch pricing entry"):
            self.module._pricing_for_model("unknown-model")

        estimate = self.module.estimate_request_cost(
            [
                {
                    "custom_id": "demo::1",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": "gpt-5.4"},
                }
            ],
            [{"original_stem": "What is the capital of France?"}],
            model_name="gpt-5.4-mini",
        )
        self.assertGreater(estimate["estimated_input_tokens"], 0)
        self.assertGreater(estimate["estimated_output_tokens"], 0)
        self.assertGreater(estimate["estimated_total_cost_usd"], 0.0)

    def test_validation_flags_cover_expected_failure_modes(self):
        normalized, flags = self.module.validate_paraphrased_stem(
            "Which city is the capital of France?",
            "Answer: (B) Paris\nIt is the capital.",
        )

        self.assertEqual(normalized, "Answer: (B) Paris It is the capital.")
        self.assertIn("multi_line_output", flags)
        self.assertIn("contains_option_marker", flags)
        self.assertIn("contains_answer_label", flags)

        _, empty_flags = self.module.validate_paraphrased_stem("Original question text", "")
        self.assertIn("empty_output", empty_flags)

        _, unchanged_flags = self.module.validate_paraphrased_stem("Original question text", "Original question text")
        self.assertIn("unchanged_stem", unchanged_flags)

        _, short_flags = self.module.validate_paraphrased_stem(
            "This is a fairly long source question stem for the validator.",
            "Short.",
        )
        self.assertIn("too_short", short_flags)

        _, long_flags = self.module.validate_paraphrased_stem(
            "Short source stem?",
            "This paraphrased stem is intentionally much longer than the source stem so that the validator marks it as too long.",
        )
        self.assertIn("too_long", long_flags)

    def test_paraphrased_prompt_text_changes_only_the_stem_section(self):
        item = self.full_items[0]
        new_stem = f"Paraphrased version: {item['original_stem']}"
        new_prompt = self.module._render_neutral_prompt_text(dict(item["neutral_base"]), new_stem)

        self.assertNotEqual(item["original_prompt_text"], new_prompt)
        self.assertIn(item["answer_options"], item["original_prompt_text"])
        self.assertIn(item["answer_options"], new_prompt)

        original_prefix, original_suffix = item["original_prompt_text"].split(item["answer_options"], 1)
        new_prefix, new_suffix = new_prompt.split(item["answer_options"], 1)
        self.assertIn(item["original_stem"], original_prefix)
        self.assertIn(new_stem, new_prefix)
        self.assertEqual(original_suffix, new_suffix)

    def test_collect_payload_shape_uses_jsonl_and_no_csv_outputs(self):
        with tempfile.TemporaryDirectory() as scratch_tmp, tempfile.TemporaryDirectory() as tracked_tmp:
            rc = self.module.main(
                [
                    "--scratch-output-dir",
                    scratch_tmp,
                    "--tracked-output-dir",
                    tracked_tmp,
                    "--max-items",
                    "3",
                    "--force",
                    "prepare",
                ]
            )
            self.assertEqual(rc, 0)

            tracked_dir = Path(tracked_tmp)
            tracked_names = {path.name for path in tracked_dir.iterdir()}
            self.assertIn(self.module.TRACKED_MANIFEST_NAME, tracked_names)
            self.assertIn(self.module.TRACKED_SUMMARY_NAME, tracked_names)
            self.assertFalse(any(name.endswith(".csv") for name in tracked_names))

            prepared_rows = read_jsonl(str(Path(scratch_tmp) / self.module.SCRATCH_PREPARED_ITEMS_NAME))
            self.assertEqual(len(prepared_rows), 3)


if __name__ == "__main__":
    unittest.main()
