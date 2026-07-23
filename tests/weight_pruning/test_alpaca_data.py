import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools" / "weight_pruning")
)

from alpaca_data import (  # noqa: E402
    LEGACY_ALPACA_EVAL_DATA,
    load_alpaca_eval_prompts,
    render_alpaca_user_prompt,
    resolve_alpaca_eval_data_path,
)


class AlpacaDataTests(unittest.TestCase):
    def test_renders_standard_templates_with_and_without_input(self):
        without_input = render_alpaca_user_prompt(
            {"instruction": "Name a prime.", "input": ""},
            row_label="row 1",
        )
        self.assertIn("### Instruction:\nName a prime.", without_input)
        self.assertNotIn("### Input:", without_input)
        self.assertTrue(without_input.endswith("### Response:"))

        with_input = render_alpaca_user_prompt(
            {"instruction": "Translate.", "input": "bonjour"},
            row_label="row 2",
        )
        self.assertIn("paired with an input", with_input)
        self.assertIn("### Input:\nbonjour", with_input)
        self.assertTrue(with_input.endswith("### Response:"))

    def test_loads_json_jsonl_and_csv(self):
        rows = [
            {"instruction": "First", "input": "", "output": "ignored"},
            {"instruction": "Second", "input": "context"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "alpaca.json"
            json_path.write_text(json.dumps(rows), encoding="utf-8")
            jsonl_path = root / "alpaca.jsonl"
            jsonl_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            csv_path = root / "alpaca.csv"
            csv_path.write_text(
                "instruction,input,output\nFirst,,ignored\nSecond,context,\n",
                encoding="utf-8",
            )

            for path in (json_path, jsonl_path, csv_path):
                with self.subTest(path=path.name):
                    prompts = load_alpaca_eval_prompts(path, nsamples=5, seed=7)
                    self.assertEqual(len(prompts), 2)
                    self.assertTrue(all("### Response:" in item for item in prompts))

    def test_sampling_is_seeded_and_caps_at_available_rows(self):
        rows = [{"instruction": f"instruction-{index}"} for index in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpaca.json"
            path.write_text(json.dumps(rows), encoding="utf-8")

            first = load_alpaca_eval_prompts(path, nsamples=4, seed=5)
            repeated = load_alpaca_eval_prompts(path, nsamples=4, seed=5)
            other_seed = load_alpaca_eval_prompts(path, nsamples=4, seed=17)
            all_rows = load_alpaca_eval_prompts(path, nsamples=100, seed=5)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(all_rows), 10)

    def test_legacy_prompt_is_accepted_verbatim(self):
        self.assertEqual(
            render_alpaca_user_prompt({"prompt": "Already rendered"}, row_label="legacy"),
            "Already rendered",
        )

    def test_frozen_raw_prompt_is_authoritative(self):
        self.assertEqual(
            render_alpaca_user_prompt(
                {
                    "raw_prompt": "Frozen rendered prompt",
                    "instruction": "This would render differently.",
                    "input": "context",
                },
                row_label="frozen",
            ),
            "Frozen rendered prompt",
        )

    def test_manifest_runs_require_explicit_path(self):
        with self.assertRaisesRegex(ValueError, "--alpaca_eval_data is required"):
            resolve_alpaca_eval_data_path(None, manifest_run=True)
        self.assertEqual(
            resolve_alpaca_eval_data_path(None, manifest_run=False),
            LEGACY_ALPACA_EVAL_DATA,
        )
        self.assertEqual(
            resolve_alpaca_eval_data_path("~/alpaca.jsonl", manifest_run=True),
            Path("~/alpaca.jsonl").expanduser(),
        )

    def test_missing_empty_and_malformed_inputs_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                load_alpaca_eval_prompts(root / "missing.json", nsamples=1, seed=0)

            empty = root / "empty.json"
            empty.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contains no rows"):
                load_alpaca_eval_prompts(empty, nsamples=1, seed=0)

            malformed = root / "bad.jsonl"
            malformed.write_text('{"instruction": "ok"}\nnot json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_alpaca_eval_prompts(malformed, nsamples=1, seed=0)

            missing_instruction = root / "missing_instruction.csv"
            missing_instruction.write_text("output\nanswer\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty 'instruction'"):
                load_alpaca_eval_prompts(
                    missing_instruction,
                    nsamples=1,
                    seed=0,
                )

    def test_generation_uses_explicit_file_without_dataset_or_network(self):
        import eval_utils

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpaca.json"
            path.write_text(
                json.dumps([{"instruction": "Give a short greeting."}]),
                encoding="utf-8",
            )
            args = Namespace(
                alpaca_eval_data=str(path),
                alpaca_eval_seed=17,
                model="plain-base-model",
                no_chat_template=False,
                preserve_manifest="preserve.jsonl",
                prune_manifest="prune.jsonl",
                seed=5,
            )
            calls = []

            def fake_generate(prompt, *, max_new_tokens, do_sample):
                calls.append((prompt, max_new_tokens, do_sample))
                return [{"generated_text": prompt + "hello"}]

            with mock.patch.object(
                eval_utils.transformers,
                "pipeline",
                return_value=fake_generate,
            ), mock.patch.object(
                eval_utils,
                "load_alpaca_eval_prompts",
                wraps=eval_utils.load_alpaca_eval_prompts,
            ) as load_prompts, mock.patch.object(eval_utils.wandb, "log"), mock.patch.object(
                eval_utils,
                "measure_refusal_rate_explanation",
                return_value=(0.0, [False]),
            ):
                output = eval_utils.generate_responses_for_alpaca(
                    object(),
                    object(),
                    args,
                    nsamples=10,
                )

        self.assertEqual(len(calls), 1)
        self.assertIn("### Instruction:\nGive a short greeting.", calls[0][0])
        self.assertEqual(calls[0][1], 256)
        self.assertFalse(calls[0][2])
        self.assertEqual(output.loc[0, "output"], "hello")
        self.assertIn("### Response:", output.loc[0, "prompt"])
        self.assertEqual(load_prompts.call_args.kwargs["seed"], 17)


if __name__ == "__main__":
    unittest.main()
