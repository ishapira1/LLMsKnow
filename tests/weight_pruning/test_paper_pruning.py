import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools" / "weight_pruning")
)

from paper_pruning import (  # noqa: E402
    EncodedCompletion,
    ManifestError,
    PreparedExample,
    completion_nll_from_logits,
    backward_example,
    dump_scores,
    encode_completion,
    exact_global_topk,
    evaluate_manifest_mean_loss,
    expected_score_dir,
    load_manifest,
    mask_output_dir,
    prepare_examples,
    score_manifest,
    set_difference,
    run_manifest_global_pruning,
    validate_score_cache,
    write_evaluation_metadata,
)


class CharacterTokenizer:
    """Tiny fast-tokenizer stand-in with unambiguous character offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [(ord(char) % 31) + 1 for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize and add_generation_prompt
        body = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        return body + "<assistant>"


class CrossingTokenizer(CharacterTokenizer):
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # One final token deliberately covers the last prompt character and target.
        result = {"input_ids": [1, 2]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(0, max(1, len(text) - 2)), (len(text) - 2, len(text))]
        return result


class BinaryTokenizer(CharacterTokenizer):
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(char) % 2 for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


class SpecialResponseTokenizer(CharacterTokenizer):
    all_special_ids = [99]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert text.endswith("<eot>")
        boundary = len(text) - len("<eot>")
        result = {"input_ids": [1] * boundary + [99]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(boundary)
            ] + [(boundary, len(text))]
        return result


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 2, bias=False)
        nn.init.constant_(self.proj.weight, 1.0)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock()])


class ToyCausalLM(nn.Module):
    """Two-token LM whose opposite targets have exactly cancelling gradients."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(2, 1)
        nn.init.constant_(self.embed.weight, 1.0)
        self.model = ToyBackbone()
        self.config = SimpleNamespace(use_cache=True)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, use_cache=False):
        hidden = self.embed(input_ids)
        return SimpleNamespace(logits=self.model.layers[0].proj(hidden))


def toy_example(target):
    encoded = EncodedCompletion(
        input_ids=torch.tensor([0, target]),
        response_start=1,
        rendered_prompt="P",
        target_text=str(target),
    )
    return PreparedExample(record={}, completion=encoded)


def load_only_tensor(directory):
    with open(Path(directory) / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    item = next(iter(metadata["tensors"].values()))
    return torch.load(Path(directory) / item["file"], map_location="cpu", weights_only=True)


class ManifestRenderingTests(unittest.TestCase):
    def test_raw_and_chat_mask_only_target_suffix(self):
        tokenizer = CharacterTokenizer()
        row = {
            "_manifest_line": 1,
            "raw_prompt": "Question\nAnswer: ",
            "messages": [{"role": "user", "content": "Question"}],
            "target_text": "B",
        }
        raw = encode_completion(row, tokenizer, "raw")
        chat = encode_completion(row, tokenizer, "chat")
        self.assertEqual(raw.response_start, len(row["raw_prompt"]))
        self.assertEqual(raw.input_ids.numel() - raw.response_start, 1)
        self.assertEqual(chat.input_ids.numel() - chat.response_start, 1)
        self.assertTrue(chat.rendered_prompt.endswith("<assistant>"))

    def test_boundary_crossing_fails_closed(self):
        with self.assertRaisesRegex(ManifestError, "crosses the prompt/target boundary"):
            encode_completion(
                {"_manifest_line": 1, "raw_prompt": "Answer:", "target_text": "B"},
                CrossingTokenizer(),
                "raw",
            )

    def test_response_special_tokens_are_never_scored(self):
        with self.assertRaisesRegex(ManifestError, "special/control token IDs"):
            encode_completion(
                {"_manifest_line": 1, "raw_prompt": "Answer: ", "target_text": "<eot>"},
                SpecialResponseTokenizer(),
                "raw",
            )

    def test_manifest_never_silently_shrinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            path.write_text('{"raw_prompt":"P", "target_text":"A"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "requested 2"):
                load_manifest(path, nsamples=2)

    def test_canonical_revision_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            path.write_text(
                '{"raw_prompt":"P", "target_text":"A", "revision":"wrong"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "does not match --revision"):
                load_manifest(path, nsamples=1, expected_revision="expected")

    def test_calibration_seed_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            path.write_text(
                '{"raw_prompt":"P", "target_text":"A", "calibration_seed":17}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "does not match --seed=5"):
                load_manifest(
                    path,
                    nsamples=1,
                    expected_calibration_seed=5,
                )

    def test_choice_sensitivity_keeps_alpaca_rows_on_completion_nll(self):
        rows = [
            {
                "_manifest_line": 1,
                "raw_prompt": "Question\nAnswer: ",
                "target_text": "A",
                "target_letter": "A",
                "choice_letters": ["A", "B"],
            },
            {
                "_manifest_line": 2,
                "raw_prompt": "Instruction\nAnswer: ",
                "target_text": "A complete response",
                "choice_letters": [],
                "task_format": "instruction_following",
            },
        ]
        prepared = prepare_examples(
            rows,
            CharacterTokenizer(),
            score_format="raw",
            loss_mode="choice_token",
            max_length=4096,
        )
        self.assertEqual(len(prepared[0].choices), 2)
        self.assertIsNone(prepared[0].completion)
        self.assertEqual(prepared[1].choices, ())
        self.assertIsNotNone(prepared[1].completion)


class LossTests(unittest.TestCase):
    def test_completion_nll_matches_manual_response_only_loss(self):
        logits = torch.tensor(
            [[[4.0, 0.0], [0.0, 3.0], [2.0, 1.0]]], requires_grad=True
        )
        ids = torch.tensor([[0, 1, 0]])
        actual = completion_nll_from_logits(logits, ids, response_start=1)
        manual = torch.stack(
            [
                -torch.log_softmax(logits[0, 0], dim=-1)[1],
                -torch.log_softmax(logits[0, 1], dim=-1)[0],
            ]
        ).mean()
        torch.testing.assert_close(actual, manual)

    def test_choice_loss_gradient_matches_two_class_cross_entropy(self):
        model = ToyCausalLM()
        model.requires_grad_(False)
        model.model.layers[0].proj.weight.requires_grad_(True)
        target_zero = toy_example(0).completion
        target_one = toy_example(1).completion
        choice_example = PreparedExample(
            record={},
            choices=(target_zero, target_one),
            target_choice_index=0,
        )
        evaluated = evaluate_manifest_mean_loss(model, [choice_example], "choice_token")
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        backward_loss = backward_example(model, choice_example, "choice_token")
        self.assertAlmostEqual(evaluated, backward_loss, places=7)
        choice_gradient = model.model.layers[0].proj.weight.grad.detach().clone()

        model.zero_grad(set_to_none=True)
        backward_example(model, toy_example(0), "completion_nll")
        completion_gradient = model.model.layers[0].proj.weight.grad.detach().clone()
        torch.testing.assert_close(choice_gradient, completion_gradient)

    def test_no_grad_completion_mean_matches_scoring_loss(self):
        model = ToyCausalLM()
        examples = [toy_example(0), toy_example(1), toy_example(0)]
        evaluated = evaluate_manifest_mean_loss(model, examples, "completion_nll")
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

        losses = []
        model.requires_grad_(False)
        model.model.layers[0].proj.weight.requires_grad_(True)
        for example in examples:
            model.zero_grad(set_to_none=True)
            losses.append(backward_example(model, example, "completion_nll"))
        self.assertAlmostEqual(evaluated, sum(losses) / len(losses), places=7)


class AttributionTests(unittest.TestCase):
    def _score(self, examples, directory, variant="paper", role_abs=False):
        score_manifest(
            model=ToyCausalLM(),
            examples=examples,
            output_dir=directory,
            role="preserve" if role_abs else "prune",
            loss_mode="completion_nll",
            no_abs=True,
            role_abs=role_abs,
            attribution_variant=variant,
            layers=None,
        )
        return load_only_tensor(directory)

    def test_duplicate_dataset_invariance(self):
        with tempfile.TemporaryDirectory() as temporary:
            one = self._score([toy_example(0)], Path(temporary) / "one")
            duplicate = self._score(
                [toy_example(0), toy_example(0)], Path(temporary) / "duplicate"
            )
            torch.testing.assert_close(one, duplicate)

    def test_signed_gradients_cancel_before_preservation_abs(self):
        with tempfile.TemporaryDirectory() as temporary:
            paper = self._score(
                [toy_example(0), toy_example(1)],
                Path(temporary) / "paper",
                role_abs=True,
            )
            released = self._score(
                [toy_example(0), toy_example(1)],
                Path(temporary) / "released",
                variant="released_abs",
                role_abs=True,
            )
            torch.testing.assert_close(paper, torch.zeros_like(paper), atol=1e-7, rtol=0)
            self.assertTrue(torch.all(released > 0))


class GlobalSelectionTests(unittest.TestCase):
    def test_selection_is_global_and_set_difference_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save(torch.tensor([100.0, 90.0]), root / "a.pt")
            torch.save(torch.tensor([80.0, 70.0]), root / "b.pt")
            metadata = {
                "tensors": {
                    "model.layers.0.a": {"file": "a.pt"},
                    "model.layers.1.b": {"file": "b.pt"},
                }
            }
            selected = exact_global_topk(root, metadata, 2, largest=True)
            self.assertEqual(selected.keys(), {"model.layers.0.a"})
            torch.testing.assert_close(selected["model.layers.0.a"], torch.tensor([0, 1]))

            remaining = set_difference(
                selected, {"model.layers.0.a": torch.tensor([1])}
            )
            torch.testing.assert_close(remaining["model.layers.0.a"], torch.tensor([0]))

    def test_second_global_slice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save(torch.tensor([4.0, 3.0]), root / "a.pt")
            torch.save(torch.tensor([2.0, 1.0]), root / "b.pt")
            metadata = {
                "tensors": {
                    "model.layers.0.a": {"file": "a.pt"},
                    "model.layers.1.b": {"file": "b.pt"},
                }
            }
            selected = exact_global_topk(root, metadata, 2, largest=True, rank_start=2)
            self.assertEqual(selected.keys(), {"model.layers.1.b"})
            torch.testing.assert_close(selected["model.layers.1.b"], torch.tensor([0, 1]))

    def test_nonfinite_score_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save(torch.tensor([1.0, float("nan")]), root / "score.pt")
            metadata = {
                "tensors": {"model.layers.0.a": {"file": "score.pt"}}
            }
            with self.assertRaisesRegex(RuntimeError, "non-finite score values"):
                exact_global_topk(root, metadata, 1, largest=True)


class ArtifactIdentityTests(unittest.TestCase):
    def _args(self, root, prune_manifest, preserve_manifest, q=1e-5):
        return SimpleNamespace(
            model="org/model",
            revision="abc123",
            tokenizer=None,
            tokenizer_revision=None,
            prune_manifest=str(prune_manifest),
            preserve_manifest=str(preserve_manifest),
            nsamples=1,
            nsamples_preserve=1,
            seed=5,
            score_format="raw",
            loss_mode="completion_nll",
            attribution_variant="paper",
            no_abs=True,
            abs_prune=False,
            abs_preserve=True,
            layers=None,
            max_score_length=4096,
            artifact_root=str(root),
            score_cache=None,
            q=q,
            p=5e-5,
            neg_prune=True,
            freeze_first_top_q=False,
            control="none",
            dump_score=False,
            use_saved_scores=False,
            dump_mask=False,
            dump_indices=False,
            alpha=0,
            match_bins=20,
        )

    def test_score_path_exposes_scientific_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prune = root / "prune.jsonl"
            preserve = root / "preserve.jsonl"
            prune.write_text('{"raw_prompt":"P ","target_text":"A"}\n', encoding="utf-8")
            preserve.write_text('{"raw_prompt":"P ","target_text":"A"}\n', encoding="utf-8")
            path = str(expected_score_dir(self._args(root / "artifacts", prune, preserve)))
            for component in (
                "revision_abc123",
                "format_raw",
                "loss_completion_nll",
                "attribution_paper",
                "seed_5",
                "noabs_1_absprune_0_abspreserve_1",
            ):
                self.assertIn(component, path)

    def test_external_same_basename_caches_have_identity_distinct_mask_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prune_a = root / "prune_a.jsonl"
            prune_b = root / "prune_b.jsonl"
            preserve = root / "preserve.jsonl"
            prune_a.write_text('{"raw_prompt":"A ","target_text":"A"}\n', encoding="utf-8")
            prune_b.write_text('{"raw_prompt":"B ","target_text":"A"}\n', encoding="utf-8")
            preserve.write_text('{"raw_prompt":"P ","target_text":"A"}\n', encoding="utf-8")
            args_a = self._args(root / "artifacts", prune_a, preserve)
            args_b = self._args(root / "artifacts", prune_b, preserve)
            cache_a = root / "external_a" / "same_name"
            cache_b = root / "external_b" / "same_name"
            self.assertNotEqual(
                mask_output_dir(args_a, cache_a),
                mask_output_dir(args_b, cache_b),
            )

    def test_q_zero_is_direct_unmodified_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prune = root / "prune.jsonl"
            preserve = root / "preserve.jsonl"
            prune.write_text('{"raw_prompt":"P ","target_text":"A"}\n', encoding="utf-8")
            preserve.write_text('{"raw_prompt":"P ","target_text":"A"}\n', encoding="utf-8")
            args = self._args(root / "artifacts", prune, preserve, q=0)
            result = run_manifest_global_pruning(args, ToyCausalLM(), CharacterTokenizer())
            self.assertTrue(result["baseline"])
            self.assertEqual(result["surviving_count"], 0)
            self.assertIsNone(result["score_dir"])
            self.assertFalse((Path(args.artifact_root) / "scores").exists())
            self.assertTrue(Path(result["mask_dir"], "metadata.json").exists())
            evaluation_path = write_evaluation_metadata(
                args,
                result,
                preservation_loss=0.25,
                wikitext_perplexity=7.5,
                sparsity=0.0,
            )
            with open(evaluation_path, "r", encoding="utf-8") as handle:
                evaluation = json.load(handle)
            self.assertEqual(
                set(
                    (
                        "preservation_loss",
                        "wikitext_perplexity",
                        "sparsity",
                        "model",
                        "revision",
                        "score_format",
                        "loss_mode",
                        "p",
                        "q",
                        "seed",
                        "control",
                    )
                ) - evaluation.keys(),
                set(),
            )
            self.assertEqual(evaluation["q"], 0.0)

    def test_sharded_scores_equal_monolithic_and_reuse_applies_exact_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prune = root / "prune.jsonl"
            preserve = root / "preserve.jsonl"
            prune.write_text(
                '{"raw_prompt":"P","target_text":"0","calibration_seed":5}\n',
                encoding="utf-8",
            )
            preserve.write_text(
                '{"raw_prompt":"P","target_text":"1","calibration_seed":5}\n',
                encoding="utf-8",
            )
            tokenizer = BinaryTokenizer()

            sharded_args = self._args(root / "sharded", prune, preserve, q=0.5)
            sharded_args.score_role = "prune"
            sharded_dir = dump_scores(sharded_args, ToyCausalLM(), tokenizer)
            sharded_args.score_role = "preserve"
            dump_scores(sharded_args, ToyCausalLM(), tokenizer)
            validate_score_cache(sharded_args)

            mono_args = self._args(root / "mono", prune, preserve, q=0.5)
            mono_args.score_role = "both"
            mono_dir = dump_scores(mono_args, ToyCausalLM(), tokenizer)
            for role in ("prune", "preserve"):
                torch.testing.assert_close(
                    load_only_tensor(sharded_dir / role),
                    load_only_tensor(mono_dir / role),
                )

            sharded_args.use_saved_scores = True
            sharded_args.dump_mask = True
            model = ToyCausalLM()
            result = run_manifest_global_pruning(sharded_args, model, tokenizer)
            self.assertEqual(result["nominal_prune_count"], 1)
            self.assertEqual(result["surviving_count"], 1)
            self.assertEqual(int((model.model.layers[0].proj.weight == 0).sum()), 1)
            self.assertTrue(Path(result["mask_dir"], "indices.pt").exists())

            # Explicit caches cannot be reused after any scientific input changes.
            sharded_args.score_cache = str(sharded_dir)
            prune.write_text('{"raw_prompt":"changed","target_text":"0"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                validate_score_cache(sharded_args)


if __name__ == "__main__":
    unittest.main()
