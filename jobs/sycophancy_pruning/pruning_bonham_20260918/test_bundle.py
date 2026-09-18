#!/usr/bin/env python3
"""Fast, scheduler-free contract tests for Bonham."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import torch

import core


def _question() -> core.Question:
    return core.Question(
        dataset_id="commonsense_qa",
        source_example_id="q-1",
        source_split="train",
        question="Which process lets plants turn light into stored energy?",
        labels=("A", "B", "C", "D"),
        answers=("respiration", "photosynthesis", "digestion", "fermentation"),
        gold="B",
    )


class PromptRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = core.load_config()

    def test_frozen_registry_counts(self) -> None:
        self.assertEqual(4, len(self.config["construction_templates"]["incorrect_suggestion"]))
        self.assertEqual(4, len(self.config["construction_templates"]["doubt_correct"]))
        self.assertEqual(8, len(self.config["close_paraphrase_templates"]["incorrect_suggestion"]))
        self.assertEqual(8, len(self.config["close_paraphrase_templates"]["doubt_correct"]))
        self.assertEqual(24, len(self.config["naturalistic_templates"]))
        self.assertEqual(12, len(self.config["source_templates"]))

    def test_bias_sentence_is_identical_between_turn_formats(self) -> None:
        question = _question()
        wrong = core.designated_wrong(question)
        for bias_type in core.BIAS_TYPES:
            for template_index in range(4):
                sentence = core.construction_bias(
                    self.config, question, bias_type, template_index, wrong
                )
                single = core.render_messages(
                    question,
                    bias_sentence=sentence,
                    turn_format="single_turn",
                    assistant_answer=None,
                    answer_instruction=self.config["answer_instruction"],
                )
                multi = core.render_messages(
                    question,
                    bias_sentence=sentence,
                    turn_format="multi_turn",
                    assistant_answer=question.gold,
                    answer_instruction=self.config["answer_instruction"],
                )
                self.assertIn(sentence, single[0]["content"])
                self.assertEqual(
                    sentence,
                    multi[-1]["content"].split("\n\n", maxsplit=1)[0],
                )

    def test_all_prompt_regimes_render_letter_and_text(self) -> None:
        question = _question()
        wrong = core.designated_wrong(question)
        regime_sizes = {"seen": 4, "close_paraphrase": 8, "naturalistic": 24}
        for regime, count in regime_sizes.items():
            for bias_type in core.BIAS_TYPES:
                for index in range(count):
                    sentence, family = core.evaluation_bias(
                        self.config,
                        question,
                        bias_type=bias_type,
                        regime=regime,
                        template_index=index,
                        wrong=wrong,
                    )
                    target = wrong if bias_type == "incorrect_suggestion" else question.gold
                    self.assertIn(f"option {target}", sentence)
                    self.assertIn("(“", sentence)
                    self.assertTrue(family)

    def test_source_cell_template_quota(self) -> None:
        indices = core.source_template_indices()
        counts = Counter(indices)
        self.assertEqual(64, len(indices))
        self.assertEqual(16, sum(counts[index] for index in range(3)))
        self.assertEqual(48, sum(counts[index] for index in range(3, 12)))
        self.assertEqual(set(range(12)), set(indices))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_balanced_evaluation_assignment(self) -> None:
        assignments = core.balanced_template_assignments(
            [f"q-{index}" for index in range(500)], 24, "bonham-test"
        )
        counts = Counter(assignments.values())
        self.assertEqual(500, len(assignments))
        self.assertEqual(set(range(24)), set(counts))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


class ScoreAndSelectorTests(unittest.TestCase):
    def test_preservation_is_mean_absolute_per_example_not_absolute_mean(self) -> None:
        weight = torch.tensor([2.0])
        accumulator = torch.zeros(1, dtype=torch.float32)
        gradients = (torch.tensor([1.0]), torch.tensor([-1.0]))
        for gradient in gradients:
            core.per_example_preservation_update(accumulator, weight, gradient)
        correct = accumulator / len(gradients)
        historical_wrong = (-weight * torch.stack(gradients).mean(dim=0)).abs()
        self.assertEqual(2.0, float(correct.item()))
        self.assertEqual(0.0, float(historical_wrong.item()))

    def test_selector_protection_determinism_exact_n_and_nested_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prune_dir = root / "prune"
            preserve_dir = root / "preserve"
            prune_dir.mkdir()
            preserve_dir.mkdir()
            names = ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.up_proj.weight")
            prune_values = {
                names[0]: torch.tensor([100.0, 9.0, 8.0, 7.0, 6.0, 5.0]),
                names[1]: torch.tensor([10.0, 10.0, 4.0, 3.0, 2.0, 1.0]),
            }
            preserve_values = {
                names[0]: torch.tensor([99.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                names[1]: torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 98.0]),
            }
            for directory, tensors, aggregation in (
                (prune_dir, prune_values, "signed_mean_negative_weight_times_gradient"),
                (preserve_dir, preserve_values, "mean_absolute_per_example_weight_times_gradient"),
            ):
                metadata = {"model_id": "toy/model", "aggregation": aggregation, "tensors": {}}
                for index, (name, tensor) in enumerate(tensors.items()):
                    filename = f"tensor_{index}.pt"
                    torch.save(tensor, directory / filename)
                    metadata["tensors"][name] = {"file": filename, "shape": list(tensor.shape)}
                (directory / "metadata.json").write_text(
                    json.dumps(metadata, sort_keys=True), encoding="utf-8"
                )

            two, two_meta = core.select_mask(
                prune_dir, preserve_dir, p=1 / 6, n=2, ordering_seed="test-seed"
            )
            three, three_meta = core.select_mask(
                prune_dir, preserve_dir, p=1 / 6, n=3, ordering_seed="test-seed"
            )
            repeat, _ = core.select_mask(
                prune_dir, preserve_dir, p=1 / 6, n=3, ordering_seed="test-seed"
            )
            protected = {(names[0], 0), (names[1], 5)}
            self.assertTrue(core.mask_coordinates(three).isdisjoint(protected))
            self.assertEqual(2, len(core.mask_coordinates(two)))
            self.assertEqual(3, len(core.mask_coordinates(three)))
            self.assertTrue(core.mask_coordinates(two) < core.mask_coordinates(three))
            self.assertEqual(core.mask_coordinates(three), core.mask_coordinates(repeat))
            self.assertEqual(2, two_meta["n"])
            self.assertEqual(3, three_meta["n"])


if __name__ == "__main__":
    unittest.main()
