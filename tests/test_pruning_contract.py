from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from llmssycoph.llm.scoring import score_choices
from llmssycoph.pruning.cli import parse_args
from llmssycoph.pruning.data import CalibrationExample, EvalPair, PruningDatasets, build_pruning_datasets
from llmssycoph.pruning.losses import choice_token_loss
from llmssycoph.pruning.masks import (
    apply_mask,
    build_magnitude_mask,
    build_random_mask,
    count_masked_weights,
    restore_masked_values,
    select_pruning_mask,
)
from llmssycoph.pruning.metrics import choose_selected_sparsity, compute_item_metrics, summarize_item_metrics
from llmssycoph.pruning.runner import run as run_pruning
from llmssycoph.pruning.scores import collect_prunable_linear_weights, score_weight_importance


class FakeTokenizer:
    name_or_path = "fake-tokenizer"

    def __init__(self):
        self.vocab = {"A": 0, "B": 1, "C": 2, "prompt": 3, "<gen>": 4}

    def __call__(self, text, add_special_tokens=False):
        token = str(text).strip()
        return SimpleNamespace(input_ids=[self.vocab[token]])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
        del messages, tokenize, add_generation_prompt, return_tensors
        return torch.tensor([[3, 4]], dtype=torch.long)

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        inverse = {value: key for key, value in self.vocab.items()}
        return "".join(inverse.get(int(token_id), "?") for token_id in token_ids)


class TinyChoiceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 5, bias=False)
        with torch.no_grad():
            self.proj.weight[:, 0] = torch.tensor([0.2, 1.1, -0.4, 0.0, 0.0])
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, use_cache=False, output_hidden_states=False, return_dict=True):
        del attention_mask, use_cache, output_hidden_states, return_dict
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 5, dtype=self.proj.weight.dtype)
        hidden = torch.ones(batch, 1, dtype=self.proj.weight.dtype)
        logits[:, -1, :] = self.proj(hidden)
        return SimpleNamespace(logits=logits)


class FakeLLM:
    def __init__(self):
        self.model = TinyChoiceModel()
        self.tokenizer = FakeTokenizer()

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer


def make_row(condition: str, prompt_suffix: str = "") -> dict:
    prompt = f"Which option is correct?\n(A) alpha\n(B) beta\n(C) gamma"
    if condition != "neutral":
        prompt = f"{prompt}\n{prompt_suffix or condition}"
    return {
        "prompt": [{"type": "human", "content": prompt}],
        "base": {
            "dataset": "arc_challenge",
            "question": "Which option is correct?\n(A) alpha\n(B) beta\n(C) gamma",
            "correct_answer": "alpha",
            "incorrect_answer": "beta",
            "correct_letter": "A",
            "incorrect_letter": "B",
            "suggested_label": "B" if condition in {"incorrect_suggestion", "incorrect_suggestion_strong", "suggest_random"} else "A",
            "suggested_answer": "beta" if condition in {"incorrect_suggestion", "incorrect_suggestion_strong", "suggest_random"} else "alpha",
            "letters": "ABC",
            "answers": "(A) alpha\n(B) beta\n(C) gamma",
            "answers_list": ["alpha", "beta", "gamma"],
            "task_format": "multiple_choice",
            "mc_mode": "strict_mc",
            "instruction_policy": "answer_only",
            "source_split": "train",
            "source_example_id": "ex1",
        },
        "metadata": {"prompt_template": condition, "template_type": condition},
    }


def make_group(source_split: str = "train") -> dict:
    rows = {
        condition: make_row(condition)
        for condition in [
            "neutral",
            "incorrect_suggestion",
            "incorrect_suggestion_strong",
            "suggest_correct",
            "suggest_correct_strong",
            "suggest_random",
            "doubt_correct",
        ]
    }
    for row in rows.values():
        row["base"]["source_split"] = source_split
    return {
        "question_id": f"q_{source_split}",
        "dataset": "arc_challenge",
        "question": "Which option is correct?",
        "correct_answer": "alpha",
        "incorrect_answer": "beta",
        "source_split": source_split,
        "rows_by_type": rows,
    }


class PruningContractTests(unittest.TestCase):
    def test_choice_token_loss_matches_existing_score_choices_distribution(self):
        model = TinyChoiceModel()
        tokenizer = FakeTokenizer()
        messages = [{"type": "human", "content": "prompt"}]

        loss = choice_token_loss(model, tokenizer, messages, choices=["A", "B", "C"], target_choice="B")
        probabilities = score_choices(model, tokenizer, messages, ["A", "B", "C"])

        self.assertAlmostEqual(float(loss.item()), -math.log(probabilities["B"]), places=6)

    def test_snip_sign_and_mask_selection_contract(self):
        model = TinyChoiceModel()
        tokenizer = FakeTokenizer()
        messages = [{"type": "human", "content": "prompt"}]
        example = {
            "loss_type": "choice_token",
            "messages": messages,
            "choices": ["A", "B", "C"],
            "target_choice": "A",
        }

        model.zero_grad(set_to_none=True)
        manual_loss = choice_token_loss(model, tokenizer, messages, choices=["A", "B", "C"], target_choice="A")
        manual_loss.backward()
        expected = model.proj.weight.detach().float().cpu() * model.proj.weight.grad.detach().float().cpu()
        scored = score_weight_importance(model, tokenizer, [example], desc="unit-test")

        self.assertTrue(torch.allclose(scored["proj.weight"], expected))

        syc_scores = {"proj.weight": torch.tensor([[-0.5], [0.2], [-0.1], [0.3], [0.4]])}
        pres_scores = {"proj.weight": torch.zeros(5, 1)}
        selection = select_pruning_mask(
            syc_scores,
            pres_scores,
            sparsity=0.2,
            preserve_exclude_fraction=0.0,
        )
        self.assertEqual(selection.selected_count, 1)
        self.assertTrue(selection.masks["proj.weight"][0, 0])

    def test_mask_selection_is_global_and_preservation_exclusion_handles_ties(self):
        syc_scores = {
            "layer_a.weight": torch.tensor([[-10.0], [-9.0], [-8.0], [5.0], [6.0]]),
            "layer_b.weight": torch.tensor([[-1.0], [-0.5], [0.1], [0.2], [0.3]]),
        }
        pres_scores = {name: torch.zeros_like(score) for name, score in syc_scores.items()}
        selection = select_pruning_mask(
            syc_scores,
            pres_scores,
            sparsity=0.3,
            preserve_exclude_fraction=0.0,
        )

        self.assertEqual(selection.selected_count, 3)
        self.assertEqual(int(selection.masks["layer_a.weight"].sum().item()), 3)
        self.assertEqual(int(selection.masks["layer_b.weight"].sum().item()), 0)

        tie_selection = select_pruning_mask(
            syc_scores,
            pres_scores,
            sparsity=0.2,
            preserve_exclude_fraction=0.2,
        )
        self.assertEqual(tie_selection.requested_count, 2)
        self.assertEqual(tie_selection.selected_count, 2)

    def test_mask_apply_restore_and_controls_match_count(self):
        model = TinyChoiceModel()
        prunable = collect_prunable_linear_weights(model)
        random_mask = build_random_mask(prunable, count=2, seed=7)
        magnitude_mask = build_magnitude_mask(prunable, count=2)

        self.assertEqual(count_masked_weights(random_mask), 2)
        self.assertEqual(count_masked_weights(magnitude_mask), 2)

        before = model.proj.weight.detach().clone()
        originals = apply_mask(model, random_mask)
        self.assertNotEqual(float(model.proj.weight.detach().sum()), float(before.sum()))
        restore_masked_values(model, random_mask, originals)
        self.assertTrue(torch.equal(model.proj.weight.detach(), before))

    def test_dataset_assembly_targets_and_controls(self):
        args = SimpleNamespace(
            prune_family="incorrect_suggestion",
            eval_families=[
                "incorrect_suggestion",
                "incorrect_suggestion_rephrase_1",
                "incorrect_suggestion_rephrase_2",
                "model_congruent_suggestion",
            ],
            test_frac=0.2,
            val_frac=0.2,
            split_seed=5,
            max_calibration_records=None,
            max_preservation_records=None,
            max_eval_records=None,
            wrong_control_min_examples=50,
            seed=5,
        )

        def choice_scorer(_messages, choices):
            return {"A": 0.8, "B": 0.1, "C": 0.1}

        groups = [make_group("train"), make_group("validation"), make_group("test")]
        with patch("llmssycoph.pruning.data._load_prepared_groups", return_value=groups), patch(
            "llmssycoph.pruning.data._load_text_preservation", return_value=[]
        ):
            datasets = build_pruning_datasets(args, choice_scorer=choice_scorer)

        self.assertEqual(datasets.sycophancy[0].target_choice, "B")
        self.assertIn("neutral", {example.condition for example in datasets.preservation})
        self.assertIn("model_congruent_suggestion", {example.condition for example in datasets.preservation})
        self.assertIn("truthful_correction", {example.condition for example in datasets.truthful_correction})
        self.assertEqual({pair.condition for pair in datasets.eval_pairs}, set(args.eval_families))

    def test_metrics_formulas(self):
        pair = SimpleNamespace(
            pair_id="p1",
            dataset="arc_challenge",
            split="test",
            condition="incorrect_suggestion",
            question_id="q1",
            choices=["A", "B", "C"],
            correct_letter="A",
            incorrect_letter="B",
            target_letter="B",
        )
        row = compute_item_metrics(
            pair,
            neutral_probabilities={"A": 0.7, "B": 0.2, "C": 0.1},
            biased_probabilities={"A": 0.3, "B": 0.6, "C": 0.1},
            sparsity=0.001,
            mask_name="sycophancy",
        )
        self.assertAlmostEqual(row["delta_p_b"], 0.4)
        self.assertAlmostEqual(row["gap_closure"], 0.8)
        self.assertEqual(row["flip_rate_to_b"], 1)

        summary = summarize_item_metrics(__import__("pandas").DataFrame([row]))
        self.assertEqual(int(summary.iloc[0]["n_pairs"]), 1)

    def test_selected_sparsity_uses_validation_mean_across_datasets(self):
        summary = pd.DataFrame(
            [
                {
                    "mask_name": "sycophancy",
                    "sparsity": 0.0,
                    "split": "val",
                    "dataset": "small",
                    "condition": "incorrect_suggestion",
                    "n_pairs": 1,
                    "mean_delta_p_b": 1.0,
                    "neutral_accuracy": 1.0,
                    "preservation_loss_increase": 0.0,
                },
                {
                    "mask_name": "sycophancy",
                    "sparsity": 0.0,
                    "split": "val",
                    "dataset": "large",
                    "condition": "incorrect_suggestion",
                    "n_pairs": 99,
                    "mean_delta_p_b": 1.0,
                    "neutral_accuracy": 1.0,
                    "preservation_loss_increase": 0.0,
                },
                {
                    "mask_name": "sycophancy",
                    "sparsity": 0.1,
                    "split": "val",
                    "dataset": "small",
                    "condition": "incorrect_suggestion",
                    "n_pairs": 1,
                    "mean_delta_p_b": 0.0,
                    "neutral_accuracy": 1.0,
                    "preservation_loss_increase": 0.0,
                },
                {
                    "mask_name": "sycophancy",
                    "sparsity": 0.1,
                    "split": "val",
                    "dataset": "large",
                    "condition": "incorrect_suggestion",
                    "n_pairs": 99,
                    "mean_delta_p_b": 1.0,
                    "neutral_accuracy": 1.0,
                    "preservation_loss_increase": 0.0,
                },
                {
                    "mask_name": "sycophancy",
                    "sparsity": 0.2,
                    "split": "val",
                    "dataset": "small",
                    "condition": "incorrect_suggestion",
                    "n_pairs": 1,
                    "mean_delta_p_b": 0.6,
                    "neutral_accuracy": 1.0,
                    "preservation_loss_increase": 0.0,
                },
                {
                    "mask_name": "sycophancy",
                    "sparsity": 0.2,
                    "split": "val",
                    "dataset": "large",
                    "condition": "incorrect_suggestion",
                    "n_pairs": 99,
                    "mean_delta_p_b": 0.6,
                    "neutral_accuracy": 1.0,
                    "preservation_loss_increase": 0.0,
                },
            ]
        )

        selected = choose_selected_sparsity(
            summary,
            syc_reduction_target=0.30,
            preservation_loss_budget=0.10,
            neutral_accuracy_drop_budget=0.05,
        )
        self.assertEqual(selected, 0.2)

    def test_pruning_cli_defaults(self):
        args = parse_args([])
        self.assertEqual(args.model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(args.datasets, ["arc_challenge", "commonsense_qa"])
        self.assertEqual(args.prune_family, "incorrect_suggestion")
        self.assertEqual(args.target_loss, "choice_token")
        self.assertIn(1e-3, args.sparsities)

    def test_slurm_wrapper_has_required_mail_settings(self):
        job_dir = Path(__file__).resolve().parents[1] / "jobs" / "sycophancy_pruning"
        sbatch_files = sorted(job_dir.glob("*.sbatch"))
        self.assertGreaterEqual(len(sbatch_files), 1)
        for path in sbatch_files:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("#SBATCH --mail-type=END,FAIL", text)
                self.assertIn("#SBATCH --mail-user=itaishapira@g.harvard.edu", text)

    def test_cluster_job_shell_syntax(self):
        job_dir = Path(__file__).resolve().parents[1] / "jobs" / "sycophancy_pruning"
        shell_files = sorted(list(job_dir.glob("*.sbatch")) + list(job_dir.glob("*.sh")))
        self.assertGreaterEqual(len(shell_files), 1)
        for path in shell_files:
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_mocked_runner_smoke_writes_paper_style_artifacts(self):
        def example(condition: str, target: str) -> CalibrationExample:
            return CalibrationExample(
                example_id=f"train:q1:{condition}",
                dataset="arc_challenge",
                split="train",
                condition=condition,
                question_id="q1",
                loss_type="choice_token",
                messages=[{"type": "human", "content": "prompt"}],
                choices=["A", "B", "C"],
                target_choice=target,
                correct_letter="A",
                incorrect_letter="B",
            )

        def pair(split: str, suffix: str, neutral: str, biased: str) -> EvalPair:
            return EvalPair(
                pair_id=f"{split}:{suffix}:incorrect_suggestion",
                dataset="arc_challenge",
                split=split,
                condition="incorrect_suggestion",
                question_id=f"q_{suffix}",
                neutral_messages=[{"type": "human", "content": neutral}],
                biased_messages=[{"type": "human", "content": biased}],
                choices=["A", "B", "C"],
                correct_letter="A",
                incorrect_letter="B",
                target_letter="B",
            )

        datasets = PruningDatasets(
            sycophancy=[example("incorrect_suggestion", "B")],
            preservation=[example("neutral", "A"), example("truthful_correction", "A")],
            truthful_correction=[example("truthful_correction", "A")],
            neutral_wrong=[],
            eval_pairs=[
                pair("val", "high", "neutral-high", "biased-high"),
                pair("val", "low", "neutral-low", "biased-low"),
                pair("test", "high", "neutral-high", "biased-high"),
                pair("test", "low", "neutral-low", "biased-low"),
            ],
            groups_by_split={"train": [], "val": [], "test": []},
        )

        def fake_choice_probabilities(_model, _tokenizer, messages, *, choices):
            del choices
            text = messages[0]["content"]
            if "neutral-high" in text:
                return {"A": 0.90, "B": 0.05, "C": 0.05}
            if "neutral-low" in text:
                return {"A": 0.60, "B": 0.25, "C": 0.15}
            if "biased-high" in text:
                return {"A": 0.40, "B": 0.55, "C": 0.05}
            if "biased-low" in text:
                return {"A": 0.45, "B": 0.40, "C": 0.15}
            return {"A": 0.70, "B": 0.20, "C": 0.10}

        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--run_name",
                    "paper_smoke",
                    "--sparsities",
                    "0,0.2,0.4",
                    "--device",
                    "cpu",
                ]
            )
            with patch("llmssycoph.pruning.runner.load_llm", return_value=FakeLLM()), patch(
                "llmssycoph.pruning.runner.build_pruning_datasets", return_value=datasets
            ), patch(
                "llmssycoph.pruning.runner.choice_token_probabilities", side_effect=fake_choice_probabilities
            ):
                run_dir = run_pruning(args)

            expected = [
                "run_config.json",
                "status.json",
                "logs/run.log",
                "calibration/sycophancy.jsonl",
                "calibration/preservation.jsonl",
                "calibration/truthful_correction.jsonl",
                "calibration/neutral_wrong.jsonl",
                "calibration/eval_pairs.jsonl",
                "masks/selected_sycophancy.pt",
                "masks/random.pt",
                "masks/magnitude.pt",
                "masks/truthful_correction.pt",
                "metrics/sweep_metrics.csv",
                "metrics/item_metrics.csv",
                "metrics/control_metrics.csv",
                "reports/summary.md",
                "plots/delta_p_by_neutral_confidence_bucket.png",
                "plots/delta_p_by_neutral_confidence_bucket.pdf",
            ]
            for relative_path in expected:
                self.assertTrue((run_dir / relative_path).exists(), relative_path)

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "completed")
            controls = pd.read_csv(run_dir / "metrics" / "control_metrics.csv")
            self.assertIn("insufficient_examples", set(controls["status"]))
            self.assertIn("truthful_correction", set(controls["control_name"]))


if __name__ == "__main__":
    unittest.main()
