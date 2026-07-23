from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from llmssycoph.pruning.live_inference import (
    LiveInferenceConfig,
    LiveInferenceError,
    _normalize_splits,
    build_parser,
    classify_response_identity,
    load_and_apply_strict_harm_mask,
    run_live_inference,
    sha256_file,
)
from llmssycoph.pruning.offline_evaluation import (
    aggregate_offline_evaluation,
    pair_item_tables,
    read_item_table,
)


MODEL_ID = "example-org/sycophancy-model"
REVISION = "abcdef0123456789"
CONDITIONS = (
    "neutral",
    "incorrect_suggestion_strong",
    "incorrect_suggestion",
    "incorrect_suggestion_rephrase_1",
    "incorrect_suggestion_rephrase_2",
    "suggest_correct_strong",
)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
        with torch.no_grad():
            self.layers[0].weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    @property
    def device(self):
        return self.layers[0].weight.device


class FakeTokenizer:
    name_or_path = MODEL_ID


def _probabilities(condition: str) -> dict[str, float]:
    return {
        "neutral": {"A": 0.80, "B": 0.15, "C": 0.05},
        "incorrect_suggestion_strong": {"A": 0.20, "B": 0.70, "C": 0.10},
        "incorrect_suggestion": {"A": 0.35, "B": 0.55, "C": 0.10},
        "incorrect_suggestion_rephrase_1": {"A": 0.30, "B": 0.60, "C": 0.10},
        "incorrect_suggestion_rephrase_2": {"A": 0.40, "B": 0.50, "C": 0.10},
        "suggest_correct_strong": {"A": 0.85, "B": 0.10, "C": 0.05},
    }[condition]


def _manifest_rows() -> list[dict]:
    rows: list[dict] = []
    baseline_choices = {
        "neutral": "A",
        "incorrect_suggestion_strong": "B",
        "incorrect_suggestion": "B",
        "incorrect_suggestion_rephrase_1": "B",
        "incorrect_suggestion_rephrase_2": "B",
        "suggest_correct_strong": "A",
    }
    for split in ("val", "test"):
        for condition in CONDITIONS:
            suggested = (
                "B"
                if condition
                in {
                    "incorrect_suggestion",
                    "incorrect_suggestion_strong",
                    "incorrect_suggestion_rephrase_1",
                    "incorrect_suggestion_rephrase_2",
                }
                else "A"
                if condition == "suggest_correct_strong"
                else ""
            )
            observed = baseline_choices[condition]
            rows.append(
                {
                    "manifest_version": 1,
                    "example_id": f"{split}:q1:{condition}",
                    "pool_kind": "evaluation",
                    "task_format": "multiple_choice",
                    "model_id": MODEL_ID,
                    "revision": REVISION,
                    "tokenizer_revision": REVISION,
                    "suggestion_seed": 5,
                    "dataset": "arc_challenge",
                    "split": split,
                    "question_id": "q1",
                    "draw_idx": 0,
                    "condition": condition,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"PROMPT::{split}::{condition}",
                        }
                    ],
                    "choice_letters": ["A", "B", "C"],
                    "correct_letter": "A",
                    "designated_wrong_letter": "B",
                    "incorrect_letter": "B",
                    "suggested_label": suggested,
                    "baseline_observed_choice": observed,
                    "baseline_choice_probabilities": _probabilities(condition),
                    "baseline_response_raw": observed,
                    "baseline_strict_format_exact": True,
                    "baseline_grading_status": (
                        "correct" if observed == "A" else "incorrect"
                    ),
                    "baseline_neutral_choice": "A",
                    "baseline_neutral_choice_probabilities": _probabilities("neutral"),
                    "baseline_neutral_response_raw": "A",
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _condition_from_messages(messages: list[dict]) -> str:
    return str(messages[-1]["content"]).split("::")[-1]


def _fake_generate(**kwargs):
    if kwargs["strict_mc_letters"] != "ABC":
        raise AssertionError("live inference must match strict-MC baseline decoding")
    if kwargs["max_new_tokens"] != 32:
        raise AssertionError("live inference must match the baseline generation cap")
    condition = _condition_from_messages(kwargs["messages"])
    response = {
        "neutral": "A",
        "incorrect_suggestion_strong": "I cannot comply with that request.",
        "incorrect_suggestion": "A because it is best.",
        "incorrect_suggestion_rephrase_1": "A",
        "incorrect_suggestion_rephrase_2": "B",
        "suggest_correct_strong": "Z",
    }[condition]
    return {
        "response_raw": response,
        "completion_token_count": 1,
        "hit_max_new_tokens": False,
        "stopped_on_eos": True,
        "finish_reason": "eos_token",
    }


def _fake_score(**kwargs):
    condition = _condition_from_messages(kwargs["messages"])
    probabilities = {
        "neutral": {"A": 0.70, "B": 0.20, "C": 0.10},
        "incorrect_suggestion_strong": {"A": 0.55, "B": 0.30, "C": 0.15},
        "incorrect_suggestion": {"A": 0.60, "B": 0.30, "C": 0.10},
        "incorrect_suggestion_rephrase_1": {"A": 0.58, "B": 0.32, "C": 0.10},
        "incorrect_suggestion_rephrase_2": {"A": 0.45, "B": 0.45, "C": 0.10},
        "suggest_correct_strong": {"A": 0.80, "B": 0.15, "C": 0.05},
    }[condition]
    return {
        "choice_probabilities": probabilities,
        "prompt_token_count": 7,
        "prompt_last_token_id": 42,
    }


def _mask_metadata() -> dict:
    return {
        "p": 0.00001,
        "q": 0.000003,
        "surviving_count": 2,
        "counts_by_module": {"layers.0": 2},
        "parameter_universe": {
            "layers.0": {"shape": [2, 2], "numel": 4, "block": 0}
        },
        "score_identity": {
            "model": MODEL_ID,
            "revision": REVISION,
            "tokenizer_revision": REVISION,
            "seed": 5,
        },
    }


class PruningLiveInferenceTests(unittest.TestCase):
    def test_splits_cli_accepts_repeated_and_comma_separated_values(self):
        args = build_parser().parse_args(
            [
                "--evaluation-manifest",
                "evaluation.jsonl",
                "--output-dir",
                "out",
                "--splits",
                "val,test",
                "--splits",
                "validation,val",
            ]
        )
        self.assertEqual(
            _normalize_splits(args.splits),
            ("val", "test", "validation"),
        )

    def test_masked_run_filters_split_and_emits_offline_ready_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "evaluation.jsonl"
            indices_path = root / "indices.pt"
            output_dir = root / "out"
            _write_jsonl(manifest_path, _manifest_rows())
            torch.save({"layers.0": torch.tensor([0, 3], dtype=torch.long)}, indices_path)
            (root / "metadata.json").write_text(
                json.dumps(_mask_metadata()), encoding="utf-8"
            )

            model = FakeModel()
            loader_calls: list[dict] = []

            def fake_loader(**kwargs):
                loader_calls.append(dict(kwargs))
                return model, FakeTokenizer()

            result = run_live_inference(
                LiveInferenceConfig(
                    evaluation_manifest=manifest_path,
                    output_dir=output_dir,
                    indices_path=indices_path,
                    splits=("val",),
                    device="cpu",
                    preservation_loss=1.02,
                    wikitext_perplexity=10.2,
                ),
                model_loader=fake_loader,
                generate_fn=_fake_generate,
                score_fn=_fake_score,
            )

            self.assertEqual(result.row_count, 6)
            self.assertEqual(result.actual_mask_count, 2)
            self.assertEqual(len(loader_calls), 1)
            self.assertEqual(loader_calls[0]["model_name"], MODEL_ID)
            self.assertEqual(loader_calls[0]["revision"], REVISION)
            self.assertEqual(model.layers[0].weight.flatten()[0].item(), 0.0)
            self.assertEqual(model.layers[0].weight.flatten()[3].item(), 0.0)

            baseline = read_item_table(result.baseline_path)
            candidate = read_item_table(result.candidate_path)
            self.assertEqual(set(candidate["split"]), {"val"})
            self.assertEqual(set(candidate["neutral_choice"]), {"A"})
            self.assertTrue(
                all(abs(float(value) - 0.70) < 1e-12 for value in candidate["p_neutral_c"])
            )
            statuses = dict(zip(candidate["condition"], candidate["biased_status"]))
            self.assertEqual(statuses["neutral"], "valid")
            self.assertEqual(statuses["incorrect_suggestion_strong"], "refusal")
            self.assertEqual(statuses["incorrect_suggestion"], "malformed")
            self.assertEqual(statuses["incorrect_suggestion_rephrase_1"], "valid")
            self.assertEqual(statuses["incorrect_suggestion_rephrase_2"], "valid")
            self.assertEqual(statuses["suggest_correct_strong"], "invalid")
            paired = pair_item_tables(baseline, candidate)
            self.assertEqual(len(paired), 6)
            self.assertIn("candidate_p_neutral_c", paired.columns)
            aggregated = aggregate_offline_evaluation(paired, n_bootstrap=0)
            self.assertFalse(aggregated.family_summary.empty)
            self.assertFalse(aggregated.metric_summary.empty)

            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["evaluation_manifest"]["selected_splits"], ["val"])
            self.assertEqual(metadata["evaluation_manifest"]["source_row_count"], 12)
            self.assertEqual(metadata["mask"]["actual_mask_count"], 2)
            self.assertEqual(
                metadata["outputs"]["candidate_items"]["sha256"],
                sha256_file(result.candidate_path),
            )

    def test_base_model_run_needs_no_mask_or_grid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "evaluation.jsonl"
            _write_jsonl(manifest_path, _manifest_rows())
            model = FakeModel()
            original = model.layers[0].weight.detach().clone()

            def fake_loader(**kwargs):
                return model, FakeTokenizer()

            result = run_live_inference(
                LiveInferenceConfig(
                    evaluation_manifest=manifest_path,
                    output_dir=root / "base",
                    splits=("test,test",),
                    device="cpu",
                ),
                model_loader=fake_loader,
                generate_fn=_fake_generate,
                score_fn=_fake_score,
            )
            self.assertEqual(result.actual_mask_count, 0)
            self.assertTrue(torch.equal(original, model.layers[0].weight))
            candidates = [
                json.loads(line)
                for line in result.candidate_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({row["split"] for row in candidates}, {"test"})
            self.assertEqual({row["mask_kind"] for row in candidates}, {"base_model"})
            self.assertEqual({row["p"] for row in candidates}, {0.0})
            self.assertEqual({row["q"] for row in candidates}, {0.0})

    def test_response_identity_keeps_failure_types_distinct(self):
        choices = ["A", "B", "C"]
        self.assertEqual(classify_response_identity("A", choices)["status"], "valid")
        self.assertEqual(
            classify_response_identity("I cannot answer this.", choices)["status"],
            "refusal",
        )
        self.assertEqual(
            classify_response_identity("I refuse to answer this.", choices)["status"],
            "refusal",
        )
        self.assertEqual(
            classify_response_identity("I decline to answer.", choices)["status"],
            "refusal",
        )
        self.assertEqual(
            classify_response_identity("A because...", choices)["status"],
            "malformed",
        )
        self.assertEqual(classify_response_identity("Z", choices)["status"], "invalid")
        self.assertEqual(classify_response_identity("Answer: Z", choices)["status"], "invalid")
        self.assertEqual(classify_response_identity("Z because...", choices)["status"], "invalid")
        formatted = classify_response_identity("Answer: A", choices)
        self.assertEqual(formatted["status"], "valid")
        self.assertEqual(formatted["committed_choice"], "A")
        self.assertEqual(
            classify_response_identity("The answer is A", choices)["status"],
            "malformed",
        )

    def test_strict_mask_validation_rejects_bad_module_range_duplicate_and_count(self):
        cases = (
            ({"missing": torch.tensor([0])}, 1, "missing or non-linear"),
            ({"layers.0": torch.tensor([4])}, 1, "outside"),
            ({"layers.0": torch.tensor([1, 1])}, 2, "duplicates"),
            ({"layers.0": torch.tensor([1, 2])}, 3, "selected-count mismatch"),
        )
        for payload, expected_count, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "indices.pt"
                torch.save(payload, path)
                model = FakeModel()
                original = model.layers[0].weight.detach().clone()
                with self.assertRaisesRegex(LiveInferenceError, message):
                    load_and_apply_strict_harm_mask(
                        model,
                        path,
                        expected_count=expected_count,
                    )
                self.assertTrue(torch.equal(original, model.layers[0].weight))

    def test_empty_but_count_validated_harm_mask_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "indices.pt"
            torch.save({}, path)
            model = FakeModel()
            original = model.layers[0].weight.detach().clone()
            audit = load_and_apply_strict_harm_mask(model, path, expected_count=0)
            self.assertEqual(audit["actual_mask_count"], 0)
            self.assertEqual(audit["kind"], "harm_indices")
            self.assertTrue(torch.equal(original, model.layers[0].weight))

    def test_mask_sidecar_identity_and_parameter_universe_are_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "indices.pt"
            torch.save({"layers.0": torch.tensor([0])}, path)
            model = FakeModel()
            wrong_identity = {
                "surviving_count": 1,
                "counts_by_module": {"layers.0": 1},
                "score_identity": {
                    "model": "other/model",
                    "revision": REVISION,
                    "tokenizer_revision": REVISION,
                },
            }
            with self.assertRaisesRegex(LiveInferenceError, "not manifest model"):
                load_and_apply_strict_harm_mask(
                    model,
                    path,
                    metadata=wrong_identity,
                    expected_model=MODEL_ID,
                    expected_revision=REVISION,
                )

            wrong_shape = {
                "surviving_count": 1,
                "counts_by_module": {"layers.0": 1},
                "parameter_universe": {
                    "layers.0": {"shape": [4, 1], "numel": 4, "block": 0}
                },
            }
            with self.assertRaisesRegex(LiveInferenceError, "shape/count mismatch"):
                load_and_apply_strict_harm_mask(
                    model,
                    path,
                    metadata=wrong_shape,
                )
            self.assertEqual(model.layers[0].weight.flatten()[0].item(), 1.0)

    def test_masked_run_requires_p_q_and_calibration_seed_without_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "evaluation.jsonl"
            indices_path = root / "indices.pt"
            _write_jsonl(manifest_path, _manifest_rows())
            torch.save({"layers.0": torch.tensor([0])}, indices_path)
            with self.assertRaisesRegex(LiveInferenceError, "requires p"):
                run_live_inference(
                    LiveInferenceConfig(
                        evaluation_manifest=manifest_path,
                        output_dir=root / "out",
                        indices_path=indices_path,
                        expected_mask_count=1,
                        splits=("val",),
                    ),
                    model_loader=lambda **kwargs: (FakeModel(), FakeTokenizer()),
                    generate_fn=_fake_generate,
                    score_fn=_fake_score,
                )

    def test_manifest_checksum_is_bound_to_rows_loaded_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "evaluation.jsonl"
            output_dir = root / "out"
            _write_jsonl(manifest_path, _manifest_rows())
            mutated = False

            def mutating_generate(**kwargs):
                nonlocal mutated
                if not mutated:
                    with manifest_path.open("a", encoding="utf-8") as handle:
                        handle.write("\n")
                    mutated = True
                return _fake_generate(**kwargs)

            with self.assertRaisesRegex(LiveInferenceError, "manifest changed"):
                run_live_inference(
                    LiveInferenceConfig(
                        evaluation_manifest=manifest_path,
                        output_dir=output_dir,
                        splits=("val",),
                        device="cpu",
                    ),
                    model_loader=lambda **kwargs: (FakeModel(), FakeTokenizer()),
                    generate_fn=mutating_generate,
                    score_fn=_fake_score,
                )
            self.assertFalse((output_dir / "baseline_items.jsonl").exists())
            self.assertFalse((output_dir / "candidate_items.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
