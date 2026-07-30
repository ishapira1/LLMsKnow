from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from llmssycoph.interventions.activations import (
    block_for_residual_layer,
    extract_prompt_state,
    residual_addition_hook,
    residual_additions_hooks,
    residual_replacement_hook,
    score_with_multilayer_residual_additions,
    score_with_residual_additions,
)
from llmssycoph.interventions.data import build_intervention_pairs
from llmssycoph.interventions.directions import (
    fit_direction_arrays,
    orthogonal_component,
    parallel_component,
    save_direction_artifact,
)
from llmssycoph.interventions.experiment import (
    aggregate_intervention_results,
    select_validation_dose,
    select_validation_layers,
    write_jsonl,
)
from llmssycoph.interventions.metrics import (
    correct_endorsed_margin,
    distribution_shift,
    make_result_row,
    normalized_recovery,
)


try:
    import torch
except ImportError:  # pragma: no cover - project runtime normally provides torch
    torch = None


class _Tokenized:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _FakeTokenizer:
    name_or_path = "fake-tokenizer"
    chat_template = "fake-template"
    init_kwargs = {"_commit_hash": "fake-tokenizer-revision"}

    _ids = {
        "A": 1,
        " A": 2,
        "\nA": 3,
        "B": 4,
        " B": 5,
        "\nB": 6,
    }

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return _Tokenized([self._ids[str(text)]])

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        reverse = {value: key for key, value in self._ids.items()}
        return "".join(reverse.get(int(value), "?") for value in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors,
    ):
        del messages, tokenize, add_generation_prompt, return_tensors
        return torch.tensor([[0, 0]], dtype=torch.long)


if torch is not None:

    class _FakeBlock(torch.nn.Module):
        def __init__(self, increment: float):
            super().__init__()
            self.increment = float(increment)

        def forward(self, hidden):
            return (hidden + self.increment,)


    class _FakeNorm(torch.nn.Module):
        def forward(self, hidden):
            return hidden


    class _FakeDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_FakeBlock(0.25), _FakeBlock(0.5)])
            self.norm = _FakeNorm()


    class _FakeLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _FakeDecoder()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(
                _name_or_path="fake-model", _commit_hash="fake-model-revision"
            )
            weights = torch.zeros((10, 4), dtype=torch.float32)
            weights[1:4, 0] = 1.0
            weights[4:7, 0] = -1.0
            self.register_buffer("lm_weights", weights)

        @property
        def device(self):
            return self.anchor.device

        @property
        def dtype(self):
            return self.anchor.dtype

        def forward(
            self,
            input_ids,
            attention_mask=None,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        ):
            del attention_mask, use_cache, return_dict
            hidden = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            all_hidden = []
            for block in self.model.layers:
                all_hidden.append(hidden)
                hidden = block(hidden)[0]
            hidden = self.model.norm(hidden)
            all_hidden.append(hidden)
            logits = torch.einsum("bsh,vh->bsv", hidden, self.lm_weights)
            return SimpleNamespace(
                logits=logits,
                hidden_states=tuple(all_hidden) if output_hidden_states else None,
            )


def _record(
    condition: str,
    *,
    record_id: int,
    selected: str,
    probabilities: dict[str, float],
) -> dict:
    return {
        "record_id": record_id,
        "split": "test",
        "question_id": "q1",
        "draw_idx": 0,
        "template_type": condition,
        "task_format": "multiple_choice",
        "mc_mode": "strict_mc",
        "prompt_messages": [{"type": "human", "content": condition}],
        "letters": "AB",
        "correct_letter": "A",
        "incorrect_letter": "B",
        "suggested_label": "B" if "incorrect" in condition else "",
        "committed_answer": selected,
        "response": selected,
        "response_raw": selected,
        "choice_probabilities": probabilities,
        "correctness": int(selected == "A"),
        "usable_for_metrics": True,
    }


@unittest.skipIf(torch is None, "torch is required")
class ActivationInterventionTests(unittest.TestCase):
    def setUp(self):
        self.model = _FakeLM().eval()
        self.tokenizer = _FakeTokenizer()
        self.messages = [{"type": "human", "content": "question"}]

    def test_final_hidden_state_maps_to_final_norm(self):
        self.assertIs(block_for_residual_layer(self.model, 1), self.model.model.layers[0])
        self.assertIs(block_for_residual_layer(self.model, 2), self.model.model.norm)

    def test_zero_hook_is_exact_and_positive_addition_changes_choice_margin(self):
        baseline = extract_prompt_state(
            self.model,
            self.tokenizer,
            self.messages,
            choices=["A", "B"],
            residual_layers=[1, 2],
        )
        probabilities, _ = score_with_residual_additions(
            self.model,
            self.tokenizer,
            self.messages,
            choices=["A", "B"],
            residual_layer=2,
            addition_vectors=np.asarray([[0, 0, 0, 0], [2, 0, 0, 0]], dtype=np.float32),
        )
        self.assertEqual(probabilities[0], baseline.choice_probabilities)
        self.assertGreater(probabilities[1]["A"], probabilities[0]["A"])

    def test_hook_is_removed_after_context(self):
        module = self.model.model.norm
        before = len(module._forward_hooks)
        with residual_addition_hook(
            self.model,
            residual_layer=2,
            addition_vectors=torch.zeros((1, 4)),
        ):
            self.assertEqual(len(module._forward_hooks), before + 1)
        self.assertEqual(len(module._forward_hooks), before)

    def test_multilayer_hooks_are_simultaneous_and_removed(self):
        first = self.model.model.layers[0]
        final = self.model.model.norm
        before = (len(first._forward_hooks), len(final._forward_hooks))
        additions = {
            1: torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            2: torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
        with residual_additions_hooks(
            self.model,
            additions_by_layer=additions,
        ):
            self.assertEqual(len(first._forward_hooks), before[0] + 1)
            self.assertEqual(len(final._forward_hooks), before[1] + 1)
        self.assertEqual(
            (len(first._forward_hooks), len(final._forward_hooks)),
            before,
        )

        baseline = extract_prompt_state(
            self.model,
            self.tokenizer,
            self.messages,
            choices=["A", "B"],
            residual_layers=[1, 2],
        )
        probabilities, _ = score_with_multilayer_residual_additions(
            self.model,
            self.tokenizer,
            self.messages,
            choices=["A", "B"],
            addition_vectors_by_layer={
                1: np.asarray([[1, 0, 0, 0]], dtype=np.float32),
                2: np.asarray([[1, 0, 0, 0]], dtype=np.float32),
            },
        )
        self.assertGreater(probabilities[0]["A"], baseline.choice_probabilities["A"])

    def test_multilayer_zero_hook_is_exact(self):
        baseline = extract_prompt_state(
            self.model,
            self.tokenizer,
            self.messages,
            choices=["A", "B"],
            residual_layers=[1, 2],
        )
        probabilities, _ = score_with_multilayer_residual_additions(
            self.model,
            self.tokenizer,
            self.messages,
            choices=["A", "B"],
            addition_vectors_by_layer={
                1: np.zeros((1, 4), dtype=np.float32),
                2: np.zeros((1, 4), dtype=np.float32),
            },
        )
        self.assertEqual(probabilities[0], baseline.choice_probabilities)

    def test_replacement_hook_sets_exact_target_at_the_intervention_site(self):
        observed = []
        target = torch.tensor([[3.0, -2.0, 1.0, 0.5]])
        module = self.model.model.norm
        with residual_replacement_hook(
            self.model,
            residual_layer=2,
            replacement_vectors=target,
        ):
            observer = module.register_forward_hook(
                lambda _module, _inputs, output: observed.append(output[:, -1, :].detach().clone())
            )
            try:
                self.model(torch.tensor([[0, 0]]), output_hidden_states=False)
            finally:
                observer.remove()
        torch.testing.assert_close(observed[0], target)


class PairingAndGeometryTests(unittest.TestCase):
    def test_pairing_identifies_hidden_truth_flip(self):
        conditions = (
            "neutral",
            "incorrect_suggestion",
            "incorrect_suggestion_strong",
            "suggest_correct_strong",
        )
        records = [
            _record("neutral", record_id=1, selected="A", probabilities={"A": 0.9, "B": 0.1}),
            _record(
                "incorrect_suggestion",
                record_id=2,
                selected="B",
                probabilities={"A": 0.4, "B": 0.6},
            ),
            _record(
                "incorrect_suggestion_strong",
                record_id=3,
                selected="B",
                probabilities={"A": 0.1, "B": 0.9},
            ),
            _record(
                "suggest_correct_strong",
                record_id=4,
                selected="A",
                probabilities={"A": 0.95, "B": 0.05},
            ),
        ]
        probe = pd.DataFrame(
            [{"source_record_id": 3, "probe_argmax_choice": "A", "probe_score_gap_correct_minus_selected": 2.0}]
        )
        pairs, coverage = build_intervention_pairs(
            records,
            probe_scores=probe,
            required_conditions=conditions,
        )
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["sycophantic_flip"])
        self.assertTrue(pairs[0]["hidden_truth_flip"])
        self.assertFalse(pairs[0]["probe_follows_user"])
        self.assertTrue(bool(coverage.iloc[0]["included"]))

    def test_parallel_and_orthogonal_components_recompose(self):
        vector = np.asarray([2.0, 3.0, 4.0])
        direction = np.asarray([1.0, 0.0, 0.0])
        parallel = parallel_component(vector, direction)
        orthogonal = orthogonal_component(vector, direction)
        np.testing.assert_allclose(parallel + orthogonal, vector)
        self.assertAlmostEqual(float(np.dot(orthogonal, direction)), 0.0)

    def test_direction_fit_balances_option_position_strata(self):
        neutral = np.asarray(
            [
                [[2.0, 0.0]],
                [[4.0, 0.0]],
                [[0.0, 6.0]],
                [[0.0, 8.0]],
            ]
        )
        biased = np.zeros_like(neutral)
        arrays, metadata = fit_direction_arrays(
            neutral,
            biased,
            layers=[1],
            option_position_strata=["A>B", "A>B", "B>A", "B>A"],
            seed=5,
            n_control_directions=3,
        )
        np.testing.assert_allclose(arrays["restoration_raw"][0], [1.5, 3.5])
        self.assertEqual(metadata["n_option_position_strata"], 2)
        self.assertEqual(arrays["null_unit"].shape, (1, 3, 2))
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = save_direction_artifact(
                Path(temporary_directory), arrays=arrays, metadata=metadata
            )
            self.assertEqual(artifact.control_vector("null_unit", 1, 2).shape, (2,))
            self.assertGreater(artifact.control_scalar("random_scale", 1, 1), 0.0)


class MetricAndSelectionTests(unittest.TestCase):
    def test_paired_metrics_and_normalized_recovery(self):
        baseline = {"A": 0.2, "B": 0.8}
        intervened = {"A": 0.8, "B": 0.2}
        neutral = {"A": 0.9, "B": 0.1}
        row = make_result_row(
            probabilities=intervened,
            baseline_probabilities=baseline,
            neutral_baseline_probabilities=neutral,
            correct_choice="A",
            endorsed_choice="B",
        )
        self.assertGreater(row["delta_margin"], 0.0)
        self.assertEqual(row["reverses_endorsed_to_correct"], 1.0)
        self.assertGreater(row["normalized_recovery"], 0.0)
        kl, tv = distribution_shift(intervened, baseline)
        self.assertGreater(kl, 0.0)
        self.assertAlmostEqual(tv, 0.6)
        self.assertAlmostEqual(
            normalized_recovery(
                correct_endorsed_margin(intervened, correct_choice="A", endorsed_choice="B"),
                biased_margin=correct_endorsed_margin(
                    baseline, correct_choice="A", endorsed_choice="B"
                ),
                neutral_margin=correct_endorsed_margin(
                    neutral, correct_choice="A", endorsed_choice="B"
                ),
            ),
            row["normalized_recovery"],
        )

    def test_log_score_margin_does_not_clip_extreme_probabilities(self):
        row = make_result_row(
            probabilities={"A": 0.0, "B": 1.0},
            baseline_probabilities={"A": 0.5, "B": 0.5},
            log_scores={"A": -100.0, "B": 0.0},
            baseline_log_scores={"A": 0.0, "B": 0.0},
            correct_choice="A",
            endorsed_choice="B",
        )
        self.assertEqual(row["margin_correct_minus_endorsed"], -100.0)
        self.assertEqual(row["delta_margin"], -100.0)
        self.assertEqual(row["margin_source"], "choice_log_scores")

    def test_two_stage_selection_uses_patch_window_then_selective_did(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for layer, forward, reverse in ((1, 1.0, -1.0), (2, 2.0, -0.5)):
                rows = [
                    {
                        "layer": layer,
                        "intervention": "patch_paired_full",
                        "condition": "incorrect_suggestion_strong",
                        "neutral_correct": True,
                        "delta_margin": forward,
                        "accuracy_change": 0.0,
                        "alpha": 1.0,
                        "protocol": "patch-localize",
                    },
                    {
                        "layer": layer,
                        "intervention": "patch_reverse_full",
                        "condition": "neutral",
                        "neutral_correct": True,
                        "delta_margin": reverse,
                        "accuracy_change": 0.0,
                        "alpha": 1.0,
                        "protocol": "patch-localize",
                    },
                    {
                        "layer": layer,
                        "intervention": "patch_random_matched",
                        "condition": "incorrect_suggestion_strong",
                        "neutral_correct": True,
                        "delta_margin": 0.1,
                        "accuracy_change": 0.0,
                        "alpha": 1.0,
                        "question_id": "q1",
                        "draw_idx": 0,
                        "protocol": "patch-localize",
                    },
                    {
                        "layer": layer,
                        "intervention": "patch_wrong_question",
                        "condition": "incorrect_suggestion_strong",
                        "neutral_correct": True,
                        "delta_margin": 0.2,
                        "accuracy_change": 0.0,
                        "alpha": 1.0,
                        "protocol": "patch-localize",
                    },
                ]
                for row_index, row in enumerate(rows):
                    row.setdefault("question_id", "q1")
                    row.setdefault("draw_idx", 0)
                    row.setdefault("hidden_truth_flip", False)
                hidden_rows = []
                for row in rows:
                    hidden = dict(row)
                    hidden["question_id"] = "q_hidden"
                    hidden["hidden_truth_flip"] = True
                    if hidden["intervention"] == "patch_paired_full":
                        hidden["delta_margin"] = 100.0 if layer == 1 else -100.0
                    elif hidden["intervention"] == "patch_reverse_full":
                        hidden["delta_margin"] = -100.0 if layer == 1 else 100.0
                    hidden_rows.append(hidden)
                rows.extend(hidden_rows)
                write_jsonl(
                    root
                    / "layers"
                    / f"layer_{layer:03d}"
                    / "val"
                    / "item_results_patch_localize.jsonl",
                    rows,
                )
                self._write_manifest(root, layer, "patch-localize", model_layer_count=3)
            layer_selection = select_validation_layers(output_root=root, top_k=2)
            self.assertEqual(layer_selection["candidate_layers"], [2, 1])

            for layer in (1, 2):
                dose_rows = []
                for alpha in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    treatment_effect = (0.4 if layer == 1 else 0.8) * alpha
                    for intervention, multiplier in (
                        ("steer_restoration_meandiff", 1.0),
                        ("steer_rademacher_null", 0.1),
                        ("steer_random_direction", 0.05),
                    ):
                        for condition, neutral_shift in (
                            ("incorrect_suggestion_strong", treatment_effect * multiplier),
                            ("neutral", 0.05 * alpha * multiplier),
                            ("suggest_correct_strong", 0.05 * alpha * multiplier),
                        ):
                            dose_rows.append(
                                self._dose_row(
                                    layer=layer,
                                    alpha=alpha,
                                    intervention=intervention,
                                    condition=condition,
                                    delta_margin=neutral_shift,
                                )
                            )
                    # This frozen random_all subgroup would make layer 1 appear
                    # overwhelmingly best if it leaked into validation selection.
                    if layer == 1:
                        for condition, hidden_shift in (
                            ("incorrect_suggestion_strong", 100.0 * alpha),
                            ("neutral", 0.0),
                            ("suggest_correct_strong", 0.0),
                        ):
                            dose_rows.append(
                                self._dose_row(
                                    layer=layer,
                                    alpha=alpha,
                                    intervention="steer_restoration_meandiff",
                                    condition=condition,
                                    delta_margin=hidden_shift,
                                    question_id="q_hidden",
                                    hidden_truth_flip=True,
                                )
                            )
                write_jsonl(
                    root
                    / "layers"
                    / f"layer_{layer:03d}"
                    / "val"
                    / "item_results_dose_tune.jsonl",
                    dose_rows,
                )
                self._write_manifest(root, layer, "dose-tune", model_layer_count=3)
            selection = select_validation_dose(output_root=root)
            self.assertEqual(selection["selected_layer"], 2)
            self.assertEqual(selection["selected_alpha"], 1.0)
            self.assertLess(
                selection["selected_dose_row"]["negative_alpha_mitigation_did"], 0.0
            )
            self.assertGreaterEqual(
                selection["selected_dose_row"]["dose_response_spearman"], 0.70
            )
            self.assertEqual(
                selection["primary_selection_estimand"],
                "delta_margin(strong_wrong) - delta_margin(neutral)",
            )

    def test_streaming_aggregate_writes_bootstrap_contrasts_and_probe_moderator(self):
        import json

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            selection = {
                "selected_layer": 1,
                "selected_alpha": 1.0,
                "test_alphas": [0.0, 1.0, -1.0],
                "source_run_dir": "/tmp/source",
                "directions_manifest_sha256": "manifest-hash",
                "directions_npz_sha256": "npz-hash",
                "model_name": "fake/model",
                "dataset_name": "fake_dataset",
                "test_confirmation_allowed": True,
            }
            (root / "selected_intervention.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            rows = []
            for question_id, hidden_truth, probe_user, restoration_correct_probability in (
                ("q_hidden", True, False, 0.85),
                ("q_user", False, True, 0.65),
            ):
                flags = {
                    "neutral_correct": True,
                    "sycophantic_flip": True,
                    "hidden_truth_flip": hidden_truth,
                    "probe_follows_user": probe_user,
                    "probe_other": False,
                    "sycophantic_flip_probe_user": probe_user,
                    "sycophantic_flip_probe_other": False,
                    "high_confidence_neutral_correct": True,
                    "neutral_wrong_to_correct_suggestion_correct": False,
                    "baseline_replay_matched": True,
                    "probe_score_gap_correct_minus_selected": 1.0 if hidden_truth else -1.0,
                    "neutral_p_correct_saved": 0.9,
                }
                for intervention, condition, alpha, p_correct in (
                    ("steer_restoration_meandiff", "incorrect_suggestion_strong", 1.0, restoration_correct_probability),
                    ("steer_restoration_meandiff", "neutral", 1.0, 0.82),
                    ("steer_restoration_meandiff", "suggest_correct_strong", 1.0, 0.9),
                    ("steer_rademacher_null", "incorrect_suggestion_strong", 1.0, 0.58),
                    ("steer_rademacher_null", "neutral", 1.0, 0.8),
                    ("steer_rademacher_null", "suggest_correct_strong", 1.0, 0.88),
                    ("steer_random_direction", "incorrect_suggestion_strong", 1.0, 0.56),
                    ("steer_random_direction", "neutral", 1.0, 0.8),
                    ("steer_random_direction", "suggest_correct_strong", 1.0, 0.88),
                    ("patch_paired_full", "incorrect_suggestion_strong", 1.0, 0.9),
                    ("patch_reverse_full", "neutral", 1.0, 0.2),
                    ("patch_wrong_question", "incorrect_suggestion_strong", 1.0, 0.65),
                    ("patch_random_matched", "incorrect_suggestion_strong", 1.0, 0.55),
                ):
                    baseline = {"A": 0.4, "B": 0.6} if condition == "incorrect_suggestion_strong" else {"A": 0.8, "B": 0.2}
                    probabilities = {"A": p_correct, "B": 1.0 - p_correct}
                    row = make_result_row(
                        probabilities=probabilities,
                        baseline_probabilities=baseline,
                        log_scores={"A": float(np.log(p_correct)), "B": float(np.log(1.0 - p_correct))},
                        baseline_log_scores={"A": float(np.log(baseline["A"])), "B": float(np.log(baseline["B"]))},
                        neutral_baseline_probabilities={"A": 0.8, "B": 0.2},
                        neutral_baseline_log_scores={"A": float(np.log(0.8)), "B": float(np.log(0.2))},
                        correct_choice="A",
                        endorsed_choice="B",
                        condition_suggested_choice=("A" if condition == "suggest_correct_strong" else "B"),
                        metadata={
                            **flags,
                            "model_name": "fake/model",
                            "dataset_name": "fake_dataset",
                            "split": "test",
                            "protocol": "confirm",
                            "layer": 1,
                            "condition": condition,
                            "intervention": intervention,
                            "direction_family": intervention,
                            "alpha": alpha,
                            "confirmatory_status": "confirmatory",
                            "question_id": question_id,
                            "draw_idx": 0,
                            "control_seed": 0 if "random" in intervention or "null" in intervention else None,
                        },
                    )
                    row.pop("probabilities")
                    row.pop("baseline_probabilities")
                    rows.append(row)
            item_path = root / "layers" / "layer_001" / "test" / "item_results_confirm.jsonl"
            write_jsonl(item_path, rows)
            manifest = {
                "protocol": "confirm",
                "source_run_dir": "/tmp/source",
                "directions_manifest_sha256": "manifest-hash",
                "directions_npz_sha256": "npz-hash",
                "model_name": "fake/model",
                "dataset_name": "fake_dataset",
                "conditions": [
                    "neutral",
                    "incorrect_suggestion",
                    "incorrect_suggestion_strong",
                    "suggest_correct_strong",
                ],
                "max_questions": None,
                "model_layer_count": 3,
                "layer": 1,
            }
            item_path.with_name("manifest_confirm.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            outputs = aggregate_intervention_results(
                output_root=root, n_bootstrap=20, seed=5
            )
            contrasts = pd.read_csv(outputs["contrasts"])
            moderator = pd.read_csv(outputs["moderator"])
            self.assertIn("restoration_minus_random_did", set(contrasts["contrast"]))
            self.assertIn(
                "hidden_truth_flip_minus_probe_follows_user",
                set(moderator["moderator_contrast"]),
            )

    @staticmethod
    def _write_manifest(root: Path, layer: int, protocol: str, model_layer_count: int) -> None:
        protocol_slug = protocol.replace("-", "_")
        path = root / "layers" / f"layer_{layer:03d}" / "val" / f"manifest_{protocol_slug}.json"
        path.write_text(
            __import__("json").dumps(
                {
                    "protocol": protocol,
                    "source_run_dir": "/tmp/source",
                    "directions_manifest_sha256": "manifest-hash",
                    "directions_npz_sha256": "npz-hash",
                    "model_name": "fake/model",
                    "dataset_name": "fake_dataset",
                    "conditions": [
                        "neutral",
                        "incorrect_suggestion",
                        "incorrect_suggestion_strong",
                        "suggest_correct_strong",
                    ],
                    "layer": layer,
                    "model_layer_count": model_layer_count,
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _dose_row(
        *,
        layer: int,
        alpha: float,
        intervention: str,
        condition: str,
        delta_margin: float,
        question_id: str = "q1",
        hidden_truth_flip: bool = False,
    ) -> dict:
        return {
            "model_name": "fake/model",
            "dataset_name": "fake_dataset",
            "split": "val",
            "protocol": "dose-tune",
            "layer": layer,
            "alpha": alpha,
            "question_id": question_id,
            "draw_idx": 0,
            "condition": condition,
            "intervention": intervention,
            "delta_margin": delta_margin,
            "accuracy_change": 0.0,
            "delta_p_correct": 0.0,
            "delta_p_condition_suggested": 0.0,
            "condition_suggestion_agreement_change": 0.0,
            "baseline_margin_correct_minus_endorsed": -1.0,
            "neutral_correct": True,
            "sycophantic_flip": True,
            "hidden_truth_flip": hidden_truth_flip,
            "probe_follows_user": False,
            "probe_other": False,
            "sycophantic_flip_probe_user": False,
            "sycophantic_flip_probe_other": False,
            "high_confidence_neutral_correct": True,
            "neutral_wrong_to_correct_suggestion_correct": False,
            "baseline_replay_matched": True,
            "probe_score_gap_correct_minus_selected": 1.0,
            "neutral_p_correct_saved": 0.9,
        }


if __name__ == "__main__":
    unittest.main()
