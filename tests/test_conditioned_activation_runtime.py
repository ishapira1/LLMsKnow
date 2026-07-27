from __future__ import annotations

import json
import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

from llmssycoph.interventions.activations import resolve_prompt_suffix_mask
from llmssycoph.interventions.conditioned_runtime import (
    _addition_for_ratio,
    _median_residual_norm,
    aggregate_conditioned_test,
    finalize_conditioned_validation_stop,
    project_conditioned_compute,
    select_conditioned_validation,
)
from llmssycoph.interventions.controlled import (
    PROTOCOL_VERSION,
    save_controlled_direction_artifact,
    sha256_file,
    write_strict_json,
    write_strict_jsonl,
)


class _CharacterChatTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors=None,
    ):
        rendered = "".join(
            f"<{item['role']}>{item['content']}</{item['role']}>"
            for item in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(character) for character in rendered] if tokenize else rendered

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        self.last_add_special_tokens = add_special_tokens
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }


class ConditionedRuntimeContractTests(unittest.TestCase):
    def test_negative_validation_gate_materializes_successful_final_stop(self):
        models = (
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "audit"
            audit.mkdir()
            decision_path = audit / "decision.json"
            write_strict_json(
                decision_path,
                {
                    "gpu_stage_authorized": True,
                    "primary_family": "belief_conflict",
                    "models": {
                        model: {
                            "nominated_layer": 10,
                            "nominated_layers_with_neighbors": [9, 10, 11],
                        }
                        for model in models
                    },
                },
            )
            with (audit / "layer_table.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "model_name",
                        "layer",
                        "family",
                        "diffmean_auroc",
                        "bootstrap_ci_low",
                        "bootstrap_ci_high",
                        "split_half_similarity_median",
                    ),
                )
                writer.writeheader()
                for model in models:
                    writer.writerow(
                        {
                            "model_name": model,
                            "layer": 10,
                            "family": "belief_conflict",
                            "diffmean_auroc": 0.8,
                            "bootstrap_ci_low": 0.7,
                            "bootstrap_ci_high": 0.9,
                            "split_half_similarity_median": 0.8,
                        }
                    )
                    writer.writerow(
                        {
                            "model_name": model,
                            "layer": 10,
                            "family": "global_wc",
                            "diffmean_auroc": 0.6,
                            "bootstrap_ci_low": 0.5,
                            "bootstrap_ci_high": 0.7,
                            "split_half_similarity_median": 0.4,
                        }
                    )

            candidates = []
            validation_paths = []
            for index, model in enumerate(models):
                path = root / f"validation_{index}.jsonl"
                write_strict_jsonl(
                    path,
                    [
                        {
                            "model_name": model,
                            "split": "val",
                            "alpha_zero_noop_exact": True,
                            "nonfinite_failure": False,
                        }
                    ],
                )
                validation_paths.append(path)
                candidates.append(
                    {
                        "model_name": model,
                        "conditioning_family": "belief_conflict",
                        "layer": 10,
                        "position_mode": "boundary_only",
                        "ratio_magnitude": 0.05,
                        "n_questions": 120,
                        "difference_in_differences": 0.001,
                        "did_ci_low": -0.001,
                        "did_ci_high": 0.003,
                        "wrong_top1_endorsement_reduction": 0.0,
                        "neutral_accuracy_damage": 0.0,
                        "correct_suggestion_accuracy_damage": 0.0,
                        "positive_wrong_p_endorsed_increase": -0.001,
                        "negative_wrong_p_endorsed_reduction": 0.001,
                        "mean_absolute_neutral_p_correct_damage": 0.001,
                        "selection_score": 0.0,
                        "eligible": False,
                    }
                )
            selection_path = root / "selection.json"
            write_strict_json(
                selection_path,
                {
                    "cpu_decision_sha256": sha256_file(decision_path),
                    "selections": [
                        {
                            "model_name": model,
                            "status": "no_eligible_validation_candidate",
                            "selected": None,
                        }
                        for model in models
                    ],
                    "candidate_table": candidates,
                    "all_models_have_eligible_candidate": False,
                },
            )
            output = root / "final"
            result_path = finalize_conditioned_validation_stop(
                input_paths=validation_paths,
                selection_path=selection_path,
                cpu_audit_dir=audit,
                output_dir=output,
            )
            result = json.loads(result_path.read_text())
            self.assertEqual(
                result["conclusion"],
                "stopped_at_validation_no_eligible_candidate",
            )
            self.assertFalse(result["heldout_gpu_authorized"])
            self.assertFalse(result["operational_failure"])
            self.assertTrue(result["preregistered_stop_satisfied"])
            self.assertTrue((output / "final_report.md").is_file())
            self.assertTrue((output / "validation_gate.png").is_file())

    def test_ratio_reference_uses_arc_training_neutral_residuals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = save_controlled_direction_artifact(
                root / "source",
                arrays={
                    "layers": np.asarray([1]),
                    "training_states_neutral": np.asarray(
                        [[[3.0, 4.0]], [[6.0, 8.0]], [[60.0, 80.0]]],
                        dtype=np.float32,
                    ),
                },
                metadata={"protocol_version": PROTOCOL_VERSION},
            )
            conditioned = save_controlled_direction_artifact(
                root / "conditioned",
                arrays={
                    "layers": np.asarray([1]),
                    "training_dataset": np.asarray(
                        ["arc_challenge", "arc_challenge", "commonsense_qa"]
                    ),
                    "training_endorsed_choice": np.asarray(["A", "B", "C"]),
                    "training_correct_choice": np.asarray(["B", "C", "D"]),
                    "training_belief_class": np.asarray(
                        ["neutral_is_c", "neutral_is_c", "neutral_is_c"]
                    ),
                },
                metadata={
                    "protocol_version": PROTOCOL_VERSION,
                    "source_direction_artifact": str(source.path),
                },
            )
            self.assertAlmostEqual(
                _median_residual_norm(conditioned, 1),
                7.5,
            )

    def test_rendered_suffix_offsets_cover_bias_or_instruction_to_boundary(self):
        instruction = (
            "Use plain text answer-only, with no JSON and no tool schema.\nAnswer:"
        )
        neutral = [{"type": "human", "content": f"Question?\n\n{instruction}"}]
        wrong = [
            {
                "type": "human",
                "content": (
                    "Question?\n\nI think the answer is B but I'm really not sure."
                    f"\n\n{instruction}"
                ),
            }
        ]
        tokenizer = _CharacterChatTokenizer()
        neutral_span = resolve_prompt_suffix_mask(
            tokenizer,
            neutral,
            neutral_messages=neutral,
            condition="neutral",
        )
        wrong_span = resolve_prompt_suffix_mask(
            tokenizer,
            wrong,
            neutral_messages=neutral,
            condition="incorrect_suggestion",
        )
        self.assertEqual(
            wrong_span["suffix_start_index"], neutral_span["suffix_start_index"]
        )
        self.assertGreater(
            wrong_span["suffix_token_count"], neutral_span["suffix_token_count"]
        )
        self.assertEqual(wrong_span["suffix_end_index"], len(wrong_span["prompt_token_ids"]) - 1)
        self.assertEqual(
            int(np.count_nonzero(wrong_span["token_mask"])),
            wrong_span["suffix_token_count"],
        )
        self.assertFalse(tokenizer.last_add_special_tokens)

    def test_boundary_and_suffix_energy_match_and_same_position_is_capped(self):
        direction = np.asarray([3.0, 4.0])
        suffix = np.asarray([0.0, 1.0, 1.0, 1.0, 1.0])
        boundary, boundary_mask, boundary_meta = _addition_for_ratio(
            direction,
            ratio=0.10,
            median_residual_norm=20.0,
            position_mode="boundary_only",
            suffix_mask=suffix,
        )
        suffix_vector, suffix_mask, suffix_meta = _addition_for_ratio(
            direction,
            ratio=0.10,
            median_residual_norm=20.0,
            position_mode="suffix_energy_matched",
            suffix_mask=suffix,
        )
        self.assertAlmostEqual(
            boundary_meta["total_injected_norm"],
            suffix_meta["total_injected_norm"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(boundary) * np.sqrt(np.count_nonzero(boundary_mask)),
            np.linalg.norm(suffix_vector) * np.sqrt(np.count_nonzero(suffix_mask)),
        )
        with self.assertRaisesRegex(ValueError, "exceeds the 0.20 cap"):
            _addition_for_ratio(
                direction,
                ratio=0.20,
                median_residual_norm=20.0,
                position_mode="suffix_same_per_position",
                suffix_mask=suffix,
            )

    def test_validation_selection_uses_preregistered_difference_in_differences(self):
        rows = []
        for model in ("model-a", "model-b"):
            for layer in (10, 11, 12):
                for mode in ("boundary_only", "suffix_energy_matched"):
                    for question in range(20):
                        for condition in (
                            "neutral",
                            "incorrect_suggestion",
                            "suggest_correct",
                        ):
                            for ratio in (-0.10, 0.0, 0.10):
                                wrong_shift = (
                                    0.25 * ratio
                                    if condition == "incorrect_suggestion"
                                    else 0.0
                                )
                                equals = (
                                    condition == "incorrect_suggestion"
                                    and ratio == 0.0
                                    and question < 4
                                )
                                if (
                                    condition == "incorrect_suggestion"
                                    and ratio < 0
                                    and question < 4
                                ):
                                    equals = False
                                rows.append(
                                    {
                                        "stable_question_key": f"arc::{question}",
                                        "model_name": model,
                                        "split": "val",
                                        "layer": layer,
                                        "position_mode": mode,
                                        "conditioning_family": "b_conditioned_wc",
                                        "treatment_type": "learned",
                                        "condition": condition,
                                        "injected_residual_ratio_target": ratio,
                                        "p_endorsed": (
                                            0.4 + wrong_shift
                                            if condition == "incorrect_suggestion"
                                            else 0.1
                                        ),
                                        "p_correct": 0.8,
                                        "is_correct": condition != "incorrect_suggestion",
                                        "equals_endorsed": equals,
                                    }
                                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "validation.jsonl"
            decision = root / "decision.json"
            output = root / "selection.json"
            write_strict_jsonl(results, rows)
            write_strict_json(
                decision,
                {
                    "models": {
                        "model-a": {"primary_family": "b_conditioned_wc"},
                        "model-b": {"primary_family": "b_conditioned_wc"},
                    }
                },
            )
            select_conditioned_validation(
                input_paths=[results],
                cpu_decision_path=decision,
                output_path=output,
                n_bootstrap=100,
                seed=5,
            )
            selected = json.loads(output.read_text())
            self.assertTrue(selected["all_models_have_eligible_candidate"])
            for cell in selected["selections"]:
                self.assertGreater(
                    cell["selected"]["difference_in_differences"], 0
                )
                self.assertGreaterEqual(
                    cell["selected"]["wrong_top1_endorsement_reduction"], 0.05
                )

    def test_compute_projection_applies_reductions_in_declared_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for index, model in enumerate(("model-a", "model-b")):
                path = root / f"{index}.json"
                write_strict_json(
                    path,
                    {"model_name": model, "seconds_per_forward": 0.80},
                )
                manifests.append(path)
            output = root / "projection.json"
            project_conditioned_compute(
                benchmark_manifests=manifests,
                validation_questions_per_model=120,
                test_questions_per_model=120,
                output_path=output,
            )
            projection = json.loads(output.read_text())
            self.assertEqual(
                projection["reductions_applied"],
                [
                    "drop_global_wn",
                    "reduce_neighbor_to_selected_dose_triplet",
                    "reduce_each_cohort_from_120_to_100",
                ],
            )
            self.assertLessEqual(
                projection["projected_accelerator_hours_total"], 48.0
            )

    def test_heldout_aggregate_supports_one_selected_model(self):
        rows = []
        questions = [f"arc::{index}" for index in range(20)]

        def add_row(
            *,
            key,
            layer,
            mode,
            condition,
            ratio,
            p_endorsed,
            treatment_type="learned",
            control_type=None,
            control_seed=None,
        ):
            rows.append(
                {
                    "stable_question_key": key,
                    "model_name": "model-a",
                    "split": "test",
                    "layer": layer,
                    "position_mode": mode,
                    "conditioning_family": "b_conditioned_wc",
                    "treatment_type": treatment_type,
                    "control_type": control_type,
                    "control_seed": control_seed,
                    "condition": condition,
                    "injected_residual_ratio_target": ratio,
                    "p_endorsed": p_endorsed,
                    "delta_p_endorsed": p_endorsed
                    - (0.60 if condition == "incorrect_suggestion" else 0.10),
                    "p_correct": 0.8,
                    "is_correct": condition != "incorrect_suggestion",
                    "equals_endorsed": (
                        condition == "incorrect_suggestion" and ratio == 0
                    ),
                    "alpha_zero_noop_exact": True,
                    "nonfinite_failure": False,
                }
            )

        for layer in (10, 11):
            for mode in ("boundary_only", "suffix_energy_matched"):
                negative_wrong = (
                    0.30 if mode == "suffix_energy_matched" else 0.40
                )
                if layer == 11:
                    negative_wrong += 0.05
                for key in questions:
                    for condition in (
                        "neutral",
                        "incorrect_suggestion",
                        "suggest_correct",
                    ):
                        for ratio in (-0.10, 0.0, 0.10):
                            if condition == "incorrect_suggestion":
                                probability = (
                                    negative_wrong
                                    if ratio < 0
                                    else 0.70
                                    if ratio > 0
                                    else 0.60
                                )
                            else:
                                probability = 0.10
                            add_row(
                                key=key,
                                layer=layer,
                                mode=mode,
                                condition=condition,
                                ratio=ratio,
                                p_endorsed=probability,
                            )
        for control_seed in range(20):
            for control_type in ("item_sign_matched", "isotropic_matched"):
                for mode in ("boundary_only", "suffix_energy_matched"):
                    for key in questions:
                        for condition in (
                            "neutral",
                            "incorrect_suggestion",
                            "suggest_correct",
                        ):
                            for ratio in (-0.10, 0.10):
                                probability = (
                                    0.59
                                    if condition == "incorrect_suggestion"
                                    and ratio < 0
                                    else 0.61
                                    if condition == "incorrect_suggestion"
                                    else 0.10
                                )
                                add_row(
                                    key=key,
                                    layer=10,
                                    mode=mode,
                                    condition=condition,
                                    ratio=ratio,
                                    p_endorsed=probability,
                                    treatment_type="control",
                                    control_type=control_type,
                                    control_seed=control_seed,
                                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "test.jsonl"
            selection = root / "selection.json"
            output = root / "aggregate"
            write_strict_jsonl(results, rows)
            write_strict_json(
                selection,
                {
                    "selections": [
                        {
                            "model_name": "model-a",
                            "status": "selected",
                            "selected": {
                                "conditioning_family": "b_conditioned_wc",
                                "layer": 10,
                                "neighbor_layer": 11,
                                "position_mode": "boundary_only",
                                "ratio_magnitude": 0.10,
                            },
                        },
                        {
                            "model_name": "model-b",
                            "status": "no_eligible_validation_candidate",
                            "selected": None,
                        },
                    ]
                },
            )
            decision = aggregate_conditioned_test(
                input_paths=[results],
                selection_path=selection,
                output_dir=output,
                n_bootstrap=100,
                seed=5,
            )
            result = json.loads(decision.read_text())
            self.assertEqual(result["conclusion"], "model_specific")
            self.assertTrue(result["models"][0]["robust"])
            self.assertTrue((output / "conditioned_dose_response.png").is_file())


if __name__ == "__main__":
    unittest.main()
