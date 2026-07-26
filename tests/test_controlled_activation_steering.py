from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from llmssycoph.interventions.controlled import (
    PRIMARY_ALPHA_GRID,
    assert_noop_contract,
    assert_prompt_only_messages,
    canonical_choice_map,
    canonicalize_choice_mapping,
    fit_controlled_direction_arrays,
    geometry_pair_rows,
    git_fingerprint,
    identity_framing_ratio,
    intervention_specs,
    load_controlled_direction_artifact,
    make_controlled_result_row,
    save_controlled_direction_artifact,
    sha256_file,
    validate_question_manifest,
    write_strict_jsonl,
)
from llmssycoph.interventions.controlled_cli import build_parser
from llmssycoph.interventions.controlled_runtime import (
    _validate_direction_artifact_reuse,
    aggregate_controlled_results,
)
from llmssycoph.interventions.data import build_intervention_pairs


try:
    import torch

    from llmssycoph.interventions.activations import residual_addition_hook
except ImportError:  # pragma: no cover
    torch = None
    residual_addition_hook = None


def _manifest_row(
    source_id: str,
    split: str,
    *,
    review_status: str = "approved",
) -> dict:
    return {
        "dataset": "commonsense_qa",
        "source_example_id": source_id,
        "split": split,
        "correct_choice": "A",
        "endorsed_choice": "B",
        "semantic_b_review_status": review_status,
        "semantic_b_reviewer": "unit-test-reviewer",
        "semantic_b_reviewed_at": "2026-07-25T00:00:00+00:00",
    }


def _prompt_record(condition: str, *, usable: bool) -> dict:
    suggested = "A" if condition == "suggest_correct" else "B" if "incorrect" in condition else ""
    return {
        "record_id": hash(condition) % 10000,
        "split": "train",
        "question_id": "q_1",
        "source_example_id": "stable-1",
        "dataset": "commonsense_qa",
        "draw_idx": 0,
        "template_type": condition,
        "task_format": "multiple_choice",
        "mc_mode": "strict_mc",
        "prompt_messages": [{"type": "human", "content": condition}],
        "letters": "AB",
        "correct_letter": "A",
        "incorrect_letter": "B",
        "suggested_label": suggested,
        "usable_for_metrics": usable,
        "grading_status": "invalid" if not usable else "correct",
        "committed_answer": "",
        "choice_probabilities": {},
    }


class ControlledManifestTests(unittest.TestCase):
    def test_manifest_requires_stable_disjoint_question_keys(self):
        rows = [
            _manifest_row("one", "train"),
            _manifest_row("two", "val"),
            _manifest_row("three", "test"),
        ]
        summary = validate_question_manifest(rows, require_human_approval=True)
        self.assertEqual(summary["n_questions"], 3)
        with self.assertRaises(ValueError):
            validate_question_manifest(
                rows + [_manifest_row("one", "test")],
                require_human_approval=True,
            )

    def test_manifest_human_review_is_a_hard_gate(self):
        with self.assertRaisesRegex(ValueError, "not human-approved"):
            validate_question_manifest(
                [_manifest_row("one", "train", review_status="pending")],
                require_human_approval=True,
            )
        validate_question_manifest(
            [_manifest_row("one", "train", review_status="pending")],
            require_human_approval=False,
        )
        missing_provenance = _manifest_row("one", "train")
        missing_provenance["semantic_b_reviewer"] = ""
        with self.assertRaisesRegex(ValueError, "lacks reviewer identity"):
            validate_question_manifest(
                [missing_provenance],
                require_human_approval=True,
            )

    def test_direction_pairing_does_not_use_generated_answer_usability(self):
        records = [
            _prompt_record(condition, usable=False)
            for condition in (
                "neutral",
                "incorrect_suggestion",
                "incorrect_suggestion_strong",
                "suggest_correct",
            )
        ]
        pairs, coverage = build_intervention_pairs(
            records,
            required_conditions=(
                "neutral",
                "incorrect_suggestion",
                "incorrect_suggestion_strong",
                "suggest_correct",
            ),
            require_metric_usable=False,
        )
        self.assertEqual(len(pairs), 1)
        self.assertTrue(bool(coverage.iloc[0]["included"]))
        legacy_pairs, _ = build_intervention_pairs(
            records,
            required_conditions=(
                "neutral",
                "incorrect_suggestion",
                "incorrect_suggestion_strong",
                "suggest_correct",
            ),
            require_metric_usable=True,
        )
        self.assertEqual(legacy_pairs, [])

    def test_prompt_only_contract_rejects_assistant_content(self):
        assert_prompt_only_messages(
            [{"type": "human", "content": "Question"}],
            context="unit",
        )
        with self.assertRaisesRegex(ValueError, "assistant message"):
            assert_prompt_only_messages(
                [
                    {"type": "human", "content": "Question"},
                    {"type": "assistant", "content": "A"},
                ],
                context="unit",
            )


class ControlledDirectionTests(unittest.TestCase):
    def test_numeric_option_labels_are_canonicalized_positionally(self):
        numeric_map = canonical_choice_map(["1", "2", "3", "4"])
        self.assertEqual(
            numeric_map,
            {"1": "A", "2": "B", "3": "C", "4": "D"},
        )
        self.assertEqual(
            canonical_choice_map(["A", "B", "C", "D", "E"]),
            {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E"},
        )
        self.assertEqual(
            canonicalize_choice_mapping(
                {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4},
                numeric_map,
            ),
            {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
        )
        with self.assertRaises(ValueError):
            canonical_choice_map(["A", "C"])

    def setUp(self):
        rng = np.random.default_rng(17)
        self.layers = [1, 2]
        neutral = rng.normal(size=(8, 2, 6)).astype(np.float32)
        item_noise = rng.normal(scale=0.05, size=neutral.shape).astype(np.float32)
        wrong_shift = np.asarray([1.0, -0.5, 0.25, 0.0, 0.5, -0.25], dtype=np.float32)
        correct_shift = np.asarray([0.1, 0.2, 0.0, -0.1, 0.05, 0.1], dtype=np.float32)
        strong_extra = np.asarray([0.4, -0.1, 0.2, 0.1, 0.0, -0.2], dtype=np.float32)
        wrong = neutral + wrong_shift + item_noise
        correct = neutral + correct_shift + item_noise * 0.5
        strong = wrong + strong_extra + item_noise * 0.25
        self.states = {
            "neutral": neutral,
            "incorrect_suggestion": wrong,
            "incorrect_suggestion_strong": strong,
            "suggest_correct": correct,
        }
        self.arrays, self.metadata = fit_controlled_direction_arrays(
            self.states,
            layers=self.layers,
            question_keys=[f"commonsense_qa::q{index}" for index in range(8)],
            control_seeds=[0, 1, 2],
        )

    def test_wn_is_raw_paired_mean_with_positive_pressure_sign(self):
        expected = (
            self.states["incorrect_suggestion"] - self.states["neutral"]
        ).astype(np.float64).mean(axis=0)
        np.testing.assert_allclose(self.arrays["wn_raw"], expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(
            self.metadata["positive_alpha_meaning"],
            "more_like_ordinary_wrong_pressure",
        )
        self.assertEqual(
            self.metadata["alpha_one_meaning"],
            "one raw paired mean activation shift",
        )

    def test_controls_are_deterministic_and_norm_matched(self):
        arrays_again, _ = fit_controlled_direction_arrays(
            self.states,
            layers=self.layers,
            question_keys=[f"commonsense_qa::q{index}" for index in range(8)],
            control_seeds=[0, 1, 2],
        )
        for name in (
            "isotropic_matched",
            "coordinate_sign_matched",
            "item_sign_raw",
            "item_sign_matched",
        ):
            np.testing.assert_array_equal(self.arrays[name], arrays_again[name])
        wn_norms = np.linalg.norm(self.arrays["wn_raw"], axis=1)
        for name in ("isotropic_matched", "coordinate_sign_matched", "item_sign_matched"):
            control_norms = np.linalg.norm(self.arrays[name], axis=2)
            np.testing.assert_allclose(
                control_norms,
                np.repeat(wn_norms[:, None], control_norms.shape[1], axis=1),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_artifact_refuses_nonfinite_and_preserves_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = save_controlled_direction_artifact(
                Path(temporary) / "directions",
                arrays=self.arrays,
                metadata=self.metadata,
            )
            loaded = load_controlled_direction_artifact(artifact.path)
            np.testing.assert_array_equal(loaded.raw_direction("wn", 1), self.arrays["wn_raw"][0])
            specs = intervention_specs(
                loaded,
                layer=1,
                alphas=[-1, 0, 1],
                control_seeds=[0, 1],
            )
            self.assertTrue(any(spec["direction_name"] == "coordinate_sign" for spec in specs))
            self.assertTrue(
                any(
                    spec["direction_name"] == "wn"
                    and spec["alpha"] == 1
                    and np.array_equal(
                        spec["addition_vector"],
                        loaded.raw_direction("wn", 1),
                    )
                    for spec in specs
                )
            )
        bad = dict(self.states)
        bad["neutral"] = bad["neutral"].copy()
        bad["neutral"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            fit_controlled_direction_arrays(
                bad,
                layers=self.layers,
                question_keys=[f"commonsense_qa::q{index}" for index in range(8)],
                control_seeds=[0],
            )

    def test_cache_reuse_requires_exact_runtime_and_git_identity(self):
        runtime = {
            "model_name_or_path": "fake/model",
            "model_commit_hash": "revision-1",
            "tokenizer_name_or_path": "fake/model",
            "tokenizer_commit_hash": "revision-1",
            "chat_template_sha256": "template-hash",
        }
        config = {
            "models": {
                "fake": {
                    "identifier": "fake/model",
                    "revision": "revision-1",
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            manifest_path = root / "questions.jsonl"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest_path.write_text("{}\n", encoding="utf-8")
            artifact = SimpleNamespace(
                metadata={
                    "model_name": "fake/model",
                    "configured_model_revision": "revision-1",
                    "config_sha256": sha256_file(config_path),
                    "question_manifest_sha256": sha256_file(manifest_path),
                    "intervention_site": (
                        "post_block_residual_final_rendered_prompt_token"
                    ),
                    "runtime": dict(runtime),
                    "provenance": git_fingerprint(Path.cwd()),
                }
            )
            source = SimpleNamespace(model_name="fake/model")
            _validate_direction_artifact_reuse(
                artifact,
                config=config,
                config_path=config_path,
                question_manifest_path=manifest_path,
                source=source,
                runtime=runtime,
            )
            changed_runtime = {**runtime, "chat_template_sha256": "changed"}
            with self.assertRaisesRegex(ValueError, "cache identity mismatch"):
                _validate_direction_artifact_reuse(
                    artifact,
                    config=config,
                    config_path=config_path,
                    question_manifest_path=manifest_path,
                    source=source,
                    runtime=changed_runtime,
                )


class ControlledScoringAndGeometryTests(unittest.TestCase):
    def test_noop_contract_and_wide_output(self):
        probabilities = [{"A": 0.75, "B": 0.25}, {"A": 0.5, "B": 0.5}]
        result = assert_noop_contract(
            probabilities,
            probabilities,
            exact=True,
            max_probability_error=0.0,
        )
        self.assertEqual(result["max_abs_probability_error"], 0.0)
        with self.assertRaises(AssertionError):
            assert_noop_contract(
                probabilities,
                [{"A": 0.74, "B": 0.26}, probabilities[1]],
                exact=False,
                max_probability_error=0.005,
            )
        row = make_controlled_result_row(
            metadata={"injected_norm": 2.0},
            probabilities={"A": 0.75, "B": 0.25},
            log_scores={"A": 3.0, "B": 2.0},
            baseline_probabilities={"A": 0.5, "B": 0.5},
            baseline_log_scores={"A": 2.0, "B": 2.0},
            correct_choice="A",
            endorsed_choice="B",
            median_residual_norm=4.0,
        )
        self.assertEqual(row["option_log_score_A"], 3.0)
        self.assertEqual(row["prob_B"], 0.25)
        self.assertEqual(row["injected_norm_ratio"], 0.5)
        self.assertEqual(row["error_indicator"], 0)
        self.assertEqual(row["targeted_error_indicator"], 0)
        targeted = make_controlled_result_row(
            metadata={"injected_norm": 0.0},
            probabilities={"A": 0.25, "B": 0.75},
            log_scores={"A": 2.0, "B": 3.0},
            baseline_probabilities={"A": 0.5, "B": 0.5},
            baseline_log_scores={"A": 2.0, "B": 2.0},
            correct_choice="A",
            endorsed_choice="B",
            median_residual_norm=4.0,
        )
        self.assertEqual(targeted["error_indicator"], 1)
        self.assertEqual(targeted["targeted_error_indicator"], 1)

    def test_strict_json_writer_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                write_strict_jsonl(
                    Path(temporary) / "bad.jsonl",
                    [{"value": float("inf")}],
                )

    def test_aggregate_rejects_cross_shard_alpha_zero_drift(self):
        base = {
            "stable_question_key": "commonsense_qa::one",
            "dataset": "commonsense_qa",
            "split": "val",
            "condition": "neutral",
            "model_name": "fake-model",
            "layer": 1,
            "direction_name": "wn",
            "scale_convention": "native",
            "control_seed": None,
            "alpha": 0.0,
            "treatment_type": "learned",
            "is_correct": True,
            "equals_endorsed": False,
            "p_correct": 0.8,
            "p_endorsed": 0.2,
            "delta_p_correct": 0.0,
            "delta_p_endorsed": 0.0,
            "delta_log_score_margin": 0.0,
            "log_score_margin_correct_minus_endorsed": 1.0,
            "valid_answer": True,
            "scoring_mode": "strict_choice",
            "predicted_option": "A",
            "prob_A": 0.8,
            "prob_B": 0.2,
        }
        rows = [
            base,
            {
                **base,
                "layer": 2,
                "predicted_option": "B",
                "prob_A": 0.7,
                "prob_B": 0.3,
                "log_score_margin_correct_minus_endorsed": 0.8,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "rows.jsonl"
            write_strict_jsonl(input_path, rows)
            with self.assertRaisesRegex(AssertionError, "Cross-shard"):
                aggregate_controlled_results(
                    input_paths=[input_path],
                    output_dir=Path(temporary) / "aggregate",
                    n_bootstrap=10,
                    seed=5,
                )

    def test_exploratory_aggregate_reports_cross_shard_alpha_zero_drift(self):
        base = {
            "stable_question_key": "commonsense_qa::one",
            "dataset": "commonsense_qa",
            "split": "test",
            "condition": "neutral",
            "model_name": "fake-model",
            "layer": 1,
            "direction_fit_scope": "pooled",
            "direction_name": "wn",
            "scale_convention": "native",
            "control_seed": None,
            "alpha": 0.0,
            "treatment_type": "learned",
            "is_correct": True,
            "equals_endorsed": False,
            "p_correct": 0.8,
            "p_endorsed": 0.2,
            "delta_p_correct": 0.0,
            "delta_p_endorsed": 0.0,
            "delta_log_score_margin": 0.0,
            "log_score_margin_correct_minus_endorsed": 1.0,
            "error_indicator": 0,
            "targeted_error_indicator": 0,
            "valid_answer": True,
            "scoring_mode": "strict_choice",
            "predicted_option": "A",
            "prob_A": 0.8,
            "prob_B": 0.2,
        }
        rows = [
            base,
            {
                **base,
                "layer": 2,
                "predicted_option": "B",
                "prob_A": 0.7,
                "prob_B": 0.3,
                "log_score_margin_correct_minus_endorsed": 0.8,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "rows.jsonl"
            write_strict_jsonl(input_path, rows)
            output_dir = Path(temporary) / "aggregate"
            aggregate_controlled_results(
                input_paths=[input_path],
                output_dir=output_dir,
                n_bootstrap=0,
                seed=5,
                enforce_cross_shard_replay=False,
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            replay = manifest["cross_shard_replay"]
            self.assertFalse(replay["passed"])
            self.assertFalse(replay["enforced"])
            self.assertEqual(
                replay["interpretation"],
                "exploratory_within_shard_paired_effects_only",
            )

    def test_aggregate_replays_alpha_zero_controls_before_compaction(self):
        base = {
            "stable_question_key": "commonsense_qa::one",
            "dataset": "commonsense_qa",
            "split": "val",
            "condition": "neutral",
            "model_name": "fake-model",
            "layer": 1,
            "direction_name": "isotropic",
            "scale_convention": "wn_norm_matched",
            "control_seed": 0,
            "alpha": 0.0,
            "treatment_type": "control",
            "is_correct": True,
            "equals_endorsed": False,
            "p_correct": 0.8,
            "p_endorsed": 0.2,
            "delta_p_correct": 0.0,
            "delta_p_endorsed": 0.0,
            "delta_log_score_margin": 0.0,
            "log_score_margin_correct_minus_endorsed": 1.0,
            "scoring_mode": "strict_choice",
            "predicted_option": "A",
            "prob_A": 0.8,
            "prob_B": 0.2,
        }
        rows = [
            base,
            {
                **base,
                "layer": 2,
                "predicted_option": "B",
                "prob_A": 0.7,
                "prob_B": 0.3,
                "log_score_margin_correct_minus_endorsed": 0.8,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "rows.jsonl"
            write_strict_jsonl(input_path, rows)
            with self.assertRaisesRegex(AssertionError, "Cross-shard"):
                aggregate_controlled_results(
                    input_paths=[input_path],
                    output_dir=Path(temporary) / "aggregate",
                    n_bootstrap=10,
                    seed=5,
                )

    def test_geometry_uses_derangements_and_training_mean(self):
        rng = np.random.default_rng(11)
        neutral = rng.normal(size=(7, 5))
        wrong = neutral + np.asarray([0.2, 0.0, -0.1, 0.05, 0.1])
        states = {
            "neutral": neutral,
            "incorrect_suggestion": wrong,
            "incorrect_suggestion_strong": wrong * 1.02,
            "suggest_correct": neutral + 0.01,
        }
        frame = geometry_pair_rows(
            states,
            training_mean=np.zeros(5),
            median_residual_norm=2.0,
            permutation_seeds=[0, 1, 2],
        )
        unmatched = frame[frame["group"].str.contains("different_questions")]
        self.assertTrue((unmatched["left_index"] != unmatched["right_index"]).all())
        ratio = identity_framing_ratio(frame)
        self.assertTrue(np.isfinite(ratio))
        self.assertGreaterEqual(ratio, 0.0)

    def test_cli_exposes_all_protocol_stages_and_full_alpha_grid(self):
        parser = build_parser()
        commands = parser._subparsers._group_actions[0].choices
        self.assertEqual(
            {
                "validate-source",
                "inspect-examples",
                "fit-directions",
                "audit-mean-cancellation",
                "screen-layers",
                "tiny-dry-run",
                "run-selected",
                "score-fixed-probe",
                "run-geometry",
                "run-alpaca-guardrail",
                "aggregate",
            },
            set(commands),
        )
        self.assertEqual(len(PRIMARY_ALPHA_GRID), 21)
        self.assertEqual(PRIMARY_ALPHA_GRID[0], -128.0)
        self.assertEqual(PRIMARY_ALPHA_GRID[-1], 128.0)

    def test_aggregate_selects_validation_layer_and_writes_plots(self):
        rows = []
        for layer in (1, 2):
            for question_index in range(4):
                dataset = (
                    "commonsense_qa"
                    if question_index % 2 == 0
                    else "arc_challenge"
                )
                for alpha in (-1.0, 0.0, 1.0):
                    for condition in ("neutral", "incorrect_suggestion"):
                        learned_effect = (
                            (0.2 if layer == 2 else 0.08) * alpha
                            if condition == "incorrect_suggestion"
                            else -0.005 * abs(alpha)
                        )
                        rows.append(
                            {
                                "stable_question_key": (
                                    f"{dataset}::q{question_index}"
                                ),
                                "dataset": dataset,
                                "split": "val",
                                "condition": condition,
                                "model_name": "fake-model",
                                "layer": layer,
                                "direction_name": "wn",
                                "scale_convention": "native",
                                "control_seed": None,
                                "alpha": alpha,
                                "treatment_type": "learned",
                                "is_correct": True,
                                "equals_endorsed": False,
                                "p_correct": 0.8,
                                "p_endorsed": 0.4 + learned_effect,
                                "delta_p_correct": (
                                    learned_effect if condition == "neutral" else 0.0
                                ),
                                "delta_p_endorsed": (
                                    learned_effect
                                    if condition == "incorrect_suggestion"
                                    else 0.0
                                ),
                                "delta_log_score_margin": -learned_effect,
                                "valid_answer": True,
                                "scoring_mode": "strict_choice",
                            }
                        )
                        rows.append(
                            {
                                **rows[-1],
                                "direction_name": "isotropic",
                                "scale_convention": "wn_norm_matched",
                                "control_seed": 0,
                                "treatment_type": "control",
                                "p_endorsed": 0.4 + 0.005 * alpha,
                                "delta_p_endorsed": (
                                    0.005 * alpha
                                    if condition == "incorrect_suggestion"
                                    else 0.0
                                ),
                                "delta_log_score_margin": -0.005 * alpha,
                            }
                        )
                        if alpha != 0.0:
                            rows.append(
                                {
                                    "stable_question_key": (
                                        f"{dataset}::q{question_index}"
                                    ),
                                    "dataset": dataset,
                                    "split": "val",
                                    "condition": condition,
                                    "model_name": "fake-model",
                                    "layer": layer,
                                    "direction_name": "wn",
                                    "scale_convention": "native",
                                    "control_seed": None,
                                    "alpha": alpha,
                                    "treatment_type": "learned",
                                    "scoring_mode": "free_generation",
                                    "generation_steering_mode": (
                                        "final_prompt_only"
                                    ),
                                    "valid_answer": condition == "neutral",
                                    "answer_format_failure": (
                                        condition != "neutral"
                                    ),
                                    "repetition_failure": False,
                                    "collapse_failure": False,
                                    "nonfinite_failure": False,
                                    "hit_max_new_tokens": False,
                                }
                            )
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "rows.jsonl"
            write_strict_jsonl(input_path, rows)
            output = aggregate_controlled_results(
                input_paths=[input_path],
                output_dir=Path(temporary) / "aggregate",
                n_bootstrap=20,
                seed=5,
            )
            self.assertTrue(output.exists())
            selection = json.loads(
                (output.parent / "layer_selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selection["selections"][0]["selected_layer"], 2)
            selected_dose = selection["selections"][0]["selected_dose"]
            aggregate_summary = pd.read_csv(output)
            self.assertIn(
                "targeted_error_share_among_errors",
                aggregate_summary.columns,
            )
            self.assertIn(
                "pooled_arc_csqa",
                set(aggregate_summary["dataset"]),
            )
            control_summary = aggregate_summary[
                aggregate_summary["treatment_type"].eq("control")
                & ~aggregate_summary["alpha"].eq(0.0)
            ]
            self.assertEqual(
                set(control_summary["interval_status"]),
                {"not_bootstrapped_compacted_control"},
            )
            aggregation_manifest = json.loads(
                (output.parent / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreater(
                aggregation_manifest["aggregation_memory_policy"][
                    "input_wide_rows"
                ],
                aggregation_manifest["aggregation_memory_policy"][
                    "retained_strict_rows"
                ],
            )
            self.assertEqual(selected_dose["selected"]["negative_alpha"], -1.0)
            self.assertEqual(selected_dose["selected"]["positive_alpha"], 1.0)
            self.assertTrue(selected_dose["fallback_is_descriptive_only"])
            selected_candidate = selection["selections"][0]["selected_candidate"]
            self.assertEqual(selected_candidate["neutral_invalid_rate"], 0.0)
            self.assertEqual(selected_candidate["neutral_damage_composite"], 0.0)
            self.assertEqual(selected_candidate["overall_degeneration_rate"], 0.5)
            self.assertTrue(selected_candidate["passes_invalid_rate"])
            self.assertFalse(selected_candidate["passes_degeneration_gate"])
            self.assertEqual(
                selected_dose["selected"]["neutral_invalid_rate"],
                0.0,
            )
            self.assertEqual(
                selected_dose["selected"]["overall_degeneration_rate"],
                0.5,
            )
            self.assertTrue((output.parent / "plots" / "dose_response.png").exists())
            self.assertTrue((output.parent / "plots" / "selectivity_pareto.png").exists())


@unittest.skipIf(torch is None, "torch is required")
class PerExampleHookTests(unittest.TestCase):
    class _Block(torch.nn.Module):
        def forward(self, hidden):
            return hidden

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(
                layers=torch.nn.ModuleList([PerExampleHookTests._Block(), PerExampleHookTests._Block()]),
                norm=PerExampleHookTests._Block(),
            )

    def test_per_example_prompt_boundary_indices(self):
        model = self._Model()
        hidden = torch.zeros((2, 3, 2))
        vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with residual_addition_hook(
            model,
            residual_layer=1,
            addition_vectors=vectors,
            token_index=[0, 2],
        ):
            output = model.model.layers[0](hidden)
        torch.testing.assert_close(output[0, 0], vectors[0])
        torch.testing.assert_close(output[1, 2], vectors[1])
        torch.testing.assert_close(output[0, 2], torch.zeros(2))

    def test_zero_hook_returns_bitwise_identical_tensor(self):
        model = self._Model()
        hidden = torch.randn((1, 3, 2))
        with residual_addition_hook(
            model,
            residual_layer=1,
            addition_vectors=torch.zeros((1, 2)),
            token_index=-1,
        ):
            output = model.model.layers[0](hidden)
        self.assertIs(output, hidden)
        self.assertTrue(torch.equal(output, hidden))


if __name__ == "__main__":
    unittest.main()
