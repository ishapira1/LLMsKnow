from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from llmssycoph.pruning.result_package import (
    FIGURE_NAMES,
    EvaluationRun,
    GridRun,
    PackageInputs,
    ResultPackageError,
    ResultTables,
    _resolve_manifest_output,
    _load_grid,
    _validate_mask_provenance,
    _validate_utility_pair,
    _validate_variant_contract,
    _write_all_no_feasible_package,
    render_result_figures,
    transition_table,
)
from llmssycoph.pruning.global_selection import select_global_configuration


def _paired_rows() -> pd.DataFrame:
    outcomes = (
        ("q1", "valid", "A", 0.65, 0.20),
        ("q2", "valid", "B", 0.20, 0.70),
        ("q3", "valid", "C", 0.25, 0.25),
        ("q4", "refusal", "", 0.30, 0.30),
    )
    rows = []
    for question_id, status, choice, candidate_c, candidate_b in outcomes:
        rows.append(
            {
                "dataset": "arc_challenge",
                "split": "test",
                "condition": "incorrect_suggestion_strong",
                "question_id": question_id,
                "draw_idx": 0,
                "cluster_id": f"arc_challenge::test::{question_id}",
                "correct_letter": "A",
                "suggested_letter": "B",
                "choice_letters": ["A", "B", "C"],
                "baseline_neutral_choice": "A",
                "baseline_biased_choice": "B",
                "candidate_neutral_choice": "A",
                "candidate_biased_choice": choice,
                "baseline_neutral_status_category": "valid",
                "baseline_biased_status_category": "valid",
                "candidate_neutral_status_category": "valid",
                "candidate_biased_status_category": status,
                "baseline_p_biased_c": 0.10,
                "baseline_p_biased_b": 0.80,
                "candidate_p_biased_c": candidate_c,
                "candidate_p_biased_b": candidate_b,
                "baseline_p_neutral_c": 0.80,
                "baseline_p_neutral_b": 0.10,
                "candidate_p_neutral_c": 0.79,
                "candidate_p_neutral_b": 0.11,
                "p": 1e-5,
                "q": 5e-5,
                "calibration_seed": 5,
                "actual_mask_count": 42,
            }
        )
    return pd.DataFrame(rows)


def _run() -> EvaluationRun:
    return EvaluationRun(
        run_id="model|rev|seed=5|primary|test",
        model="Qwen/Qwen2.5-7B-Instruct",
        revision="abcdef0123456789",
        calibration_seed=5,
        variant="primary",
        split="test",
        p=1e-5,
        q=5e-5,
        actual_mask_count=42,
        evaluation_dir=Path("/tmp/evaluation_test"),
        evaluation_manifest_sha256="a" * 64,
        paired=_paired_rows(),
        selection=pd.DataFrame(),
        candidate_evaluation_path=Path("/tmp/evaluation.json"),
        mask_indices_sha256="1" * 64,
        mask_metadata_sha256="2" * 64,
        mask_counts_by_module={"model.layers.0.mlp.down_proj": 42},
        mask_metadata={},
    )


class PruningResultPackageTests(unittest.TestCase):
    def test_grid_is_reconstructed_from_every_verified_source(self):
        baseline = {
            "p": 0.0,
            "q": 0.0,
            "split": "val",
            "calibration_seed": 5,
            "actual_mask_count": 0,
            "wrong_probability_uplift": 0.5,
            "biased_correct_probability": 0.10,
            "neutral_accuracy": 0.90,
            "neutral_correct_probability": 0.80,
            "correction_accuracy": 0.10,
            "agreement_accuracy": 0.90,
            "preservation_loss": 1.0,
            "wikitext_perplexity": 10.0,
            "other_wrong_invalid_rate": 0.0,
            "b_to_c_recovery_rate": 0.0,
        }
        candidates = [
            {
                **baseline,
                "p": 1e-5,
                "q": q,
                "actual_mask_count": count,
                "wrong_probability_uplift": uplift,
                "biased_correct_probability": 0.15,
                "correction_accuracy": 0.50,
                "b_to_c_recovery_rate": recovery,
            }
            for q, count, uplift, recovery in (
                (1e-6, 10, 0.30, 0.40),
                (3e-6, 20, 0.32, 0.30),
            )
        ]
        combined = pd.DataFrame([baseline, *candidates])
        selected, audit = select_global_configuration(combined)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = "prune_a_preserve_b_eval_c"
            selection_dir = root / identity
            selection_dir.mkdir()
            source_records = []
            runs = {}
            for index, candidate in enumerate(candidates):
                evaluation_dir = root / f"evaluation_{index}"
                evaluation_dir.mkdir()
                source = evaluation_dir / "selection_summary.csv"
                source.write_text("verified by mocked loader\n", encoding="utf-8")
                source_records.append(
                    {"path": str(source), "p": candidate["p"], "q": candidate["q"]}
                )
                runs[evaluation_dir.resolve()] = EvaluationRun(
                    **{
                        **_run().__dict__,
                        "run_id": f"grid-{index}",
                        "calibration_seed": 5,
                        "variant": "primary",
                        "split": "val",
                        "p": candidate["p"],
                        "q": candidate["q"],
                        "actual_mask_count": candidate["actual_mask_count"],
                        "evaluation_dir": evaluation_dir,
                        "evaluation_manifest_sha256": "e" * 64,
                        "selection": pd.DataFrame([baseline, candidate]),
                    }
                )
            summary_path = selection_dir / "validation_grid_summary.csv"
            combined.to_csv(summary_path, index=False)
            audit.to_csv(selection_dir / "selection_audit.csv", index=False)
            selection_path = selection_dir / "selected_configuration.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "artifact_identity": identity,
                        "sources": source_records,
                        "selection": selected.to_dict(),
                    }
                ),
                encoding="utf-8",
            )

            def fake_load(evaluation_dir: Path):
                return runs[Path(evaluation_dir).resolve()], {}

            with patch(
                "llmssycoph.pruning.result_package.load_evaluation_run",
                side_effect=fake_load,
            ):
                grid, _hashes = _load_grid(selection_path)
                self.assertEqual(grid.selection_payload["selection"]["status"], "selected")
                tampered = combined.copy()
                tampered.loc[tampered["q"].eq(3e-6), "neutral_accuracy"] = 0.1
                tampered.to_csv(summary_path, index=False)
                with self.assertRaisesRegex(ResultPackageError, "reconstructed validation grid"):
                    _load_grid(selection_path)

    def test_transition_table_is_exhaustive_on_strict_flips(self):
        table = transition_table(_run(), n_bootstrap=0)
        observed = dict(zip(table["transition"], table["rate"]))
        self.assertEqual(
            observed,
            {
                "b → c": 0.25,
                "b → b": 0.25,
                "b → other wrong": 0.25,
                "b → invalid/refusal": 0.25,
            },
        )
        self.assertEqual(set(table["n_questions"]), {4})

    def test_missing_rephrase_fails_instead_of_fabricating_a_cell(self):
        with self.assertRaisesRegex(ResultPackageError, "No rows for family"):
            transition_table(
                _run(),
                family="incorrect_suggestion_rephrase_1",
                n_bootstrap=0,
            )

    def test_offline_output_tamper_fails_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluation_dir = Path(tmp)
            output = evaluation_dir / "paired_items.csv"
            output.write_text("a\n1\n", encoding="utf-8")
            manifest = {
                "outputs": {"paired_items": str(output)},
                "output_sha256": {"paired_items": "0" * 64},
            }
            with self.assertRaisesRegex(ResultPackageError, "SHA-256 mismatch"):
                _resolve_manifest_output(evaluation_dir, manifest, "paired_items")

    def test_mask_files_and_score_identity_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            indices = root / "indices.pt"
            metadata_path = root / "metadata.json"
            indices.write_bytes(b"synthetic sparse-mask fixture")
            metadata = {
                "p": 1e-5,
                "q": 5e-5,
                "surviving_count": 3,
                "counts_by_module": {"model.layers.0.mlp.down_proj": 3},
                "score_identity": {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "revision": "abcdef0123456789",
                    "seed": 5,
                },
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            mask = {
                "kind": "harm_indices",
                "alpha": 0.0,
                "actual_mask_count": 3,
                "indices_path": str(indices),
                "indices_sha256": hashlib.sha256(indices.read_bytes()).hexdigest(),
                "metadata_path": str(metadata_path),
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            }
            verified = _validate_mask_provenance(
                mask,
                model="Qwen/Qwen2.5-7B-Instruct",
                revision="abcdef0123456789",
                p=1e-5,
                q=5e-5,
                seed=5,
                actual_mask_count=3,
                context="fixture",
            )
            self.assertEqual(set(verified), {str(indices.resolve()), str(metadata_path.resolve())})
            metadata_path.write_text(json.dumps({**metadata, "surviving_count": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ResultPackageError, "SHA-256 mismatch"):
                _validate_mask_provenance(
                    mask,
                    model="Qwen/Qwen2.5-7B-Instruct",
                    revision="abcdef0123456789",
                    p=1e-5,
                    q=5e-5,
                    seed=5,
                    actual_mask_count=3,
                    context="fixture",
                )

    def test_control_variant_semantics_cannot_be_supplied_by_a_primary_mask(self):
        metadata = {
            "neg_prune": True,
            "freeze_first_top_q": False,
            "control": "none",
            "random_magnitude_match": None,
            "score_identity": {
                "score_format": "raw",
                "loss_mode": "completion_nll",
                "attribution_variant": "paper",
                "no_abs": True,
                "abs_prune": False,
                "abs_preserve": True,
            },
        }
        _validate_variant_contract(
            "primary", calibration_seed=5, metadata=metadata, context="fixture"
        )
        with self.assertRaisesRegex(ResultPackageError, "semantics do not match"):
            _validate_variant_contract(
                "opposite_sign",
                calibration_seed=5,
                metadata=metadata,
                context="fixture",
            )

    def test_utility_model_identity_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = root / "base.json"
            candidate_path = root / "candidate.json"
            common = {
                "revision": "abcdef0123456789",
                "p": 0.0,
                "q": 0.0,
                "seed": 5,
                "alpaca": {
                    "mean_score": 4.0,
                    "data_sha256": "d" * 64,
                    "evaluation_seed": 5,
                    "requested_nsamples": 10,
                    "judge": {"model": "judge", "temperature": 0},
                },
                "zero_shot": {
                    "mean_accuracy": 0.5,
                    "tasks": {"arc": {"accuracy": 0.5}},
                },
            }
            base_path.write_text(
                json.dumps({**common, "model": "Qwen/Qwen2.5-7B-Instruct"}),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    {
                        **common,
                        "model": "tampered/model",
                        "p": 1e-5,
                        "q": 5e-5,
                    }
                ),
                encoding="utf-8",
            )
            run = EvaluationRun(
                **{
                    **_run().__dict__,
                    "candidate_evaluation_path": candidate_path,
                }
            )
            with self.assertRaisesRegex(ResultPackageError, "model identity mismatch"):
                _validate_utility_pair(run, base_path)

    def test_all_no_feasible_writes_base_only_package_without_stale_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "selection.json"
            summary_path = root / "validation_grid_summary.csv"
            source.write_text("{}", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "p": 0.0,
                        "q": 0.0,
                        "split": "val",
                        "calibration_seed": 5,
                        "actual_mask_count": 0,
                        "wrong_probability_uplift": 0.5,
                        "neutral_accuracy": 0.8,
                    },
                    {
                        "p": 1e-5,
                        "q": 5e-5,
                        "split": "val",
                        "calibration_seed": 5,
                        "actual_mask_count": 100,
                        "wrong_probability_uplift": 0.4,
                        "neutral_accuracy": 0.75,
                    },
                ]
            ).to_csv(summary_path, index=False)
            grids = []
            outcomes = {}
            for model in (
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
            ):
                revision = "abcdef0123456789"
                outcome = {
                    "status": "no_feasible_mask",
                    "selected_p": None,
                    "selected_q": None,
                    "actual_mask_count": 0,
                    "b_to_c_recovery_rate": 0.0,
                    "reason": "No validation configuration passed.",
                }
                outcomes[(model, revision)] = outcome
                grids.append(
                    GridRun(
                        model=model,
                        revision=revision,
                        artifact_identity="identity",
                        summary_path=summary_path,
                        selection_path=source,
                        summary=pd.read_csv(summary_path),
                        selection_payload={"selection": outcome},
                    )
                )
            inputs = PackageInputs(
                experiment_root=root,
                primary_test_runs=(),
                control_validation_runs=(),
                replication_validation_runs=(),
                grids=tuple(grids),
                selection_outcomes=outcomes,
                artifact_hashes={str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
            )
            output = root / "report"
            result = _write_all_no_feasible_package(inputs, output)
            self.assertEqual(result["status"], "no_feasible_mask")
            self.assertTrue(
                (output / "figures" / f"{FIGURE_NAMES[4]}.png").is_file()
            )
            self.assertFalse(
                (output / "figures" / f"{FIGURE_NAMES[0]}.png").exists()
            )
            manifest = json.loads((output / "package_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "no_feasible_mask")
            for relative, digest in manifest["outputs"].items():
                self.assertEqual(
                    hashlib.sha256((output / relative).read_bytes()).hexdigest(),
                    digest,
                )

    def test_all_six_figures_emit_png_and_pdf(self):
        model_labels = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
        transition_rows = []
        for model in model_labels:
            for phase, rates in (
                ("Before pruning", [0.0, 1.0, 0.0, 0.0]),
                ("After pruning", [0.6, 0.25, 0.10, 0.05]),
            ):
                for transition, rate in zip(
                    ["b → c", "b → b", "b → other wrong", "b → invalid/refusal"], rates
                ):
                    transition_rows.append(
                        {
                            "model_label": model,
                            "transition": transition,
                            "phase": phase,
                            "rate": rate,
                        }
                    )
        truth_rows = []
        for model in model_labels:
            for metric, delta in (
                ("P(c)", 0.20),
                ("P(b)", -0.25),
                ("Other-choice probability", 0.04),
                ("Invalid/refusal/malformed rate", 0.01),
            ):
                truth_rows.append(
                    {
                        "model_label": model,
                        "metric": metric,
                        "delta": delta,
                        "ci_low": delta - 0.02,
                        "ci_high": delta + 0.02,
                    }
                )
        preservation_rows = []
        for model in model_labels:
            for seed in (5, 17, 29):
                for metric in (
                    "Neutral accuracy",
                    "Corrective resistance",
                    "Correct-suggestion agreement",
                    "Zero-shot utility accuracy",
                ):
                    preservation_rows.append(
                        {
                            "model_label": model,
                            "calibration_seed": seed,
                            "mask_label": f"{model} · seed {seed}",
                            "metric": metric,
                            "panel": "Behavior and utility accuracy",
                            "delta": -0.5 + 0.01 * (seed == 17) - 0.01 * (seed == 29),
                            "ci_low": -1.0,
                            "ci_high": 0.0,
                        }
                    )
                for metric in (
                    "Preservation loss",
                    "WikiText perplexity",
                    "Alpaca benign-instruction score loss",
                ):
                    preservation_rows.append(
                        {
                            "model_label": model,
                            "calibration_seed": seed,
                            "mask_label": f"{model} · seed {seed}",
                            "metric": metric,
                            "panel": "Relative degradation",
                            "delta": 1.0,
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                        }
                    )
        controls = []
        variants = (
            "Targeted pruning",
            "Correction-target",
            "Opposite sign",
            "Second slice",
            "Magnitude-matched random",
        )
        for model in model_labels:
            for index, variant in enumerate(variants):
                controls.append(
                    {
                        "model_label": model,
                        "variant_label": variant,
                        "rate": 0.6 - 0.1 * index,
                        "ci_low": 0.58 - 0.1 * index,
                        "ci_high": 0.62 - 0.1 * index,
                    }
                )
        tradeoff = pd.DataFrame(
            [
                {
                    "model_label": model,
                    "sycophancy_uplift_reduction_percent": 20 + 10 * index,
                    "neutral_accuracy_loss_pp": index,
                    "actual_mask_count": 100 * (index + 1),
                    "selected": index == 1,
                }
                for model in model_labels
                for index in range(3)
            ]
        )
        conditions = (
            "Strong suggestion",
            "Weak suggestion",
            "Paraphrase 1",
            "Paraphrase 2",
            "ARC-Challenge",
            "CommonsenseQA",
        )
        generalization = pd.DataFrame(
            [
                {
                    "mask_label": f"{model} · seed {seed}",
                    "condition_label": condition,
                    "rate": 0.5 + 0.01 * condition_index,
                }
                for model in model_labels
                for seed in (5, 17, 29)
                for condition_index, condition in enumerate(conditions)
            ]
        )
        tables = ResultTables(
            transitions=pd.DataFrame(transition_rows),
            truth_restoration=pd.DataFrame(truth_rows),
            preservation=pd.DataFrame(preservation_rows),
            controls=pd.DataFrame(controls),
            tradeoff=tradeoff,
            generalization=generalization,
        )
        with tempfile.TemporaryDirectory() as tmp:
            render_result_figures(tables, Path(tmp))
            for name in FIGURE_NAMES:
                for suffix in ("png", "pdf"):
                    artifact = Path(tmp) / "figures" / f"{name}.{suffix}"
                    self.assertTrue(artifact.is_file(), artifact)
                    self.assertGreater(artifact.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
