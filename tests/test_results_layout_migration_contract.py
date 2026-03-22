from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from llmssycoph.results_layout_migration import (
    build_migration_manifest,
    discover_unmanaged_legacy_run_dirs,
    discover_run_roots,
    execute_manifest,
    infer_dataset_dir,
    verify_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["dataset"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_notebook(path: Path, old_path: str) -> None:
    payload = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": [f"{old_path}\n"]}],
                "source": [f'RUN_DIR = Path("{old_path}")\n'],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _write_json(path, payload)


def _legacy_output_run_dir(results_root: Path, model_dir: str, run_name: str) -> str:
    return f"output/sycophancy_bias_probe/{model_dir}/{run_name}"


def _create_run(
    results_root: Path,
    model_dir: str,
    run_name: str,
    *,
    model_name: str,
    dataset_name: str | None = None,
    ays_mc_datasets: list[str] | None = None,
    sampled_dataset: str | None = None,
    internal: bool = False,
    include_notebook: bool = False,
    workspace_root: Path | None = None,
) -> Path:
    run_root = results_root / model_dir / run_name
    config_root = run_root / "internal" if internal else run_root
    legacy_run_dir = _legacy_output_run_dir(results_root, model_dir, run_name)
    run_config = {
        "model": model_name,
        "run_name": run_name,
        "out_dir": "output/sycophancy_bias_probe",
        "run_dir": legacy_run_dir,
    }
    if dataset_name is not None:
        run_config["dataset_name"] = dataset_name
    if ays_mc_datasets is not None:
        run_config["ays_mc_datasets"] = ays_mc_datasets
    _write_json(config_root / "run_config.json", run_config)
    _write_json(
        config_root / "status.json",
        {
            "status": "completed",
            "run_name": run_name,
            "run_dir": legacy_run_dir,
        },
    )
    _write_json(
        run_root / "run_summary.json",
        {
            "model_name": model_name,
            **({"dataset_name": dataset_name} if dataset_name is not None else {}),
        },
    )
    _write_json(
        run_root / "sampling_manifest.json",
        {
            "sampling_hash": "abc123",
            "source_cache_run_dir": legacy_run_dir,
        },
    )
    if sampled_dataset is not None:
        _write_csv(run_root / "sampling" / "sampled_responses.csv", [{"dataset": sampled_dataset}])
    if include_notebook:
        workspace_root = workspace_root or results_root.parent.parent
        old_abs = workspace_root / legacy_run_dir
        _write_json(
            run_root / "analysis" / "analysis_notebook_status.json",
            {
                "status": "completed",
                "run_dir": str(old_abs),
                "notebook_path": str(old_abs / "analysis" / "analysis_full_mc_report.ipynb"),
            },
        )
        _write_json(
            run_root / "probes" / "chosen_probe" / "probe_no_bias" / "manifest.json",
            {
                "artifact_dir": f"{legacy_run_dir}/probes/chosen_probe/probe_no_bias",
                "model_path": f"{legacy_run_dir}/probes/chosen_probe/probe_no_bias/model.pkl",
            },
        )
        _write_notebook(run_root / "analysis" / "analysis_full_mc_report.ipynb", str(old_abs))
    return run_root


class ResultsLayoutMigrationContractTests(unittest.TestCase):
    def test_discover_run_roots_handles_direct_internal_and_nested_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            results_root = workspace_root / "results" / "sycophancy_bias_probe"
            direct_run = _create_run(
                results_root,
                "model_alpha",
                "run_direct",
                model_name="alpha/model",
                dataset_name="aqua_mc",
            )
            internal_run = _create_run(
                results_root,
                "model_beta",
                "run_internal",
                model_name="beta/model",
                dataset_name="commonsense_qa",
                internal=True,
            )
            nested_run = _create_run(
                results_root / "model_gamma" / "container",
                "ignored",
                "run_nested",
                model_name="gamma/model",
                dataset_name="arc_challenge",
            )

            roots = discover_run_roots(results_root)
            self.assertEqual(
                {root.resolve() for root in roots},
                {
                    direct_run.resolve(),
                    internal_run.resolve(),
                    nested_run.resolve(),
                },
            )

    def test_infer_dataset_dir_uses_metadata_then_sampled_data_then_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            results_root = workspace_root / "results" / "sycophancy_bias_probe"
            explicit_run = _create_run(
                results_root,
                "model_alpha",
                "run_explicit",
                model_name="alpha/model",
                dataset_name="commonsense_qa",
            )
            sampled_run = _create_run(
                results_root,
                "model_beta",
                "run_sampled",
                model_name="beta/model",
                sampled_dataset="truthful_qa",
            )
            fallback_run = _create_run(
                results_root,
                "model_gamma",
                "run_fallback",
                model_name="gamma/model",
            )

            self.assertEqual(infer_dataset_dir(explicit_run), ("commonsense_qa", "run_config.dataset_name"))
            self.assertEqual(infer_dataset_dir(sampled_run), ("truthful_qa", "sampled_responses.dataset"))
            self.assertEqual(infer_dataset_dir(fallback_run), ("legacy_unknown", "fallback"))

    def test_build_migration_manifest_marks_destination_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            results_root = workspace_root / "results" / "sycophancy_bias_probe"
            source_run = _create_run(
                results_root,
                "model_alpha",
                "run_collision",
                model_name="alpha/model",
                dataset_name="aqua_mc",
            )
            collision_target = results_root / "alpha_model" / "aqua_mc" / "run_collision"
            collision_target.mkdir(parents=True, exist_ok=True)

            manifest = build_migration_manifest(results_root=results_root, workspace_root=workspace_root)
            entry = next(item for item in manifest["entries"] if item["run_name"] == source_run.name)
            self.assertEqual(entry["collision_status"], "collision")
            self.assertEqual(manifest["collision_count"], 1)

    def test_discover_unmanaged_legacy_run_dirs_reports_status_only_run_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            results_root = workspace_root / "results" / "sycophancy_bias_probe"
            unmanaged_run = results_root / "alpha_model" / "run_without_config"
            _write_json(
                unmanaged_run / "status.json",
                {
                    "status": "running",
                    "updated_at_utc": "2026-03-22T01:08:06Z",
                },
            )
            (unmanaged_run / ".run.lock").write_text("", encoding="utf-8")

            unmanaged = discover_unmanaged_legacy_run_dirs(results_root)
            self.assertEqual(len(unmanaged), 1)
            self.assertEqual(Path(unmanaged[0]["path"]).resolve(), unmanaged_run.resolve())
            self.assertEqual(unmanaged[0]["status"], "running")

            manifest = build_migration_manifest(results_root=results_root, workspace_root=workspace_root)
            self.assertEqual(manifest["unmanaged_legacy_dir_count"], 1)

    def test_execute_manifest_moves_runs_and_rewrites_structured_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            results_root = workspace_root / "results" / "sycophancy_bias_probe"
            source_run = _create_run(
                results_root,
                "mistralai_Mistral_7B_Instruct_v0_2",
                "run_to_move",
                model_name="mistralai/Mistral-7B-Instruct-v0.2",
                dataset_name="aqua_mc",
                include_notebook=True,
                workspace_root=workspace_root,
            )

            manifest = build_migration_manifest(results_root=results_root, workspace_root=workspace_root)
            execute_manifest(manifest, workspace_root=workspace_root)
            issues = verify_manifest(manifest, workspace_root=workspace_root)
            self.assertEqual(issues, [])

            destination_run = results_root / "mistralai_Mistral_7B_Instruct_v0_2" / "aqua_mc" / "run_to_move"
            self.assertFalse(source_run.exists())
            self.assertTrue(destination_run.exists())

            run_config = json.loads((destination_run / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                run_config["run_dir"],
                "results/sycophancy_bias_probe/mistralai_Mistral_7B_Instruct_v0_2/aqua_mc/run_to_move",
            )
            self.assertEqual(run_config["out_dir"], "results/sycophancy_bias_probe")

            status = json.loads((destination_run / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(
                status["run_dir"],
                "results/sycophancy_bias_probe/mistralai_Mistral_7B_Instruct_v0_2/aqua_mc/run_to_move",
            )

            sampling_manifest = json.loads((destination_run / "sampling_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sampling_manifest["source_cache_run_dir"],
                "results/sycophancy_bias_probe/mistralai_Mistral_7B_Instruct_v0_2/aqua_mc/run_to_move",
            )

            notebook_status = json.loads(
                (destination_run / "analysis" / "analysis_notebook_status.json").read_text(encoding="utf-8")
            )
            self.assertTrue(notebook_status["run_dir"].endswith("/results/sycophancy_bias_probe/mistralai_Mistral_7B_Instruct_v0_2/aqua_mc/run_to_move"))

            notebook_text = (destination_run / "analysis" / "analysis_full_mc_report.ipynb").read_text(encoding="utf-8")
            self.assertIn("/results/sycophancy_bias_probe/mistralai_Mistral_7B_Instruct_v0_2/aqua_mc/run_to_move", notebook_text)
            self.assertNotIn("/output/sycophancy_bias_probe/mistralai_Mistral_7B_Instruct_v0_2/run_to_move", notebook_text)

    def test_execute_manifest_rewrites_stale_aux_paths_for_already_canonical_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            results_root = workspace_root / "results" / "sycophancy_bias_probe"
            run_root = results_root / "alpha_model" / "aqua_mc" / "run_canonical"
            canonical_run_dir = "results/sycophancy_bias_probe/alpha_model/aqua_mc/run_canonical"
            legacy_results_run_dir = "results/sycophancy_bias_probe/alpha_model/run_canonical"
            legacy_output_run_dir = "output/sycophancy_bias_probe/alpha_model/run_canonical"

            _write_json(
                run_root / "run_config.json",
                {
                    "model": "alpha/model",
                    "run_name": "run_canonical",
                    "dataset_name": "aqua_mc",
                    "out_dir": "results/sycophancy_bias_probe",
                    "run_dir": canonical_run_dir,
                },
            )
            _write_json(
                run_root / "status.json",
                {
                    "status": "completed",
                    "run_name": "run_canonical",
                    "run_dir": canonical_run_dir,
                },
            )
            _write_json(
                run_root / "run_summary.json",
                {
                    "model_name": "alpha/model",
                    "dataset_name": "aqua_mc",
                },
            )
            _write_json(
                run_root / "sampling_manifest.json",
                {
                    "source_cache_run_dir": legacy_output_run_dir,
                },
            )
            _write_notebook(run_root / "analysis" / "analysis_full_mc_report.ipynb", legacy_results_run_dir)

            manifest = build_migration_manifest(results_root=results_root, workspace_root=workspace_root)
            self.assertEqual(manifest["collision_count"], 0)
            execute_manifest(manifest, workspace_root=workspace_root)
            self.assertEqual(verify_manifest(manifest, workspace_root=workspace_root), [])

            sampling_manifest = json.loads((run_root / "sampling_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(sampling_manifest["source_cache_run_dir"], canonical_run_dir)

            notebook_text = (run_root / "analysis" / "analysis_full_mc_report.ipynb").read_text(encoding="utf-8")
            self.assertIn(canonical_run_dir, notebook_text)
            self.assertNotIn(legacy_results_run_dir, notebook_text)


if __name__ == "__main__":
    unittest.main()
