from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_activation_steering_full_gate import (
    APPROVAL_PHRASE,
    PROTOCOL_VERSION,
    REQUIRED_CONDITIONS,
    REQUIRED_REVIEW_ASSERTIONS,
    approval_template,
    sha256_file,
    validate_approval,
    validate_inspection,
    validate_tiny_compute,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dry_run": {
            "model_key": "qwen25_7b",
            "layers": [17, 18],
        },
        "models": {
            "qwen25_7b": {
                "identifier": "Qwen/Qwen2.5-7B-Instruct",
                "revision": "revision",
            }
        },
        "compute_projection": {
            "full_strict_choice_rows": 4_737_600,
            "full_fixed_probe_candidate_passes": 23_688_000,
        },
        "numeric_gates": {
            "cross_batch_max_probability_error": 0.005,
            "cross_batch_max_margin_error": 0.05,
        },
    }


def _noop_row() -> dict:
    exact = {
        "top_choice_agreement": 1.0,
        "max_abs_probability_error": 0.0,
        "exact_required": True,
        "threshold": 0.0,
    }
    return {
        "same_shape": exact,
        "mixed_batch_zero": exact,
        "cross_batch": {
            "top_choice_agreement": 1.0,
            "max_abs_probability_error": 0.004,
            "exact_required": False,
            "threshold": 0.005,
        },
        "same_shape_max_margin_error": 0.0,
        "mixed_batch_zero_max_margin_error": 0.0,
        "cross_batch_max_margin_error": 0.04,
    }


class FullSubmissionGateTests(unittest.TestCase):
    def test_submitter_reserves_a_hash_bound_non_overwriting_run_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        submitter = (
            repo_root
            / "jobs"
            / "sycophancy_bias_probe"
            / "activation_steering_controlled_sharded_20260725"
            / "submit_activation_steering_controlled_sharded_20260725.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if ! mkdir "$INTERVENTION_ROOT"; then', submitter)
        self.assertIn("full_identity_sha256", submitter)
        self.assertIn('"$CONFIG" "$MANIFEST" "$ALPACA_MANIFEST"', submitter)
        self.assertIn("submission_reservation.env", submitter)
        self.assertLess(
            submitter.index('if [[ "$DRY_RUN" == "1" ]]'),
            submitter.index('if ! mkdir "$INTERVENTION_ROOT"; then'),
        )

    def test_inspection_requires_complete_prompt_boundary_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "question_manifest_snapshot.jsonl"
            manifest_rows = [
                {
                    "dataset": "commonsense_qa",
                    "source_example_id": f"source-{index}",
                }
                for index in range(8)
            ]
            _write_jsonl(snapshot, manifest_rows)
            inspection_rows = []
            for question in manifest_rows:
                stable_key = (
                    f"{question['dataset']}::{question['source_example_id']}"
                )
                for condition in REQUIRED_CONDITIONS:
                    for layer in (17, 18):
                        inspection_rows.append(
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "stable_question_key": stable_key,
                                "condition": condition,
                                "layer": layer,
                                "prompt_token_count": 10,
                                "final_20_token_ids": [7, 8, 9],
                                "selected_activation_token_index": 9,
                                "selected_activation_token_id": 9,
                                "rendered_chat": "<chat>",
                                "prompt_messages": [
                                    {"type": "human", "content": "question"}
                                ],
                                "activation_shape": [3584],
                                "activation_norm": 42.0,
                                "item_delta_norm": 2.0,
                                "item_delta_projection_on_wn_unit": 1.0,
                                "wn_direction_norm": 0.5,
                                "choice_probabilities": {"A": 0.6, "B": 0.4},
                                "choice_log_scores": {"A": -0.2, "B": -0.6},
                                "choice_token_ids": {"A": [1], "B": [2]},
                                "representative_injected_norms": {"-4.0": 2.0},
                            }
                        )
            output = root / "preflight_examples.jsonl"
            _write_jsonl(output, inspection_rows)
            report = root / "manifest.json"
            _write_json(
                report,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "stage": "inspect_examples",
                    "question_manifest_sha256": sha256_file(snapshot),
                    "question_manifest_snapshot": snapshot.name,
                    "question_manifest_snapshot_sha256": sha256_file(snapshot),
                    "n_questions": 8,
                    "n_rows": len(inspection_rows),
                    "layers": [17, 18],
                    "output_sha256": sha256_file(output),
                },
            )
            evidence = validate_inspection(report, config=_config())
            self.assertEqual(evidence["n_rows"], 64)
            inspection_rows.pop()
            _write_jsonl(root / "truncated.jsonl", inspection_rows)
            report_payload = json.loads(report.read_text(encoding="utf-8"))
            report_payload["output_sha256"] = sha256_file(root / "truncated.jsonl")
            _write_json(root / "bad_manifest.json", report_payload)
            (root / "preflight_examples.jsonl").write_bytes(
                (root / "truncated.jsonl").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "must contain 64 rows"):
                validate_inspection(root / "bad_manifest.json", config=_config())

    def test_tiny_compute_replays_hashes_splits_and_noop_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            shared_manifest_row = {
                "dataset": "commonsense_qa",
                "source_example_id": "shared",
            }
            for split in ("train", "val", "test"):
                run_dir = root / split
                snapshot = run_dir / "question_manifest_snapshot.jsonl"
                results = run_dir / "question_results.jsonl"
                noops = run_dir / "noop_sentinels.jsonl"
                _write_jsonl(
                    snapshot,
                    [shared_manifest_row],
                )
                _write_jsonl(results, [{"split": split, "scoring_mode": "strict_choice"}])
                _write_jsonl(noops, [_noop_row()])
                manifest = run_dir / "manifest.json"
                _write_json(
                    manifest,
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "stage": "tiny_dry_run",
                        "question_manifest_sha256": sha256_file(snapshot),
                        "question_manifest_snapshot": snapshot.name,
                        "question_manifest_snapshot_sha256": sha256_file(snapshot),
                        "score_fixed_probe": True,
                        "generation_diagnostics": True,
                        "requested_max_batch_size": 8,
                        "effective_max_batch_size": 8,
                        "batch_policy": {
                            "requested_batch_size": 8,
                            "effective_batch_size": 8,
                            "forced_batch_size_one": False,
                            "failure": None,
                        },
                        "n_result_rows": 1,
                        "n_strict_choice_rows": 1,
                        "n_noop_rows": 1,
                        "question_results_sha256": sha256_file(results),
                        "noop_sentinels_sha256": sha256_file(noops),
                    },
                )
                inputs.append(
                    {"path": str(manifest), "sha256": sha256_file(manifest)}
                )
            real_gate = root / "real_model_bf16_gate.json"
            _write_json(
                real_gate,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "stage": "real_model_bf16_gate",
                    "status": "passed",
                    "question_manifest_sha256": sha256_file(
                        root / "train" / "question_manifest_snapshot.jsonl"
                    ),
                    "runtime": {
                        "device": "cuda",
                        "model_dtype": "torch.bfloat16",
                        "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                        "model_commit_hash": "revision",
                        "tokenizer_commit_hash": "revision",
                        "chat_template_sha256": "template",
                    },
                    "provenance": {
                        "git_commit": "a" * 40,
                        "dirty": False,
                    },
                },
            )
            report = root / "compute_projection.json"
            _write_json(
                report,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "stage": "tiny_compute_projection",
                    "status": "requires_researcher_review",
                    "authorizes_full_submission": False,
                    "input_manifests": inputs,
                    "full_strict_choice_rows": 4_737_600,
                    "full_fixed_probe_candidate_passes": 23_688_000,
                    "projected_gpu_hours": 100.0,
                    "any_forced_batch_size_one": False,
                    "real_model_bf16_gate": {
                        "path": str(real_gate),
                        "sha256": sha256_file(real_gate),
                    },
                    "batch_policies": [
                        {
                            "requested_batch_size": 8,
                            "effective_batch_size": 8,
                            "forced_batch_size_one": False,
                            "failure": None,
                        }
                        for _ in range(3)
                    ],
                },
            )
            evidence = validate_tiny_compute(report, config=_config())
            self.assertEqual(evidence["n_noop_rows"], 3)
            self.assertEqual(evidence["projected_gpu_hours"], 100.0)
            train_manifest_path = root / "train" / "manifest.json"
            train_manifest = json.loads(
                train_manifest_path.read_text(encoding="utf-8")
            )
            fallback_policy = {
                "requested_batch_size": 8,
                "effective_batch_size": 1,
                "forced_batch_size_one": True,
                "failure": {
                    "stable_question_key": "commonsense_qa::train",
                    "condition": "neutral",
                    "layer": 17,
                    "error": "cross-batch replay exceeded threshold",
                },
            }
            train_manifest["effective_max_batch_size"] = 1
            train_manifest["batch_policy"] = fallback_policy
            _write_json(train_manifest_path, train_manifest)
            inputs[0]["sha256"] = sha256_file(train_manifest_path)
            report_payload = json.loads(report.read_text(encoding="utf-8"))
            report_payload["input_manifests"] = inputs
            report_payload["batch_policies"][0] = fallback_policy
            report_payload["any_forced_batch_size_one"] = True
            _write_json(root / "fallback_projection.json", report_payload)
            fallback_evidence = validate_tiny_compute(
                root / "fallback_projection.json",
                config=_config(),
            )
            self.assertTrue(fallback_evidence["any_forced_batch_size_one"])

    def test_researcher_approval_binds_commit_and_all_evidence_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = {
                "config_sha256": "a" * 64,
                "full_question_manifest_sha256": "b" * 64,
                "alpaca_manifest_sha256": "c" * 64,
                "inspection_report_sha256": "d" * 64,
                "tiny_compute_report_sha256": "e" * 64,
            }
            payload = approval_template(commit="f" * 40, evidence_hashes=evidence)
            payload.update(
                {
                    "status": "approved",
                    "approval_phrase": APPROVAL_PHRASE,
                    "reviewer": "researcher@example.edu",
                    "reviewed_at": "2026-07-25T15:00:00-04:00",
                }
            )
            payload.update({field: True for field in REQUIRED_REVIEW_ASSERTIONS})
            approval = root / "approval.json"
            _write_json(approval, payload)
            result = validate_approval(
                approval,
                commit="f" * 40,
                evidence_hashes=evidence,
            )
            self.assertEqual(result["reviewer"], "researcher@example.edu")
            payload["tiny_compute_report_sha256"] = "0" * 64
            _write_json(root / "bad_approval.json", payload)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_approval(
                    root / "bad_approval.json",
                    commit="f" * 40,
                    evidence_hashes=evidence,
                )


if __name__ == "__main__":
    unittest.main()
