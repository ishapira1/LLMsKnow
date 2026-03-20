from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from llmssycoph.constants import (
    GENERATION_SPEC_VERSION,
    GRADING_SPEC_VERSION,
    MC_MODE_STRICT,
    PROMPT_SPEC_VERSION,
    RESUME_COMPAT_KEYS,
)
from llmssycoph.runtime import (
    acquire_run_lock,
    assert_resume_compatible,
    make_run_dir,
    model_slug,
    preferred_run_artifact_path,
    release_run_lock,
    run_lock_path,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_pickle_atomic,
    write_run_status,
)


def make_args(**overrides):
    payload = {key: None for key in RESUME_COMPAT_KEYS}
    payload.update(
        {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "model_backend": "huggingface",
            "benchmark_source": "answer_json",
            "input_jsonl": "answer.jsonl",
            "dataset_name": "all",
            "ays_mc_datasets": ["truthful_qa_mc", "aqua_mc"],
            "sycophancy_repo": "meg-tong/sycophancy-eval",
            "mc_mode": MC_MODE_STRICT,
            "prompt_spec_version": PROMPT_SPEC_VERSION,
            "grading_spec_version": GRADING_SPEC_VERSION,
            "generation_spec_version": GENERATION_SPEC_VERSION,
            "bias_types": "incorrect_suggestion,doubt_correct,suggest_correct",
            "test_frac": 0.2,
            "split_seed": 0,
            "max_questions": 24,
            "smoke_test": True,
            "smoke_questions": 24,
            "n_draws": 4,
            "sample_batch_size": 4,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_new_tokens": 32,
            "sampling_only": False,
            "probe_construction": "auto",
            "probe_example_weighting": "model_probability",
            "probe_layer_min": 1,
            "probe_layer_max": 32,
            "probe_val_frac": 0.2,
            "probe_seed": 0,
            "probe_selection_max_samples": 2000,
            "probe_train_max_samples": None,
            "seed": 0,
        }
    )
    payload.update(overrides)
    return SimpleNamespace(**payload)


class RuntimeContractTests(unittest.TestCase):
    def test_make_run_dir_and_model_slug_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_run_dir(tmpdir, "mistralai/Mistral-7B-Instruct-v0.2", "smoke_run")
            expected = Path(tmpdir) / model_slug("mistralai/Mistral-7B-Instruct-v0.2") / "smoke_run"
            self.assertEqual(run_dir, expected)
            self.assertTrue(run_dir.is_dir())
            self.assertEqual(
                preferred_run_artifact_path(run_dir, "warnings_log"),
                run_dir / "logs" / "warnings.log",
            )
            self.assertEqual(
                preferred_run_artifact_path(run_dir, "reports_summary_csv"),
                run_dir / "reports" / "summary.csv",
            )
            self.assertEqual(
                preferred_run_artifact_path(run_dir, "warnings_summary"),
                run_dir / "logs" / "warnings_summary.json",
            )

            # Explicit run names are resume-friendly.
            same_dir = make_run_dir(tmpdir, "mistralai/Mistral-7B-Instruct-v0.2", "smoke_run")
            self.assertEqual(same_dir, run_dir)

            with self.assertRaises(ValueError):
                make_run_dir(tmpdir, "model", "bad/name")

    def test_assert_resume_compatible_detects_mismatches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_run_dir(tmpdir, "model", "resume_case")
            args = make_args()
            cfg_path = preferred_run_artifact_path(run_dir, "run_config")
            write_json_atomic(cfg_path, {key: getattr(args, key, None) for key in RESUME_COMPAT_KEYS})

            assert_resume_compatible(run_dir, args)

            mismatched_args = make_args(n_draws=8)
            with self.assertRaises(ValueError):
                assert_resume_compatible(run_dir, mismatched_args)

            mismatched_seed_args = make_args(seed=1)
            with self.assertRaises(ValueError):
                assert_resume_compatible(run_dir, mismatched_seed_args)

            mismatched_mc_mode_args = make_args(mc_mode="mc_with_rationale")
            with self.assertRaises(ValueError):
                assert_resume_compatible(run_dir, mismatched_mc_mode_args)

            mismatched_sampling_only_args = make_args(sampling_only=True)
            with self.assertRaises(ValueError):
                assert_resume_compatible(run_dir, mismatched_sampling_only_args)

    def test_assert_resume_compatible_canonicalizes_list_like_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_run_dir(tmpdir, "model", "resume_canonical_case")
            args = make_args(
                ays_mc_datasets="truthful_qa_mc, aqua_mc",
                bias_types="incorrect_suggestion, doubt_correct, suggest_correct",
            )
            cfg_path = preferred_run_artifact_path(run_dir, "run_config")
            write_json_atomic(
                cfg_path,
                {
                    **{key: getattr(args, key, None) for key in RESUME_COMPAT_KEYS},
                    "ays_mc_datasets": ["truthful_qa_mc", "aqua_mc"],
                    "bias_types": ["incorrect_suggestion", "doubt_correct", "suggest_correct"],
                },
            )

            assert_resume_compatible(run_dir, args)

    def test_assert_resume_compatible_treats_missing_sampling_only_as_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_run_dir(tmpdir, "model", "resume_sampling_only_default_case")
            args = make_args(sampling_only=False)
            cfg_path = preferred_run_artifact_path(run_dir, "run_config")
            payload = {key: getattr(args, key, None) for key in RESUME_COMPAT_KEYS if key != "sampling_only"}
            write_json_atomic(cfg_path, payload)

            assert_resume_compatible(run_dir, args)

    def test_assert_resume_compatible_infers_missing_model_backend_from_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_run_dir(tmpdir, "gpt-5.4-nano", "resume_model_backend_default_case")
            args = make_args(model="gpt-5.4-nano", model_backend="openai")
            cfg_path = preferred_run_artifact_path(run_dir, "run_config")
            payload = {key: getattr(args, key, None) for key in RESUME_COMPAT_KEYS if key != "model_backend"}
            write_json_atomic(cfg_path, payload)

            assert_resume_compatible(run_dir, args)

    def test_lock_and_status_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_run_dir(tmpdir, "model", "lock_case")
            args = make_args()
            lock_path = run_lock_path(run_dir)

            acquire_run_lock(lock_path, run_dir)
            self.assertTrue(lock_path.exists())

            with self.assertRaises(RuntimeError):
                acquire_run_lock(lock_path, run_dir)

            write_run_status(run_dir, args=args, status="failed", lock_path=lock_path, error="boom")
            status_payload = json.loads(
                preferred_run_artifact_path(run_dir, "status").read_text(encoding="utf-8")
            )
            self.assertEqual(status_payload["status"], "failed")
            self.assertEqual(status_payload["error"], "boom")
            self.assertEqual(status_payload["lock_path"], str(lock_path))

            release_run_lock(lock_path)
            self.assertFalse(lock_path.exists())

    def test_completed_status_clears_error_and_all_atomic_writes_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_run_dir(tmpdir, "model", "artifact_case")
            args = make_args()
            lock_path = run_lock_path(run_dir)

            write_run_status(run_dir, args=args, status="failed", lock_path=lock_path, error="boom")
            write_run_status(run_dir, args=args, status="completed", lock_path=lock_path, error=None)
            status_payload = json.loads(
                preferred_run_artifact_path(run_dir, "status").read_text(encoding="utf-8")
            )
            self.assertEqual(status_payload["status"], "completed")
            self.assertNotIn("error", status_payload)

            json_path = run_dir / "payload.json"
            jsonl_path = run_dir / "payload.jsonl"
            csv_path = run_dir / "payload.csv"
            pickle_path = run_dir / "payload.pkl"

            write_json_atomic(json_path, {"a": 1, "b": "two"})
            write_jsonl_atomic(jsonl_path, [{"x": 1}, {"x": 2}])
            write_csv_atomic(csv_path, pd.DataFrame([{"x": 1}, {"x": 2}]))
            write_pickle_atomic(pickle_path, {"nested": ["a", "b"]})

            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), {"a": 1, "b": "two"})
            self.assertEqual(
                [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()],
                [{"x": 1}, {"x": 2}],
            )
            self.assertEqual(pd.read_csv(csv_path).to_dict(orient="records"), [{"x": 1}, {"x": 2}])
            with open(pickle_path, "rb") as handle:
                self.assertEqual(pickle.load(handle), {"nested": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
