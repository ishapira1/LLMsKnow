from pathlib import Path
import re
import unittest


class WeightPruningSingleRepoContractTests(unittest.TestCase):
    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def bundle(self) -> Path:
        return (
            self.root
            / "jobs"
            / "sycophancy_pruning"
            / "paper_global_sharded_20260722"
        )

    def test_active_bundle_has_no_external_pruning_checkout_dependency(self):
        paths = [
            self.bundle,
            self.root / "scripts" / "snapshot_pruning_tokenization.py",
            self.root / "scripts" / "resolve_paper_pruning_artifact.py",
        ]
        forbidden = ("HARM_REPO_DIR", "--harm-repo", "harm_pruning_WIP")
        for path in paths:
            files = sorted(path.rglob("*")) if path.is_dir() else [path]
            for file_path in files:
                if not file_path.is_file():
                    continue
                text = file_path.read_text(encoding="utf-8")
                for value in forbidden:
                    self.assertNotIn(value, text, f"{value!r} remains in {file_path}")

    def test_required_in_repo_runtime_files_exist(self):
        runtime = self.root / "tools" / "weight_pruning"
        required = {
            "alpaca_data.py",
            "cohere_support.py",
            "data_utils.py",
            "eval_utils.py",
            "paper_pruning.py",
            "prune.py",
            "prune_utils.py",
        }
        self.assertEqual(
            required,
            {path.name for path in runtime.glob("*.py")},
        )

    def test_every_batch_job_is_harvard_guarded_and_emails_status(self):
        for path in sorted(self.bundle.glob("*.sbatch")):
            text = path.read_text(encoding="utf-8")
            self.assertRegex(
                text,
                r"(?m)^#SBATCH --job-name=weight_pruning_[A-Za-z0-9_]+$",
                path.name,
            )
            self.assertIn("#SBATCH --mail-type=END,FAIL", text, path.name)
            self.assertIn(
                "#SBATCH --mail-user=itaishapira@g.harvard.edu", text, path.name
            )
            self.assertIn("#SBATCH --nodes=1", text, path.name)
            self.assertIn("#SBATCH --ntasks=1", text, path.name)
            self.assertIn("#SBATCH --open-mode=append", text, path.name)
            self.assertNotIn("${SLURM_ARRAY_TASK_ID:-", text, path.name)
            if "#SBATCH --partition=gpu,seas_gpu,gpu_h200" in text:
                self.assertIn("#SBATCH --gres=gpu:1", text, path.name)
                self.assertIn("require_gpu_allocation", text, path.name)
            elif "#SBATCH --array=" in text:
                self.assertIn("require_slurm_array_task", text, path.name)
            else:
                self.assertIn("require_slurm_job", text, path.name)

    def test_submitter_uses_external_logs_and_setup_dependencies(self):
        submitter = (
            self.bundle / "submit_paper_global_sharded_20260722.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("require_harvard_scheduler", submitter)
        self.assertIn("require_pushed_main_checkout", submitter)
        self.assertIn('refs/remotes/origin/main', submitter)
        self.assertIn('git ls-remote --exit-code origin refs/heads/main', submitter)
        self.assertIn('--untracked-files=all -- "${protected_paths[@]}"', submitter)
        self.assertIn('LOG_ROOT="$WEIGHT_PRUNING_LOG_ROOT"', submitter)
        self.assertIn('setup_dependency="$prepare_id:$sampling_id"', submitter)
        self.assertIn('submit_manifests "$setup_dependency"', submitter)
        self.assertIn('submit_tokens "$manifest_id"', submitter)
        self.assertIn("SBATCH_SUBMIT_DELAY_SECONDS", submitter)
        self.assertIn("latest_setup.env", submitter)
        self.assertNotIn("jobs/sycophancy_pruning/logs", submitter.replace(
            'WEIGHT_PRUNING_LOG_ROOT="$ROOT_DIR/jobs/sycophancy_pruning/logs/$BUNDLE_NAME"',
            "",
        ))
        for match in re.finditer(r"--job-name=(?:\"([^\"]+)\"|([^\s]+))", submitter):
            job_name = match.group(1) or match.group(2)
            self.assertTrue(job_name.startswith("weight_pruning_"), job_name)

    def test_common_uses_existing_large_storage_configuration(self):
        common = (self.bundle / "common.sh").read_text(encoding="utf-8")
        self.assertIn('source .env', common)
        self.assertIn('STORAGE_ROOT="${SYCOPHANCY_STORAGE_ROOT:-', common)
        self.assertIn('WEIGHT_PRUNING_MIN_FREE_GB:-1600', common)
        self.assertIn("EXPECTED_SLURM_CLUSTER", common)
        self.assertNotIn("SYCOPHANCY_STORAGE_ROOT_OVERRIDE", common)


if __name__ == "__main__":
    unittest.main()
