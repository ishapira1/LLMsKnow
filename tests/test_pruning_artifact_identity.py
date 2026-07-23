from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_SH = (
    REPO_ROOT
    / "jobs"
    / "sycophancy_pruning"
    / "paper_global_sharded_20260722"
    / "common.sh"
)


class PruningArtifactIdentityTests(unittest.TestCase):
    def _identity(self, prune: Path, preserve: Path, evaluation: Path) -> str:
        command = (
            'source "$COMMON_SH"; '
            'pruning_manifest_identity "$PRUNE" "$PRESERVE" "$EVALUATION"'
        )
        environment = os.environ.copy()
        environment.update(
            {
                "COMMON_SH": str(COMMON_SH),
                "ENV_PYTHON": sys.executable,
                "PRUNE": str(prune),
                "PRESERVE": str(preserve),
                "EVALUATION": str(evaluation),
            }
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_identity_contains_all_three_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prune = root / "pruning.jsonl"
            preserve = root / "preservation.jsonl"
            evaluation = root / "evaluation.jsonl"
            payloads = {
                prune: b'{"kind":"prune"}\n',
                preserve: b'{"kind":"preserve"}\n',
                evaluation: b'{"kind":"evaluation"}\n',
            }
            for path, payload in payloads.items():
                path.write_bytes(payload)

            expected = "prune_{}_preserve_{}_eval_{}".format(
                hashlib.sha256(payloads[prune]).hexdigest()[:12],
                hashlib.sha256(payloads[preserve]).hexdigest()[:12],
                hashlib.sha256(payloads[evaluation]).hexdigest()[:12],
            )
            self.assertEqual(self._identity(prune, preserve, evaluation), expected)

            evaluation.write_text('{"kind":"changed"}\n', encoding="utf-8")
            self.assertNotEqual(self._identity(prune, preserve, evaluation), expected)

    def test_grid_collector_rejects_an_identity_mismatched_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid_root = root / "prune_old_preserve_old_eval_old"
            grid_root.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "collect_pruning_grid.py"),
                    "--grid-root",
                    str(grid_root),
                    "--output-dir",
                    str(root / "selection"),
                    "--artifact-identity",
                    "prune_new_preserve_new_eval_new",
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Grid root does not match --artifact-identity", result.stderr)


if __name__ == "__main__":
    unittest.main()
