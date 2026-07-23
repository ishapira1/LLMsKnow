from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prepare_weight_pruning_alpaca.py"
SPEC = importlib.util.spec_from_file_location("prepare_weight_pruning_alpaca", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareWeightPruningAlpacaTests(unittest.TestCase):
    def test_source_is_immutably_pinned(self) -> None:
        self.assertEqual(
            MODULE.SOURCE_COMMIT,
            "761dc5bfbdeeffa89b8bff5d038781a4055f796a",
        )
        self.assertIn(MODULE.SOURCE_COMMIT, MODULE.SOURCE_URL)
        self.assertRegex(MODULE.EXPECTED_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(MODULE.EXPECTED_ROWS, 52_002)
        self.assertEqual(
            MODULE.WIKITEXT_REVISION,
            "b08601e04326c79dfdd32d625aee71d232d685c3",
        )
        self.assertEqual(MODULE.WIKITEXT_EXPECTED_ROWS, 4_358)

    def test_validation_checks_checksum_rows_and_instruction(self) -> None:
        rows = [
            {"instruction": "First task", "input": "", "output": "First reply"},
            {"instruction": "Second task", "input": "context", "output": "Reply"},
        ]
        payload = json.dumps(rows).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "alpaca.json"
            path.write_bytes(payload)
            original_sha = MODULE.EXPECTED_SHA256
            original_rows = MODULE.EXPECTED_ROWS
            try:
                MODULE.EXPECTED_SHA256 = hashlib.sha256(payload).hexdigest()
                MODULE.EXPECTED_ROWS = len(rows)
                MODULE.validate(path)

                MODULE.EXPECTED_ROWS += 1
                with self.assertRaisesRegex(RuntimeError, "row-count mismatch"):
                    MODULE.validate(path)

                MODULE.EXPECTED_ROWS = len(rows)
                MODULE.EXPECTED_SHA256 = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    MODULE.validate(path)
            finally:
                MODULE.EXPECTED_SHA256 = original_sha
                MODULE.EXPECTED_ROWS = original_rows

    def test_wikitext_prefetch_uses_pinned_revision_and_cache(self) -> None:
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return [None] * MODULE.WIKITEXT_EXPECTED_ROWS

        fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "datasets"
            with patch.dict(sys.modules, {"datasets": fake_datasets}):
                rows = MODULE.prepare_wikitext(cache_dir)

        self.assertEqual(rows, MODULE.WIKITEXT_EXPECTED_ROWS)
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args, (MODULE.WIKITEXT_REPOSITORY, MODULE.WIKITEXT_CONFIG))
        self.assertEqual(kwargs["split"], MODULE.WIKITEXT_SPLIT)
        self.assertEqual(kwargs["revision"], MODULE.WIKITEXT_REVISION)
        self.assertEqual(Path(kwargs["cache_dir"]), cache_dir)


if __name__ == "__main__":
    unittest.main()
