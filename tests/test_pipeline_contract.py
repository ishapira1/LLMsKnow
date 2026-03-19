from __future__ import annotations

import subprocess
import sys
import unittest
import ast
from pathlib import Path

from sycophancy_bias_probe.pipeline import _next_record_id


ROOT = Path(__file__).resolve().parents[1]


class PipelineContractTests(unittest.TestCase):
    def _local_import_graph(self, entry: Path) -> list[str]:
        root = ROOT.resolve()
        visited = set()
        stack = [entry.resolve()]
        local_files: list[str] = []

        module_map = {}
        for package_file in (root / "sycophancy_bias_probe").rglob("*.py"):
            rel = package_file.relative_to(root).with_suffix("")
            module_map[".".join(rel.parts)] = package_file.resolve()
        module_map["script"] = (root / "script.py").resolve()

        while stack:
            path = stack.pop()
            if path in visited or not path.exists():
                continue
            visited.add(path)
            local_files.append(path.relative_to(root).as_posix())

            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            package_name = ".".join(path.relative_to(root).with_suffix("").parts[:-1])
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = module_map.get(alias.name)
                        if target is not None:
                            stack.append(target)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        package_parts = package_name.split(".") if package_name else []
                        base_parts = package_parts[:-node.level + 1] if node.level > 1 else package_parts
                        base = (
                            ".".join([*base_parts, node.module])
                            if node.module and base_parts
                            else (node.module or ".".join(base_parts))
                        )
                    else:
                        base = node.module or ""

                    candidates = [base] if base else []
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        candidates.append(f"{base}.{alias.name}" if base else alias.name)

                    for candidate in candidates:
                        target = module_map.get(candidate)
                        if target is not None:
                            stack.append(target)

        return sorted(local_files)

    def test_next_record_id_contract(self):
        self.assertEqual(_next_record_id(), 0)
        self.assertEqual(_next_record_id([{"record_id": 2}], [{"record_id": "7"}]), 8)
        self.assertEqual(_next_record_id([{"record_id": "bad"}], [{"other": 3}]), 0)

    def test_runner_and_pipeline_import_commands(self):
        commands = [
            "import run_sycophancy_bias_probe; print(callable(run_sycophancy_bias_probe.main))",
            "from sycophancy_bias_probe.pipeline import run_pipeline; print(callable(run_pipeline))",
        ]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-c", command],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertEqual(result.stdout.strip(), "True")

    def test_main_entrypoint_has_no_legacy_runtime_dependencies(self):
        local_files = self._local_import_graph(ROOT / "run_sycophancy_bias_probe.py")
        self.assertFalse(
            any(path.startswith("src/") for path in local_files),
            msg=f"main entrypoint should not depend on src/: {local_files}",
        )
        self.assertNotIn(
            "script.py",
            local_files,
            msg=f"main entrypoint should not depend on script.py: {local_files}",
        )


if __name__ == "__main__":
    unittest.main()
