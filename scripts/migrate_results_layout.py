from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_bootstrap_src_path()


def main() -> int:
    from llmssycoph.results_layout_migration import main as migration_main

    return migration_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
