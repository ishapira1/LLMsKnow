from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parent / "src"
    src_dir_string = str(src_dir)
    if src_dir_string not in sys.path:
        sys.path.insert(0, src_dir_string)


_bootstrap_src_path()


if __name__ == "__main__":
    from llmssycoph.interventions.cli import main

    raise SystemExit(main())
