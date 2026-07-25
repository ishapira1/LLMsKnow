from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()


if __name__ == "__main__":
    from llmssycoph.interventions.controlled_cli import main

    raise SystemExit(main())
