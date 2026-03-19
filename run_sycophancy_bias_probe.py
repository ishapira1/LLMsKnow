from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parent / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_bootstrap_src_path()

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    from llmssycoph.cli import parse_args as _parse_args

    _parse_args()
    raise SystemExit(0)


def main() -> None:
    from llmssycoph.cli import parse_args
    from llmssycoph.pipeline import run_pipeline

    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
