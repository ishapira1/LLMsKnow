from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_bootstrap_src_path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a spec-driven analysis notebook for a completed multiple-choice run.",
    )
    parser.add_argument("--run_dir", required=True, help="Absolute or repo-relative path to the run directory.")
    parser.add_argument(
        "--spec",
        default="full_mc_report",
        help="Notebook spec name. The default is the full multiple-choice analysis report.",
    )
    parser.add_argument(
        "--notebook_name",
        default=None,
        help="Optional notebook filename. Defaults to analysis_<spec>.ipynb",
    )
    parser.add_argument(
        "--raise_on_error",
        action="store_true",
        help="Raise the underlying exception instead of writing a failure status file and returning cleanly.",
    )
    return parser


def main() -> None:
    from llmssycoph.analysis import safe_generate_analysis_notebook

    args = build_parser().parse_args()
    status = safe_generate_analysis_notebook(
        args.run_dir,
        spec=args.spec,
        notebook_name=args.notebook_name,
        raise_on_error=args.raise_on_error,
    )
    print(status)


if __name__ == "__main__":
    main()
