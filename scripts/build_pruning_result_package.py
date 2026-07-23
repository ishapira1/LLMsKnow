#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llmssycoph.pruning.result_package import build_minimum_result_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fail-closed minimum sycophancy-pruning result package from "
            "the canonical experiment-root analysis, prediction, registry, and utility artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a nonempty output directory after strict path-safety checks.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_minimum_result_package(
        args.experiment_root,
        args.output_dir,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "input_identity_sha256": summary["input_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
