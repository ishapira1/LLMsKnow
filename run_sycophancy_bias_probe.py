from __future__ import annotations

import sys

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    from sycophancy_bias_probe.cli import parse_args as _parse_args

    _parse_args()
    raise SystemExit(0)

def main() -> None:
    from sycophancy_bias_probe.cli import parse_args
    from sycophancy_bias_probe.pipeline import run_pipeline

    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
