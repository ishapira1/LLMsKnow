#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmssycoph.fixed_development_cohort import (
    audit_development_cohort,
    freeze_development_cohort,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "openai_api"
    / "full_dataset_sample_size_validation_gpt54nano_20260729"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "openai_sycophancy_development_cohort_gpt54nano_v1.jsonl"
)
DEFAULT_SPEC = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "openai_sycophancy_development_cohort_gpt54nano_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or audit the reusable neutral-correct train cohort."
    )
    parser.add_argument("mode", choices=("freeze", "audit"))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()

    if args.mode == "freeze":
        result = freeze_development_cohort(
            source_root=args.source_root,
            manifest_path=args.manifest,
            spec_path=args.spec,
        )
    else:
        result = audit_development_cohort(
            manifest_path=args.manifest,
            spec_path=args.spec,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
