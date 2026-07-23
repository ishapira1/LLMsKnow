#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llmssycoph.pruning.strict_manifests import (  # noqa: E402
    DEFAULT_SIZES,
    ManifestBuildError,
    SeedManifestBuild,
    build_alpaca_utility_manifest,
    build_evaluation_manifest,
    build_seed_manifests,
    calibration_question_uids,
    load_sampling_record_inputs,
    read_json_or_jsonl,
    write_manifest_bundle,
)


EXPECTED_SEEDS = (5, 17, 29)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict, behavior-filtered sycophancy pruning and mixed preservation manifests "
            "from LLMsKnow sampling_records.jsonl artifacts."
        )
    )
    parser.add_argument("--model-id", required=True, help="Exact Hugging Face model ID.")
    parser.add_argument(
        "--revision",
        required=True,
        help="Pinned model/tokenizer revision SHA recorded in every manifest row.",
    )
    parser.add_argument(
        "--seed-records",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help=(
            "Sampling-record file or root for a calibration seed. Repeat for ARC and CSQA and for "
            "seeds 5, 17, and 29, e.g. --seed-records 5=/path/to/arc/run."
        ),
    )
    parser.add_argument(
        "--alpaca-data",
        type=Path,
        required=True,
        help="Local JSON array or JSONL with Alpaca instruction/input/output fields.",
    )
    parser.add_argument(
        "--alpaca-utility-size",
        type=int,
        default=1000,
        help=(
            "Maximum rows in the fixed held-out Alpaca utility manifest after excluding all "
            "calibration scoring rows (default: 1000)."
        ),
    )
    parser.add_argument(
        "--evaluation-records",
        action="append",
        type=Path,
        metavar="PATH",
        help=(
            "Seed-5 sampling-record file or root used to build the single fixed val/test evaluation "
            "manifest. Repeat for ARC and CSQA."
        ),
    )
    parser.add_argument(
        "--skip-evaluation-manifest",
        action="store_true",
        help="Explicit test-only opt-out from the required fixed held-out evaluation manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument(
        "--expected-seeds",
        default=",".join(str(seed) for seed in EXPECTED_SEEDS),
        help="Comma-separated required calibration seeds (default: 5,17,29).",
    )
    return parser


def _parse_seed_paths(values: List[str]) -> Dict[int, List[Path]]:
    result: Dict[int, List[Path]] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not seed_text.strip() or not path_text.strip():
            raise ManifestBuildError(
                f"Invalid --seed-records value {value!r}; expected SEED=PATH."
            )
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ManifestBuildError(f"Invalid calibration seed {seed_text!r}.") from exc
        result.setdefault(seed, []).append(Path(path_text))
    return result


def _validate_source_metadata(
    source_audit,
    *,
    model_id: str,
    revision: str,
    expected_seed: int,
) -> None:
    for source in source_audit:
        config = source.get("run_config")
        if not config:
            raise ManifestBuildError(
                f"Source {source['path']} has no discoverable run_config.json; "
                "paper-faithful manifests require verifiable sampling provenance."
            )
        configured_model = str(config.get("model", "") or "")
        if configured_model != model_id:
            raise ManifestBuildError(
                f"Source model mismatch for {source['path']}: expected {model_id!r}, "
                f"run_config has {configured_model!r}."
            )
        configured_revision = str(config.get("revision", "") or "")
        if configured_revision != revision:
            raise ManifestBuildError(
                f"Source revision mismatch for {source['path']}: expected {revision!r}, "
                f"run_config has {configured_revision!r}."
            )
        configured_seed = config.get("seed")
        if configured_seed is None or int(configured_seed) != int(expected_seed):
            raise ManifestBuildError(
                f"Source seed mismatch for {source['path']}: expected {expected_seed}, "
                f"run_config has {configured_seed}."
            )
        split_seed = config.get("split_seed")
        if split_seed is None or int(split_seed) != 5:
            raise ManifestBuildError(
                f"Source split_seed mismatch for {source['path']}: expected locked split_seed=5, "
                f"run_config has {split_seed}."
            )
        if config.get("behavior_generation") is not True:
            raise ManifestBuildError(
                f"Source {source['path']} must have run_config behavior_generation=true."
            )
        if config.get("benchmark_source") != "ays_mc_single_turn":
            raise ManifestBuildError(
                f"Source {source['path']} must use benchmark_source='ays_mc_single_turn'."
            )
        if config.get("mc_mode") != "strict_mc":
            raise ManifestBuildError(
                f"Source {source['path']} must use mc_mode='strict_mc'."
            )
        if config.get("sampling_only") is not True:
            raise ManifestBuildError(
                f"Source {source['path']} must have sampling_only=true."
            )
        modes = source.get("sampling_modes")
        if modes != ["generation_with_choice_probabilities"]:
            raise ManifestBuildError(
                f"Source {source['path']} must contain only actual generation records with "
                f"audit probabilities; sampling_modes={modes!r}."
            )
        if int(source.get("rows_with_choice_probabilities", 0)) != int(source.get("rows", 0)):
            raise ManifestBuildError(
                f"Source {source['path']} is missing choice probabilities for one or more rows."
            )


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_paths = _parse_seed_paths(args.seed_records)
    expected_seeds = tuple(int(value.strip()) for value in args.expected_seeds.split(",") if value.strip())
    if set(seed_paths) != set(expected_seeds):
        raise ManifestBuildError(
            f"Expected exactly calibration seeds {sorted(expected_seeds)}, got {sorted(seed_paths)}."
        )
    if args.skip_evaluation_manifest and args.evaluation_records:
        raise ManifestBuildError(
            "Use either --evaluation-records or --skip-evaluation-manifest, not both."
        )
    if not args.skip_evaluation_manifest and not args.evaluation_records:
        raise ManifestBuildError(
            "--evaluation-records is required. Use --skip-evaluation-manifest only for unit-test fixtures."
        )
    alpaca_rows = read_json_or_jsonl(args.alpaca_data.expanduser().resolve())
    builds: Dict[int, SeedManifestBuild] = {}
    source_audits = {}
    for seed in sorted(seed_paths):
        records, source_audit = load_sampling_record_inputs(seed_paths[seed])
        _validate_source_metadata(
            source_audit,
            model_id=args.model_id,
            revision=args.revision,
            expected_seed=seed,
        )
        builds[seed] = build_seed_manifests(
            records,
            alpaca_rows,
            model_id=args.model_id,
            revision=args.revision,
            calibration_seed=seed,
            calibration_split=args.calibration_split,
            sizes=DEFAULT_SIZES,
        )
        source_audits[seed] = source_audit
    evaluation = None
    evaluation_source_audit = None
    if args.evaluation_records:
        evaluation_records, evaluation_source_audit = load_sampling_record_inputs(args.evaluation_records)
        _validate_source_metadata(
            evaluation_source_audit,
            model_id=args.model_id,
            revision=args.revision,
            expected_seed=5,
        )
        evaluation = build_evaluation_manifest(
            evaluation_records,
            model_id=args.model_id,
            revision=args.revision,
            suggestion_seed=5,
            calibration_question_uids=calibration_question_uids(builds),
        )
    alpaca_utility = build_alpaca_utility_manifest(
        alpaca_rows,
        builds,
        max_examples=args.alpaca_utility_size,
    )
    index = write_manifest_bundle(
        args.output_dir.expanduser().resolve(),
        builds,
        source_audits=source_audits,
        evaluation=evaluation,
        evaluation_source_audit=evaluation_source_audit,
        alpaca_utility=alpaca_utility,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "model_id": args.model_id,
                "revision": args.revision,
                "seeds": sorted(builds),
                "sizes": [name for name, _ in DEFAULT_SIZES],
                "manifest_index": index,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
