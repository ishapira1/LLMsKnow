#!/usr/bin/env python3
"""Execute Bonham HumanEval+/MBPP+ generations in the pinned EvalPlus SIF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import campaign
from core import (
    atomic_json,
    atomic_jsonl,
    canonical_shard_directories,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from bonham_runtime.evaluation.evalplus_sandbox import (
    EVALPLUS_RESULT_NAME,
    EVALPLUS_TASK_COUNTS,
    EvalPlusSandboxSpec,
    build_singularity_command,
    prepare_evalplus_workspace,
    validate_evalplus_results,
)


BENCHMARK_FILES = {
    "humaneval": "evalplus_humaneval_samples.jsonl",
    "mbpp": "evalplus_mbpp_samples.jsonl",
}
SHARD_COUNT = 4


class EvalPlusError(campaign.CampaignError):
    pass


def _selected_models(args: argparse.Namespace) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(getattr(args, "model_key", None) or ()))
    return requested or campaign.MODEL_KEYS


def _scope_suffix(model_keys: tuple[str, ...]) -> str:
    if model_keys == campaign.MODEL_KEYS:
        return ""
    return "_" + "_".join(model_keys)


def _validate_prepared_scope(
    root: Path,
    model_keys: tuple[str, ...],
) -> Mapping[str, Any]:
    suffix = _scope_suffix(model_keys)
    index_path = root / "evalplus" / "inputs" / f"index{suffix}.jsonl"
    complete_path = root / "evalplus" / "inputs" / f"COMPLETE{suffix}.json"
    if not index_path.is_file() or not complete_path.is_file():
        raise EvalPlusError("Prepared EvalPlus scope is incomplete")
    receipt = read_json(complete_path)
    entries = read_jsonl(index_path)
    expected_inventory = {
        (model_key, state_id, benchmark, shard)
        for model_key in model_keys
        for state_id in campaign.PRIMARY_STATE_IDS
        for benchmark in BENCHMARK_FILES
        for shard in range(SHARD_COUNT)
    }
    if (
        receipt.get("status") != "complete"
        or receipt.get("experiment") != campaign.EXPERIMENT
        or tuple(receipt.get("model_keys", ())) != model_keys
        or int(receipt.get("shard_count", -1)) != len(expected_inventory)
        or int(receipt.get("shards_per_benchmark_state", -1)) != SHARD_COUNT
        or dict(receipt.get("task_counts", {})) != dict(EVALPLUS_TASK_COUNTS)
        or receipt.get("index_sha256") != sha256_file(index_path)
        or len(entries) != len(expected_inventory)
    ):
        raise EvalPlusError("Prepared EvalPlus receipt differs from the frozen scope")
    observed_inventory = set()
    observed_task_counts = {}
    for entry in entries:
        key = (
            str(entry.get("model_key", "")),
            str(entry.get("state_id", "")),
            str(entry.get("benchmark", "")),
            int(entry.get("shard", -1)),
        )
        if key not in expected_inventory or key in observed_inventory:
            raise EvalPlusError(f"Prepared EvalPlus shard inventory is invalid: {key}")
        observed_inventory.add(key)
        model_key, state_id, benchmark, shard = key
        expected_path = (
            root
            / "evalplus"
            / "inputs"
            / model_key
            / state_id
            / benchmark
            / f"shard_{shard:04d}.jsonl"
        )
        path = Path(str(entry.get("path", "")))
        rows = read_jsonl(path)
        task_ids = [str(row.get("task_id", "")) for row in rows]
        if (
            path.resolve() != expected_path.resolve()
            or entry.get("sha256") != sha256_file(path)
            or int(entry.get("task_count", -1)) != len(rows)
            or not rows
            or any(not task_id for task_id in task_ids)
            or len(task_ids) != len(set(task_ids))
        ):
            raise EvalPlusError(f"Prepared EvalPlus shard changed: {path}")
        count_key = (model_key, state_id, benchmark)
        observed_task_counts[count_key] = observed_task_counts.get(count_key, 0) + len(rows)
    if observed_inventory != expected_inventory:
        raise EvalPlusError("Prepared EvalPlus shard inventory is incomplete")
    for model_key in model_keys:
        for state_id in campaign.PRIMARY_STATE_IDS:
            for benchmark, expected in EVALPLUS_TASK_COUNTS.items():
                if observed_task_counts.get((model_key, state_id, benchmark)) != int(expected):
                    raise EvalPlusError(
                        f"Prepared {benchmark} task count is incomplete for {model_key}/{state_id}"
                    )
    return receipt


def _sample_rows(root: Path, model_key: str, state_id: str, benchmark: str) -> list[Mapping[str, Any]]:
    filename = BENCHMARK_FILES[benchmark]
    rows = []
    family_root = root / "evaluations" / "results" / model_key / state_id / "capabilities"
    for directory in canonical_shard_directories(family_root):
        complete_path = directory / "COMPLETE"
        sample_path = directory / filename
        if not sample_path.is_file():
            continue
        complete = read_json(complete_path)
        if dict(complete.get("file_sha256", {})).get(filename) != sha256_file(sample_path):
            raise EvalPlusError(f"EvalPlus generation handoff changed: {sample_path}")
        rows.extend(read_jsonl(sample_path))
    expected = int(EVALPLUS_TASK_COUNTS[benchmark])
    task_ids = [str(row.get("task_id", "")) for row in rows]
    if len(rows) != expected or any(not value for value in task_ids) or len(set(task_ids)) != expected:
        raise EvalPlusError(
            f"Incomplete {benchmark} generation for {model_key}/{state_id}: {len(rows)}/{expected}"
        )
    return sorted(rows, key=lambda row: str(row["task_id"]))


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.result_root)
    model_keys = _selected_models(args)
    suffix = _scope_suffix(model_keys)
    index_path = root / "evalplus" / "inputs" / f"index{suffix}.jsonl"
    complete_path = root / "evalplus" / "inputs" / f"COMPLETE{suffix}.json"
    if index_path.exists() or complete_path.exists():
        receipt = _validate_prepared_scope(root, model_keys)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    entries = []
    canonical_inventory = {}
    for model_key in model_keys:
        for state_id in campaign.PRIMARY_STATE_IDS:
            for benchmark in BENCHMARK_FILES:
                rows = _sample_rows(root, model_key, state_id, benchmark)
                inventory = tuple(str(row["task_id"]) for row in rows)
                prior = canonical_inventory.setdefault((model_key, benchmark), inventory)
                if inventory != prior:
                    raise EvalPlusError(f"{benchmark} task IDs differ across states for {model_key}")
                ordered = sorted(
                    rows,
                    key=lambda row: stable_evalplus_rank(benchmark, str(row["task_id"])),
                )
                buckets = [[] for _ in range(SHARD_COUNT)]
                for position, row in enumerate(ordered):
                    buckets[position % SHARD_COUNT].append(row)
                for shard, bucket in enumerate(buckets):
                    destination = (
                        root
                        / "evalplus"
                        / "inputs"
                        / model_key
                        / state_id
                        / benchmark
                        / f"shard_{shard:04d}.jsonl"
                    )
                    atomic_jsonl(destination, bucket)
                    entries.append(
                        {
                            "model_key": model_key,
                            "state_id": state_id,
                            "benchmark": benchmark,
                            "shard": shard,
                            "task_count": len(bucket),
                            "path": str(destination.resolve()),
                            "sha256": sha256_file(destination),
                        }
                    )
    expected_entries = (
        len(model_keys)
        * len(campaign.PRIMARY_STATE_IDS)
        * len(BENCHMARK_FILES)
        * SHARD_COUNT
    )
    if len(entries) != expected_entries:
        raise EvalPlusError("EvalPlus input shard census is incomplete")
    atomic_jsonl(index_path, entries)
    receipt = {
        "status": "complete",
        "experiment": campaign.EXPERIMENT,
        "shard_count": len(entries),
        "model_keys": list(model_keys),
        "task_counts": dict(EVALPLUS_TASK_COUNTS),
        "shards_per_benchmark_state": SHARD_COUNT,
        "index_sha256": sha256_file(index_path),
    }
    atomic_json(complete_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def stable_evalplus_rank(benchmark: str, task_id: str) -> str:
    return hashlib.sha256(
        f"{campaign.EXPERIMENT}|evalplus|{benchmark}|{task_id}".encode("utf-8")
    ).hexdigest()


def _outcomes(result_path: Path, benchmark: str) -> list[Mapping[str, Any]]:
    payload = read_json(result_path)
    evaluations = payload.get("eval")
    if not isinstance(evaluations, Mapping):
        raise EvalPlusError("Official EvalPlus output lacks its eval mapping")
    rows = []
    for task_id in sorted(evaluations):
        samples = evaluations[task_id]
        if not isinstance(samples, list) or len(samples) != 1 or not isinstance(samples[0], Mapping):
            raise EvalPlusError(f"Malformed EvalPlus result for {task_id}")
        sample = dict(samples[0])
        base_status = str(sample.get("base_status", ""))
        plus_status = str(sample.get("plus_status", ""))
        if not base_status or not plus_status:
            raise EvalPlusError(f"EvalPlus result lacks statuses for {task_id}")
        rows.append(
            {
                "benchmark": "HumanEval+" if benchmark == "humaneval" else "MBPP+",
                "task_id": str(task_id),
                "base_status": base_status,
                "plus_status": plus_status,
                "base_pass": base_status.casefold() == "pass",
                "plus_pass": plus_status.casefold() == "pass",
            }
        )
    return rows


def run_shard(args: argparse.Namespace) -> None:
    root = Path(args.result_root).resolve()
    samples = (
        root
        / "evalplus"
        / "inputs"
        / args.model_key
        / args.state_id
        / args.benchmark
        / f"shard_{int(args.shard):04d}.jsonl"
    )
    rows = read_jsonl(samples)
    destination = (
        root
        / "evalplus"
        / "shards"
        / args.model_key
        / args.state_id
        / args.benchmark
        / f"shard_{int(args.shard):04d}"
    )
    if (destination / "COMPLETE.json").is_file():
        complete = read_json(destination / "COMPLETE.json")
        if complete.get("records_sha256") != sha256_file(destination / "records.jsonl"):
            raise EvalPlusError(f"Completed EvalPlus shard changed: {destination}")
        return
    if destination.exists():
        raise EvalPlusError(f"Incomplete EvalPlus destination exists: {destination}")
    attempt = destination.with_name(destination.name + f".partial.{os.getpid()}")
    attempt.mkdir(parents=True)
    specification = EvalPlusSandboxSpec(
        benchmark=args.benchmark,
        samples_path=samples,
        samples_sha256=sha256_file(samples),
        image_path=Path(args.image).resolve(),
        image_sha256=str(args.image_sha256),
        work_root=attempt / "workspace",
        expected_task_count=len(rows),
        timeout_seconds=int(args.timeout_seconds),
        memory_mib=16384,
        process_limit=128,
    )
    workspace = prepare_evalplus_workspace(specification)
    command = build_singularity_command(
        specification, workspace, singularity_binary=str(args.singularity_binary)
    )
    if args.parallel_workers is not None:
        workers = int(args.parallel_workers)
        if not 1 <= workers <= specification.process_limit:
            raise EvalPlusError("Invalid EvalPlus parallel worker count")
        command = (*command, "--parallel", str(workers))
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=int(args.timeout_seconds) + 180,
    )
    (attempt / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (attempt / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise EvalPlusError(
            f"EvalPlus returned {completed.returncode}: {completed.stderr[-1000:]}"
        )
    official = workspace / EVALPLUS_RESULT_NAME
    validated = validate_evalplus_results(specification, workspace, official)
    output_rows = _outcomes(official, args.benchmark)
    records = attempt / "records.jsonl"
    atomic_jsonl(records, output_rows)
    receipt = {
        "status": "complete",
        "experiment": campaign.EXPERIMENT,
        "model_key": args.model_key,
        "state_id": args.state_id,
        "benchmark": args.benchmark,
        "shard": int(args.shard),
        "task_count": len(output_rows),
        "records_sha256": sha256_file(records),
        "samples_sha256": sha256_file(samples),
        "image_sha256": args.image_sha256,
        "official_result_sha256": validated["result_sha256"],
        "parallel_workers": args.parallel_workers,
        "wall_seconds": time.monotonic() - started,
        "command_sha256": sha256_json(list(command)),
    }
    atomic_json(attempt / "COMPLETE.json", receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt, destination)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.result_root)
    model_keys = _selected_models(args)
    canonical = {}
    publications = []
    for model_key in model_keys:
        for state_id in campaign.PRIMARY_STATE_IDS:
            combined = []
            source_receipts = []
            for benchmark in BENCHMARK_FILES:
                benchmark_rows = []
                for shard in range(SHARD_COUNT):
                    directory = (
                        root
                        / "evalplus"
                        / "shards"
                        / model_key
                        / state_id
                        / benchmark
                        / f"shard_{shard:04d}"
                    )
                    complete = read_json(directory / "COMPLETE.json")
                    records = directory / "records.jsonl"
                    samples = (
                        root
                        / "evalplus"
                        / "inputs"
                        / model_key
                        / state_id
                        / benchmark
                        / f"shard_{shard:04d}.jsonl"
                    )
                    if (
                        complete.get("records_sha256") != sha256_file(records)
                        or complete.get("samples_sha256") != sha256_file(samples)
                        or complete.get("image_sha256") != args.image_sha256
                    ):
                        raise EvalPlusError(f"EvalPlus shard receipt mismatch: {directory}")
                    input_ids = tuple(sorted(str(row["task_id"]) for row in read_jsonl(samples)))
                    output_rows = read_jsonl(records)
                    output_ids = tuple(sorted(str(row["task_id"]) for row in output_rows))
                    if input_ids != output_ids:
                        raise EvalPlusError(f"EvalPlus task inventory mismatch: {directory}")
                    benchmark_rows.extend(output_rows)
                    source_receipts.append(sha256_file(directory / "COMPLETE.json"))
                expected = int(EVALPLUS_TASK_COUNTS[benchmark])
                task_ids = tuple(sorted(str(row["task_id"]) for row in benchmark_rows))
                if len(task_ids) != expected or len(set(task_ids)) != expected:
                    raise EvalPlusError(f"Incomplete {benchmark} union for {model_key}/{state_id}")
                if task_ids != canonical.setdefault((model_key, benchmark), task_ids):
                    raise EvalPlusError("EvalPlus task IDs differ across states")
                combined.extend(benchmark_rows)
            destination = root / "evalplus" / "results" / model_key / state_id
            results_path = destination / "results.jsonl"
            atomic_jsonl(results_path, combined)
            receipt = {
                "status": "complete",
                "model_key": model_key,
                "state_id": state_id,
                "task_count": len(combined),
                "results_sha256": sha256_file(results_path),
                "image_sha256": args.image_sha256,
                "source_receipts_sha256": sha256_json(source_receipts),
                "pass_at_1": {
                    display: sum(bool(row["plus_pass"]) for row in combined if row["benchmark"] == display)
                    / sum(1 for row in combined if row["benchmark"] == display)
                    for display in ("HumanEval+", "MBPP+")
                },
            }
            atomic_json(destination / "COMPLETE.json", receipt)
            publications.append(receipt)
    final = {
        "status": "complete",
        "experiment": campaign.EXPERIMENT,
        "publication_count": len(publications),
        "model_keys": list(model_keys),
        "image_sha256": args.image_sha256,
        "publications_sha256": sha256_json(publications),
    }
    suffix = _scope_suffix(model_keys)
    atomic_json(root / "evalplus" / "results" / f"COMPLETE{suffix}.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("prepare")
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument(
        "--model-key", choices=campaign.MODEL_KEYS, action="append", default=[]
    )
    command.set_defaults(func=prepare)
    command = subparsers.add_parser("run-shard")
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--model-key", choices=campaign.MODEL_KEYS, required=True)
    command.add_argument("--state-id", choices=campaign.PRIMARY_STATE_IDS, required=True)
    command.add_argument("--benchmark", choices=tuple(BENCHMARK_FILES), required=True)
    command.add_argument("--shard", type=int, choices=range(SHARD_COUNT), required=True)
    command.add_argument("--image", type=Path, required=True)
    command.add_argument("--image-sha256", required=True)
    command.add_argument("--singularity-binary", default="singularity")
    command.add_argument("--timeout-seconds", type=int, default=7200)
    command.add_argument("--parallel-workers", type=int)
    command.set_defaults(func=run_shard)
    command = subparsers.add_parser("aggregate")
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--image-sha256", required=True)
    command.add_argument(
        "--model-key", choices=campaign.MODEL_KEYS, action="append", default=[]
    )
    command.set_defaults(func=aggregate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
