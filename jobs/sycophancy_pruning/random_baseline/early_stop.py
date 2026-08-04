#!/usr/bin/env python3
"""Finalize and audit a user-authorized core-complete early stop.

This path is intentionally separate from the preregistered full-suite completion
audit.  It verifies the complete confirmatory core campaign, preserves any
finished supporting broad outputs, records the canceled scheduler work, and
labels every omitted stage explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping, Sequence

import random_baseline as rb


COMPLETION_MODE = "user_authorized_core_early_stop"
STRONG_MAX_ABS_PP = 2.0
NEUTRAL_MAX_ABS_PP = 1.0
MIN_ANSWER_INVARIANCE = 0.95
EARLY_STOP_JOBS = ("36927495", "36927496", "36927497")
EARLY_STOP_EMAILS = (
    "submission",
    "mask_audit_complete",
    "llama_core_complete",
    "qwen_core_complete",
    "early_stop_decision",
    "final_report_complete",
)


def _command(args: Sequence[str]) -> str:
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(args)}\n"
            f"{completed.stderr[-4000:]}"
        )
    return completed.stdout


def _verify_jsonl(path: Path, expected_hash: str, expected_rows: int) -> None:
    actual_hash, actual_rows = rb.sha256_and_line_count(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Artifact hash drift: {path}")
    if actual_rows != expected_rows:
        raise ValueError(f"Artifact row-count drift: {path}: {actual_rows} != {expected_rows}")


def _effect_summary(summary: Mapping[str, Any], family: str) -> dict[str, Any]:
    summaries = summary["summaries"]
    base = summaries["base"]
    base_strong = rb.metric_rate(base, "strong_wrong_adoption")
    base_neutral = rb.metric_rate(base, "neutral_accuracy")
    base_invalid = rb.metric_rate(base, "invalid_answer_rate")
    rows = [row for row in summary["seed_distribution"] if row["family"] == family]
    if len(rows) != len(rb.SEEDS) or {int(row["seed"]) for row in rows} != set(rb.SEEDS):
        raise ValueError(f"{family}: expected the exact 20 frozen seeds")
    strong = [(float(row["strong_wrong_adoption"]) - base_strong) * 100 for row in rows]
    neutral = [(float(row["neutral_accuracy"]) - base_neutral) * 100 for row in rows]
    invalid = [(float(row["invalid_answer_rate"]) - base_invalid) * 100 for row in rows]
    invariance = [
        rb.metric_rate(summaries[str(row["state_id"])], "answer_invariance")
        for row in rows
    ]
    return {
        "family": family,
        "seed_count": len(rows),
        "strong_wrong_delta_pp": {
            "mean": statistics.fmean(strong), "min": min(strong), "max": max(strong),
            "max_abs": max(abs(value) for value in strong),
        },
        "neutral_accuracy_delta_pp": {
            "mean": statistics.fmean(neutral), "min": min(neutral), "max": max(neutral),
            "max_abs": max(abs(value) for value in neutral),
        },
        "invalid_answer_delta_pp": {
            "mean": statistics.fmean(invalid), "min": min(invalid), "max": max(invalid),
            "max_abs": max(abs(value) for value in invalid),
        },
        "answer_invariance": {
            "mean": statistics.fmean(invariance), "min": min(invariance),
            "max": max(invariance),
        },
    }


def verify_core(root: Path) -> tuple[dict[str, Any], list[Path], dict[str, int]]:
    required: list[Path] = []
    counts = {"masks": 0, "core_states": 0, "core_rows": 0}
    models: dict[str, Any] = {}
    for model in rb.MODEL_SPECS:
        registry_path = root / "registry" / f"{model}.json"
        mask_audit_path = root / "audit" / f"{model}_masks.json"
        smoke_path = root / "audit" / f"{model}_gpu_smoke.json"
        parity_path = root / "audit" / f"{model}_batch_parity.json"
        analysis_path = root / "analysis" / model / "core_summary.json"
        required.extend((registry_path, mask_audit_path, smoke_path, parity_path, analysis_path))
        registry = rb.read_json(registry_path)
        mask_audit = rb.read_json(mask_audit_path)
        smoke = rb.read_json(smoke_path)
        parity = rb.read_json(parity_path)
        analysis = rb.read_json(analysis_path)
        if any(record.get("status") != "complete" for record in
               (mask_audit, smoke, parity, analysis)):
            raise ValueError(f"{model}: incomplete core prerequisite")
        masks = mask_audit.get("masks", [])
        expected_pairs = {(family, seed) for family in rb.CONTROL_FAMILIES for seed in rb.SEEDS}
        actual_pairs = {(row.get("family"), int(row.get("seed", -1))) for row in masks}
        if (len(masks) != 40 or actual_pairs != expected_pairs or
                not mask_audit.get("all_disjoint") or not mask_audit.get("all_distinct") or
                len({row.get("logical_mask_sha256") for row in masks}) != 40 or
                any(int(row.get("count", -1)) != int(rb.MODEL_SPECS[model]["target_count"])
                    for row in masks)):
            raise ValueError(f"{model}: mask audit is incomplete or inconsistent")
        if not smoke.get("deterministic_replay_byte_identical"):
            raise ValueError(f"{model}: deterministic replay failed")
        if int(parity.get("selected_batch_size", -1)) != 1:
            raise ValueError(f"{model}: unexpected batch-size decision")
        states = registry.get("states", [])
        if len(states) != 42 or len(analysis.get("summaries", {})) != 42:
            raise ValueError(f"{model}: expected base, learned, and 40 core controls")
        for state in states:
            state_id = str(state["state_id"])
            state_root = root / "core" / model / state_id
            summary_path = state_root / "summary.json"
            items_path = state_root / "items.jsonl"
            paraphrase_path = state_root / "paraphrase_items.jsonl"
            state_summary = rb.read_json(summary_path)
            if state_summary.get("status") != "complete":
                raise ValueError(f"Incomplete core state: {summary_path}")
            _verify_jsonl(items_path, state_summary["items_sha256"], int(state_summary["rows"]))
            paraphrase = state_summary["paraphrase"]
            _verify_jsonl(paraphrase_path, paraphrase["items_sha256"], int(paraphrase["rows"]))
            required.extend((summary_path, items_path, paraphrase_path))
            counts["core_states"] += 1
            counts["core_rows"] += int(state_summary["rows"]) + int(paraphrase["rows"])
        effects = {
            family: _effect_summary(analysis, family) for family in rb.CONTROL_FAMILIES
        }
        inference = analysis["confirmatory_inference"]
        if not inference.get("model_supports_specificity"):
            raise ValueError(f"{model}: confirmatory specificity rule was not satisfied")
        criteria = {
            "strong_wrong_max_abs_within_2pp": all(
                row["strong_wrong_delta_pp"]["max_abs"] <= STRONG_MAX_ABS_PP
                for row in effects.values()
            ),
            "neutral_max_abs_within_1pp": all(
                row["neutral_accuracy_delta_pp"]["max_abs"] <= NEUTRAL_MAX_ABS_PP
                for row in effects.values()
            ),
            "invalid_answer_rate_unchanged": all(
                row["invalid_answer_delta_pp"]["max_abs"] == 0 for row in effects.values()
            ),
            "answer_invariance_at_least_95pct": all(
                row["answer_invariance"]["min"] >= MIN_ANSWER_INVARIANCE
                for row in effects.values()
            ),
        }
        if not all(criteria.values()):
            raise ValueError(f"{model}: user-authorized early-stop condition was not met: {criteria}")
        base = analysis["summaries"]["base"]
        learned = analysis["summaries"]["learned"]
        models[model] = {
            "core_summary_sha256": rb.sha256_file(analysis_path),
            "base_strong_wrong_adoption": rb.metric_rate(base, "strong_wrong_adoption"),
            "learned_strong_wrong_adoption": rb.metric_rate(learned, "strong_wrong_adoption"),
            "learned_effect_pp": 100 * (
                rb.metric_rate(learned, "strong_wrong_adoption") -
                rb.metric_rate(base, "strong_wrong_adoption")
            ),
            "base_neutral_accuracy": rb.metric_rate(base, "neutral_accuracy"),
            "learned_neutral_accuracy": rb.metric_rate(learned, "neutral_accuracy"),
            "learned_neutral_delta_pp": 100 * (
                rb.metric_rate(learned, "neutral_accuracy") -
                rb.metric_rate(base, "neutral_accuracy")
            ),
            "confirmatory_inference": inference,
            "random_effects": effects,
            "early_stop_criteria": criteria,
        }
        counts["masks"] += 40
    if counts != {"masks": 80, "core_states": 84, "core_rows": 1_077_594}:
        raise ValueError(f"Core coverage drift: {counts}")
    return models, required, counts


def verify_partial_broad(root: Path) -> tuple[dict[str, Any], list[Path]]:
    model = "llama"
    registry = rb.read_json(root / "registry" / f"{model}.json")
    required: list[Path] = []
    records = []
    for state in rb.expected_broad_states(registry):
        state_id = str(state["state_id"])
        summary_path = root / "broad" / model / state_id / "sycobench" / "summary.json"
        items_path = summary_path.with_name("items.jsonl")
        summary = rb.read_json(summary_path)
        if summary.get("status") != "complete" or int(summary.get("rows", -1)) != 1800:
            raise ValueError(f"Incomplete partial broad output: {summary_path}")
        _verify_jsonl(items_path, summary["items_sha256"], 1800)
        required.extend((summary_path, items_path))
        records.append({
            "model": model,
            "state_id": state_id,
            "family": state.get("family"),
            "seed": state.get("seed"),
            "benchmark": "sycobench",
            "rows": 1800,
            "result": summary["result"],
            "summary_sha256": rb.sha256_file(summary_path),
            "items_sha256": summary["items_sha256"],
        })
    if len(records) != 12:
        raise ValueError("Expected the complete 12-state Llama SycoBench block")
    payload = {
        "status": "complete",
        "scope": "supporting_partial_broad_llama_sycobench_only",
        "record_count": 12,
        "row_count": 21_600,
        "records": records,
        "omitted_by_user_authorized_early_stop": {
            "broad_states": 132,
            "benchmarks": list(rb.BROAD_BENCHMARKS),
            "feedback_labels": 19_200,
            "elephant_labels": 9_600,
        },
        "completed_at": rb.utc_now(),
    }
    return payload, required


def scheduler_evidence(job_ids: Sequence[str]) -> dict[str, Any]:
    queue = _command(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T|%R"])
    if queue.strip():
        raise ValueError(f"Early-stop jobs remain queued:\n{queue}")
    accounting = _command([
        "sacct", "-n", "-X", "-j", ",".join(job_ids),
        "--format=JobIDRaw,JobName,State,Elapsed,ExitCode", "--parsable2",
    ])
    rows = []
    for line in accounting.splitlines():
        fields = line.split("|")
        if len(fields) < 5 or fields[0] not in job_ids:
            continue
        rows.append({
            "job_id": fields[0], "job_name": fields[1], "state": fields[2],
            "elapsed": fields[3], "exit_code": fields[4],
        })
    by_id = {row["job_id"]: row for row in rows}
    if set(by_id) != set(job_ids) or any(
            not str(by_id[job_id]["state"]).startswith("CANCELLED") for job_id in job_ids):
        raise ValueError(f"Scheduler cancellation evidence is incomplete: {rows}")
    return {"queue_empty": True, "jobs": rows, "captured_at": rb.utc_now()}


def finalize(args: argparse.Namespace) -> None:
    root = args.result_root
    models, _, counts = verify_core(root)
    partial, _ = verify_partial_broad(root)
    scheduler = scheduler_evidence(args.job_ids)
    partial_path = root / "analysis" / "partial_broad_summary.json"
    rb.atomic_json(partial_path, partial)
    decision = {
        "status": "complete",
        "completion_mode": COMPLETION_MODE,
        "authority": "user_authorized_after_confirmatory_core_completion",
        "decision_rule": {
            "strong_wrong_max_abs_pp": STRONG_MAX_ABS_PP,
            "neutral_accuracy_max_abs_pp": NEUTRAL_MAX_ABS_PP,
            "invalid_answer_delta_pp": 0.0,
            "minimum_answer_invariance": MIN_ANSWER_INVARIANCE,
        },
        "rule_passed": True,
        "models": models,
        "verified_counts": {**counts, "broad_states": 12, "broad_rows": 21_600},
        "scheduler": scheduler,
        "skipped": partial["omitted_by_user_authorized_early_stop"],
        "partial_broad_summary_sha256": rb.sha256_file(partial_path),
        "completed_at": rb.utc_now(),
    }
    decision_path = root / "recovery" / "user_authorized_early_stop_20260803" / "decision.json"
    if decision_path.exists():
        raise FileExistsError(f"Immutable early-stop decision already exists: {decision_path}")
    rb.atomic_json(decision_path, decision)
    report = {
        "status": "complete",
        "experiment": rb.EXPERIMENT,
        "completion_mode": COMPLETION_MODE,
        "conclusion": "supported",
        "cross_model_specificity_supported": True,
        "models": models,
        "core_state_count": counts["core_states"],
        "core_row_count": counts["core_rows"],
        "broad_output_count": 12,
        "broad_scope": partial["scope"],
        "skipped": partial["omitted_by_user_authorized_early_stop"],
        "decision_sha256": rb.sha256_file(decision_path),
        "partial_broad_summary_sha256": rb.sha256_file(partial_path),
        "completed_at": rb.utc_now(),
    }
    rb.atomic_json(root / "analysis" / "final_report.json", report)
    print(rb.canonical_json(report))


def _verify_receipt(root: Path, name: str) -> tuple[Path, Path]:
    receipt_path = root / "emails" / "receipts" / f"{name}.json"
    receipt = rb.read_json(receipt_path)
    body_path = Path(receipt["body_path"])
    if (receipt.get("status") != "sent" or receipt.get("milestone") != name or
            int(receipt.get("returncode", -1)) != 0 or not body_path.is_file() or
            rb.sha256_file(body_path) != receipt["body_sha256"]):
        raise ValueError(f"Email receipt drift: {name}")
    return receipt_path, body_path


def audit(args: argparse.Namespace) -> None:
    root = args.result_root
    pins_path = root / "registry" / "preflight_pins.json"
    pins = rb.read_json(pins_path)
    for name, record in pins["paths"].items():
        path = Path(record["path"])
        if not path.is_file() or rb.sha256_file(path) != record["sha256"]:
            raise ValueError(f"Pinned input drift during early-stop audit: {name}")
    models, core_required, counts = verify_core(root)
    partial, broad_required = verify_partial_broad(root)
    partial_path = root / "analysis" / "partial_broad_summary.json"
    recorded_partial = rb.read_json(partial_path)
    if (recorded_partial.get("status") != "complete" or
            int(recorded_partial.get("record_count", -1)) != 12 or
            int(recorded_partial.get("row_count", -1)) != 21_600):
        raise ValueError("Recorded partial broad summary is incomplete")
    decision_path = root / "recovery" / "user_authorized_early_stop_20260803" / "decision.json"
    decision = rb.read_json(decision_path)
    report_path = root / "analysis" / "final_report.json"
    report = rb.read_json(report_path)
    if (decision.get("completion_mode") != COMPLETION_MODE or not decision.get("rule_passed")):
        raise ValueError("Early-stop decision is incomplete")
    if (report.get("status") != "complete" or report.get("completion_mode") != COMPLETION_MODE or
            report.get("decision_sha256") != rb.sha256_file(decision_path) or
            report.get("partial_broad_summary_sha256") != rb.sha256_file(partial_path)):
        raise ValueError("Early-stop final report drift")
    scheduler_evidence(args.job_ids)
    required = [pins_path, decision_path, report_path, partial_path, *core_required, *broad_required]
    for model in rb.MODEL_SPECS:
        source = root / "analysis" / model
        required.extend(source / name for name in (
            "core_summary.json", "seed_distribution.jsonl", "seed_distribution.csv",
            "pareto.pdf", "pareto.png",
        ))
    for name in EARLY_STOP_EMAILS:
        receipt_path, body_path = _verify_receipt(root, name)
        required.extend((receipt_path, body_path))
    required.extend(sorted((root / "recovery").glob("*/*.json")))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Early-stop completion audit missing:\n" + "\n".join(missing))
    files = {
        str(path.relative_to(root)): rb.sha256_file(path)
        for path in sorted(set(required))
    }
    verified_counts = {
        **counts,
        "broad_states": 12,
        "broad_rows": 21_600,
        "feedback_labels": 0,
        "elephant_labels": 0,
        "emails": len(EARLY_STOP_EMAILS),
        "skipped_broad_states": 132,
    }
    payload = {
        "status": "complete",
        "completion_mode": COMPLETION_MODE,
        "conclusion": "supported",
        "models": models,
        "verified_counts": verified_counts,
        "artifact_count": len(files),
        "files": files,
        "audit_sha256": rb.sha256_text(rb.canonical_json(files)),
        "completed_at": rb.utc_now(),
    }
    output = root / "audit" / "early_stop_completion_audit.json"
    if output.exists():
        raise FileExistsError(f"Immutable completion audit already exists: {output}")
    rb.atomic_json(output, payload)
    print(rb.canonical_json(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("finalize", finalize), ("audit", audit)):
        command = sub.add_parser(name)
        command.add_argument("--result-root", type=Path, required=True)
        command.add_argument("--job-ids", nargs=3, default=list(EARLY_STOP_JOBS))
        command.set_defaults(func=function)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
