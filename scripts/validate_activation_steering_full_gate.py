#!/usr/bin/env python3
"""Validate every evidence artifact required for a full steering submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


PROTOCOL_VERSION = "controlled_prompt_only_v1_20260725"
APPROVAL_PHRASE = "APPROVE_CONTROLLED_ACTIVATION_STEERING_FULL"
REQUIRED_CONDITIONS = {
    "neutral",
    "incorrect_suggestion",
    "incorrect_suggestion_strong",
    "suggest_correct",
}
REQUIRED_REVIEW_ASSERTIONS = (
    "inspection_review_passed",
    "semantic_b_review_complete",
    "tiny_compute_review_passed",
    "projected_compute_accepted",
    "real_model_bf16_gate_passed",
    "diff_review_passed",
    "bash_syntax_passed",
    "submitter_dry_run_passed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    assert_finite(payload, location=str(path))
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            assert_finite(row, location=f"{path}:{line_number}")
            rows.append(row)
    return rows


def assert_finite(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite number at {location}: {value!r}")


def require_protocol(payload: Mapping[str, Any], *, stage: str, path: Path) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Protocol mismatch in {path}.")
    if payload.get("stage") != stage:
        raise ValueError(
            f"Stage mismatch in {path}: {payload.get('stage')!r} != {stage!r}."
        )


def resolve_snapshot(
    manifest: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path:
    relative = str(manifest.get("question_manifest_snapshot", "") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"Invalid question-manifest snapshot in {report_path}.")
    snapshot = report_path.parent / relative
    if not snapshot.is_file():
        raise FileNotFoundError(f"Missing question-manifest snapshot: {snapshot}")
    expected = str(manifest.get("question_manifest_snapshot_sha256", "") or "")
    actual = sha256_file(snapshot)
    if expected != actual or manifest.get("question_manifest_sha256") != actual:
        raise ValueError(f"Question-manifest snapshot hash mismatch in {report_path}.")
    return snapshot


def validate_inspection(
    report_path: Path,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    report = read_json(report_path)
    require_protocol(report, stage="inspect_examples", path=report_path)
    snapshot = resolve_snapshot(report, report_path=report_path)
    manifest_rows = read_jsonl(snapshot)
    expected_questions = 8
    if len(manifest_rows) != expected_questions:
        raise ValueError(
            f"Inspection snapshot must contain {expected_questions} questions."
        )
    if int(report.get("n_questions", -1)) != expected_questions:
        raise ValueError("Inspection report question count is not eight.")
    dry_run = dict(config.get("dry_run", {}) or {})
    model_config = dict(
        dict(config.get("models", {}) or {}).get(
            str(dry_run.get("model_key", "")),
            {},
        )
        or {}
    )
    runtime = dict(report.get("runtime", {}) or {})
    if (
        str(runtime.get("device", "")).lower() != "cuda"
        or "bfloat16" not in str(runtime.get("model_dtype", "")).lower()
        or runtime.get("model_name_or_path") != model_config.get("identifier")
        or runtime.get("model_commit_hash") != model_config.get("revision")
        or runtime.get("tokenizer_name_or_path") != model_config.get("identifier")
        or runtime.get("tokenizer_commit_hash") != model_config.get("revision")
        or not str(runtime.get("chat_template_sha256", "") or "")
        or report.get("model_revision") != model_config.get("revision")
        or report.get("tokenizer_revision") != model_config.get("revision")
    ):
        raise ValueError("Inspection runtime identity is invalid.")
    inspection_commit = str(report.get("git_commit", "") or "")
    if not inspection_commit or report.get("dirty") is not False:
        raise ValueError("Inspection did not use a clean Git revision.")
    layers = [int(value) for value in report.get("layers", [])]
    configured_layers = [
        int(value) for value in dry_run.get("layers", [])
    ]
    if not layers or layers != configured_layers:
        raise ValueError(
            f"Inspection layers differ from the frozen dry-run layers: {layers!r}."
        )
    output_path = report_path.parent / "preflight_examples.jsonl"
    if not output_path.is_file():
        raise FileNotFoundError(f"Missing inspection rows: {output_path}")
    if report.get("output_sha256") != sha256_file(output_path):
        raise ValueError("Inspection-row hash mismatch.")
    rows = read_jsonl(output_path)
    expected_rows = expected_questions * len(REQUIRED_CONDITIONS) * len(layers)
    if int(report.get("n_rows", -1)) != expected_rows or len(rows) != expected_rows:
        raise ValueError(
            f"Inspection must contain {expected_rows} rows; found {len(rows)}."
        )
    expected_keys = {
        (
            f"{row['dataset']}::{row['source_example_id']}",
            condition,
            layer,
        )
        for row in manifest_rows
        for condition in REQUIRED_CONDITIONS
        for layer in layers
    }
    observed_keys: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows):
        key = (
            str(row.get("stable_question_key", "")),
            str(row.get("condition", "")),
            int(row.get("layer", -1)),
        )
        observed_keys.add(key)
        prompt_count = int(row.get("prompt_token_count", 0))
        final_ids = list(row.get("final_20_token_ids", []) or [])
        if (
            row.get("protocol_version") != PROTOCOL_VERSION
            or prompt_count <= 0
            or not final_ids
            or int(row.get("selected_activation_token_index", -1))
            != prompt_count - 1
            or int(row.get("selected_activation_token_id", -1)) != int(final_ids[-1])
            or not str(row.get("rendered_chat", "") or "")
            or not list(row.get("prompt_messages", []) or [])
            or not list(row.get("activation_shape", []) or [])
        ):
            raise ValueError(f"Incomplete prompt-boundary evidence at inspection row {index}.")
        for field in (
            "activation_norm",
            "item_delta_norm",
            "item_delta_projection_on_wn_unit",
            "wn_direction_norm",
        ):
            value = float(row.get(field, math.nan))
            if not math.isfinite(value) or (field.endswith("_norm") and value < 0):
                raise ValueError(f"Invalid {field} at inspection row {index}.")
        for field in (
            "choice_probabilities",
            "choice_log_scores",
            "choice_token_ids",
            "representative_injected_norms",
        ):
            if not dict(row.get(field, {}) or {}):
                raise ValueError(f"Missing {field} at inspection row {index}.")
    if observed_keys != expected_keys:
        raise ValueError("Inspection rows do not form the required question/condition/layer grid.")
    return {
        "report_sha256": sha256_file(report_path),
        "question_manifest_snapshot_sha256": sha256_file(snapshot),
        "n_questions": expected_questions,
        "n_rows": expected_rows,
        "layers": layers,
        "git_commit": inspection_commit,
    }


def validate_noop_rows(
    path: Path,
    *,
    probability_threshold: float,
    margin_threshold: float,
) -> int:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No no-op sentinels found in {path}.")
    for index, row in enumerate(rows):
        same_shape = dict(row.get("same_shape", {}) or {})
        mixed = dict(row.get("mixed_batch_zero", {}) or {})
        cross = dict(row.get("cross_batch", {}) or {})
        if (
            same_shape.get("exact_required") is not True
            or float(same_shape.get("max_abs_probability_error", math.inf)) != 0.0
            or float(same_shape.get("top_choice_agreement", 0.0)) != 1.0
            or mixed.get("exact_required") is not True
            or float(mixed.get("max_abs_probability_error", math.inf)) != 0.0
            or float(mixed.get("top_choice_agreement", 0.0)) != 1.0
            or float(cross.get("max_abs_probability_error", math.inf))
            > probability_threshold
            or float(cross.get("top_choice_agreement", 0.0)) != 1.0
            or float(row.get("same_shape_max_margin_error", math.inf)) != 0.0
            or float(row.get("mixed_batch_zero_max_margin_error", math.inf)) != 0.0
            or float(row.get("cross_batch_max_margin_error", math.inf))
            > margin_threshold
        ):
            raise ValueError(f"No-op sentinel gate failed at {path}:{index + 1}.")
    return len(rows)


def validate_tiny_compute(
    report_path: Path,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    report = read_json(report_path)
    require_protocol(report, stage="tiny_compute_projection", path=report_path)
    if (
        report.get("status") != "requires_researcher_review"
        or report.get("authorizes_full_submission") is not False
    ):
        raise ValueError("Tiny projection must remain non-authorizing machine evidence.")
    projection = dict(config.get("compute_projection", {}) or {})
    for field in ("full_strict_choice_rows", "full_fixed_probe_candidate_passes"):
        if int(report.get(field, -1)) != int(projection.get(field, -2)):
            raise ValueError(f"Tiny projection {field} differs from the frozen config.")
    projected_hours = float(report.get("projected_gpu_hours", math.nan))
    if not math.isfinite(projected_hours) or projected_hours <= 0:
        raise ValueError("Tiny projection has no valid positive GPU-hour estimate.")
    inputs = list(report.get("input_manifests", []) or [])
    if len(inputs) != 3:
        raise ValueError("Tiny projection must bind train, validation, and test manifests.")
    numeric = dict(config.get("numeric_gates", {}) or {})
    probability_threshold = float(
        numeric.get("cross_batch_max_probability_error", 0.005)
    )
    margin_threshold = float(numeric.get("cross_batch_max_margin_error", 0.05))
    observed_splits: set[str] = set()
    manifest_hashes: list[str] = []
    question_manifest_hashes: set[str] = set()
    batch_policies: list[dict[str, Any]] = []
    no_op_rows = 0
    for item in inputs:
        manifest_path = Path(str(dict(item).get("path", ""))).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing tiny manifest: {manifest_path}")
        manifest_hash = sha256_file(manifest_path)
        if dict(item).get("sha256") != manifest_hash:
            raise ValueError(f"Tiny manifest hash mismatch: {manifest_path}")
        manifest_hashes.append(manifest_hash)
        manifest = read_json(manifest_path)
        require_protocol(manifest, stage="tiny_dry_run", path=manifest_path)
        snapshot = resolve_snapshot(manifest, report_path=manifest_path)
        question_manifest_hashes.add(sha256_file(snapshot))
        if (
            manifest.get("score_fixed_probe") is not True
            or manifest.get("generation_diagnostics") is not True
            or int(manifest.get("n_strict_choice_rows", 0)) <= 0
        ):
            raise ValueError(f"Incomplete tiny-run diagnostics in {manifest_path}.")
        batch_policy = dict(manifest.get("batch_policy", {}) or {})
        requested_batch = int(batch_policy.get("requested_batch_size", 0))
        effective_batch = int(batch_policy.get("effective_batch_size", 0))
        forced_batch_one = batch_policy.get("forced_batch_size_one")
        failure = batch_policy.get("failure")
        if (
            requested_batch <= 0
            or requested_batch
            != int(manifest.get("requested_max_batch_size", 0))
            or effective_batch
            != int(manifest.get("effective_max_batch_size", 0))
        ):
            raise ValueError(f"Inconsistent tiny batch policy in {manifest_path}.")
        if forced_batch_one is True:
            if effective_batch != 1 or not isinstance(failure, dict) or not failure:
                raise ValueError(
                    f"Batch-one fallback lacks failure evidence in {manifest_path}."
                )
        elif forced_batch_one is False:
            if effective_batch != requested_batch or failure is not None:
                raise ValueError(f"Unexpected tiny batch fallback in {manifest_path}.")
        else:
            raise ValueError(f"Missing forced-batch policy flag in {manifest_path}.")
        batch_policies.append(batch_policy)
        results_path = manifest_path.parent / "question_results.jsonl"
        noop_path = manifest_path.parent / "noop_sentinels.jsonl"
        if (
            not results_path.is_file()
            or manifest.get("question_results_sha256") != sha256_file(results_path)
            or not noop_path.is_file()
            or manifest.get("noop_sentinels_sha256") != sha256_file(noop_path)
        ):
            raise ValueError(f"Tiny output hash mismatch beside {manifest_path}.")
        result_rows = read_jsonl(results_path)
        if len(result_rows) != int(manifest.get("n_result_rows", -1)):
            raise ValueError(f"Tiny result-row count mismatch beside {manifest_path}.")
        observed_splits.update(str(row.get("split", "")) for row in result_rows)
        count = validate_noop_rows(
            noop_path,
            probability_threshold=probability_threshold,
            margin_threshold=margin_threshold,
        )
        if count != int(manifest.get("n_noop_rows", -1)):
            raise ValueError(f"Tiny no-op row count mismatch beside {manifest_path}.")
        no_op_rows += count
    if observed_splits != {"train", "val", "test"}:
        raise ValueError(f"Tiny evidence has wrong split coverage: {observed_splits!r}.")
    if len(question_manifest_hashes) != 1:
        raise ValueError("Tiny train/validation/test runs used different manifests.")
    if report.get("batch_policies") != batch_policies:
        raise ValueError("Tiny compute report batch policies do not match its inputs.")
    any_forced = any(
        bool(policy["forced_batch_size_one"]) for policy in batch_policies
    )
    if bool(report.get("any_forced_batch_size_one")) != any_forced:
        raise ValueError("Tiny compute report misstates batch-one fallback.")
    real_gate_item = dict(report.get("real_model_bf16_gate", {}) or {})
    real_gate_path = Path(str(real_gate_item.get("path", ""))).expanduser().resolve()
    if not real_gate_path.is_file():
        raise FileNotFoundError(f"Missing real-model BF16 gate: {real_gate_path}")
    real_gate_hash = sha256_file(real_gate_path)
    if real_gate_item.get("sha256") != real_gate_hash:
        raise ValueError("Real-model BF16 gate hash mismatch.")
    real_gate = read_json(real_gate_path)
    require_protocol(
        real_gate,
        stage="real_model_bf16_gate",
        path=real_gate_path,
    )
    if real_gate.get("status") != "passed":
        raise ValueError("Real-model BF16 gate is not passing.")
    if real_gate.get("question_manifest_sha256") not in question_manifest_hashes:
        raise ValueError("Real-model BF16 gate used a different question manifest.")
    runtime = dict(real_gate.get("runtime", {}) or {})
    dry_run = dict(config.get("dry_run", {}) or {})
    model_config = dict(
        dict(config.get("models", {}) or {}).get(
            str(dry_run.get("model_key", "")),
            {},
        )
        or {}
    )
    if (
        str(runtime.get("device", "")).lower() != "cuda"
        or "bfloat16" not in str(runtime.get("model_dtype", "")).lower()
        or runtime.get("model_name_or_path") != model_config.get("identifier")
        or runtime.get("model_commit_hash") != model_config.get("revision")
        or runtime.get("tokenizer_commit_hash") != model_config.get("revision")
        or not str(runtime.get("chat_template_sha256", "") or "")
    ):
        raise ValueError("Real-model BF16 gate runtime identity is invalid.")
    provenance = dict(real_gate.get("provenance", {}) or {})
    if (
        not str(provenance.get("git_commit", "") or "")
        or provenance.get("dirty") is not False
    ):
        raise ValueError("Real-model BF16 gate did not use a clean Git revision.")
    return {
        "report_sha256": sha256_file(report_path),
        "input_manifest_sha256s": manifest_hashes,
        "projected_gpu_hours": projected_hours,
        "any_forced_batch_size_one": any_forced,
        "real_model_bf16_gate_sha256": real_gate_hash,
        "real_model_git_commit": str(provenance["git_commit"]),
        "n_noop_rows": no_op_rows,
    }


def git_state(repo_dir: Path) -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_dir),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        text=True,
    ).strip()
    return commit, status


def approval_template(
    *,
    commit: str,
    evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "full_submission_researcher_approval",
        "status": "pending_researcher_approval",
        "approval_phrase": "",
        "reviewer": "",
        "reviewed_at": "",
        "approved_git_commit": commit,
        **dict(evidence_hashes),
        **{field: False for field in REQUIRED_REVIEW_ASSERTIONS},
        "review_note": "",
    }


def validate_approval(
    approval_path: Path,
    *,
    commit: str,
    evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    approval = read_json(approval_path)
    require_protocol(
        approval,
        stage="full_submission_researcher_approval",
        path=approval_path,
    )
    if (
        approval.get("status") != "approved"
        or approval.get("approval_phrase") != APPROVAL_PHRASE
    ):
        raise ValueError("Exact full-submission researcher approval is absent.")
    reviewer = str(approval.get("reviewer", "") or "").strip()
    reviewed_at = str(approval.get("reviewed_at", "") or "").strip()
    if not reviewer or not reviewed_at:
        raise ValueError("Full-submission approval lacks reviewer provenance.")
    timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Full-submission approval timestamp must include a timezone.")
    if approval.get("approved_git_commit") != commit:
        raise ValueError("Full-submission approval is for a different Git commit.")
    for field, expected in evidence_hashes.items():
        if approval.get(field) != expected:
            raise ValueError(f"Full-submission approval hash mismatch: {field}.")
    missing = [field for field in REQUIRED_REVIEW_ASSERTIONS if approval.get(field) is not True]
    if missing:
        raise ValueError(f"Required researcher assertions are not true: {missing!r}.")
    return {
        "approval_sha256": sha256_file(approval_path),
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
    }


def require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing gate input: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--question-manifest", type=Path, required=True)
    parser.add_argument("--alpaca-manifest", type=Path, required=True)
    parser.add_argument("--inspection-report", type=Path, required=True)
    parser.add_argument("--tiny-compute-report", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--print-approval-template", action="store_true")
    args = parser.parse_args()

    require_files(
        (
            args.config,
            args.question_manifest,
            args.alpaca_manifest,
            args.inspection_report,
            args.tiny_compute_report,
        )
    )
    config = read_json(args.config)
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Controlled config protocol mismatch.")
    commit, status = git_state(args.repo_dir.resolve())
    if commit != args.expected_git_commit:
        raise ValueError(
            f"Submission commit mismatch: current={commit} expected={args.expected_git_commit}."
        )
    inspection = validate_inspection(args.inspection_report.resolve(), config=config)
    tiny = validate_tiny_compute(args.tiny_compute_report.resolve(), config=config)
    if inspection["git_commit"] != commit:
        raise ValueError("Inspection evidence was produced by a different Git commit.")
    if tiny["real_model_git_commit"] != commit:
        raise ValueError("Real-model BF16 evidence was produced by a different Git commit.")
    evidence_hashes = {
        "config_sha256": sha256_file(args.config),
        "full_question_manifest_sha256": sha256_file(args.question_manifest),
        "alpaca_manifest_sha256": sha256_file(args.alpaca_manifest),
        "inspection_report_sha256": inspection["report_sha256"],
        "tiny_compute_report_sha256": tiny["report_sha256"],
    }
    if args.print_approval_template:
        print(
            json.dumps(
                approval_template(commit=commit, evidence_hashes=evidence_hashes),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if status:
        raise ValueError(
            "Full submission requires a clean worktree; current changes are:\n" + status
        )
    if args.approval is None or not args.approval.is_file():
        raise ValueError("Missing full-submission researcher approval JSON.")
    approval = validate_approval(
        args.approval.resolve(),
        commit=commit,
        evidence_hashes=evidence_hashes,
    )
    print(
        json.dumps(
            {
                "status": "valid_for_full_submission",
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": commit,
                "inspection": inspection,
                "tiny_compute": tiny,
                "approval": approval,
                **evidence_hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
