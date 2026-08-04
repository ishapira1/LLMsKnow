#!/usr/bin/env python3
"""Receipt-deduplicated campaign status emails."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("submitted", "manifests", "screening", "final"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--email-to", required=True)
    args = parser.parse_args()
    receipt = args.result_root / "email_receipts" / f"{args.stage}.json"
    if receipt.exists():
        return
    if args.stage == "submitted":
        subject = "[dense_behavioral_diversity] submitted | RUNNING"
        body = (
            "Dense behavioral-diversity campaign submitted.\n"
            "Design: 12 pressure families; 256 bad + 256 good examples per family; "
            "512 genuine correct updates; four family-weight profiles; 40 screened masks.\n"
            f"Artifacts: {args.result_root}\n"
        )
    elif args.stage == "manifests":
        audit = read(args.result_root / "inputs" / "input_audit.json")
        subject = "[dense_behavioral_diversity] dense manifests | COMPLETE"
        body = (
            f"Candidate questions: {audit['candidate_questions']}\n"
            f"Pruning rows: {audit['prune_rows']}\n"
            f"Preservation rows: {audit['preserve_rows']}\n"
            f"Per pressure family: {audit['rows_per_pressure_family_per_role']} bad + good\n"
            f"Genuine correction rows: {audit['guidance_counts']['correct_update']}\n"
            f"Total Alpaca preservation rows: {audit['alpaca_total']}\n"
            f"Artifacts: {args.result_root}\n"
        )
    elif args.stage == "screening":
        selection = read(args.result_root / "selection" / "finalists.json")
        subject = "[dense_behavioral_diversity] Pareto screening | COMPLETE"
        lines = [
            f"Screened masks: {selection['screened_candidates']}",
            f"General-guardrail masks: {selection['general_guardrail_candidates']}",
            f"Strict Pareto-feasible masks: {selection['full_pareto_candidates']}",
        ]
        for row in selection["finalists"]:
            metrics = row["metrics"]
            lines.append(
                f"{row['selection_role']}: {row['config_id']} | weights={row['mask_count']} | "
                f"wrong-adoption={pct(metrics['wrong_suggestion_adoption']['rate'])} | "
                f"doubt-flip={pct(metrics['doubt_correct_wrong_flip']['rate'])} | "
                f"valid-update={pct(metrics['correct_update']['rate'])} | "
                f"neutral={pct(metrics['neutral_accuracy'])}"
            )
        lines.append(f"Artifacts: {args.result_root}")
        body = "\n".join(lines) + "\n"
    else:
        report = read(args.result_root / "analysis" / "final_report.json")
        subject = "[dense_behavioral_diversity] full Pareto audit | COMPLETE"
        lines = [report["conclusion"], ""]
        for row in report["table"]:
            lines.append(
                f"{row['state_id']}: wrong-adoption={pct(row['wrong_suggestion_adoption'])}, "
                f"doubt-flip={pct(row['doubt_wrong_flip'])}, valid-update={pct(row['correct_update'])}, "
                f"neutral={pct(row['neutral_accuracy'])}, MMLU={pct(row['mmlu_accuracy'])}, "
                f"ICL={pct(row['icl_macro_accuracy'])}"
            )
        lines.append(f"Artifacts: {args.result_root}")
        body = "\n".join(lines) + "\n"
    body_hash = hashlib.sha256(body.encode()).hexdigest()
    completed = subprocess.run(
        ["/usr/bin/mail", "-s", subject, args.email_to],
        input=body,
        text=True,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    atomic_json(
        receipt,
        {
            "status": "sent",
            "stage": args.stage,
            "subject": subject,
            "recipient": args.email_to,
            "body_sha256": body_hash,
            "sent_at_epoch": int(time.time()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )


if __name__ == "__main__":
    main()
