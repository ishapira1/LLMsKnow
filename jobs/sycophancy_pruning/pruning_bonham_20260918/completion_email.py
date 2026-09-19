#!/usr/bin/env python3
"""Send one authenticated Bonham completion email after the final audit passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Mapping

from core import atomic_json, atomic_text, read_json, sha256_file


class CompletionEmailError(RuntimeError):
    pass


def _identity(root: Path, recipient: str) -> Mapping[str, Any]:
    audit_path = root / "audit" / "COMPLETE.json"
    audit = read_json(audit_path)
    if audit.get("status") != "complete":
        raise CompletionEmailError("Final audit receipt does not report complete status")
    return {
        "experiment": str(audit.get("experiment", "")),
        "recipient": recipient,
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
        "report_complete_sha256": str(audit.get("report_complete_sha256", "")),
    }


def build_body(root: Path, identity: Mapping[str, Any]) -> str:
    audit = read_json(Path(str(identity["audit_path"])))
    models = ", ".join(sorted(dict(audit.get("models", {}))))
    return "\n".join(
        (
            "The Bonham paper-ready sparse-pruning campaign has passed its final audit.",
            "",
            f"Experiment: {identity['experiment']}",
            f"Models: {models}",
            f"States: {len(audit.get('state_ids', []))}",
            f"Raw evaluation records: {audit.get('raw_evaluation_record_count')}",
            f"Protection fraction p: {audit.get('protection_fraction')}",
            f"Every primary mask has exactly 1,000 weights: {audit.get('primary_masks_exactly_1000')}",
            f"Final audit: {identity['audit_path']}",
            f"Audit SHA-256: {identity['audit_sha256']}",
            f"Paper-ready reports: {(root / 'reports').resolve()}",
            "",
            "This message was emitted only after audit/COMPLETE.json was authenticated.",
        )
    ) + "\n"


def _send_slurm_notification(
    *, root: Path, recipient: str, body: str, sbatch_binary: str
) -> Mapping[str, Any]:
    """Request and await a dedicated Slurm END notification.

    Cannon's interactive ``mail`` command can exist without an SMTP
    configuration.  Slurm mail is the authenticated notification path that is
    already used by every Bonham batch stage, so prefer it whenever ``sbatch``
    is available.  ``--wait`` means the receipt is written only after the
    notification job has completed and Slurm has emitted its END event.
    """

    notification_dir = root / "notifications"
    body_path = notification_dir / "FINAL_EMAIL_BODY.txt"
    atomic_text(body_path, body)
    output_pattern = notification_dir / "final_email_slurm_%j.out"
    command = [
        sbatch_binary,
        "--parsable",
        "--wait",
        "--account=barak_lab",
        "--partition=test",
        "--time=00:02:00",
        "--mem=256M",
        "--job-name=bonh_final_pass",
        "--mail-type=END,FAIL",
        f"--mail-user={recipient}",
        f"--output={output_pattern}",
        f"--error={output_pattern}",
        f"--wrap=/bin/cat {shlex.quote(str(body_path))}",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise CompletionEmailError(
            f"Slurm notification returned an invalid job id: {completed.stdout!r}"
        )
    return {
        "delivery": "slurm_end_notification",
        "mailer": sbatch_binary,
        "slurm_notification_job_id": job_id,
        "notification_body_path": str(body_path.resolve()),
    }


def _send_direct_mail(
    *, recipient: str, subject: str, body: str, mail_binary: str
) -> Mapping[str, Any]:
    subprocess.run(
        [mail_binary, "-s", subject, recipient],
        input=body,
        text=True,
        check=True,
    )
    return {"delivery": "direct_mail", "mailer": mail_binary}


def send_completion_email(args: argparse.Namespace) -> None:
    root = Path(args.result_root).resolve()
    identity = _identity(root, str(args.recipient))
    ledger = root / "notifications" / "FINAL_EMAIL.json"
    if ledger.is_file():
        existing = read_json(ledger)
        existing_identity = dict(existing.get("identity", {}))
        if existing.get("status") == "sent" and existing_identity == dict(identity):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return
        if existing_identity != dict(identity):
            raise CompletionEmailError(
                "Completion-email ledger belongs to a different authenticated audit"
            )

    subject = str(args.subject)
    body = build_body(root, identity)
    sbatch_binary = shutil.which("sbatch")
    mail_binary = shutil.which("mail") or shutil.which("mailx")
    if sbatch_binary is None and mail_binary is None:
        raise CompletionEmailError("Neither sbatch nor mail/mailx is available")
    pending = {
        "status": "sending",
        "identity": dict(identity),
        "subject": subject,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(ledger, pending)
    try:
        if sbatch_binary is not None:
            delivery = _send_slurm_notification(
                root=root,
                recipient=str(args.recipient),
                body=body,
                sbatch_binary=str(sbatch_binary),
            )
        else:
            assert mail_binary is not None
            delivery = _send_direct_mail(
                recipient=str(args.recipient),
                subject=subject,
                body=body,
                mail_binary=str(mail_binary),
            )
    except Exception as error:
        atomic_json(
            ledger,
            {
                **pending,
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    complete = {
        **pending,
        **delivery,
        "status": "sent",
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    atomic_json(ledger, complete)
    print(json.dumps(complete, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--recipient", default="itaishapira@g.harvard.edu")
    parser.add_argument(
        "--subject", default="Bonham sparse-pruning experiment: final audit passed"
    )
    send_completion_email(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
