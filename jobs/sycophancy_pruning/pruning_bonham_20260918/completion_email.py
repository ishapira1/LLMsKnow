#!/usr/bin/env python3
"""Send one authenticated Bonham completion email after the final audit passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

from core import atomic_json, read_json, sha256_file


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
        raise CompletionEmailError(
            "Completion-email ledger exists without an identical authenticated sent receipt"
        )

    subject = str(args.subject)
    body = build_body(root, identity)
    mail_binary = shutil.which("mail") or shutil.which("mailx")
    if mail_binary is None:
        raise CompletionEmailError("Neither mail nor mailx is available")
    pending = {
        "status": "sending",
        "identity": dict(identity),
        "subject": subject,
        "mailer": str(mail_binary),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(ledger, pending)
    subprocess.run(
        [str(mail_binary), "-s", subject, str(args.recipient)],
        input=body,
        text=True,
        check=True,
    )
    complete = {
        **pending,
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
