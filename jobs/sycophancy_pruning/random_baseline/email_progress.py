#!/usr/bin/env python3
"""Archive and idempotently send random_baseline milestone/failure emails."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import random_baseline as rb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--milestone", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--to", default="itaishapira@g.harvard.edu")
    parser.add_argument("--mail-command", default="mail")
    parser.add_argument("--allow-resend", action="store_true")
    args = parser.parse_args()
    safe = "".join(character if character.isalnum() or character in "_-" else "_"
                   for character in args.milestone)
    body_path = args.result_root / "emails" / "bodies" / f"{safe}.txt"
    receipt_path = args.result_root / "emails" / "receipts" / f"{safe}.json"
    if receipt_path.exists() and not args.allow_resend:
        existing = rb.read_json(receipt_path)
        if existing.get("status") != "sent":
            raise RuntimeError(f"Existing failed receipt: {receipt_path}")
        print(json.dumps({"status": "reused", "receipt": str(receipt_path)}))
        return 0
    body = (f"Experiment: {rb.EXPERIMENT}\nMilestone: {args.milestone}\n"
            f"Time (UTC): {rb.utc_now()}\nHost: {os.uname().nodename}\n"
            f"Slurm job: {os.environ.get('SLURM_JOB_ID', 'local')}\n\n{args.body}\n")
    rb.atomic_text(body_path, body)
    completed = subprocess.run([args.mail_command, "-s", args.subject, args.to],
                               input=body, text=True, capture_output=True, check=False)
    receipt = {"status": "sent" if completed.returncode == 0 else "failed",
               "milestone": args.milestone, "to": args.to, "subject": args.subject,
               "body_path": str(body_path), "body_sha256": rb.sha256_file(body_path),
               "mail_command": args.mail_command, "returncode": completed.returncode,
               "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:],
               "sent_at_epoch": int(time.time())}
    rb.atomic_json(receipt_path, receipt)
    if completed.returncode:
        raise RuntimeError(f"Mail command failed: {completed.stderr}")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
