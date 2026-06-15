from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from ..cli import resolve_hf_cache_dir
from ..llm.registry import load_llm
from ..logging_utils import log_status, warn_status
from ..runtime import model_slug, utc_now_iso, write_json_atomic
from .cli import build_parser, finalize_args
from .data import _base, _letters, _load_prepared_groups, _messages
from .losses import choice_token_probabilities, final_token_logit_diagnostics


def _csv_set(value: str) -> Set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def build_preflight_parser():
    parser = build_parser()
    parser.description = "Preflight real-model strict-MC scoring before sycophancy pruning."
    parser.add_argument(
        "--sample_per_dataset",
        type=int,
        default=4,
        help="Neutral rows to score per dataset in addition to known source-example ids.",
    )
    parser.add_argument(
        "--known_source_example_ids",
        default="Mercury_7081270",
        help="Comma-separated source_example_id values that must be scored when present.",
    )
    return parser


def parse_preflight_args(argv: Optional[Sequence[str]] = None):
    return finalize_args(build_preflight_parser().parse_args(argv))


def _run_dir(args: Any) -> Path:
    run_name = str(args.run_name or "preflight").strip()
    path = Path(args.out_dir) / model_slug(args.model) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_rows(args: Any) -> List[Dict[str, Any]]:
    known_ids = _csv_set(args.known_source_example_ids)
    groups = _load_prepared_groups(args)
    rows: List[Dict[str, Any]] = []
    seen = set()
    per_dataset_counts: Dict[str, int] = {}

    def add_row(group: Mapping[str, Any], reason: str) -> None:
        rows_by_type = dict(group.get("rows_by_type", {}) or {})
        neutral = rows_by_type.get("neutral")
        if neutral is None:
            return
        base = _base(neutral)
        dataset = str(base.get("dataset", group.get("dataset", "")) or "")
        source_example_id = str(base.get("source_example_id", "") or "")
        key = (dataset, source_example_id)
        if key in seen:
            return
        choices = _letters(neutral)
        if not choices:
            return
        seen.add(key)
        rows.append(
            {
                "dataset": dataset,
                "source_example_id": source_example_id,
                "source_split": str(base.get("source_split", "") or ""),
                "condition": "neutral",
                "reason": reason,
                "choices": choices,
                "messages": _messages(neutral),
            }
        )

    for group in groups:
        rows_by_type = dict(group.get("rows_by_type", {}) or {})
        neutral = rows_by_type.get("neutral")
        if neutral is None:
            continue
        source_example_id = str(_base(neutral).get("source_example_id", "") or "")
        if source_example_id in known_ids:
            add_row(group, "known_source_example_id")

    for group in groups:
        rows_by_type = dict(group.get("rows_by_type", {}) or {})
        neutral = rows_by_type.get("neutral")
        if neutral is None:
            continue
        dataset = str(_base(neutral).get("dataset", group.get("dataset", "")) or "")
        count = per_dataset_counts.get(dataset, 0)
        if count >= int(args.sample_per_dataset):
            continue
        add_row(group, "dataset_sample")
        per_dataset_counts[dataset] = count + 1

    return rows


def run(args: Any) -> Path:
    run_dir = _run_dir(args)
    report_path = run_dir / "preflight_report.json"
    write_json_atomic(
        run_dir / "status.json",
        {
            "status": "running",
            "updated_at_utc": utc_now_iso(),
            "model": args.model,
            "datasets": list(args.datasets),
            "run_name": args.run_name,
            "torch_dtype": args.torch_dtype,
        },
    )
    try:
        hf_cache_dir = resolve_hf_cache_dir(args.hf_cache_dir)
        log_status(
            "pruning/preflight.py",
            f"loading model={args.model} device={args.resolved_device} torch_dtype={args.torch_dtype}",
        )
        llm = load_llm(
            args.model,
            device=args.resolved_device,
            device_map_auto=bool(args.device_map_auto),
            hf_cache_dir=hf_cache_dir,
            torch_dtype=args.torch_dtype,
        )
        model, tokenizer = llm.get_model_and_tokenizer()
        cases = _candidate_rows(args)
        if not cases:
            raise RuntimeError("Preflight found no neutral rows to score.")

        results = []
        failures = []
        for case in cases:
            row = {
                key: value
                for key, value in case.items()
                if key not in {"messages"}
            }
            try:
                diagnostics = final_token_logit_diagnostics(
                    model,
                    tokenizer,
                    case["messages"],
                    choices=case["choices"],
                )
                probabilities = choice_token_probabilities(
                    model,
                    tokenizer,
                    case["messages"],
                    choices=case["choices"],
                )
                row.update(diagnostics)
                row["probabilities"] = probabilities
                row["status"] = "ok"
                if not bool(diagnostics.get("logits_all_finite", False)):
                    row["status"] = "failed"
                    row["error"] = "non_finite_logits"
                    failures.append(row)
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                failures.append(row)
            results.append(row)

        report = {
            "status": "failed" if failures else "completed",
            "updated_at_utc": utc_now_iso(),
            "model": args.model,
            "datasets": list(args.datasets),
            "run_name": args.run_name,
            "torch_dtype": args.torch_dtype,
            "resolved_device": args.resolved_device,
            "hf_cache_dir": hf_cache_dir,
            "n_cases": len(results),
            "n_failures": len(failures),
            "results": results,
        }
        write_json_atomic(report_path, report)
        if failures:
            raise RuntimeError(f"Preflight failed on {len(failures)}/{len(results)} real-model scoring cases.")
        write_json_atomic(
            run_dir / "status.json",
            {
                "status": "completed",
                "updated_at_utc": utc_now_iso(),
                "model": args.model,
                "datasets": list(args.datasets),
                "run_name": args.run_name,
                "torch_dtype": args.torch_dtype,
                "n_cases": len(results),
                "report_path": str(report_path),
            },
        )
        log_status("pruning/preflight.py", f"completed preflight: {report_path}")
        return run_dir
    except Exception as exc:
        warn_status("pruning/preflight.py", "preflight_failed", str(exc))
        write_json_atomic(
            run_dir / "status.json",
            {
                "status": "failed",
                "updated_at_utc": utc_now_iso(),
                "model": args.model,
                "datasets": list(args.datasets),
                "run_name": args.run_name,
                "torch_dtype": args.torch_dtype,
                "error": str(exc),
                "report_path": str(report_path),
            },
        )
        raise


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_preflight_args(argv))


__all__ = ["build_preflight_parser", "main", "parse_preflight_args", "run"]
