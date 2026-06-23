from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _fmt_pct(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "NA"
    return f"{100.0 * numeric:+.1f} pp"


def _fmt_rate(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "NA"
    return f"{100.0 * numeric:.1f}%"


def _mean(values: Iterable[Any]) -> Optional[float]:
    clean = [value for value in (_safe_float(value) for value in values) if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _discover_comparison_dirs(comparison_root: Path) -> List[Path]:
    return sorted(
        path.parent
        for path in comparison_root.rglob("behavior_summary.csv")
        if path.is_file()
    )


def _load_summary_rows(comparison_dir: Path) -> List[Dict[str, Any]]:
    metadata_path = comparison_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    rows: List[Dict[str, Any]] = []
    for row in _read_csv(comparison_dir / "behavior_summary.csv"):
        enriched = dict(row)
        enriched["comparison_dir"] = str(comparison_dir)
        enriched["model"] = metadata.get("model", "")
        enriched["dataset_name"] = metadata.get("dataset_name", "")
        enriched["request_anti_sycophancy_request"] = metadata.get(
            "request_anti_sycophancy_request",
            comparison_dir.name,
        )
        rows.append(enriched)
    return rows


def load_main_rows(comparison_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for comparison_dir in _discover_comparison_dirs(comparison_root):
        for row in _load_summary_rows(comparison_dir):
            if row.get("metric_family") != "family_mitigation":
                continue
            rows.append(
                {
                    "dataset_name": row.get("dataset_name", ""),
                    "model": row.get("model", ""),
                    "request": row.get("request_anti_sycophancy_request", ""),
                    "template_type": row.get("template_type", ""),
                    "n_pairs": int(float(row.get("n_pairs", 0) or 0)),
                    "baseline_accuracy": _safe_float(row.get("baseline_accuracy")),
                    "request_accuracy": _safe_float(row.get("request_accuracy")),
                    "baseline_sycophancy_drop": _safe_float(row.get("baseline_sycophancy_drop")),
                    "request_sycophancy_drop": _safe_float(row.get("request_sycophancy_drop")),
                    "mitigation": _safe_float(row.get("mitigation")),
                    "comparison_dir": row.get("comparison_dir", ""),
                }
            )
    return rows


def build_aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("dataset_name", "")),
                str(row.get("model", "")),
                str(row.get("request", "")),
            )
        ].append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for (dataset_name, model, request), group_rows in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "dataset_name": dataset_name,
                "model": model,
                "request": request,
                "template_type": "ALL_NON_NEUTRAL_MEAN",
                "n_pairs": sum(int(row.get("n_pairs", 0) or 0) for row in group_rows),
                "baseline_accuracy": _mean(row.get("baseline_accuracy") for row in group_rows),
                "request_accuracy": _mean(row.get("request_accuracy") for row in group_rows),
                "baseline_sycophancy_drop": _mean(row.get("baseline_sycophancy_drop") for row in group_rows),
                "request_sycophancy_drop": _mean(row.get("request_sycophancy_drop") for row in group_rows),
                "mitigation": _mean(row.get("mitigation") for row in group_rows),
                "comparison_dir": "",
            }
        )
        random_all_rows = [row for row in group_rows if row.get("template_type") == "random_all"]
        if random_all_rows:
            row = random_all_rows[0]
            aggregate_rows.append(
                {
                    "dataset_name": dataset_name,
                    "model": model,
                    "request": request,
                    "template_type": "random_all",
                    "n_pairs": row.get("n_pairs", 0),
                    "baseline_accuracy": row.get("baseline_accuracy"),
                    "request_accuracy": row.get("request_accuracy"),
                    "baseline_sycophancy_drop": row.get("baseline_sycophancy_drop"),
                    "request_sycophancy_drop": row.get("request_sycophancy_drop"),
                    "mitigation": row.get("mitigation"),
                    "comparison_dir": row.get("comparison_dir", ""),
                }
            )
    return aggregate_rows


def render_email_body(
    *,
    comparison_root: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
) -> str:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "Anti-sycophancy request experiment summary",
        f"Created at UTC: {created}",
        f"Comparison root: {comparison_root}",
        "",
        "Main metric:",
        "  Subset = baseline no-request neutral is usable and correct.",
        "  baseline_drop = 1 - accuracy(biased family, no request)",
        "  request_drop = 1 - accuracy(biased family, request)",
        "  mitigation = baseline_drop - request_drop = request_accuracy - baseline_accuracy",
        "",
    ]
    if not aggregate_rows:
        lines.append("No family_mitigation rows were found.")
        return "\n".join(lines) + "\n"

    lines.append("Topline rows:")
    lines.append(
        "dataset | model | request | family | n | baseline_drop | request_drop | mitigation"
    )
    lines.append("-" * 110)
    for row in aggregate_rows:
        model_short = str(row.get("model", "")).replace("meta-llama/", "").replace("Qwen/", "")
        lines.append(
            " | ".join(
                [
                    str(row.get("dataset_name", "")),
                    model_short,
                    str(row.get("request", "")),
                    str(row.get("template_type", "")),
                    str(row.get("n_pairs", "")),
                    _fmt_rate(row.get("baseline_sycophancy_drop")),
                    _fmt_rate(row.get("request_sycophancy_drop")),
                    _fmt_pct(row.get("mitigation")),
                ]
            )
        )

    strongest = sorted(
        [row for row in detail_rows if _safe_float(row.get("mitigation")) is not None],
        key=lambda row: float(row.get("mitigation") or 0.0),
        reverse=True,
    )[:12]
    if strongest:
        lines.extend(["", "Largest positive mitigation rows:"])
        lines.append("dataset | model | request | family | n | mitigation")
        lines.append("-" * 90)
        for row in strongest:
            model_short = str(row.get("model", "")).replace("meta-llama/", "").replace("Qwen/", "")
            lines.append(
                " | ".join(
                    [
                        str(row.get("dataset_name", "")),
                        model_short,
                        str(row.get("request", "")),
                        str(row.get("template_type", "")),
                        str(row.get("n_pairs", "")),
                        _fmt_pct(row.get("mitigation")),
                    ]
                )
            )

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize anti-sycophancy request comparison outputs.")
    parser.add_argument("--comparison_root", required=True, help="Root containing comparison output directories.")
    parser.add_argument("--output_txt", required=True, help="Text summary path.")
    parser.add_argument("--output_csv", required=True, help="CSV summary path.")
    parser.add_argument("--detail_csv", default=None, help="Optional per-family detail CSV path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison_root = Path(args.comparison_root).expanduser().resolve()
    output_txt = Path(args.output_txt).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    detail_csv = Path(args.detail_csv).expanduser().resolve() if args.detail_csv else None

    detail_rows = load_main_rows(comparison_root)
    aggregate_rows = build_aggregate_rows(detail_rows)
    fieldnames = [
        "dataset_name",
        "model",
        "request",
        "template_type",
        "n_pairs",
        "baseline_accuracy",
        "request_accuracy",
        "baseline_sycophancy_drop",
        "request_sycophancy_drop",
        "mitigation",
        "comparison_dir",
    ]
    _write_csv(output_csv, aggregate_rows, fieldnames)
    if detail_csv is not None:
        _write_csv(detail_csv, detail_rows, fieldnames)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(
        render_email_body(
            comparison_root=comparison_root,
            aggregate_rows=aggregate_rows,
            detail_rows=detail_rows,
        ),
        encoding="utf-8",
    )
    print(
        "[anti-sycophancy-summary] wrote "
        f"text={output_txt} csv={output_csv} detail_rows={len(detail_rows)} aggregate_rows={len(aggregate_rows)}"
    )


if __name__ == "__main__":
    main()
