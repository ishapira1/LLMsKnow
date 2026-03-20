from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm


_RUN_LOG_PATH: Optional[Path] = None
_WARNING_LOG_PATH: Optional[Path] = None
_WARNING_RECORDS: List[Dict[str, Any]] = []
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RESET = "\033[0m"


def configure_run_logging(log_path: Path, warning_log_path: Optional[Path] = None) -> None:
    global _RUN_LOG_PATH, _WARNING_LOG_PATH, _WARNING_RECORDS

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    _RUN_LOG_PATH = log_path
    _WARNING_LOG_PATH = warning_log_path
    _WARNING_RECORDS = []


def clear_run_logging() -> None:
    global _RUN_LOG_PATH, _WARNING_LOG_PATH, _WARNING_RECORDS
    _RUN_LOG_PATH = None
    _WARNING_LOG_PATH = None
    _WARNING_RECORDS = []


def _append_log_line(path: Optional[Path], line: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def format_status(script_name: str, message: str) -> str:
    return f"[{Path(script_name).name}]: {message}"


def tqdm_desc(script_name: str, message: str) -> str:
    return format_status(script_name, message)


def _normalize_warning_code(warning_code: str) -> str:
    raw = str(warning_code or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def format_warning(script_name: str, warning_code: str, message: str) -> str:
    code = _normalize_warning_code(warning_code)
    if code:
        return f"[warning][{Path(script_name).name}][{code}]: {message}"
    return f"[warning][{Path(script_name).name}]: {message}"


def _record_warning(script_name: str, warning_code: str, message: str, line: str) -> None:
    normalized_code = _normalize_warning_code(warning_code)
    _WARNING_RECORDS.append(
        {
            "warning_index": int(len(_WARNING_RECORDS) + 1),
            "source": Path(script_name).name,
            "warning_code": normalized_code,
            "message": str(message),
            "formatted_line": line,
        }
    )


def get_run_warnings() -> List[Dict[str, Any]]:
    return [dict(record) for record in _WARNING_RECORDS]


def build_warning_summary_payload() -> Dict[str, Any]:
    warnings = get_run_warnings()
    by_code: Dict[str, Dict[str, Any]] = {}
    by_source: Dict[str, Dict[str, Any]] = {}

    for warning in warnings:
        code = str(warning.get("warning_code", "") or "")
        source = str(warning.get("source", "") or "")
        code_entry = by_code.setdefault(
            code,
            {
                "warning_code": code,
                "count": 0,
                "sources": set(),
                "latest_message": "",
            },
        )
        code_entry["count"] += 1
        code_entry["sources"].add(source)
        code_entry["latest_message"] = str(warning.get("message", "") or "")

        source_entry = by_source.setdefault(
            source,
            {
                "source": source,
                "count": 0,
                "warning_codes": set(),
            },
        )
        source_entry["count"] += 1
        if code:
            source_entry["warning_codes"].add(code)

    by_code_rows = []
    for row in by_code.values():
        by_code_rows.append(
            {
                "warning_code": row["warning_code"],
                "count": int(row["count"]),
                "sources": sorted(str(source) for source in row["sources"] if str(source)),
                "latest_message": row["latest_message"],
            }
        )
    by_source_rows = []
    for row in by_source.values():
        by_source_rows.append(
            {
                "source": row["source"],
                "count": int(row["count"]),
                "warning_codes": sorted(str(code) for code in row["warning_codes"] if str(code)),
            }
        )

    by_code_rows.sort(key=lambda row: (-int(row["count"]), str(row["warning_code"])))
    by_source_rows.sort(key=lambda row: (-int(row["count"]), str(row["source"])))

    return {
        "warning_summary_schema_version": 1,
        "total_warnings": int(len(warnings)),
        "unique_warning_codes": int(len(by_code_rows)),
        "unique_sources": int(len(by_source_rows)),
        "by_code": by_code_rows,
        "by_source": by_source_rows,
        "warnings": warnings,
    }


def _stdout_supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    force_color = os.environ.get("FORCE_COLOR")
    if force_color and force_color != "0":
        return True

    stream = sys.stdout
    return bool(
        hasattr(stream, "isatty")
        and stream.isatty()
        and str(os.environ.get("TERM", "")).lower() != "dumb"
    )


def _format_console_line(line: str, ansi_code: str) -> str:
    if not _stdout_supports_color():
        return line
    return f"{ansi_code}{line}{_ANSI_RESET}"


def log_status(script_name: str, message: str) -> str:
    line = format_status(script_name, message)
    tqdm.write(line)
    _append_log_line(_RUN_LOG_PATH, line)
    return line


def warn_status(script_name: str, warning_code: str, message: str) -> str:
    line = format_warning(script_name, warning_code, message)
    tqdm.write(_format_console_line(line, _ANSI_YELLOW))
    _append_log_line(_RUN_LOG_PATH, line)
    _append_log_line(_WARNING_LOG_PATH, line)
    _record_warning(script_name, warning_code, message, line)
    return line


def ok_status(script_name: str, message: str) -> str:
    line = format_status(script_name, message)
    tqdm.write(_format_console_line(line, _ANSI_GREEN))
    _append_log_line(_RUN_LOG_PATH, line)
    return line


__all__ = [
    "build_warning_summary_payload",
    "clear_run_logging",
    "configure_run_logging",
    "format_warning",
    "format_status",
    "get_run_warnings",
    "log_status",
    "ok_status",
    "tqdm_desc",
    "warn_status",
]
