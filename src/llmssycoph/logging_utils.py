from __future__ import annotations

from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm


_RUN_LOG_PATH: Optional[Path] = None
_WARNING_LOG_PATH: Optional[Path] = None


def configure_run_logging(log_path: Path, warning_log_path: Optional[Path] = None) -> None:
    global _RUN_LOG_PATH, _WARNING_LOG_PATH

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    _RUN_LOG_PATH = log_path
    _WARNING_LOG_PATH = warning_log_path


def clear_run_logging() -> None:
    global _RUN_LOG_PATH, _WARNING_LOG_PATH
    _RUN_LOG_PATH = None
    _WARNING_LOG_PATH = None


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


def log_status(script_name: str, message: str) -> str:
    line = format_status(script_name, message)
    tqdm.write(line)
    _append_log_line(_RUN_LOG_PATH, line)
    return line


def warn_status(script_name: str, warning_code: str, message: str) -> str:
    line = format_warning(script_name, warning_code, message)
    tqdm.write(line)
    _append_log_line(_RUN_LOG_PATH, line)
    _append_log_line(_WARNING_LOG_PATH, line)
    return line


__all__ = [
    "clear_run_logging",
    "configure_run_logging",
    "format_warning",
    "format_status",
    "log_status",
    "tqdm_desc",
    "warn_status",
]
