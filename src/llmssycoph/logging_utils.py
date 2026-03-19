from __future__ import annotations

from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm


_RUN_LOG_PATH: Optional[Path] = None


def configure_run_logging(log_path: Path) -> None:
    global _RUN_LOG_PATH

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    _RUN_LOG_PATH = log_path


def clear_run_logging() -> None:
    global _RUN_LOG_PATH
    _RUN_LOG_PATH = None


def format_status(script_name: str, message: str) -> str:
    return f"[{Path(script_name).name}]: {message}"


def tqdm_desc(script_name: str, message: str) -> str:
    return format_status(script_name, message)


def log_status(script_name: str, message: str) -> str:
    line = format_status(script_name, message)
    tqdm.write(line)
    if _RUN_LOG_PATH is not None:
        with open(_RUN_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return line


__all__ = [
    "clear_run_logging",
    "configure_run_logging",
    "format_status",
    "log_status",
    "tqdm_desc",
]
