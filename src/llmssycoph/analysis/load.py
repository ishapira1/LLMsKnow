from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from ..runtime import resolve_run_artifact_path
from .core import AnalysisContext, AnalysisError, AnalysisNotSupportedError, maybe_read_csv


MC_PROBABILITY_COLUMNS = tuple(f"P({letter})" for letter in "ABCDE")
REQUIRED_SAMPLED_COLUMNS = (
    "question_id",
    "template_type",
    "response",
    "P(selected)",
    "task_format",
) + MC_PROBABILITY_COLUMNS


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(column for column in required if column not in df.columns)


def _assert_multiple_choice(sampled_responses: pd.DataFrame) -> None:
    missing = _missing_columns(sampled_responses, REQUIRED_SAMPLED_COLUMNS)
    if missing:
        raise AnalysisNotSupportedError(
            "This analysis module currently supports only multiple-choice runs. "
            f"sampled_responses.csv is missing required MC columns: {missing}"
        )

    task_formats = {
        str(value).strip().lower()
        for value in sampled_responses["task_format"].dropna().tolist()
        if str(value).strip()
    }
    if task_formats and task_formats != {"multiple_choice"}:
        raise AnalysisNotSupportedError(
            "This analysis module currently supports only multiple-choice runs. "
            f"Found task_format values: {sorted(task_formats)}"
        )


def load_analysis_context(run_dir: Path | str) -> AnalysisContext:
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise AnalysisError(f"Run directory does not exist: {run_dir}")

    run_config_path = resolve_run_artifact_path(run_dir, "run_config")
    run_summary_path = resolve_run_artifact_path(run_dir, "run_summary")
    sampled_responses_path = resolve_run_artifact_path(run_dir, "sampled_responses")
    probe_scores_path = resolve_run_artifact_path(run_dir, "probe_scores_by_prompt")

    if not run_config_path.exists():
        raise AnalysisError(f"Missing run config: {run_config_path}")
    if not sampled_responses_path.exists():
        raise AnalysisError(f"Missing sampled responses: {sampled_responses_path}")

    run_config = _load_json(run_config_path)
    run_summary = _load_json(run_summary_path)
    sampled_responses = pd.read_csv(sampled_responses_path)
    _assert_multiple_choice(sampled_responses)

    analysis_dir = run_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    tables_dir = analysis_dir / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    return AnalysisContext(
        run_dir=run_dir,
        analysis_dir=analysis_dir,
        run_config=run_config,
        run_summary=run_summary,
        sampled_responses=sampled_responses,
        probe_scores_by_prompt=maybe_read_csv(probe_scores_path),
        plots_dir=plots_dir,
        tables_dir=tables_dir,
    )
