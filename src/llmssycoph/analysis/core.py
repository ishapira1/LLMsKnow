from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


class AnalysisError(RuntimeError):
    pass


class AnalysisNotSupportedError(AnalysisError):
    pass


@dataclass
class AnalysisContext:
    run_dir: Path
    analysis_dir: Path
    run_config: Mapping[str, Any]
    run_summary: Mapping[str, Any]
    sampled_responses: pd.DataFrame
    probe_scores_by_prompt: pd.DataFrame
    plots_dir: Path
    tables_dir: Path
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return str(self.run_config.get("model") or self.run_summary.get("model_name") or "")

    @property
    def run_name(self) -> str:
        return self.run_dir.name

    def save_table(self, frame: pd.DataFrame, stem: str) -> Path:
        path = self.tables_dir / f"{stem}.csv"
        frame.to_csv(path, index=False)
        return path

    def save_plot(self, figure: Any, stem: str) -> Path:
        path = self.plots_dir / f"{stem}.pdf"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        return path

    def require_probe_scores(self) -> pd.DataFrame:
        if self.probe_scores_by_prompt.empty:
            raise AnalysisNotSupportedError(
                "This analysis requires probes/probe_scores_by_prompt.csv, but no probe scores were found."
            )
        return self.probe_scores_by_prompt


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
