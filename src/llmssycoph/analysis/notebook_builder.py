from __future__ import annotations

import json
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict

from ..runtime import write_json_atomic
from .core import AnalysisError
from .load import load_analysis_context
from .specs import AnalysisNotebookSpec, get_notebook_spec


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _split_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def _markdown_cell(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": _split_lines(text),
    }


def _code_cell(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _split_lines(text),
    }


def _repr_kwargs(kwargs: Dict[str, Any]) -> str:
    if not kwargs:
        return ""
    return ", " + ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


def build_analysis_notebook_payload(run_dir: Path | str, spec: AnalysisNotebookSpec | str) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if isinstance(spec, str):
        spec = get_notebook_spec(spec)

    cells: list[Dict[str, Any]] = [
        _markdown_cell(
            "\n".join(
                [
                    f"# {spec.title}",
                    "",
                    f"Run directory: `{run_dir}`",
                    "",
                    "This notebook is spec-driven. Each analysis cell is isolated:",
                    "- plots are saved to `analysis/plots/*.pdf`",
                    "- tables are saved to `analysis/tables/*.csv`",
                    "- cell failures are recorded in `analysis/tables/analysis_cell_failures.csv` without stopping later cells",
                ]
            )
        ),
        _code_cell(
            "\n".join(
                [
                    "import sys",
                    "from pathlib import Path",
                    "",
                    f"RUN_DIR = Path({str(run_dir)!r})",
                    "for _candidate in [RUN_DIR, *RUN_DIR.parents]:",
                    "    _src_dir = _candidate / 'src'",
                    "    if (_src_dir / 'llmssycoph').exists():",
                    "        _src_dir_str = str(_src_dir)",
                    "        if _src_dir_str not in sys.path:",
                    "            sys.path.insert(0, _src_dir_str)",
                    "        break",
                    "else:",
                    "    raise ModuleNotFoundError('Could not locate repo src/llmssycoph from the run directory.')",
                    "",
                    "from llmssycoph.analysis import load_analysis_context, safe_display_analysis_operation",
                    "",
                    "ctx = load_analysis_context(RUN_DIR)",
                    "ctx.analysis_dir",
                ]
            )
        ),
    ]

    for section in spec.sections:
        section_text = f"## {section.title}"
        if section.intro:
            section_text += f"\n\n{section.intro}"
        cells.append(_markdown_cell(section_text))
        for subsection in section.subsections:
            subsection_text = f"### {subsection.title}"
            if subsection.intro:
                subsection_text += f"\n\n{subsection.intro}"
            cells.append(_markdown_cell(subsection_text))
            for cell in subsection.cells:
                if cell.kind == "markdown":
                    cells.append(_markdown_cell(cell.text))
                    continue
                if cell.kind not in {"table", "plot"}:
                    raise AnalysisError(f"Unsupported cell kind: {cell.kind}")
                if cell.text:
                    cells.append(_markdown_cell(cell.text))
                notebook_cell_id = _cell_id()
                code = (
                    f"_ = safe_display_analysis_operation(\n"
                    f"    ctx,\n"
                    f"    {cell.function_name!r},\n"
                    f"    cell_id={notebook_cell_id!r},\n"
                    f"    output_stem={cell.output_stem!r}{_repr_kwargs(dict(cell.kwargs))},\n"
                    f")"
                )
                cells.append(
                    {
                        "cell_type": "code",
                        "id": notebook_cell_id,
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": _split_lines(code),
                    }
                )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_analysis_notebook(
    run_dir: Path | str,
    *,
    spec: AnalysisNotebookSpec | str = "full_mc_report",
    notebook_name: str | None = None,
) -> Path:
    run_dir = Path(run_dir).resolve()
    ctx = load_analysis_context(run_dir)
    if isinstance(spec, str):
        spec_obj = get_notebook_spec(spec)
    else:
        spec_obj = spec
    notebook_name = notebook_name or f"analysis_{spec_obj.name}.ipynb"
    notebook_path = ctx.analysis_dir / notebook_name
    payload = build_analysis_notebook_payload(run_dir, spec_obj)
    notebook_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return notebook_path


def safe_generate_analysis_notebook(
    run_dir: Path | str,
    *,
    spec: AnalysisNotebookSpec | str = "full_mc_report",
    notebook_name: str | None = None,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    status_path = run_dir / "analysis" / "analysis_notebook_status.json"
    try:
        load_analysis_context(run_dir)
        notebook_path = build_analysis_notebook(run_dir, spec=spec, notebook_name=notebook_name)
        status = {
            "status": "completed",
            "run_dir": str(run_dir),
            "notebook_path": str(notebook_path),
            "spec": spec.name if isinstance(spec, AnalysisNotebookSpec) else str(spec),
        }
        write_json_atomic(status_path, status)
        return status
    except Exception as exc:
        status = {
            "status": "failed",
            "run_dir": str(run_dir),
            "spec": spec.name if isinstance(spec, AnalysisNotebookSpec) else str(spec),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(status_path, status)
        if raise_on_error:
            raise
        return status
