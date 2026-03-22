from __future__ import annotations

import json
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict

from ..saving_manager import build_executive_summary_report_intro_markdown
from ..runtime import write_json_atomic
from .core import AnalysisContext, AnalysisError
from .functions import get_analysis_function_spec
from .load import load_analysis_context
from .specs import AnalysisCellSpec, AnalysisNotebookSpec, AnalysisSectionSpec, AnalysisSubsectionSpec, get_notebook_spec


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


def _dataset_name_from_context(ctx: AnalysisContext) -> str:
    for candidate in (
        ctx.run_summary.get("dataset_name"),
        ctx.run_config.get("dataset_name"),
    ):
        if str(candidate or "").strip():
            return str(candidate).strip()
    if "dataset" in ctx.sampled_responses.columns and not ctx.sampled_responses.empty:
        dataset_values = [
            str(value).strip()
            for value in ctx.sampled_responses["dataset"].dropna().astype(str).tolist()
            if str(value).strip()
        ]
        if dataset_values:
            return dataset_values[0]
    return ""


def _notebook_summary_payload(ctx: AnalysisContext) -> Dict[str, Any]:
    payload = dict(ctx.run_summary) if isinstance(ctx.run_summary, dict) else dict(ctx.run_summary or {})
    payload["run_name"] = str(payload.get("run_name") or ctx.run_name)
    payload["run_dir"] = str(payload.get("run_dir") or ctx.run_dir)
    payload["model_name"] = str(payload.get("model_name") or ctx.model_name or "")
    payload["dataset_name"] = str(payload.get("dataset_name") or _dataset_name_from_context(ctx) or "")

    headline_counts = payload.get("headline_counts")
    if not isinstance(headline_counts, dict):
        headline_counts = {}
    headline_counts.setdefault("sample_rows", int(len(ctx.sampled_responses)))
    headline_counts.setdefault(
        "question_count",
        int(ctx.sampled_responses["question_id"].nunique()) if "question_id" in ctx.sampled_responses.columns else 0,
    )
    payload["headline_counts"] = headline_counts
    return payload


def _build_notebook_intro_markdown(
    *,
    run_dir: Path,
    spec: AnalysisNotebookSpec,
    summary_payload: Dict[str, Any],
) -> str:
    report_intro = build_executive_summary_report_intro_markdown(summary_payload).rstrip()
    return "\n".join(
        [
            f"# {spec.title}",
            "",
            report_intro,
            "",
            "## Notebook Guide",
            f"Run directory: `{run_dir}`",
            "",
            "This notebook is spec-driven. Each analysis cell is isolated:",
            "- plots are saved to `analysis/plots/*.pdf`",
            "- tables are saved to `analysis/tables/*.csv`",
            "- cell failures are recorded in `analysis/tables/analysis_cell_failures.csv` without stopping later cells",
        ]
    )


def _format_section_heading(*, section_index: int, title: str) -> str:
    return f"## Section {section_index}: {title}"


def _format_subsection_heading(*, section_index: int, subsection_index: int, title: str) -> str:
    return f"### Section {section_index}.{subsection_index}: {title}"


def _context_has_probe_rows(ctx: AnalysisContext) -> bool:
    return not ctx.probe_scores_by_prompt.empty


def _cell_requires_probes(cell: AnalysisCellSpec) -> bool:
    if cell.kind not in {"table", "plot"}:
        return False
    return bool(get_analysis_function_spec(cell.function_name).requires_probes)


def _filter_spec_for_context(
    spec: AnalysisNotebookSpec,
    ctx: AnalysisContext,
) -> tuple[AnalysisNotebookSpec, str | None]:
    if _context_has_probe_rows(ctx):
        return spec, None

    filtered_sections: list[AnalysisSectionSpec] = []
    omitted_probe_cells = 0

    for section in spec.sections:
        filtered_subsections: list[AnalysisSubsectionSpec] = []
        section_had_executable_cells = False
        section_has_supported_executable_cells = False

        for subsection in section.subsections:
            original_executable_count = 0
            supported_executable_count = 0
            filtered_cells: list[AnalysisCellSpec] = []

            for cell in subsection.cells:
                if cell.kind in {"table", "plot"}:
                    original_executable_count += 1
                    if _cell_requires_probes(cell):
                        omitted_probe_cells += 1
                        continue
                    supported_executable_count += 1
                filtered_cells.append(cell)

            section_had_executable_cells = section_had_executable_cells or original_executable_count > 0
            section_has_supported_executable_cells = (
                section_has_supported_executable_cells or supported_executable_count > 0
            )

            if not filtered_cells:
                continue
            if original_executable_count > 0 and supported_executable_count == 0:
                continue

            filtered_subsections.append(
                AnalysisSubsectionSpec(
                    title=subsection.title,
                    intro=subsection.intro,
                    cells=filtered_cells,
                )
            )

        if not filtered_subsections:
            continue
        if section_had_executable_cells and not section_has_supported_executable_cells:
            continue

        filtered_sections.append(
            AnalysisSectionSpec(
                title=section.title,
                intro=section.intro,
                subsections=filtered_subsections,
            )
        )

    if omitted_probe_cells <= 0:
        return spec, None

    note_lines = [
        "## Probe Analysis Note",
        "",
        "Probe-only cells were omitted from this notebook because no prompt-level probe rows were found in `probes/probe_scores_by_prompt.csv`.",
        "The external sampling analysis remains available and should run normally for this multiple-choice commonsense run.",
    ]
    if bool(ctx.run_summary.get("sampling_only") or ctx.run_config.get("sampling_only")):
        note_lines.append("This run is marked as `sampling_only`, so skipping the probe section is expected.")

    filtered_spec = AnalysisNotebookSpec(
        name=spec.name,
        title=spec.title,
        sections=filtered_sections,
    )
    return filtered_spec, "\n".join(note_lines)


def build_analysis_notebook_payload(
    run_dir: Path | str,
    spec: AnalysisNotebookSpec | str,
    *,
    ctx: AnalysisContext | None = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if ctx is None:
        ctx = load_analysis_context(run_dir)
    if isinstance(spec, str):
        spec = get_notebook_spec(spec)
    spec, probe_note = _filter_spec_for_context(spec, ctx)

    cells: list[Dict[str, Any]] = [
        _markdown_cell(_build_notebook_intro_markdown(run_dir=run_dir, spec=spec, summary_payload=_notebook_summary_payload(ctx))),
    ]
    if probe_note:
        cells.append(_markdown_cell(probe_note))
    cells.append(
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
        )
    )

    for section_index, section in enumerate(spec.sections, start=1):
        section_text = _format_section_heading(section_index=section_index, title=section.title)
        if section.intro:
            section_text += f"\n\n{section.intro}"
        cells.append(_markdown_cell(section_text))
        for subsection_index, subsection in enumerate(section.subsections, start=1):
            subsection_text = _format_subsection_heading(
                section_index=section_index,
                subsection_index=subsection_index,
                title=subsection.title,
            )
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
    payload = build_analysis_notebook_payload(run_dir, spec_obj, ctx=ctx)
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
