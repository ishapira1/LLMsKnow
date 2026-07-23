"""Local, deterministic loading for Alpaca utility-evaluation prompts."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


LEGACY_ALPACA_EVAL_DATA = Path(
    "../data/alpaca_cleaned_no_safety_train_raw_filtered_from_0_5_26.csv"
)

_PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

_PROMPT_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def resolve_alpaca_eval_data_path(
    explicit_path: str | Path | None,
    *,
    manifest_run: bool,
) -> Path:
    """Resolve the requested data path, restricting the legacy fallback.

    Paper-faithful manifest runs must declare their utility-evaluation input so
    that the evaluation is reproducible and independent of the working
    directory.  Legacy, non-manifest invocations retain the historical path.
    """

    if explicit_path is not None:
        value = str(explicit_path).strip()
        if value:
            return Path(value).expanduser()
    if manifest_run:
        raise ValueError(
            "--alpaca_eval_data is required with manifest pruning when "
            "--eval_alpaca is enabled"
        )
    return LEGACY_ALPACA_EVAL_DATA


def render_alpaca_user_prompt(row: Mapping[str, Any], *, row_label: str) -> str:
    """Render a canonical Alpaca instruction/input row as a user prompt.

    An explicit ``raw_prompt`` is authoritative for frozen evaluation
    manifests. A legacy ``prompt`` column is accepted verbatim for old CSVs;
    otherwise standard ``instruction``/``input`` fields are rendered.
    """

    raw_prompt = _clean_text(row.get("raw_prompt"))
    if raw_prompt:
        return raw_prompt

    instruction = _clean_text(row.get("instruction"))
    if instruction:
        input_text = _clean_text(row.get("input"))
        if input_text:
            return _PROMPT_WITH_INPUT.format(
                instruction=instruction,
                input=input_text,
            )
        return _PROMPT_WITHOUT_INPUT.format(instruction=instruction)

    legacy_prompt = _clean_text(row.get("prompt"))
    if legacy_prompt:
        return legacy_prompt
    raise ValueError(
        f"Alpaca row {row_label} must contain a non-empty 'instruction' "
        "(preferred) or legacy 'prompt' field"
    )


def load_alpaca_eval_prompts(
    path: str | Path,
    *,
    nsamples: int,
    seed: int,
) -> list[str]:
    """Load and deterministically sample up to ``nsamples`` local prompts."""

    if nsamples <= 0:
        raise ValueError(f"nsamples must be positive, got {nsamples}")

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Alpaca evaluation data does not exist: {source}")

    rows = _load_rows(source)
    if not rows:
        raise ValueError(f"Alpaca evaluation data contains no rows: {source}")

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = indices[: min(nsamples, len(indices))]
    return [
        render_alpaca_user_prompt(rows[index], row_label=f"{source}:{index + 1}")
        for index in selected
    ]


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Alpaca JSON must contain a top-level array: {path}")
        return _validate_rows(payload, path)

    if suffix in {".jsonl", ".ndjson"}:
        rows: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}: {error.msg}"
                    ) from error
        return _validate_rows(rows, path)

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return _validate_rows(list(csv.DictReader(handle)), path)

    raise ValueError(
        f"Unsupported Alpaca evaluation data format {suffix!r} for {path}; "
        "expected .json, .jsonl/.ndjson, or .csv"
    )


def _validate_rows(rows: Sequence[Any], path: Path) -> list[Mapping[str, Any]]:
    validated: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Alpaca row {path}:{index} must be a JSON/object row")
        validated.append(row)
    return validated


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
