from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_model_probe_train_eval_breakdown_main_runs"
    / "claim3_model_probe_train_eval_breakdown_main_runs_long.csv"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "viewer"
    / "claim3_presentation"
    / "data"
    / "claim3_presentation_bundle_main_runs.json"
)

ALL_OPTION = "All"
AGGREGATION_MODES = ["equal_weight", "prompt_weighted"]
PROBE_TRAIN_ON_ORDER = [
    "neutral",
    "incorrect_suggestion",
    "suggest_correct",
    "doubt_correct",
]
EVAL_ON_ORDER = [
    "neutral",
    "incorrect_suggestion",
    "suggest_correct",
    "doubt_correct",
    "model_congruent_suggestion",
]
METRIC_SPECS = [
    {
        "id": "model_top1",
        "group_id": "model_performance",
        "family": "model",
        "metric_key": "top1",
        "label": "Top-1",
        "short_label": "Top-1",
    },
    {
        "id": "model_pairwise_k",
        "group_id": "model_performance",
        "family": "model",
        "metric_key": "pairwise_k",
        "label": "Pairwise K",
        "short_label": "Pairwise K",
    },
    {
        "id": "model_auc",
        "group_id": "model_performance",
        "family": "model",
        "metric_key": "auc",
        "label": "AUC",
        "short_label": "AUC",
    },
    {
        "id": "probe_top1",
        "group_id": "probe_performance",
        "family": "probe",
        "metric_key": "top1",
        "label": "Top-1",
        "short_label": "Top-1",
    },
    {
        "id": "probe_pairwise_k",
        "group_id": "probe_performance",
        "family": "probe",
        "metric_key": "pairwise_k",
        "label": "Pairwise K",
        "short_label": "Pairwise K",
    },
    {
        "id": "probe_auc",
        "group_id": "probe_performance",
        "family": "probe",
        "metric_key": "auc",
        "label": "AUC",
        "short_label": "AUC",
    },
]
METRIC_GROUPS = [
    {
        "id": "model_performance",
        "label": "Model performance",
        "metric_ids": ["model_top1", "model_pairwise_k", "model_auc"],
    },
    {
        "id": "probe_performance",
        "label": "Chosen probe performance",
        "metric_ids": ["probe_top1", "probe_pairwise_k", "probe_auc"],
    },
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a presentation-friendly JSON bundle from the per-run claim-3 "
            "model/probe train-eval breakdown export."
        ),
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help=f"Long-format breakdown CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Presentation bundle JSON path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to include. Default: test",
    )
    return parser


def _bool_from_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes"})


def _number_from_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return float(value)


def _sorted_values(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value).strip()})


def build_view_key(model: str, dataset: str, aggregation_mode: str) -> str:
    return f"model={model}|dataset={dataset}|aggregation={aggregation_mode}"


def load_long_breakdown(path: Path, *, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[df["split"].astype(str).eq(split)].copy()
    df["available"] = _bool_from_series(df["available"])
    for column in ["top1", "pairwise_k", "auc", "n_prompts", "n_candidate_rows"]:
        df[column] = _number_from_series(df[column])
    return df


def build_selector_options(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "models": [ALL_OPTION] + _sorted_values(df["model_name"].dropna().tolist()),
        "datasets": [ALL_OPTION] + _sorted_values(df["dataset"].dropna().tolist()),
        "probe_train_on": list(PROBE_TRAIN_ON_ORDER),
        "eval_on": list(EVAL_ON_ORDER),
        "aggregation_modes": list(AGGREGATION_MODES),
        "metrics": [metric["id"] for metric in METRIC_SPECS],
    }


def filter_selection(df: pd.DataFrame, *, model: str, dataset: str) -> pd.DataFrame:
    subset = df.copy()
    if model != ALL_OPTION:
        subset = subset.loc[subset["model_name"].astype(str).eq(model)].copy()
    if dataset != ALL_OPTION:
        subset = subset.loc[subset["dataset"].astype(str).eq(dataset)].copy()
    return subset


def aggregate_metric_rows(rows: pd.DataFrame, *, aggregation_mode: str) -> dict[str, Any]:
    contributing = rows.loc[rows["available"].astype(bool)].copy()
    if contributing.empty:
        return {
            "available": False,
            "runs_contributing": 0,
            "prompt_weight_total": 0.0,
            "candidate_rows_total": 0.0,
            "metrics": {"top1": None, "pairwise_k": None, "auc": None},
        }

    weights = contributing["n_prompts"].fillna(0.0).astype(float)
    prompt_weight_total = float(weights.sum())
    candidate_rows_total = float(contributing["n_candidate_rows"].fillna(0.0).sum())
    metrics: dict[str, float | None] = {}
    for metric_key in ["top1", "pairwise_k", "auc"]:
        values = contributing[metric_key].astype(float)
        if aggregation_mode == "prompt_weighted" and prompt_weight_total > 0:
            metrics[metric_key] = float((values * weights).sum() / prompt_weight_total)
        else:
            metrics[metric_key] = float(values.mean())

    return {
        "available": True,
        "runs_contributing": int(contributing["run_id"].astype(str).nunique()),
        "prompt_weight_total": prompt_weight_total,
        "candidate_rows_total": candidate_rows_total,
        "metrics": metrics,
    }


def build_view(subset: pd.DataFrame, *, model: str, dataset: str, aggregation_mode: str) -> dict[str, Any]:
    model_rows = subset.loc[subset["row_kind"].astype(str).eq("model")].copy()
    probe_rows = subset.loc[subset["row_kind"].astype(str).eq("probe")].copy()
    selected_run_ids = sorted(model_rows["run_id"].dropna().astype(str).unique().tolist())
    selected_run_count = len(selected_run_ids)

    rows: list[dict[str, Any]] = []
    semantic_index = 0
    for eval_on in EVAL_ON_ORDER:
        eval_model_rows = model_rows.loc[model_rows["eval_on"].astype(str).eq(eval_on)].copy()
        aggregated_model = aggregate_metric_rows(eval_model_rows, aggregation_mode=aggregation_mode)
        aggregated_model["runs_selected"] = selected_run_count

        for probe_train_on in PROBE_TRAIN_ON_ORDER:
            semantic_index += 1
            eval_probe_rows = probe_rows.loc[
                probe_rows["eval_on"].astype(str).eq(eval_on)
                & probe_rows["trained_on"].astype(str).eq(probe_train_on)
            ].copy()
            aggregated_probe = aggregate_metric_rows(eval_probe_rows, aggregation_mode=aggregation_mode)
            aggregated_probe["runs_selected"] = selected_run_count

            rows.append(
                {
                    "row_key": f"{probe_train_on}__{eval_on}",
                    "probe_train_on": probe_train_on,
                    "eval_on": eval_on,
                    "semantic_order": semantic_index,
                    "model": aggregated_model,
                    "probe": aggregated_probe,
                }
            )

    return {
        "selection": {
            "model": model,
            "dataset": dataset,
            "aggregation_mode": aggregation_mode,
        },
        "selected_run_ids": selected_run_ids,
        "selected_run_count": selected_run_count,
        "rows": rows,
    }


def build_views(df: pd.DataFrame, selector_options: dict[str, Any]) -> dict[str, Any]:
    views: dict[str, Any] = {}
    for model in selector_options["models"]:
        for dataset in selector_options["datasets"]:
            subset = filter_selection(df, model=model, dataset=dataset)
            for aggregation_mode in AGGREGATION_MODES:
                key = build_view_key(model, dataset, aggregation_mode)
                views[key] = build_view(subset, model=model, dataset=dataset, aggregation_mode=aggregation_mode)
    return views


def build_bundle_from_long_df(
    df: pd.DataFrame,
    *,
    source_long_path: str,
    split: str,
) -> dict[str, Any]:
    selector_options = build_selector_options(df)
    return {
        "created_at_utc": utc_now_iso(),
        "source_long_path": source_long_path,
        "split": split,
        "selector_options": selector_options,
        "metric_specs": METRIC_SPECS,
        "metric_groups": METRIC_GROUPS,
        "semantic_orders": {
            "probe_train_on": list(PROBE_TRAIN_ON_ORDER),
            "eval_on": list(EVAL_ON_ORDER),
        },
        "default_state": {
            "model": ALL_OPTION,
            "dataset": ALL_OPTION,
            "aggregation_mode": AGGREGATION_MODES[0],
            "visible_metric_ids": [metric["id"] for metric in METRIC_SPECS],
            "visible_probe_train_on": list(PROBE_TRAIN_ON_ORDER),
            "visible_eval_on": list(EVAL_ON_ORDER),
        },
        "views": build_views(df, selector_options),
    }


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_json(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, float):
        return _optional_float(value)
    return value


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_long_breakdown(input_path, split=str(args.split))
    bundle = build_bundle_from_long_df(
        df,
        source_long_path=str(input_path),
        split=str(args.split),
    )
    output_path.write_text(json.dumps(_sanitize_for_json(bundle), indent=2), encoding="utf-8")

    print(f"Wrote presentation bundle to {output_path}")
    print(
        json.dumps(
            {
                "models": bundle["selector_options"]["models"],
                "datasets": bundle["selector_options"]["datasets"],
                "view_count": len(bundle["views"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
