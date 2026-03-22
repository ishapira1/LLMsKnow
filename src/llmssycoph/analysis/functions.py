from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from ..runtime import utc_now_iso
from .core import AnalysisContext, AnalysisError, AnalysisNotSupportedError
from .dataframes import (
    build_all_probe_layer_metrics_df,
    build_candidate_probability_long_df,
    build_chosen_probe_summary_df,
    build_neutral_sampled_responses_df,
    build_paired_external_df,
    build_paired_probe_df,
    build_probe_option_long_df,
    build_probe_scores_df,
    build_sampled_responses_df,
)
from .utils import (
    bootstrap_ci,
    correct_in_top_k,
    entropy_and_effective_responses,
    js_divergence,
    probability_of_option,
    probability_rank,
    quantile_bucket_labels,
    reliability_curve,
    row_probability_values,
    score_rank,
    score_value,
    top1_top2_from_probs,
    total_variation,
)


sns.set_style("white")


@dataclass(frozen=True)
class AnalysisFunctionSpec:
    name: str
    output_kind: str
    description: str
    requires_probes: bool = False
    requires_labels: bool = False
    requires_bias_target: bool = False
    supported_task_formats: tuple[str, ...] = ("multiple_choice",)


ANALYSIS_FUNCTIONS: Dict[str, Callable[..., Any]] = {}
ANALYSIS_FUNCTION_SPECS: Dict[str, AnalysisFunctionSpec] = {}

DEFAULT_COLORS = {
    "primary": "#73b3ab",
    "secondary": "#d4651a",
    "neutral": "#b89b6f",
    "fallback": "#4f6d7a",
}

BIAS_COLORS = {
    "neutral": "#4f6d7a",
    "incorrect_suggestion": "#d4651a",
    "doubt_correct": "#b89b6f",
    "suggest_correct": "#73b3ab",
}

TEMPLATE_ORDER = ["neutral", "incorrect_suggestion", "doubt_correct", "suggest_correct"]
BIAS_ONLY_ORDER = [template for template in TEMPLATE_ORDER if template != "neutral"]
PROBE_NAME_TO_TEMPLATE = {
    "probe_no_bias": "neutral",
    "probe_bias_incorrect_suggestion": "incorrect_suggestion",
    "probe_bias_doubt_correct": "doubt_correct",
    "probe_bias_suggest_correct": "suggest_correct",
}
TEMPLATE_TO_PROBE_NAME = {value: key for key, value in PROBE_NAME_TO_TEMPLATE.items()}
DISPLAY_LABELS = {
    "neutral": "Neutral",
    "incorrect_suggestion": "Incorrect Suggestion",
    "doubt_correct": "Doubt Correct",
    "suggest_correct": "Suggest Correct",
}


def _ordered_templates(values: pd.Series | list[str], *, include_neutral: bool = True) -> list[str]:
    present = {str(value) for value in values if str(value)}
    order = TEMPLATE_ORDER if include_neutral else BIAS_ONLY_ORDER
    return [template for template in order if template in present]


def _filter_usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "usable_for_metrics" not in df.columns:
        return df.copy()
    usable = df["usable_for_metrics"].astype(str).str.strip().str.lower().eq("true")
    return df.loc[usable].copy()


def _coerce_correctness(frame: pd.DataFrame, column: str = "correctness") -> pd.DataFrame:
    out = frame.copy()
    if column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _prepare_external_rows(ctx: AnalysisContext) -> pd.DataFrame:
    df = _filter_usable_rows(build_sampled_responses_df(ctx))
    df = _coerce_correctness(df)
    if df.empty:
        return df
    prob_values = row_probability_values(df)
    top1, top2 = top1_top2_from_probs(prob_values)
    _, effective_responses = entropy_and_effective_responses(prob_values)
    df["top1_probability"] = top1
    df["top2_probability"] = top2
    df["top1_margin"] = top1 - top2
    df["effective_responses"] = effective_responses
    df["accuracy_at_2"] = correct_in_top_k(df, 2).astype(int)
    return df


def _probability_with_prefix(frame: pd.DataFrame, prefix: str, option_series: pd.Series) -> pd.Series:
    outputs = []
    for idx, option in option_series.astype(str).str.strip().str.upper().items():
        column = f"{prefix}{option}"
        outputs.append(frame.loc[idx, column] if column in frame.columns else np.nan)
    return pd.Series(outputs, index=frame.index, dtype=float)


def _paired_external_metrics(ctx: AnalysisContext) -> pd.DataFrame:
    paired = build_paired_external_df(ctx)
    if paired.empty:
        return paired
    paired = paired.copy()
    for column in ("p_correct_x", "p_correct_xprime", "p_selected_x", "p_selected_xprime"):
        if column in paired.columns:
            paired[column] = pd.to_numeric(paired[column], errors="coerce")
    for column in ("correctness_x", "correctness_xprime"):
        if column in paired.columns:
            paired[column] = pd.to_numeric(paired[column], errors="coerce")

    prob_x = paired[[f"p_x_{letter}" for letter in "ABCDE"]].to_numpy(dtype=float, copy=True)
    prob_xprime = paired[[f"p_xprime_{letter}" for letter in "ABCDE"]].to_numpy(dtype=float, copy=True)
    top1_x, top2_x = top1_top2_from_probs(prob_x)
    top1_xprime, top2_xprime = top1_top2_from_probs(prob_xprime)
    paired["top1_probability_x"] = top1_x
    paired["top1_probability_xprime"] = top1_xprime
    paired["top1_margin_x"] = top1_x - top2_x
    paired["top1_margin_xprime"] = top1_xprime - top2_xprime
    paired["effective_responses_x"] = entropy_and_effective_responses(prob_x)[1]
    paired["effective_responses_xprime"] = entropy_and_effective_responses(prob_xprime)[1]
    paired["js_divergence"] = js_divergence(prob_x, prob_xprime)
    paired["total_variation"] = total_variation(prob_x, prob_xprime)
    paired["delta_p_correct"] = paired["p_correct_x"] - paired["p_correct_xprime"]
    denom = paired["p_correct_x"] + paired["p_correct_xprime"]
    paired["delta_relative_p_correct"] = np.where(
        denom > 0,
        (paired["p_correct_x"] - paired["p_correct_xprime"]) / denom,
        np.nan,
    )
    paired["same_answer"] = paired["response_x"].eq(paired["response_xprime"])
    paired["response_changed"] = ~paired["same_answer"]
    paired["became_correct"] = paired["correctness_x"].eq(0) & paired["correctness_xprime"].eq(1)
    paired["became_incorrect"] = paired["correctness_x"].eq(1) & paired["correctness_xprime"].eq(0)

    if "incorrect_letter" in paired.columns:
        paired["bias_target_letter"] = np.where(
            paired["bias_type"].eq("incorrect_suggestion"),
            paired["incorrect_letter"],
            np.where(paired["bias_type"].eq("suggest_correct"), paired["correct_letter"], np.nan),
        )
    else:
        paired["bias_target_letter"] = np.where(paired["bias_type"].eq("suggest_correct"), paired["correct_letter"], np.nan)

    paired["p_bias_target_x"] = _probability_with_prefix(paired, "p_x_", paired["bias_target_letter"])
    paired["p_bias_target_xprime"] = _probability_with_prefix(paired, "p_xprime_", paired["bias_target_letter"])
    paired["delta_p_bias_target"] = paired["p_bias_target_xprime"] - paired["p_bias_target_x"]
    paired["m_bias_external_x"] = _probability_with_prefix(paired, "p_x_", paired["correct_letter"]) - paired["p_bias_target_x"]
    paired["adopts_bias_target"] = paired["response_xprime"].eq(paired["bias_target_letter"])

    same_wrong = (
        paired["correctness_x"].eq(0)
        & paired["correctness_xprime"].eq(0)
        & paired["response_x"].eq(paired["response_xprime"])
    )
    different_wrong = (
        paired["correctness_x"].eq(0)
        & paired["correctness_xprime"].eq(0)
        & paired["response_x"].ne(paired["response_xprime"])
    )
    paired["flip_category"] = np.select(
        [
            paired["correctness_x"].eq(1) & paired["correctness_xprime"].eq(0),
            paired["correctness_x"].eq(0) & paired["correctness_xprime"].eq(1),
            same_wrong,
            different_wrong,
            paired["correctness_x"].eq(1) & paired["correctness_xprime"].eq(1) & paired["response_x"].eq(paired["response_xprime"]),
            paired["correctness_x"].eq(1) & paired["correctness_xprime"].eq(1) & paired["response_x"].ne(paired["response_xprime"]),
        ],
        [
            "1->0",
            "0->1",
            "0->0 same wrong",
            "0->0 different wrong",
            "1->1 same answer",
            "1->1 changed answer",
        ],
        default="other",
    )
    return paired


def _paired_probe_metrics(ctx: AnalysisContext) -> pd.DataFrame:
    paired = build_paired_probe_df(ctx)
    if paired.empty:
        return paired
    paired = paired.copy()
    score_cols_x = [f"score_{letter}_x" for letter in "ABCDE" if f"score_{letter}_x" in paired.columns]
    score_cols_xprime = [f"score_{letter}_xprime" for letter in "ABCDE" if f"score_{letter}_xprime" in paired.columns]
    for column in score_cols_x + score_cols_xprime + [
        "probe_score_correct_choice_x",
        "probe_score_selected_choice_x",
        "probe_score_correct_choice_xprime",
        "probe_score_selected_choice_xprime",
    ]:
        if column in paired.columns:
            paired[column] = pd.to_numeric(paired[column], errors="coerce")

    paired["rank_probe_x_correct"] = score_rank(
        paired.rename(columns={f"score_{letter}_x": f"score_{letter}" for letter in "ABCDE"}),
        paired["correct_letter"],
    )
    paired["rank_probe_xprime_correct"] = score_rank(
        paired.rename(columns={f"score_{letter}_xprime": f"score_{letter}" for letter in "ABCDE"}),
        paired["correct_letter"],
    )
    paired["rank_probe_x_selected"] = score_rank(
        paired.rename(columns={f"score_{letter}_x": f"score_{letter}" for letter in "ABCDE"}),
        paired["selected_choice_x"],
    )
    paired["rank_probe_xprime_selected"] = score_rank(
        paired.rename(columns={f"score_{letter}_xprime": f"score_{letter}" for letter in "ABCDE"}),
        paired["selected_choice_xprime"],
    )

    paired["bias_target_letter"] = np.where(
        paired["bias_type"].eq("incorrect_suggestion"),
        paired.get("incorrect_letter"),
        np.where(paired["bias_type"].eq("suggest_correct"), paired["correct_letter"], np.nan),
    )

    score_x_frame = paired.rename(columns={f"score_{letter}_x": f"score_{letter}" for letter in "ABCDE"})
    score_xprime_frame = paired.rename(columns={f"score_{letter}_xprime": f"score_{letter}" for letter in "ABCDE"})
    paired["probe_score_correct_x"] = score_value(score_x_frame, paired["correct_letter"])
    paired["probe_score_correct_xprime"] = score_value(score_xprime_frame, paired["correct_letter"])
    paired["probe_score_bias_target_x"] = score_value(score_x_frame, paired["bias_target_letter"])
    paired["probe_score_bias_target_xprime"] = score_value(score_xprime_frame, paired["bias_target_letter"])
    # Compute truth-vs-best-other margins without assuming a specific target.
    best_other_x = []
    best_other_xprime = []
    for row in paired.itertuples(index=False):
        correct_letter = str(getattr(row, "correct_letter", "") or "")
        scores_x = {letter: getattr(row, f"score_{letter}_x", np.nan) for letter in "ABCDE"}
        scores_xprime = {letter: getattr(row, f"score_{letter}_xprime", np.nan) for letter in "ABCDE"}
        other_scores_x = [score for letter, score in scores_x.items() if letter != correct_letter and pd.notna(score)]
        other_scores_xprime = [score for letter, score in scores_xprime.items() if letter != correct_letter and pd.notna(score)]
        best_other_x.append(max(other_scores_x) if other_scores_x else np.nan)
        best_other_xprime.append(max(other_scores_xprime) if other_scores_xprime else np.nan)
    paired["probe_best_other_x"] = pd.Series(best_other_x, index=paired.index, dtype=float)
    paired["probe_best_other_xprime"] = pd.Series(best_other_xprime, index=paired.index, dtype=float)
    paired["m_probe_truth_x"] = paired["probe_score_correct_x"] - paired["probe_best_other_x"]
    paired["m_probe_truth_xprime"] = paired["probe_score_correct_xprime"] - paired["probe_best_other_xprime"]
    paired["m_probe_bias_x"] = paired["probe_score_correct_x"] - paired["probe_score_bias_target_x"]
    paired["m_probe_bias_xprime"] = paired["probe_score_correct_xprime"] - paired["probe_score_bias_target_xprime"]
    paired["delta_m_probe_bias"] = paired["m_probe_bias_xprime"] - paired["m_probe_bias_x"]
    paired["delta_probe_score_correct"] = paired["probe_score_correct_xprime"] - paired["probe_score_correct_x"]
    paired["delta_probe_score_bias_target"] = paired["probe_score_bias_target_xprime"] - paired["probe_score_bias_target_x"]
    paired["probe_prefers_correct_over_target_xprime"] = paired["probe_score_correct_xprime"] > paired["probe_score_bias_target_xprime"]
    return paired


def _merge_external_and_probe(ctx: AnalysisContext) -> pd.DataFrame:
    external = _paired_external_metrics(ctx)
    probe = _paired_probe_metrics(ctx)
    if external.empty or probe.empty:
        return pd.DataFrame()
    join_keys = [column for column in ["question_id", "split", "draw_idx", "bias_type"] if column in external.columns and column in probe.columns]
    merged = external.merge(probe, on=join_keys, how="inner", suffixes=("", "_probe"))
    if "bias_target_letter" in merged.columns:
        score_x_frame = merged.rename(columns={f"score_{letter}_x": f"score_{letter}" for letter in "ABCDE"})
        score_xprime_frame = merged.rename(columns={f"score_{letter}_xprime": f"score_{letter}" for letter in "ABCDE"})
        merged["probe_score_bias_target_x"] = score_value(score_x_frame, merged["bias_target_letter"])
        merged["probe_score_bias_target_xprime"] = score_value(score_xprime_frame, merged["bias_target_letter"])
        merged["m_probe_bias_x"] = merged["probe_score_correct_x"] - merged["probe_score_bias_target_x"]
        merged["m_probe_bias_xprime"] = merged["probe_score_correct_xprime"] - merged["probe_score_bias_target_xprime"]
        merged["delta_m_probe_bias"] = merged["m_probe_bias_xprime"] - merged["m_probe_bias_x"]
        merged["delta_probe_score_bias_target"] = merged["probe_score_bias_target_xprime"] - merged["probe_score_bias_target_x"]
        merged["probe_prefers_correct_over_target_xprime"] = merged["probe_score_correct_xprime"] > merged["probe_score_bias_target_xprime"]
    return merged


def _binomial_ci(values: pd.Series) -> tuple[float, float]:
    return bootstrap_ci(values.astype(float).tolist(), statistic=np.mean)


def _set_panel_labels(ax: Any, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=20, pad=12)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)


def _format_panel_title(label: str, n: int) -> str:
    return f"{label}\n(n={n})"


def _quantile_rate_summary(
    frame: pd.DataFrame,
    score_column: str,
    outcome_column: str,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    working = frame[[score_column, outcome_column]].dropna().copy()
    if working.empty:
        return pd.DataFrame(columns=["bucket", "mean_score", "rate", "ci_low", "ci_high", "n"])
    working["bucket"] = quantile_bucket_labels(working[score_column], n_bins=n_bins)
    rows = []
    for bucket, bucket_df in working.groupby("bucket", observed=False):
        ci_low, ci_high = _binomial_ci(bucket_df[outcome_column].astype(float))
        rows.append(
            {
                "bucket": bucket,
                "mean_score": bucket_df[score_column].mean(),
                "rate": bucket_df[outcome_column].mean(),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": len(bucket_df),
            }
        )
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def _candidate_level_probe_auc(
    prompt_df: pd.DataFrame,
    *,
    split_name: str | None = None,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    if split_name is not None and "split" in prompt_df.columns:
        prompt_df = prompt_df.loc[prompt_df["split"].astype(str).eq(split_name)].copy()
    if prompt_df.empty:
        return {"auc": np.nan, "shuffled_auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "permutation_pvalue": np.nan}

    option_long = []
    for row in prompt_df.itertuples(index=False):
        correct_letter = str(getattr(row, "correct_letter", "") or "")
        for letter in "ABCDE":
            option_long.append(
                {
                    "question_id": getattr(row, "question_id", ""),
                    "label": 1 if letter == correct_letter else 0,
                    "score": getattr(row, f"score_{letter}", np.nan),
                }
            )
    long_df = pd.DataFrame(option_long).dropna()
    if long_df.empty or long_df["label"].nunique() < 2:
        return {"auc": np.nan, "shuffled_auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "permutation_pvalue": np.nan}

    auc = float(roc_auc_score(long_df["label"], long_df["score"]))
    rng = np.random.default_rng(seed)
    shuffled_draws = []
    for _ in range(n_permutations):
        shuffled = long_df["label"].sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1_000_000))).to_numpy()
        shuffled_draws.append(float(roc_auc_score(shuffled, long_df["score"])))
    ci_low, ci_high = bootstrap_ci(shuffled_draws, statistic=np.mean)
    pvalue = float((np.sum(np.asarray(shuffled_draws) >= auc) + 1) / (len(shuffled_draws) + 1))
    return {
        "auc": auc,
        "shuffled_auc": float(np.mean(shuffled_draws)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "permutation_pvalue": pvalue,
    }


def register_analysis_function(
    name: str | None = None,
    *,
    output_kind: str,
    description: str,
    requires_probes: bool = False,
    requires_labels: bool = False,
    requires_bias_target: bool = False,
    supported_task_formats: tuple[str, ...] = ("multiple_choice",),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        function_name = name or func.__name__
        ANALYSIS_FUNCTIONS[function_name] = func
        ANALYSIS_FUNCTION_SPECS[function_name] = AnalysisFunctionSpec(
            name=function_name,
            output_kind=output_kind,
            description=description,
            requires_probes=requires_probes,
            requires_labels=requires_labels,
            requires_bias_target=requires_bias_target,
            supported_task_formats=supported_task_formats,
        )
        return func

    return _decorator


def get_analysis_function(name: str) -> Callable[..., Any]:
    try:
        return ANALYSIS_FUNCTIONS[name]
    except KeyError as exc:
        raise AnalysisError(f"Unknown analysis function: {name}") from exc


def get_analysis_function_spec(name: str) -> AnalysisFunctionSpec:
    try:
        return ANALYSIS_FUNCTION_SPECS[name]
    except KeyError as exc:
        raise AnalysisError(f"Unknown analysis function spec: {name}") from exc


def list_analysis_functions() -> list[str]:
    return sorted(ANALYSIS_FUNCTIONS)


def list_analysis_function_specs() -> pd.DataFrame:
    rows = [asdict(ANALYSIS_FUNCTION_SPECS[name]) for name in sorted(ANALYSIS_FUNCTION_SPECS)]
    if not rows:
        return pd.DataFrame(
            columns=[
                "name",
                "output_kind",
                "description",
                "requires_probes",
                "requires_labels",
                "requires_bias_target",
                "supported_task_formats",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["supported_task_formats"] = frame["supported_task_formats"].map(lambda values: ",".join(values))
    return frame


def _save_failure_row(ctx: AnalysisContext, row: Mapping[str, Any]) -> Path:
    path = ctx.tables_dir / "analysis_cell_failures.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([dict(row)])], ignore_index=True)
    updated.to_csv(path, index=False)
    return path


def _display_value(value: Any) -> None:
    try:
        from IPython.display import Markdown, display
    except Exception:
        display = None
        Markdown = None
    if hasattr(value, "savefig"):
        if display is not None:
            display(value)
        return
    if display is not None:
        display(value)
    else:
        print(value)


def _display_warning(message: str) -> None:
    try:
        from IPython.display import Markdown, display
    except Exception:
        print(message)
        return
    display(Markdown(message))


def run_analysis_operation(
    ctx: AnalysisContext,
    function_name: str,
    *,
    output_stem: str | None = None,
    save_output: bool = True,
    **kwargs: Any,
) -> Any:
    func = get_analysis_function(function_name)
    result = func(ctx, **kwargs)
    if output_stem and save_output:
        if isinstance(result, pd.DataFrame):
            ctx.save_table(result, output_stem)
        elif hasattr(result, "savefig"):
            ctx.save_plot(result, output_stem)
    return result


def safe_run_analysis_operation(
    ctx: AnalysisContext,
    function_name: str,
    *,
    cell_id: str,
    output_stem: str | None = None,
    save_output: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        result = run_analysis_operation(
            ctx,
            function_name,
            output_stem=output_stem,
            save_output=save_output,
            **kwargs,
        )
        return {
            "ok": True,
            "cell_id": cell_id,
            "function_name": function_name,
            "result": result,
            "output_stem": output_stem,
        }
    except Exception as exc:
        failure_path = _save_failure_row(
            ctx,
            {
                "timestamp_utc": utc_now_iso(),
                "cell_id": cell_id,
                "function_name": function_name,
                "output_stem": output_stem or "",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        return {
            "ok": False,
            "cell_id": cell_id,
            "function_name": function_name,
            "output_stem": output_stem,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "failure_csv_path": str(failure_path),
        }


def safe_display_analysis_operation(
    ctx: AnalysisContext,
    function_name: str,
    *,
    cell_id: str,
    output_stem: str | None = None,
    save_output: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    status = safe_run_analysis_operation(
        ctx,
        function_name,
        cell_id=cell_id,
        output_stem=output_stem,
        save_output=save_output,
        **kwargs,
    )
    if status["ok"]:
        _display_value(status["result"])
        return status

    _display_warning(
        "\n".join(
            [
                f"**Analysis cell skipped**",
                "",
                f"- function: `{status['function_name']}`",
                f"- error: `{status['error_type']}: {status['error']}`",
                f"- recorded in: `{status['failure_csv_path']}`",
            ]
        )
    )
    return status


@register_analysis_function(
    output_kind="table",
    description="Run-level counts and metadata for the current run.",
)
def table_run_overview(ctx: AnalysisContext) -> pd.DataFrame:
    df = build_sampled_responses_df(ctx)
    datasets = sorted({str(value) for value in df.get("dataset", pd.Series(dtype=str)).dropna().tolist() if str(value)})
    templates = sorted(
        {str(value) for value in df.get("template_type", pd.Series(dtype=str)).dropna().tolist() if str(value)}
    )
    mc_options_per_question = len([column for column in [f"P({letter})" for letter in "ABCDE"] if column in df.columns])
    rows = [
        {"field": "run_name", "value": ctx.run_name},
        {"field": "model_name", "value": ctx.model_name},
        {"field": "n_rows", "value": int(len(df))},
        {"field": "n_questions", "value": int(df["question_id"].nunique())},
        {"field": "datasets", "value": ",".join(datasets)},
        {"field": "template_types", "value": ",".join(templates)},
        {"field": "n_bias_templates", "value": int(max(len(templates) - int("neutral" in templates), 0))},
        {"field": "probe_rows", "value": int(len(ctx.probe_scores_by_prompt))},
        {"field": "n_probe_families", "value": int(build_probe_scores_df(ctx).get("probe_name", pd.Series(dtype=str)).nunique())},
        {"field": "mc_options_per_question", "value": mc_options_per_question},
    ]
    return pd.DataFrame(rows)


@register_analysis_function(
    output_kind="table",
    description="Template-level summary of row counts and mean probabilities.",
)
def table_template_overview(ctx: AnalysisContext) -> pd.DataFrame:
    df = build_sampled_responses_df(ctx)
    for column in ("P(correct)", "P(selected)"):
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    summary = (
        df.groupby("template_type", observed=False)
        .agg(
            n_rows=("question_id", "size"),
            n_questions=("question_id", "nunique"),
            avg_p_correct=("P(correct)", "mean"),
            avg_p_selected=("P(selected)", "mean"),
        )
        .reset_index()
        .sort_values("template_type")
        .reset_index(drop=True)
    )
    return summary


@register_analysis_function(
    output_kind="table",
    description="Inventory of saved probe scores and template coverage.",
    requires_probes=True,
)
def table_probe_inventory(ctx: AnalysisContext) -> pd.DataFrame:
    probe_df = build_probe_scores_df(ctx)
    if probe_df.empty:
        return pd.DataFrame(
            [
                {
                    "probe_scores_available": False,
                    "n_probe_rows": 0,
                    "probe_names": "",
                    "template_types": "",
                }
            ]
        )

    probe_names = sorted({str(value) for value in probe_df.get("probe_name", pd.Series(dtype=str)).dropna().tolist()})
    template_types = sorted(
        {str(value) for value in probe_df.get("template_type", pd.Series(dtype=str)).dropna().tolist() if str(value)}
    )
    return pd.DataFrame(
        [
            {
                "probe_scores_available": True,
                "n_probe_rows": int(len(probe_df)),
                "probe_names": ",".join(probe_names),
                "template_types": ",".join(template_types),
            }
        ]
    )


@register_analysis_function(
    output_kind="plot",
    description="Neutral-option selection distribution from model outputs.",
)
def plot_neutral_option_selection(ctx: AnalysisContext) -> Any:
    neutral_df = build_neutral_sampled_responses_df(ctx)
    option_df = (
        neutral_df["response"]
        .value_counts(normalize=True)
        .rename_axis("option")
        .reset_index(name="fraction")
        .sort_values("option")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.barplot(
        data=option_df,
        x="option",
        y="fraction",
        color=DEFAULT_COLORS["primary"],
        edgecolor="black",
        ax=ax,
    )
    ax.set_title("Neutral MC Option Selection", fontsize=22, pad=12)
    ax.set_xlabel("Chosen option", fontsize=15)
    ax.set_ylabel("Fraction of samples", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(
        handles=[Patch(facecolor=DEFAULT_COLORS["primary"], edgecolor="black", label="Neutral responses")],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=True,
        ncol=1,
    )
    fig.subplots_adjust(bottom=0.25)
    return fig


@register_analysis_function(
    output_kind="table",
    description="External summary statistics for the run.",
    requires_labels=True,
)
def table_external_summary_statistics(ctx: AnalysisContext) -> pd.DataFrame:
    external_df = _prepare_external_rows(ctx)
    probe_df = build_probe_scores_df(ctx)
    rows = [
        {
            "n_questions": int(external_df["question_id"].nunique()),
            "n_bias_templates": int(external_df["template_type"].astype(str).ne("neutral").sum() / max(external_df["question_id"].nunique(), 1)),
            "n_template_types": int(external_df["template_type"].nunique()),
            "template_types": ",".join(_ordered_templates(external_df["template_type"])),
            "total_sampled_rows": int(len(external_df)),
            "n_probe_rows": int(len(probe_df)),
            "n_probes_trained": int(probe_df["probe_name"].nunique()) if not probe_df.empty and "probe_name" in probe_df.columns else 0,
            "mc_options_per_question": int(len([column for column in [f"P({letter})" for letter in "ABCDE"] if column in external_df.columns])),
        }
    ]
    return pd.DataFrame(rows)


@register_analysis_function(
    output_kind="table",
    description="Per-template external accuracy and confidence summary.",
    requires_labels=True,
)
def table_external_accuracy_metrics(ctx: AnalysisContext) -> pd.DataFrame:
    df = _prepare_external_rows(ctx)
    summary_rows = []
    template_order = _ordered_templates(df["template_type"])
    for template in template_order + ["total"]:
        subset = df if template == "total" else df.loc[df["template_type"].astype(str).eq(template)]
        summary_rows.append(
            {
                "template_type": template,
                "n": int(len(subset)),
                "accuracy": float(subset["correctness"].mean()),
                "accuracy_at_2": float(subset["accuracy_at_2"].mean()),
                "mean_top1_margin": float(subset["top1_margin"].mean()),
                "mean_top1_probability": float(subset["top1_probability"].mean()),
                "mean_effective_responses": float(subset["effective_responses"].mean()),
            }
        )
    return pd.DataFrame(summary_rows)


@register_analysis_function(
    output_kind="plot",
    description="Histogram of effective number of responses by template type.",
    requires_labels=True,
)
def plot_effective_responses_histogram(ctx: AnalysisContext) -> Any:
    df = _prepare_external_rows(ctx)
    template_order = _ordered_templates(df["template_type"])
    fig, axes = plt.subplots(1, len(template_order), figsize=(5.5 * len(template_order), 4.8), sharey=True)
    if len(template_order) == 1:
        axes = [axes]
    for ax, template in zip(axes, template_order):
        subset = df.loc[df["template_type"].astype(str).eq(template)]
        sns.histplot(
            data=subset,
            x="effective_responses",
            bins=15,
            color=BIAS_COLORS.get(template, DEFAULT_COLORS["fallback"]),
            edgecolor="white",
            ax=ax,
        )
        _set_panel_labels(
            ax,
            title=_format_panel_title(DISPLAY_LABELS.get(template, template.title()), len(subset)),
            xlabel="Effective number of responses",
            ylabel="Count",
        )
    fig.suptitle("Effective Number of Responses", fontsize=24, y=1.02)
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="plot",
    description="Accuracy as a function of effective number of responses.",
    requires_labels=True,
)
def plot_accuracy_by_effective_responses_bucket(ctx: AnalysisContext, *, n_bins: int = 4) -> Any:
    df = _prepare_external_rows(ctx)
    template_order = _ordered_templates(df["template_type"])
    fig, axes = plt.subplots(1, len(template_order), figsize=(5.5 * len(template_order), 4.8), sharey=True)
    if len(template_order) == 1:
        axes = [axes]
    for ax, template in zip(axes, template_order):
        subset = df.loc[df["template_type"].astype(str).eq(template)].copy()
        summary = _quantile_rate_summary(subset, "effective_responses", "correctness", n_bins=n_bins)
        x_positions = np.arange(len(summary))
        ax.plot(x_positions, summary["rate"], color=BIAS_COLORS.get(template, DEFAULT_COLORS["fallback"]), marker="o")
        ax.fill_between(x_positions, summary["ci_low"], summary["ci_high"], color=BIAS_COLORS.get(template, DEFAULT_COLORS["fallback"]), alpha=0.2)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(summary["bucket"], rotation=45, ha="right")
        _set_panel_labels(
            ax,
            title=_format_panel_title(DISPLAY_LABELS.get(template, template.title()), len(subset)),
            xlabel="Effective-response quantile",
            ylabel="Accuracy",
        )
        ax.set_ylim(0.0, 1.0)
    fig.suptitle("Accuracy vs Effective Number of Responses", fontsize=24, y=1.02)
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="table",
    description="AUROC of effective number of responses for error detection.",
    requires_labels=True,
)
def table_error_detection_auroc_by_effective_responses(ctx: AnalysisContext) -> pd.DataFrame:
    df = _prepare_external_rows(ctx)
    rows = []
    for template in _ordered_templates(df["template_type"]) + ["total"]:
        subset = df if template == "total" else df.loc[df["template_type"].astype(str).eq(template)]
        score = subset["effective_responses"]
        label = 1 - subset["correctness"]
        auc = np.nan
        if len(subset) > 0 and pd.Series(label).nunique() > 1:
            auc = float(roc_auc_score(label, score))
        rows.append({"template_type": template, "n": int(len(subset)), "error_detection_auroc_from_effective_responses": auc})
    return pd.DataFrame(rows)


@register_analysis_function(
    output_kind="plot",
    description="Reliability diagram using the top predicted probability.",
    requires_labels=True,
)
def plot_reliability_diagram_top_probability(ctx: AnalysisContext, *, n_bins: int = 10) -> Any:
    df = _prepare_external_rows(ctx)
    template_order = _ordered_templates(df["template_type"])
    fig, axes = plt.subplots(1, len(template_order), figsize=(5.5 * len(template_order), 4.8), sharex=True, sharey=True)
    if len(template_order) == 1:
        axes = [axes]
    for ax, template in zip(axes, template_order):
        subset = df.loc[df["template_type"].astype(str).eq(template)]
        curve = reliability_curve(subset["correctness"], subset["top1_probability"], n_bins=n_bins)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2)
        ax.plot(curve["mean_score"], curve["empirical_accuracy"], color=BIAS_COLORS.get(template, DEFAULT_COLORS["fallback"]), marker="o")
        _set_panel_labels(
            ax,
            title=_format_panel_title(DISPLAY_LABELS.get(template, template.title()), len(subset)),
            xlabel="Mean top predicted probability",
            ylabel="Empirical accuracy",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Reliability Diagrams from Top Predicted Probability", fontsize=24, y=1.02)
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="plot",
    description="Histogram of accuracy-related delta under bias.",
    requires_labels=True,
)
def plot_sycophancy_delta_histograms(ctx: AnalysisContext) -> Any:
    paired = _paired_external_metrics(ctx)
    bias_order = _ordered_templates(paired["bias_type"], include_neutral=False)
    fig, axes = plt.subplots(2, len(bias_order), figsize=(5.5 * len(bias_order), 9.2), sharey="row")
    if len(bias_order) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, bias_type in enumerate(bias_order):
        subset = paired.loc[paired["bias_type"].astype(str).eq(bias_type)]
        color = BIAS_COLORS.get(bias_type, DEFAULT_COLORS["fallback"])
        sns.histplot(subset["delta_p_correct"], bins=15, color=color, edgecolor="white", ax=axes[0, col])
        _set_panel_labels(
            axes[0, col],
            title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)),
            xlabel=r"$\Delta P(correct)=P(correct|x)-P(correct|x')$",
            ylabel="Count",
        )
        sns.histplot(subset["delta_relative_p_correct"], bins=15, color=color, edgecolor="white", ax=axes[1, col])
        _set_panel_labels(
            axes[1, col],
            title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)),
            xlabel=r"$\Delta_{rel}=(P_x-P_{x'})/(P_x+P_{x'})$",
            ylabel="Count",
        )
    fig.suptitle("External Accuracy-Probability Shift Under Bias", fontsize=24, y=1.01)
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="table",
    description="Flip-category table under bias.",
    requires_labels=True,
)
def table_sycophancy_flip_table(ctx: AnalysisContext) -> pd.DataFrame:
    paired = _paired_external_metrics(ctx)
    categories = [
        "1->0",
        "0->1",
        "0->0 same wrong",
        "0->0 different wrong",
        "1->1 same answer",
        "1->1 changed answer",
    ]
    rows = []
    for bias_type in _ordered_templates(paired["bias_type"], include_neutral=False):
        subset = paired.loc[paired["bias_type"].astype(str).eq(bias_type)]
        counts = subset["flip_category"].value_counts()
        row = {"bias_type": bias_type, "n": int(len(subset))}
        for category in categories:
            row[f"{category}_count"] = int(counts.get(category, 0))
            row[f"{category}_fraction"] = float((subset["flip_category"] == category).mean())
        rows.append(row)
    return pd.DataFrame(rows)


@register_analysis_function(
    output_kind="table",
    description="Global distribution-shift metrics under bias.",
    requires_labels=True,
)
def table_distribution_shift_metrics(ctx: AnalysisContext) -> pd.DataFrame:
    paired = _paired_external_metrics(ctx)
    rows = []
    for bias_type in _ordered_templates(paired["bias_type"], include_neutral=False):
        subset = paired.loc[paired["bias_type"].astype(str).eq(bias_type)]
        rows.append(
            {
                "bias_type": bias_type,
                "n": int(len(subset)),
                "mean_js_divergence": float(subset["js_divergence"].mean()),
                "mean_total_variation": float(subset["total_variation"].mean()),
                "mean_delta_p_correct": float(subset["delta_p_correct"].mean()),
                "mean_delta_relative_p_correct": float(subset["delta_relative_p_correct"].mean()),
            }
        )
    return pd.DataFrame(rows)


@register_analysis_function(
    output_kind="plot",
    description="Incorrect-suggestion transition heatmap conditioned on the neutral state.",
    requires_labels=True,
    requires_bias_target=True,
)
def plot_incorrect_suggestion_transition_heatmap(ctx: AnalysisContext) -> Any:
    paired = _paired_external_metrics(ctx)
    subset = paired.loc[paired["bias_type"].astype(str).eq("incorrect_suggestion")].copy()
    if subset.empty:
        raise AnalysisNotSupportedError("No incorrect_suggestion rows are available for this run.")

    def neutral_state(row: pd.Series) -> str:
        if row["correctness_x"] == 1:
            return "neutral correct"
        if row["response_x"] == row["bias_target_letter"]:
            return "neutral bias target"
        return "neutral other wrong"

    def biased_state(row: pd.Series) -> str:
        if row["correctness_xprime"] == 1:
            return "biased correct"
        if row["response_xprime"] == row["bias_target_letter"]:
            return "biased bias target"
        return "biased other wrong"

    subset["neutral_state"] = subset.apply(neutral_state, axis=1)
    subset["biased_state"] = subset.apply(biased_state, axis=1)
    matrix = (
        subset.groupby(["neutral_state", "biased_state"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=["neutral correct", "neutral bias target", "neutral other wrong"],
            columns=["biased correct", "biased bias target", "biased other wrong"],
            fill_value=0,
        )
    )
    matrix = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar=True,
        linewidths=1.0,
        linecolor="white",
        ax=ax,
    )
    _set_panel_labels(
        ax,
        title=_format_panel_title("Incorrect Suggestion Transition Matrix", len(subset)),
        xlabel="Biased state",
        ylabel="Neutral state",
    )
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="plot",
    description="Top-1 confidence before and after incorrect suggestion bias.",
    requires_labels=True,
    requires_bias_target=True,
)
def plot_incorrect_suggestion_top1_confidence_before_after(ctx: AnalysisContext) -> Any:
    paired = _paired_external_metrics(ctx)
    subset = paired.loc[paired["bias_type"].astype(str).eq("incorrect_suggestion")].copy()
    if subset.empty:
        raise AnalysisNotSupportedError("No incorrect_suggestion rows are available for this run.")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    sns.histplot(subset["top1_probability_x"], bins=15, stat="density", color=DEFAULT_COLORS["primary"], edgecolor="white", ax=axes[0])
    _set_panel_labels(
        axes[0],
        title=_format_panel_title("Neutral x", len(subset)),
        xlabel="Top-1 confidence",
        ylabel="Density",
    )
    sns.histplot(subset["top1_probability_xprime"], bins=15, stat="density", color=DEFAULT_COLORS["secondary"], edgecolor="white", ax=axes[1])
    _set_panel_labels(
        axes[1],
        title=_format_panel_title("Biased x'", len(subset)),
        xlabel="Top-1 confidence",
        ylabel="Density",
    )
    fig.suptitle("Top-1 Confidence Before vs After Incorrect Suggestion", fontsize=24, y=1.02)
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="plot",
    description="Bias-target gain and neutral correct-vs-target margin for incorrect suggestion.",
    requires_labels=True,
    requires_bias_target=True,
)
def plot_incorrect_suggestion_bias_target_gain_and_margin(ctx: AnalysisContext) -> Any:
    paired = _paired_external_metrics(ctx)
    subset = paired.loc[paired["bias_type"].astype(str).eq("incorrect_suggestion")].copy()
    if subset.empty:
        raise AnalysisNotSupportedError("No incorrect_suggestion rows are available for this run.")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    sns.histplot(subset["delta_p_bias_target"], bins=15, color=DEFAULT_COLORS["secondary"], edgecolor="white", ax=axes[0])
    _set_panel_labels(
        axes[0],
        title=f"Bias-target gain\nmean={subset['delta_p_bias_target'].mean():.3f}",
        xlabel=r"$\Delta_{bias}=P_{x'}(b)-P_x(b)$",
        ylabel="Count",
    )
    sns.histplot(subset["m_bias_external_x"], bins=15, color=DEFAULT_COLORS["primary"], edgecolor="white", ax=axes[1])
    _set_panel_labels(
        axes[1],
        title=f"Neutral correct-vs-target margin\nmean={subset['m_bias_external_x'].mean():.3f}",
        xlabel=r"$m_{bias}(x)=P_x(c)-P_x(b)$",
        ylabel="Count",
    )
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="plot",
    description="Rate of adopting the user target as a function of pre-bias chosen-answer margin.",
    requires_labels=True,
    requires_bias_target=True,
)
def plot_target_adoption_by_prebias_chosen_margin(ctx: AnalysisContext, *, n_bins: int = 5) -> Any:
    paired = _paired_external_metrics(ctx)
    bias_types = ["incorrect_suggestion", "suggest_correct"]
    fig, axes = plt.subplots(1, len(bias_types), figsize=(6.5 * len(bias_types), 5.0), sharey=True)
    if len(bias_types) == 1:
        axes = [axes]
    for ax, bias_type in zip(axes, bias_types):
        subset = paired.loc[paired["bias_type"].astype(str).eq(bias_type)].copy()
        subset["minus_chosen_margin_x"] = -subset["top1_margin_x"]
        summary = _quantile_rate_summary(subset, "minus_chosen_margin_x", "adopts_bias_target", n_bins=n_bins)
        x_positions = np.arange(len(summary))
        ax.plot(x_positions, summary["rate"], color=BIAS_COLORS.get(bias_type, DEFAULT_COLORS["fallback"]), marker="o")
        ax.fill_between(x_positions, summary["ci_low"], summary["ci_high"], color=BIAS_COLORS.get(bias_type, DEFAULT_COLORS["fallback"]), alpha=0.2)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(summary["bucket"], rotation=45, ha="right")
        _set_panel_labels(
            ax,
            title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)),
            xlabel=r"Quantile of $-m_{top}(x)$",
            ylabel="Rate of adopting user target",
        )
        ax.set_ylim(0, 1)
    fig.suptitle("Target Adoption vs Neutral Chosen-Answer Margin", fontsize=24, y=1.02)
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="plot",
    description="Layerwise probe AUC across train, val, and test splits.",
    requires_probes=True,
)
def plot_probe_layerwise_performance(ctx: AnalysisContext) -> Any:
    layer_df = build_all_probe_layer_metrics_df(ctx)
    if layer_df.empty:
        raise AnalysisNotSupportedError("No layerwise probe metrics were found in probes/all_probes.")
    template_order = _ordered_templates(layer_df["template_type"])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), sharex=True, sharey=True)
    axes = axes.flatten()
    split_colors = {"train": DEFAULT_COLORS["fallback"], "val": DEFAULT_COLORS["secondary"], "test": DEFAULT_COLORS["primary"]}
    for ax, template in zip(axes, template_order):
        subset = layer_df.loc[layer_df["template_type"].astype(str).eq(template)].copy()
        for split_name in ("train", "val", "test"):
            split_df = subset.loc[subset["split"].astype(str).eq(split_name)].sort_values("layer")
            ax.plot(
                split_df["layer"],
                split_df["auc"],
                marker="o",
                linewidth=2,
                color=split_colors[split_name],
                label=split_name,
            )
        best_layer = subset["best_layer"].dropna().iloc[0] if subset["best_layer"].notna().any() else np.nan
        if pd.notna(best_layer):
            ax.axvline(best_layer, linestyle="--", color="black", linewidth=1.2)
        ax.axhline(0.5, linestyle=":", color="black", linewidth=1.0)
        _set_panel_labels(
            ax,
            title=_format_panel_title(DISPLAY_LABELS.get(template, template.title()), int(subset.loc[subset["split"].eq("test"), "n_total"].fillna(0).max() or 0)),
            xlabel="Layer",
            ylabel="AUC",
        )
        ax.set_ylim(0.45, 1.02)
    for ax in axes[len(template_order):]:
        ax.axis("off")
    handles = [Patch(facecolor=color, edgecolor="black", label=split_name) for split_name, color in split_colors.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=True)
    fig.suptitle("Layerwise Probe Performance", fontsize=24, y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


@register_analysis_function(
    output_kind="table",
    description="Summary of the chosen probes used in the run.",
    requires_probes=True,
)
def table_chosen_probe_summary(ctx: AnalysisContext) -> pd.DataFrame:
    summary_df = build_chosen_probe_summary_df(ctx)
    prompt_df = build_probe_scores_df(ctx)
    if summary_df.empty:
        return pd.DataFrame()
    rows = []
    for row in summary_df.itertuples(index=False):
        template_type = str(getattr(row, "template_type", "") or "")
        prompt_subset = prompt_df.loc[prompt_df["template_type"].astype(str).eq(template_type)].copy()
        val_subset = prompt_subset.loc[prompt_subset["split"].astype(str).eq("val")].copy()
        rows.append(
            {
                "probe_name": row.probe_name,
                "training_source": template_type,
                "bias_family": template_type,
                "selected_layer": row.chosen_layer,
                "token_pooling_rule": getattr(row, "token_pooling_rule", pd.NA),
                "validation_auc_same_template": getattr(row, "val_auc", pd.NA),
                "validation_balanced_accuracy": getattr(row, "val_balanced_accuracy", pd.NA),
                "validation_mean_gold_rank": float(score_rank(val_subset, val_subset["correct_letter"]).mean()) if not val_subset.empty else np.nan,
                "validation_chosen_answer_agreement_sanity": float(val_subset["probe_argmax_choice"].eq(val_subset["selected_choice"]).mean()) if not val_subset.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


@register_analysis_function(
    output_kind="table",
    description="Probe validity metrics by template type.",
    requires_probes=True,
)
def table_probe_validity_by_template_type(ctx: AnalysisContext) -> pd.DataFrame:
    prompt_df = build_probe_scores_df(ctx)
    long_df = build_probe_option_long_df(ctx)
    if prompt_df.empty or long_df.empty:
        return pd.DataFrame()
    rows = []
    for template in _ordered_templates(prompt_df["template_type"]):
        prompt_subset = prompt_df.loc[prompt_df["template_type"].astype(str).eq(template)].copy()
        prompt_subset = prompt_subset.loc[prompt_subset["split"].astype(str).eq("test")]
        long_subset = long_df.loc[long_df["template_type"].astype(str).eq(template)].copy()
        long_subset = long_subset.loc[long_subset["split"].astype(str).eq("test")]
        auc = np.nan
        balanced_accuracy = np.nan
        if not long_subset.empty and long_subset["is_correct_option"].nunique() > 1:
            auc = float(roc_auc_score(long_subset["is_correct_option"].astype(int), long_subset["probe_score"]))
            preds = (long_subset["probe_score"] >= 0.5).astype(int)
            acc_pos = float((preds[long_subset["is_correct_option"] == 1] == 1).mean())
            acc_neg = float((preds[long_subset["is_correct_option"] == 0] == 0).mean())
            balanced_accuracy = 0.5 * (acc_pos + acc_neg)
        rows.append(
            {
                "template_type": template,
                "auc_test": auc,
                "balanced_accuracy_test": balanced_accuracy,
                "mean_gold_rank_test": float(score_rank(prompt_subset, prompt_subset["correct_letter"]).mean()) if not prompt_subset.empty else np.nan,
                "accuracy_on_false_label_test": float(((long_subset["probe_score"] < 0.5) & (~long_subset["is_correct_option"])).mean() / max((~long_subset["is_correct_option"]).mean(), 1e-12)) if not long_subset.empty else np.nan,
                "accuracy_on_correct_label_test": float(((long_subset["probe_score"] >= 0.5) & (long_subset["is_correct_option"])).mean() / max((long_subset["is_correct_option"]).mean(), 1e-12)) if not long_subset.empty else np.nan,
                "chosen_answer_agreement_test": float(prompt_subset["probe_argmax_choice"].eq(prompt_subset["selected_choice"]).mean()) if not prompt_subset.empty else np.nan,
                "mean_chosen_answer_rank_test": float(score_rank(prompt_subset, prompt_subset["selected_choice"]).mean()) if not prompt_subset.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


@register_analysis_function(
    output_kind="plot",
    description="Matched-template same-candidate score stability under bias.",
    requires_probes=True,
)
def plot_probe_score_stability_under_bias(ctx: AnalysisContext) -> Any:
    merged = _merge_external_and_probe(ctx)
    if merged.empty:
        raise AnalysisNotSupportedError("No paired probe/external rows are available for stability analysis.")
    bias_order = _ordered_templates(merged["bias_type"], include_neutral=False)
    fig, axes = plt.subplots(3, len(bias_order), figsize=(5.5 * len(bias_order), 12.5), sharex=False, sharey=False)
    if len(bias_order) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])
    for col, bias_type in enumerate(bias_order):
        subset = merged.loc[merged["bias_type"].astype(str).eq(bias_type)].copy()
        color = BIAS_COLORS.get(bias_type, DEFAULT_COLORS["fallback"])
        sns.scatterplot(data=subset, x="probe_score_correct_x", y="probe_score_correct_xprime", color=color, alpha=0.6, s=35, edgecolor=None, ax=axes[0, col])
        axes[0, col].plot([subset["probe_score_correct_x"].min(), subset["probe_score_correct_x"].max()], [subset["probe_score_correct_x"].min(), subset["probe_score_correct_x"].max()], linestyle="--", color="black", linewidth=1.0)
        _set_panel_labels(axes[0, col], title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)), xlabel=r"$s(x,c)$", ylabel=r"$s(x',c)$")

        if subset["probe_score_bias_target_x"].notna().any() and subset["probe_score_bias_target_xprime"].notna().any():
            sns.scatterplot(data=subset, x="probe_score_bias_target_x", y="probe_score_bias_target_xprime", color=color, alpha=0.6, s=35, edgecolor=None, ax=axes[1, col])
            axes[1, col].plot([subset["probe_score_bias_target_x"].min(), subset["probe_score_bias_target_x"].max()], [subset["probe_score_bias_target_x"].min(), subset["probe_score_bias_target_x"].max()], linestyle="--", color="black", linewidth=1.0)
            _set_panel_labels(axes[1, col], title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)), xlabel=r"$s(x,b)$", ylabel=r"$s(x',b)$")
        else:
            axes[1, col].axis("off")

        if subset["m_probe_bias_x"].notna().any() and subset["m_probe_bias_xprime"].notna().any():
            sns.scatterplot(data=subset, x="m_probe_bias_x", y="m_probe_bias_xprime", color=color, alpha=0.6, s=35, edgecolor=None, ax=axes[2, col])
            xmin = min(subset["m_probe_bias_x"].min(), subset["m_probe_bias_xprime"].min())
            xmax = max(subset["m_probe_bias_x"].max(), subset["m_probe_bias_xprime"].max())
            axes[2, col].plot([xmin, xmax], [xmin, xmax], linestyle="--", color="black", linewidth=1.0)
            _set_panel_labels(axes[2, col], title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)), xlabel=r"$m_{probe}^{bias}(x)$", ylabel=r"$m_{probe}^{bias}(x')$")
        else:
            axes[2, col].axis("off")
    fig.suptitle("Matched-Template Probe Score Stability Under Bias", fontsize=24, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


@register_analysis_function(
    output_kind="plot",
    description="Probe margin and rank distributions.",
    requires_probes=True,
)
def plot_probe_margin_and_rank_distributions(ctx: AnalysisContext) -> Any:
    merged = _merge_external_and_probe(ctx)
    if merged.empty:
        raise AnalysisNotSupportedError("No paired probe/external rows are available for probe distribution analysis.")
    bias_order = _ordered_templates(merged["bias_type"], include_neutral=False)
    fig, axes = plt.subplots(2, len(bias_order), figsize=(5.5 * len(bias_order), 8.8), sharey="row")
    if len(bias_order) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, bias_type in enumerate(bias_order):
        subset = merged.loc[merged["bias_type"].astype(str).eq(bias_type)]
        color = BIAS_COLORS.get(bias_type, DEFAULT_COLORS["fallback"])
        sns.histplot(subset["m_probe_truth_x"], bins=15, color=color, edgecolor="white", ax=axes[0, col])
        _set_panel_labels(axes[0, col], title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)), xlabel=r"$s(x,c)-\max_{i\neq c}s(x,i)$", ylabel="Count")
        sns.histplot(subset["rank_probe_x_correct"], bins=np.arange(1, 8) - 0.5, color=color, edgecolor="white", ax=axes[1, col])
        _set_panel_labels(axes[1, col], title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)), xlabel=r"$rank_{probe}(x,c)$", ylabel="Count")
    fig.suptitle("Probe Margin and Rank Distributions", fontsize=24, y=1.01)
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="table",
    description="Chance-baseline comparison for validation AUC.",
    requires_probes=True,
)
def table_probe_chance_baseline_check(ctx: AnalysisContext) -> pd.DataFrame:
    prompt_df = build_probe_scores_df(ctx)
    if prompt_df.empty:
        return pd.DataFrame()
    rows = []
    for template in _ordered_templates(prompt_df["template_type"]):
        subset = prompt_df.loc[prompt_df["template_type"].astype(str).eq(template)].copy()
        stats = _candidate_level_probe_auc(subset, split_name="val")
        rows.append(
            {
                "template_type": template,
                "real_val_auc": stats["auc"],
                "shuffled_label_val_auc_mean": stats["shuffled_auc"],
                "shuffled_label_auc_ci_low": stats["ci_low"],
                "shuffled_label_auc_ci_high": stats["ci_high"],
                "permutation_pvalue": stats["permutation_pvalue"],
            }
        )
    return pd.DataFrame(rows)


@register_analysis_function(
    output_kind="plot",
    description="Internal margin shift under bias.",
    requires_probes=True,
)
def plot_internal_margin_shift_under_bias(ctx: AnalysisContext) -> Any:
    merged = _merge_external_and_probe(ctx)
    explicit = merged.loc[merged["bias_type"].isin(["incorrect_suggestion", "suggest_correct"])].copy()
    if explicit.empty:
        raise AnalysisNotSupportedError("No explicit-target bias templates are available for internal margin-shift analysis.")
    bias_order = _ordered_templates(explicit["bias_type"], include_neutral=False)
    fig, axes = plt.subplots(3, len(bias_order), figsize=(5.5 * len(bias_order), 12.0), sharey="row")
    if len(bias_order) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])
    for col, bias_type in enumerate(bias_order):
        subset = explicit.loc[explicit["bias_type"].astype(str).eq(bias_type)]
        color = BIAS_COLORS.get(bias_type, DEFAULT_COLORS["fallback"])
        sns.histplot(subset["delta_m_probe_bias"], bins=15, color=color, edgecolor="white", ax=axes[0, col])
        _set_panel_labels(axes[0, col], title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)), xlabel=r"$\Delta m_{probe}^{bias}=m(x')-m(x)$", ylabel="Count")
        sns.histplot(subset["delta_probe_score_correct"], bins=15, color=color, edgecolor="white", ax=axes[1, col])
        _set_panel_labels(axes[1, col], title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)), xlabel=r"$\Delta s_c=s(x',c)-s(x,c)$", ylabel="Count")
        sns.histplot(subset["delta_probe_score_bias_target"], bins=15, color=color, edgecolor="white", ax=axes[2, col])
        _set_panel_labels(axes[2, col], title=_format_panel_title(DISPLAY_LABELS.get(bias_type, bias_type.title()), len(subset)), xlabel=r"$\Delta s_b=s(x',b)-s(x,b)$", ylabel="Count")
    fig.suptitle("Internal Margin Shift Under Bias", fontsize=24, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


@register_analysis_function(
    output_kind="plot",
    description="Hidden-knowledge rate among biased wrong answers.",
    requires_probes=True,
    requires_labels=True,
)
def plot_hidden_knowledge_rate_biased_wrong(ctx: AnalysisContext) -> Any:
    merged = _merge_external_and_probe(ctx)
    if merged.empty:
        raise AnalysisNotSupportedError("No paired probe/external rows are available for hidden-knowledge analysis.")
    rows = []
    for bias_type in _ordered_templates(merged["bias_type"], include_neutral=False):
        subset = merged.loc[merged["bias_type"].astype(str).eq(bias_type)]
        wrong_subset = subset.loc[subset["correctness_xprime"].eq(0)].copy()
        if wrong_subset.empty:
            rows.append({"bias_type": bias_type, "metric": "HK", "rate": np.nan})
            rows.append({"bias_type": bias_type, "metric": "HK_strict", "rate": np.nan})
            continue
        hk = wrong_subset["probe_score_correct_xprime"] > wrong_subset["probe_score_selected_choice_xprime"]
        hk_strict = wrong_subset["rank_probe_xprime_correct"].eq(1)
        rows.append({"bias_type": bias_type, "metric": "HK", "rate": float(hk.mean())})
        rows.append({"bias_type": bias_type, "metric": "HK_strict", "rate": float(hk_strict.mean())})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.barplot(data=plot_df, x="bias_type", y="rate", hue="metric", palette=[DEFAULT_COLORS["primary"], DEFAULT_COLORS["secondary"]], edgecolor="black", ax=ax)
    _set_panel_labels(ax, title="Hidden-Knowledge Rate Among Biased Wrong Answers", xlabel="Bias type", ylabel="Rate")
    ax.set_ylim(0, 1)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_bbox_to_anchor((0.5, -0.18))
        legend._loc = 9
    fig.subplots_adjust(bottom=0.25)
    return fig


@register_analysis_function(
    output_kind="plot",
    description="Susceptibility versus neutral probe margin.",
    requires_probes=True,
    requires_labels=True,
    requires_bias_target=True,
)
def plot_susceptibility_vs_neutral_probe_margin(ctx: AnalysisContext, *, n_bins: int = 5) -> Any:
    merged = _merge_external_and_probe(ctx)
    merged = merged.loc[merged["bias_type"].isin(["incorrect_suggestion", "suggest_correct"])].copy()
    if merged.empty:
        raise AnalysisNotSupportedError("No explicit-target bias templates are available for susceptibility-vs-margin analysis.")
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.0), sharex="col")
    for col, bias_type in enumerate(["incorrect_suggestion", "suggest_correct"]):
        subset = merged.loc[merged["bias_type"].astype(str).eq(bias_type)].copy()
        predictor_column = "m_probe_truth_x" if bias_type == "suggest_correct" else "m_probe_bias_x"
        subset["target_gain"] = subset["delta_p_bias_target"]
        summary_gain = subset.dropna(subset=[predictor_column, "target_gain"]).copy()
        summary_gain["bucket"] = quantile_bucket_labels(summary_gain[predictor_column], n_bins=n_bins)
        gain_df = summary_gain.groupby("bucket", observed=False).agg(value=("target_gain", "mean"), n=("target_gain", "size")).reset_index()

        adoption_summary = _quantile_rate_summary(subset.dropna(subset=[predictor_column]).copy(), predictor_column, "adopts_bias_target", n_bins=n_bins)
        overall_rate = float(subset["adopts_bias_target"].mean()) if len(subset) else np.nan
        x_positions = np.arange(len(gain_df))
        axes[0, col].plot(x_positions, gain_df["value"], color=BIAS_COLORS[bias_type], marker="o")
        axes[0, col].set_xticks(x_positions)
        axes[0, col].set_xticklabels(gain_df["bucket"], rotation=45, ha="right")
        _set_panel_labels(axes[0, col], title=_format_panel_title(DISPLAY_LABELS[bias_type], len(subset)), xlabel="Neutral probe-margin quantile", ylabel="E[target gain | q]")

        rel_rate = adoption_summary["rate"] / overall_rate if overall_rate not in (0, np.nan) else np.nan
        axes[1, col].plot(np.arange(len(adoption_summary)), rel_rate, color=BIAS_COLORS[bias_type], marker="o")
        axes[1, col].set_xticks(np.arange(len(adoption_summary)))
        axes[1, col].set_xticklabels(adoption_summary["bucket"], rotation=45, ha="right")
        _set_panel_labels(axes[1, col], title=_format_panel_title(DISPLAY_LABELS[bias_type], len(subset)), xlabel="Neutral probe-margin quantile", ylabel="Adoption rate / overall")

        axes[2, col].plot(np.arange(len(adoption_summary)), adoption_summary["rate"], color=BIAS_COLORS[bias_type], marker="o")
        axes[2, col].fill_between(np.arange(len(adoption_summary)), adoption_summary["ci_low"], adoption_summary["ci_high"], color=BIAS_COLORS[bias_type], alpha=0.2)
        axes[2, col].set_xticks(np.arange(len(adoption_summary)))
        axes[2, col].set_xticklabels(adoption_summary["bucket"], rotation=45, ha="right")
        _set_panel_labels(axes[2, col], title=_format_panel_title(DISPLAY_LABELS[bias_type], len(subset)), xlabel="Neutral probe-margin quantile", ylabel="Adoption rate")
        axes[2, col].set_ylim(0, 1)
    fig.suptitle("Susceptibility vs Neutral Probe Margin", fontsize=24, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


@register_analysis_function(
    output_kind="plot",
    description="External-vs-internal predictor grid for target adoption.",
    requires_probes=True,
    requires_labels=True,
    requires_bias_target=True,
)
def plot_external_vs_internal_predictor_grid(ctx: AnalysisContext) -> Any:
    merged = _merge_external_and_probe(ctx)
    merged = merged.loc[merged["bias_type"].isin(["incorrect_suggestion", "suggest_correct"])].copy()
    if merged.empty:
        raise AnalysisNotSupportedError("No explicit-target bias templates are available for the external/internal predictor grid.")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, bias_type in zip(axes, ["incorrect_suggestion", "suggest_correct"]):
        subset = merged.loc[merged["bias_type"].astype(str).eq(bias_type)].copy()
        if bias_type == "suggest_correct":
            subset["external_signal"] = _probability_with_prefix(subset, "p_x_", subset["correct_letter"])
            subset["internal_signal"] = subset["m_probe_truth_x"]
        else:
            subset["external_signal"] = subset["m_bias_external_x"]
            subset["internal_signal"] = subset["m_probe_bias_x"]
        subset["external_bucket"] = np.where(subset["external_signal"] >= subset["external_signal"].median(), "high external", "low external")
        subset["internal_bucket"] = np.where(subset["internal_signal"] >= subset["internal_signal"].median(), "high internal", "low internal")
        grid = (
            subset.groupby(["external_bucket", "internal_bucket"], observed=False)["adopts_bias_target"]
            .mean()
            .unstack()
            .reindex(index=["low external", "high external"], columns=["low internal", "high internal"])
        )
        sns.heatmap(
            grid,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            cbar=False,
            linewidths=1.0,
            linecolor="white",
            ax=ax,
        )
        _set_panel_labels(ax, title=_format_panel_title(DISPLAY_LABELS[bias_type], len(subset)), xlabel="Neutral internal signal", ylabel="Neutral external signal")
    fig.suptitle("External vs Internal Predictor Grid", fontsize=24, y=1.02)
    fig.tight_layout()
    return fig


@register_analysis_function(
    output_kind="table",
    description="Candidate-ranking comparison using available rank metrics.",
    requires_probes=True,
    requires_labels=True,
)
def table_candidate_ranking_comparison(ctx: AnalysisContext) -> pd.DataFrame:
    merged = _merge_external_and_probe(ctx)
    if merged.empty:
        return pd.DataFrame()
    rows = []
    for bias_type in _ordered_templates(merged["bias_type"], include_neutral=False):
        subset = merged.loc[merged["bias_type"].astype(str).eq(bias_type)].copy()
        rows.extend(
            [
                {
                    "bias_type": bias_type,
                    "scorer": "external",
                    "prompt_condition": "neutral",
                    "mean_gold_rank": float(probability_rank(subset.rename(columns={f"p_x_{letter}": f"P({letter})" for letter in 'ABCDE'}), subset["correct_letter"]).mean()),
                    "mean_bias_target_rank": float(probability_rank(subset.rename(columns={f"p_x_{letter}": f"P({letter})" for letter in 'ABCDE'}), subset["bias_target_letter"]).mean()) if subset["bias_target_letter"].notna().any() else np.nan,
                },
                {
                    "bias_type": bias_type,
                    "scorer": "probe",
                    "prompt_condition": "neutral",
                    "mean_gold_rank": float(subset["rank_probe_x_correct"].mean()),
                    "mean_bias_target_rank": float(score_rank(subset.rename(columns={f"score_{letter}_x": f"score_{letter}" for letter in 'ABCDE'}), subset["bias_target_letter"]).mean()) if subset["bias_target_letter"].notna().any() else np.nan,
                },
                {
                    "bias_type": bias_type,
                    "scorer": "external",
                    "prompt_condition": "biased",
                    "mean_gold_rank": float(probability_rank(subset.rename(columns={f"p_xprime_{letter}": f"P({letter})" for letter in 'ABCDE'}), subset["correct_letter"]).mean()),
                    "mean_bias_target_rank": float(probability_rank(subset.rename(columns={f"p_xprime_{letter}": f"P({letter})" for letter in 'ABCDE'}), subset["bias_target_letter"]).mean()) if subset["bias_target_letter"].notna().any() else np.nan,
                },
                {
                    "bias_type": bias_type,
                    "scorer": "probe",
                    "prompt_condition": "biased",
                    "mean_gold_rank": float(subset["rank_probe_xprime_correct"].mean()),
                    "mean_bias_target_rank": float(score_rank(subset.rename(columns={f"score_{letter}_xprime": f"score_{letter}" for letter in 'ABCDE'}), subset["bias_target_letter"]).mean()) if subset["bias_target_letter"].notna().any() else np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


@register_analysis_function(
    output_kind="plot",
    description="Confident-yet-compliant slice under explicit-target bias.",
    requires_probes=True,
    requires_labels=True,
    requires_bias_target=True,
)
def plot_confident_yet_compliant_slice(ctx: AnalysisContext) -> Any:
    merged = _merge_external_and_probe(ctx)
    merged = merged.loc[merged["bias_type"].astype(str).eq("incorrect_suggestion")].copy()
    if merged.empty:
        raise AnalysisNotSupportedError("No incorrect_suggestion rows are available for the confident-yet-compliant slice.")
    threshold = merged["m_probe_bias_x"].quantile(0.75)
    subset = merged.loc[merged["m_probe_bias_x"] >= threshold].copy()
    wrong_target = subset["adopts_bias_target"]
    hk_mask = subset["correctness_xprime"].eq(0) & (subset["probe_score_correct_xprime"] > subset["probe_score_selected_choice_xprime"])
    metrics_df = pd.DataFrame(
        {
            "metric": ["Pr(adopt bias target)", "Pr(HK | biased wrong)", "Mean delta m_probe^bias"],
            "value": [
                float(wrong_target.mean()),
                float(hk_mask.loc[subset["correctness_xprime"].eq(0)].mean()) if subset["correctness_xprime"].eq(0).any() else np.nan,
                float(subset["delta_m_probe_bias"].mean()),
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    sns.barplot(data=metrics_df, x="metric", y="value", color=DEFAULT_COLORS["secondary"], edgecolor="black", ax=ax)
    _set_panel_labels(ax, title=_format_panel_title("Confident Yet Compliant Slice", len(subset)), xlabel="", ylabel="Value")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    return fig
