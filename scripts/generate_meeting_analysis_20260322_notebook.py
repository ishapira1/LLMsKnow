from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "analysis_20260322_v3.ipynb"


INTRO_MARKDOWN = dedent(
    """
    # analysis_20260322: Partial full-pipeline findings

    This notebook summarizes the currently completed full-pipeline runs available in the repo on March 22, 2026 for the upcoming research meeting.

    **Included runs**
    - `full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas`
    - `full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas`
    - `full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas`
    - `full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun`
    - non-probe `gpt_5_4_nano` CommonsenseQA run currently present in the active results tree

    **Goals**
    1. summarize the top-line sycophancy pattern for each dataset-model pair without mixing different pairs inside the same figure
    2. show how agreement-bias prompts move accuracy and `p(correct)` relative to neutral prompts
    3. isolate the harmful `incorrect_suggestion` transition pattern
    4. connect neutral external confidence to later susceptibility using a **friction** framing
    """
).strip()


SETUP_CODE = dedent(
    """
    import math
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from IPython.display import Markdown, display

    sns.set_style("white")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "axes.titlesize": 20,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    def resolve_repo_root() -> Path:
        cwd = Path.cwd().resolve()
        for candidate in [cwd, *cwd.parents]:
            if (candidate / "src" / "llmssycoph").exists():
                return candidate
        raise FileNotFoundError("Could not locate repo root containing src/llmssycoph.")

    REPO_ROOT = resolve_repo_root()
    ARTIFACT_DIR = REPO_ROOT / "notebooks" / "analysis_20260322_artifacts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DELTA_PLOTS_DIR = ARTIFACT_DIR / "delta_p_correct"
    TRANSITION_PLOTS_DIR = ARTIFACT_DIR / "incorrect_suggestion_transitions"
    FRICTION_PLOTS_DIR = ARTIFACT_DIR / "friction_relative_drop"
    HARMFUL_PLOTS_DIR = ARTIFACT_DIR / "harmful_incorrect_suggestion"
    for _directory in [DELTA_PLOTS_DIR, TRANSITION_PLOTS_DIR, FRICTION_PLOTS_DIR, HARMFUL_PLOTS_DIR]:
        _directory.mkdir(parents=True, exist_ok=True)

    RUN_SPECS = [
        {
            "run_key": "llama31_8b__commonsense_qa",
            "model_label": "Llama 3.1 8B Instruct",
            "dataset_label": "CommonsenseQA",
            "run_name": "full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas",
            "relative_run_dir": "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/commonsense_qa/full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas",
        },
        {
            "run_key": "llama31_8b__arc_challenge",
            "model_label": "Llama 3.1 8B Instruct",
            "dataset_label": "ARC-Challenge",
            "run_name": "full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas",
            "relative_run_dir": "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/arc_challenge/full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas",
        },
        {
            "run_key": "qwen25_7b__commonsense_qa",
            "model_label": "Qwen 2.5 7B Instruct",
            "dataset_label": "CommonsenseQA",
            "run_name": "full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas",
            "relative_run_dir": "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/commonsense_qa/full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas",
        },
        {
            "run_key": "qwen25_7b__arc_challenge",
            "model_label": "Qwen 2.5 7B Instruct",
            "dataset_label": "ARC-Challenge",
            "run_name": "full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun",
            "relative_run_dir": "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/arc_challenge/full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun",
        },
        {
            "run_key": "gpt54nano__commonsense_qa",
            "model_label": "GPT-5.4 Nano",
            "dataset_label": "CommonsenseQA",
            "run_name": "full_gpt54nano_commonsense_qa",
            "relative_run_dir": "results/sycophancy_bias_probe/gpt_5_4_nano/commonsense_qa/full_gpt54nano_commonsense_qa",
        },
    ]

    BIAS_ORDER = ["neutral", "incorrect_suggestion", "doubt_correct", "suggest_correct"]
    BIAS_ONLY_ORDER = [bias for bias in BIAS_ORDER if bias != "neutral"]
    PROXY_ORDER = ["neutral_p_selected", "neutral_chosen_margin", "neutral_effective_responses"]

    BIAS_LABELS = {
        "neutral": "Neutral",
        "incorrect_suggestion": "Incorrect Suggestion",
        "doubt_correct": "Doubt Correct",
        "suggest_correct": "Suggest Correct",
    }

    PROXY_LABELS = {
        "neutral_p_selected": "Neutral P(selected)",
        "neutral_chosen_margin": "Neutral chosen margin",
        "neutral_effective_responses": "Neutral effective responses",
    }

    COLORS = {
        "neutral": "#5b6770",
        "incorrect_suggestion": "#d4651a",
        "doubt_correct": "#8c7a5b",
        "suggest_correct": "#73b3ab",
        "stay_correct": "#73b3ab",
        "agree_suggested_bias": "#d4651a",
        "incorrect_other": "#7a8793",
        "relative_drop": "#1f4e79",
        "raw_delta": "#4c6272",
        "adopt_bias_target_rate": "#d4651a",
        "harmful_flip_rate": "#4c6272",
        "mean_relative_drop_neutral_choice": "#1f4e79",
        "mean_delta_neutral_choice_probability": "#4c6272",
    }

    NEUTRAL_STATE_ORDER = [
        "neutral_correct",
        "neutral_agree_suggested_bias",
        "neutral_incorrect_other",
    ]
    NEUTRAL_STATE_LABELS = {
        "neutral_correct": "Neutral: correct",
        "neutral_agree_suggested_bias": "Neutral: agree with suggested bias",
        "neutral_incorrect_other": "Neutral: incorrect other",
    }
    BIASED_STATE_ORDER = ["stay_correct", "agree_suggested_bias", "incorrect_other"]
    BIASED_STATE_LABELS = {
        "stay_correct": "Stay correct",
        "agree_suggested_bias": "Agree with suggested bias",
        "incorrect_other": "Incorrect other",
    }

    SAVED_ARTIFACTS = []

    def pct(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.1%}"

    def rate(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.3f}"

    def artifact_path(stem: str, subdir: Path | None = None) -> Path:
        base_dir = subdir if subdir is not None else ARTIFACT_DIR
        return base_dir / f"{stem}.png"

    def save_figure(fig: plt.Figure, stem: str, *, subdir: Path | None = None) -> Path:
        path = artifact_path(stem, subdir=subdir)
        fig.savefig(path, bbox_inches="tight", dpi=220)
        SAVED_ARTIFACTS.append(path)
        return path
    """
).strip()


HELPERS_CODE = dedent(
    """
    def load_sampled_df(run_dir: Path) -> pd.DataFrame:
        path = run_dir / "sampling" / "sampled_responses.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing sampled responses: {path}")
        df = pd.read_csv(path)
        if "usable_for_metrics" in df.columns:
            usable_mask = df["usable_for_metrics"].astype(str).str.strip().str.lower().eq("true")
            df = df.loc[usable_mask].copy()
        else:
            df = df.copy()
        if "correctness" in df.columns:
            df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce")
        for column in ["correct_letter", "incorrect_letter", "response"]:
            if column in df.columns:
                df[column] = df[column].astype(str).str.strip().str.upper()
        probability_columns = [column for column in df.columns if column.startswith("P(")]
        for column in probability_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def build_accuracy_by_template(sampled_df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            sampled_df.groupby("template_type", observed=True)
            .agg(
                n_rows=("question_id", "size"),
                n_questions=("question_id", "nunique"),
                accuracy=("correctness", "mean"),
                avg_p_selected=("P(selected)", "mean"),
                avg_p_correct=("P(correct)", "mean"),
            )
            .reset_index()
        )
        summary["template_type"] = pd.Categorical(summary["template_type"], categories=BIAS_ORDER, ordered=True)
        summary = summary.sort_values("template_type").reset_index(drop=True)
        neutral_accuracy = summary.loc[summary["template_type"].eq("neutral"), "accuracy"]
        neutral_accuracy = float(neutral_accuracy.iloc[0]) if not neutral_accuracy.empty else np.nan
        summary["delta_accuracy_vs_neutral"] = summary["accuracy"] - neutral_accuracy
        return summary

    def build_paired_df(sampled_df: pd.DataFrame) -> pd.DataFrame:
        join_keys = [column for column in ["question_id", "split", "draw_idx"] if column in sampled_df.columns]
        candidate_columns = [f"P({letter})" for letter in "ABCDE" if f"P({letter})" in sampled_df.columns]

        neutral_df = sampled_df.loc[sampled_df["template_type"].eq("neutral")].copy()
        neutral_df = neutral_df.rename(
            columns={
                "response": "response_x",
                "correctness": "correctness_x",
                "P(selected)": "p_selected_x",
                "P(correct)": "p_correct_x",
            }
        )
        for column in candidate_columns:
            neutral_df = neutral_df.rename(columns={column: f"p_x_{column[2]}"})

        neutral_keep = (
            join_keys
            + ["correct_letter", "incorrect_letter", "dataset", "response_x", "correctness_x", "p_selected_x", "p_correct_x"]
            + [f"p_x_{column[2]}" for column in candidate_columns]
        )
        neutral_keep = [column for column in neutral_keep if column in neutral_df.columns]
        neutral_df = neutral_df[neutral_keep].drop_duplicates(subset=join_keys)

        paired_frames = []
        for bias_type in BIAS_ONLY_ORDER:
            bias_df = sampled_df.loc[sampled_df["template_type"].eq(bias_type)].copy()
            bias_df = bias_df.rename(
                columns={
                    "response": "response_xprime",
                    "correctness": "correctness_xprime",
                    "P(selected)": "p_selected_xprime",
                    "P(correct)": "p_correct_xprime",
                }
            )
            for column in candidate_columns:
                bias_df = bias_df.rename(columns={column: f"p_xprime_{column[2]}"})

            bias_keep = (
                join_keys
                + ["response_xprime", "correctness_xprime", "p_selected_xprime", "p_correct_xprime"]
                + [f"p_xprime_{column[2]}" for column in candidate_columns]
            )
            bias_keep = [column for column in bias_keep if column in bias_df.columns]
            merged = neutral_df.merge(
                bias_df[bias_keep].drop_duplicates(subset=join_keys),
                on=join_keys,
                how="inner",
            )
            merged["bias_type"] = bias_type
            paired_frames.append(merged)

        if not paired_frames:
            return pd.DataFrame()

        paired = pd.concat(paired_frames, ignore_index=True)

        def probability_of_selected(row: pd.Series, prefix: str, selected_option: str) -> float:
            option = str(selected_option).strip().upper()
            column = f"{prefix}{option}"
            return float(row[column]) if column in row.index and pd.notna(row[column]) else np.nan

        def chosen_margin(row: pd.Series, prefix: str, selected_option: str) -> float:
            option = str(selected_option).strip().upper()
            selected_column = f"{prefix}{option}"
            if selected_column not in row.index or pd.isna(row[selected_column]):
                return np.nan
            other_values = []
            for letter in "ABCDE":
                if letter == option:
                    continue
                column = f"{prefix}{letter}"
                if column in row.index and pd.notna(row[column]):
                    other_values.append(float(row[column]))
            if not other_values:
                return np.nan
            return float(row[selected_column]) - max(other_values)

        def effective_responses(row: pd.Series, prefix: str) -> float:
            values = []
            for letter in "ABCDE":
                column = f"{prefix}{letter}"
                if column in row.index and pd.notna(row[column]) and float(row[column]) > 0:
                    values.append(float(row[column]))
            if not values:
                return np.nan
            values_array = np.asarray(values, dtype=float)
            entropy = -(values_array * np.log(values_array)).sum()
            return float(np.exp(entropy))

        paired["neutral_p_chosen"] = paired.apply(
            lambda row: probability_of_selected(row, "p_x_", row["response_x"]),
            axis=1,
        )
        paired["neutral_p_selected"] = paired["p_selected_x"]
        paired["neutral_chosen_margin"] = paired.apply(
            lambda row: chosen_margin(row, "p_x_", row["response_x"]),
            axis=1,
        )
        paired["biased_p_of_neutral_choice"] = paired.apply(
            lambda row: probability_of_selected(row, "p_xprime_", row["response_x"]),
            axis=1,
        )
        paired["neutral_effective_responses"] = paired.apply(
            lambda row: effective_responses(row, "p_x_"),
            axis=1,
        )
        paired["delta_p_correct"] = paired["p_correct_xprime"] - paired["p_correct_x"]
        paired["delta_neutral_choice_probability"] = paired["biased_p_of_neutral_choice"] - paired["neutral_p_chosen"]
        paired["relative_drop_neutral_choice"] = np.where(
            paired["neutral_p_chosen"] > 0,
            (paired["neutral_p_chosen"] - paired["biased_p_of_neutral_choice"]) / paired["neutral_p_chosen"],
            np.nan,
        )
        paired["answer_change"] = paired["response_x"].ne(paired["response_xprime"]).astype(int)
        paired["harmful_flip"] = (paired["correctness_x"].eq(1) & paired["correctness_xprime"].eq(0)).astype(int)
        paired["helpful_flip"] = (paired["correctness_x"].eq(0) & paired["correctness_xprime"].eq(1)).astype(int)

        paired["bias_target_letter"] = np.where(
            paired["bias_type"].eq("incorrect_suggestion"),
            paired["incorrect_letter"],
            np.where(paired["bias_type"].eq("suggest_correct"), paired["correct_letter"], np.nan),
        )
        paired["adopt_bias_target"] = paired["response_xprime"].eq(paired["bias_target_letter"]).astype(int)
        return paired

    def build_paired_summary(paired_df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            paired_df.groupby("bias_type", observed=True)
            .agg(
                n_pairs=("question_id", "size"),
                neutral_accuracy=("correctness_x", "mean"),
                biased_accuracy=("correctness_xprime", "mean"),
                mean_delta_p_correct=("delta_p_correct", "mean"),
                median_delta_p_correct=("delta_p_correct", "median"),
                harmful_flip_rate=("harmful_flip", "mean"),
                helpful_flip_rate=("helpful_flip", "mean"),
                answer_change_rate=("answer_change", "mean"),
                adopt_bias_target_rate=("adopt_bias_target", "mean"),
            )
            .reset_index()
        )
        summary["delta_accuracy_vs_neutral"] = summary["biased_accuracy"] - summary["neutral_accuracy"]
        summary["bias_type"] = pd.Categorical(summary["bias_type"], categories=BIAS_ONLY_ORDER, ordered=True)
        return summary.sort_values("bias_type").reset_index(drop=True)

    def classify_neutral_state(row: pd.Series) -> str:
        if row["response_x"] == row["correct_letter"]:
            return "neutral_correct"
        if row["response_x"] == row["incorrect_letter"]:
            return "neutral_agree_suggested_bias"
        return "neutral_incorrect_other"

    def classify_biased_state(row: pd.Series) -> str:
        if row["response_xprime"] == row["correct_letter"]:
            return "stay_correct"
        if row["response_xprime"] == row["incorrect_letter"]:
            return "agree_suggested_bias"
        return "incorrect_other"

    def build_incorrect_transition_df(paired_df: pd.DataFrame) -> pd.DataFrame:
        incorrect_df = paired_df.loc[paired_df["bias_type"].eq("incorrect_suggestion")].copy()
        incorrect_df["neutral_state"] = incorrect_df.apply(classify_neutral_state, axis=1)
        incorrect_df["biased_state"] = incorrect_df.apply(classify_biased_state, axis=1)
        transition_df = (
            incorrect_df.groupby(["neutral_state", "biased_state"], observed=True)
            .size()
            .rename("n")
            .reset_index()
        )
        transition_df["fraction_within_neutral_state"] = (
            transition_df["n"] / transition_df.groupby("neutral_state", observed=True)["n"].transform("sum")
        )
        transition_df["neutral_state"] = pd.Categorical(
            transition_df["neutral_state"], categories=NEUTRAL_STATE_ORDER, ordered=True
        )
        transition_df["biased_state"] = pd.Categorical(
            transition_df["biased_state"], categories=BIASED_STATE_ORDER, ordered=True
        )
        return transition_df.sort_values(["neutral_state", "biased_state"]).reset_index(drop=True)

    def bucket_into_quintiles(series: pd.Series) -> pd.Series:
        ranked = series.rank(method="first")
        bucket_ids = pd.qcut(ranked, q=min(5, len(ranked)), labels=False, duplicates="drop")
        return bucket_ids.astype(int) + 1

    def build_friction_df(paired_df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for proxy in PROXY_ORDER:
            proxy_df = paired_df.dropna(subset=[proxy]).copy()
            if proxy_df.empty:
                continue
            for bias_type in BIAS_ONLY_ORDER:
                subset = proxy_df.loc[proxy_df["bias_type"].eq(bias_type)].copy()
                if len(subset) < 5:
                    continue
                subset["bucket_id"] = bucket_into_quintiles(subset[proxy])
                aggregated = (
                    subset.groupby("bucket_id", observed=True)
                    .agg(
                        n=("question_id", "size"),
                        answer_change_rate=("answer_change", "mean"),
                        harmful_flip_rate=("harmful_flip", "mean"),
                        helpful_flip_rate=("helpful_flip", "mean"),
                        adopt_bias_target_rate=("adopt_bias_target", "mean"),
                        mean_delta_p_correct=("delta_p_correct", "mean"),
                        mean_relative_drop_neutral_choice=("relative_drop_neutral_choice", "mean"),
                        mean_delta_neutral_choice_probability=("delta_neutral_choice_probability", "mean"),
                        mean_proxy_value=(proxy, "mean"),
                    )
                    .reset_index()
                )
                aggregated["proxy"] = proxy
                aggregated["bias_type"] = bias_type
                frames.append(aggregated)
        if not frames:
            return pd.DataFrame()
        friction_df = pd.concat(frames, ignore_index=True)
        friction_df["proxy"] = pd.Categorical(friction_df["proxy"], categories=PROXY_ORDER, ordered=True)
        friction_df["bias_type"] = pd.Categorical(friction_df["bias_type"], categories=BIAS_ONLY_ORDER, ordered=True)
        return friction_df.sort_values(["proxy", "bias_type", "bucket_id"]).reset_index(drop=True)

    def prepare_run(run_spec: dict) -> dict:
        run_dir = REPO_ROOT / run_spec["relative_run_dir"]
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        sampled_df = load_sampled_df(run_dir)
        accuracy_df = build_accuracy_by_template(sampled_df)
        paired_df = build_paired_df(sampled_df)
        paired_summary_df = build_paired_summary(paired_df)
        transition_df = build_incorrect_transition_df(paired_df)
        friction_df = build_friction_df(paired_df)
        probe_available = (run_dir / "probes" / "probe_scores_by_prompt.csv").exists()
        run_label = f"{run_spec['model_label']} / {run_spec['dataset_label']}"
        return {
            **run_spec,
            "run_dir": run_dir,
            "run_label": run_label,
            "probe_available": probe_available,
            "sampled_df": sampled_df,
            "accuracy_df": accuracy_df,
            "paired_df": paired_df,
            "paired_summary_df": paired_summary_df,
            "transition_df": transition_df,
            "friction_df": friction_df,
        }

    def build_inventory_df(runs: dict[str, dict]) -> pd.DataFrame:
        rows = []
        for run in runs.values():
            neutral_rows = run["sampled_df"].loc[run["sampled_df"]["template_type"].eq("neutral")]
            rows.append(
                {
                    "run_label": run["run_label"],
                    "model": run["model_label"],
                    "dataset": run["dataset_label"],
                    "run_name": run["run_name"],
                    "question_count": int(neutral_rows["question_id"].nunique()),
                    "usable_prompt_rows": int(len(run["sampled_df"])),
                    "probe_available": run["probe_available"],
                    "run_dir": str(run["run_dir"]),
                }
            )
        return pd.DataFrame(rows)

    def build_topline_df(runs: dict[str, dict]) -> pd.DataFrame:
        rows = []
        for run in runs.values():
            accuracy_lookup = run["accuracy_df"].set_index("template_type")
            paired_lookup = run["paired_summary_df"].set_index("bias_type")
            rows.append(
                {
                    "run_label": run["run_label"],
                    "neutral_accuracy": float(accuracy_lookup.loc["neutral", "accuracy"]),
                    "incorrect_suggestion_accuracy": float(accuracy_lookup.loc["incorrect_suggestion", "accuracy"]),
                    "incorrect_suggestion_delta_accuracy": float(paired_lookup.loc["incorrect_suggestion", "delta_accuracy_vs_neutral"]),
                    "incorrect_suggestion_harmful_flip_rate": float(paired_lookup.loc["incorrect_suggestion", "harmful_flip_rate"]),
                    "incorrect_suggestion_adopt_bias_target_rate": float(paired_lookup.loc["incorrect_suggestion", "adopt_bias_target_rate"]),
                    "doubt_correct_accuracy": float(accuracy_lookup.loc["doubt_correct", "accuracy"]),
                    "doubt_correct_delta_accuracy": float(paired_lookup.loc["doubt_correct", "delta_accuracy_vs_neutral"]),
                    "suggest_correct_accuracy": float(accuracy_lookup.loc["suggest_correct", "accuracy"]),
                    "suggest_correct_delta_accuracy": float(paired_lookup.loc["suggest_correct", "delta_accuracy_vs_neutral"]),
                }
            )
        return pd.DataFrame(rows)

    def format_accuracy_with_delta(accuracy: float, delta: float | None = None) -> str:
        accuracy_text = f"{100 * accuracy:.1f}%"
        if delta is None or pd.isna(delta):
            return accuracy_text
        return f"{accuracy_text} [{100 * delta:+.1f}]"

    def build_topline_display_df(runs: dict[str, dict]) -> pd.DataFrame:
        rows = []
        for run in runs.values():
            accuracy_lookup = run["accuracy_df"].set_index("template_type")
            paired_lookup = run["paired_summary_df"].set_index("bias_type")
            rows.append(
                {
                    "Model": run["model_label"],
                    "Dataset": run["dataset_label"],
                    "Neutral": format_accuracy_with_delta(float(accuracy_lookup.loc["neutral", "accuracy"])),
                    "Incorrect Suggestion": format_accuracy_with_delta(
                        float(accuracy_lookup.loc["incorrect_suggestion", "accuracy"]),
                        float(paired_lookup.loc["incorrect_suggestion", "delta_accuracy_vs_neutral"]),
                    ),
                    "Doubt Correct": format_accuracy_with_delta(
                        float(accuracy_lookup.loc["doubt_correct", "accuracy"]),
                        float(paired_lookup.loc["doubt_correct", "delta_accuracy_vs_neutral"]),
                    ),
                    "Suggest Correct": format_accuracy_with_delta(
                        float(accuracy_lookup.loc["suggest_correct", "accuracy"]),
                        float(paired_lookup.loc["suggest_correct", "delta_accuracy_vs_neutral"]),
                    ),
                }
            )
        return pd.DataFrame(rows)
    """
).strip()


INVENTORY_CODE = dedent(
    """
    RUNS = {run_spec["run_key"]: prepare_run(run_spec) for run_spec in RUN_SPECS}

    inventory_df = build_inventory_df(RUNS)
    topline_df = build_topline_df(RUNS)
    topline_display_df = build_topline_display_df(RUNS)

    inventory_df.to_csv(ARTIFACT_DIR / "run_inventory.csv", index=False)
    topline_df.to_csv(ARTIFACT_DIR / "topline_summary.csv", index=False)
    topline_display_df.to_csv(ARTIFACT_DIR / "topline_summary_display.csv", index=False)

    display(Markdown("## Coverage"))
    display(inventory_df)

    display(Markdown("## Top-line table"))
    display(topline_display_df)

    worst_incorrect = topline_df.loc[topline_df["incorrect_suggestion_delta_accuracy"].idxmin()]
    best_suggest = topline_df.loc[topline_df["suggest_correct_delta_accuracy"].idxmax()]
    all_negative = int((topline_df["incorrect_suggestion_delta_accuracy"] < 0).sum())
    all_doubt_negative = int((topline_df["doubt_correct_delta_accuracy"] < 0).sum())
    all_positive = int((topline_df["suggest_correct_delta_accuracy"] > 0).sum())

    summary_lines = [
        "## Meeting framing",
        "",
        f"- `incorrect_suggestion` lowers accuracy in {all_negative}/{len(topline_df)} currently available runs.",
        f"- `doubt_correct` also lowers accuracy in {all_doubt_negative}/{len(topline_df)} currently available runs.",
        f"- `suggest_correct` raises accuracy in {all_positive}/{len(topline_df)} currently available runs.",
        (
            f"- The sharpest currently available `incorrect_suggestion` drop is "
            f"**{worst_incorrect['run_label']}**: "
            f"{rate(worst_incorrect['neutral_accuracy'])} -> "
            f"{rate(worst_incorrect['incorrect_suggestion_accuracy'])} "
            f"({worst_incorrect['incorrect_suggestion_delta_accuracy']:+.3f})."
        ),
        (
            f"- The largest currently available `suggest_correct` boost is "
            f"**{best_suggest['run_label']}**: "
            f"{rate(best_suggest['neutral_accuracy'])} -> "
            f"{rate(best_suggest['suggest_correct_accuracy'])} "
            f"({best_suggest['suggest_correct_delta_accuracy']:+.3f})."
        ),
        "- The GPT-5.4 Nano runs in this notebook are sampling-only; probe outputs are intentionally not used here so the meeting summary stays on common ground across all included runs.",
    ]
    display(Markdown("\\n".join(summary_lines)))
    """
).strip()


GENERAL_MARKDOWN = dedent(
    """
    ## General sycophancy figures

    Every plot below is **run-specific**. Different model-dataset pairs are not mixed inside the same figure.
    """
).strip()


DELTA_CODE = dedent(
    """
    def plot_delta_histograms(run: dict) -> tuple[plt.Figure, pd.DataFrame]:
        paired_df = run["paired_df"].copy()
        summary_df = (
            paired_df.groupby("bias_type", observed=True)["delta_p_correct"]
            .agg(n_pairs="size", mean="mean", median="median", std="std", min="min", max="max")
            .reset_index()
        )
        summary_df["bias_type"] = pd.Categorical(summary_df["bias_type"], categories=BIAS_ONLY_ORDER, ordered=True)
        summary_df = summary_df.sort_values("bias_type").reset_index(drop=True)

        global_min = float(paired_df["delta_p_correct"].min())
        global_max = float(paired_df["delta_p_correct"].max())
        padding = max(0.02, 0.05 * max(global_max - global_min, 0.10))
        bins = np.linspace(global_min - padding, global_max + padding, 31)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
        for ax, bias_type in zip(axes, BIAS_ONLY_ORDER):
            subset = paired_df.loc[paired_df["bias_type"].eq(bias_type)].copy()
            sns.histplot(
                subset["delta_p_correct"],
                bins=bins,
                color=COLORS[bias_type],
                edgecolor="white",
                alpha=0.9,
                ax=ax,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
            ax.axvline(subset["delta_p_correct"].mean(), color=COLORS["neutral"], linestyle=":", linewidth=2)
            ax.set_title(BIAS_LABELS[bias_type])
            ax.set_xlabel("Biased - neutral p(correct)")
            if ax is axes[0]:
                ax.set_ylabel("Count")
            else:
                ax.set_ylabel("")
            ax.text(
                0.02,
                0.95,
                f"mean = {subset['delta_p_correct'].mean():+.3f}\\nmedian = {subset['delta_p_correct'].median():+.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
            )
            sns.despine(ax=ax)

        fig.suptitle(f"Effect of presenting agreement bias on p(correct)\\n{run['run_label']}", y=1.03, fontsize=20)
        fig.tight_layout()
        return fig, summary_df

    display(Markdown("### 2. Delta correct before and after bias presentation"))
    for run in RUNS.values():
        display(Markdown(f"#### {run['run_label']}"))
        fig, summary_df = plot_delta_histograms(run)
        save_figure(fig, f"delta_histograms__{run['run_key']}", subdir=DELTA_PLOTS_DIR)
        display(summary_df.round(3))
        plt.show()
        plt.close(fig)
    """
).strip()


TRANSITION_CODE = dedent(
    """
    def plot_incorrect_transition(run: dict) -> tuple[plt.Figure, pd.DataFrame]:
        transition_df = run["transition_df"].copy()
        pivot_df = (
            transition_df.pivot(
                index="neutral_state",
                columns="biased_state",
                values="fraction_within_neutral_state",
            )
            .reindex(index=NEUTRAL_STATE_ORDER, columns=BIASED_STATE_ORDER)
            .fillna(0.0)
        )

        fig, ax = plt.subplots(figsize=(11, 6))
        bottom = np.zeros(len(pivot_df))
        x_positions = np.arange(len(pivot_df)) * 1.3
        for biased_state in BIASED_STATE_ORDER:
            values = pivot_df[biased_state].to_numpy(dtype=float)
            bars = ax.bar(
                x_positions,
                values,
                bottom=bottom,
                width=0.5,
                color=COLORS[biased_state],
                label=BIASED_STATE_LABELS[biased_state],
            )
            for bar, value, bottom_value in zip(bars, values, bottom):
                if value >= 0.06:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom_value + value / 2,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="white" if biased_state != "stay_correct" else "black",
                    )
            bottom += values

        ax.set_title(f"`incorrect_suggestion`: where neutral answers move\\n{run['run_label']}")
        ax.set_xlabel("")
        ax.set_ylabel("Fraction within neutral answer type")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([NEUTRAL_STATE_LABELS[state] for state in pivot_df.index], rotation=18, ha="right")
        ax.set_ylim(0, 1.02)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
        )
        ax.grid(False)
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig, transition_df

    display(Markdown("### 3. `incorrect_suggestion` transition view"))
    for run in RUNS.values():
        display(Markdown(f"#### {run['run_label']}"))
        fig, transition_df = plot_incorrect_transition(run)
        save_figure(fig, f"incorrect_transition__{run['run_key']}", subdir=TRANSITION_PLOTS_DIR)
        display(
            transition_df.assign(
                neutral_state=lambda df: df["neutral_state"].map(NEUTRAL_STATE_LABELS),
                biased_state=lambda df: df["biased_state"].map(BIASED_STATE_LABELS),
            ).round(3)
        )
        plt.show()
        plt.close(fig)
    """
).strip()


FRICTION_MARKDOWN = dedent(
    """
    ## Confidence and friction

    To avoid a circular argument, the notebook uses **neutral-prompt** external confidence as the predictor and **biased-prompt movement** as the outcome.

    Concretely:
    - the predictor is measured **before** the agreement-bias push is introduced
    - the main confidence proxies are label-free or nearly label-free: neutral `P(selected)`, neutral chosen-margin, and neutral effective responses
    - the main susceptibility outcome in the main friction plot is the **relative erosion of the model's original neutral choice probability**, alongside the raw delta of that same quantity
    - the harmful follow-up view shows that erosion alongside harmful flips and adoption of the suggested incorrect answer

    I am **not** using `P(correct | neutral)` as the main friction proxy here because it mixes knowledge with confidence and can look low even when the model is confidently wrong. That makes it much less clean for the meeting story.

    Quantiles are computed **within each run, bias type, and proxy separately** using equal-frequency bins. That means a small Q1-to-Q2 bump does **not** imply the bins are wrong. It usually means the relationship is only approximately monotone, with some low-confidence items still having enough internal pull to resist the bias push.
    """
).strip()


FRICTION_CODE = dedent(
    """
    def plot_friction_relative_drop(run: dict) -> tuple[plt.Figure, pd.DataFrame]:
        friction_df = run["friction_df"].copy()
        fig, axes = plt.subplots(2, 3, figsize=(18, 9.2), sharex=False, sharey="row")

        handles = []
        labels = []
        metric_specs = [
            (
                "mean_relative_drop_neutral_choice",
                "Mean relative drop of original choice probability",
                (0.0, 1.0),
            ),
            (
                "mean_delta_neutral_choice_probability",
                "Mean raw delta of original choice probability",
                (-1.0, 0.05),
            ),
        ]
        for row_idx, (metric, y_label, y_limits) in enumerate(metric_specs):
            for col_idx, proxy in enumerate(PROXY_ORDER):
                ax = axes[row_idx, col_idx]
                subset = friction_df.loc[friction_df["proxy"].eq(proxy)].copy()
                for bias_type in BIAS_ONLY_ORDER:
                    bias_subset = subset.loc[subset["bias_type"].eq(bias_type)].sort_values("bucket_id")
                    line = ax.plot(
                        bias_subset["bucket_id"],
                        bias_subset[metric],
                        marker="o",
                        linewidth=2.6,
                        color=COLORS[bias_type],
                        label=BIAS_LABELS[bias_type],
                    )[0]
                    if row_idx == 0 and col_idx == 0:
                        handles.append(line)
                        labels.append(BIAS_LABELS[bias_type])
                ax.set_title(PROXY_LABELS[proxy])
                ax.set_xlabel("Quintile (Q1 = lower values, Q5 = higher values)")
                if col_idx == 0:
                    ax.set_ylabel(y_label)
                ax.set_xticks([1, 2, 3, 4, 5])
                if metric == "mean_relative_drop_neutral_choice":
                    upper = subset["mean_relative_drop_neutral_choice"].max()
                    ax.set_ylim(0, min(y_limits[1], upper + 0.08))
                else:
                    lower = subset["mean_delta_neutral_choice_probability"].min()
                    ax.set_ylim(max(y_limits[0], lower - 0.05), y_limits[1])
                ax.grid(False)
                sns.despine(ax=ax)

        fig.suptitle(
            f"Friction view: neutral confidence vs erosion of the original choice\\n{run['run_label']}",
            y=1.04,
            fontsize=20,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=3,
            frameon=False,
        )
        fig.tight_layout()
        return fig, friction_df

    display(Markdown("### 4. Friction: neutral confidence vs erosion of the original neutral choice"))
    for run in RUNS.values():
        display(Markdown(f"#### {run['run_label']}"))
        fig, friction_df = plot_friction_relative_drop(run)
        save_figure(fig, f"friction_relative_drop__{run['run_key']}", subdir=FRICTION_PLOTS_DIR)
        display(
            friction_df[
                [
                    "proxy",
                    "bias_type",
                    "bucket_id",
                    "mean_proxy_value",
                    "mean_relative_drop_neutral_choice",
                    "mean_delta_neutral_choice_probability",
                    "mean_delta_p_correct",
                ]
            ].round(3)
        )
        plt.show()
        plt.close(fig)
    """
).strip()


HARMFUL_CODE = dedent(
    """
    def plot_incorrect_harm_all_proxies(run: dict) -> tuple[plt.Figure, pd.DataFrame]:
        subset = run["friction_df"].loc[run["friction_df"]["bias_type"].eq("incorrect_suggestion")].copy()
        metric_labels = {
            "mean_relative_drop_neutral_choice": "Relative drop of original choice",
            "mean_delta_neutral_choice_probability": "Raw delta of original choice",
            "adopt_bias_target_rate": "Adopt suggested incorrect answer",
            "harmful_flip_rate": "Correct -> incorrect",
        }

        fig, axes = plt.subplots(2, 3, figsize=(18, 9.2), sharex=False, sharey="row")
        top_metrics = ["mean_relative_drop_neutral_choice", "mean_delta_neutral_choice_probability"]
        bottom_metrics = ["adopt_bias_target_rate", "harmful_flip_rate"]
        for col_idx, proxy in enumerate(PROXY_ORDER):
            proxy_df = subset.loc[subset["proxy"].eq(proxy)].sort_values("bucket_id")
            top_ax = axes[0, col_idx]
            for metric in top_metrics:
                top_ax.plot(
                    proxy_df["bucket_id"],
                    proxy_df[metric],
                    marker="o",
                    linewidth=2.6,
                    color=COLORS[metric],
                    label=metric_labels[metric],
                )
            top_ax.set_title(PROXY_LABELS[proxy])
            top_ax.set_xlabel("Quintile (Q1 lower values, Q5 higher values)")
            if col_idx == 0:
                top_ax.set_ylabel("Change in original choice probability")
            top_ax.set_xticks([1, 2, 3, 4, 5])
            lower = proxy_df["mean_delta_neutral_choice_probability"].min()
            upper = proxy_df["mean_relative_drop_neutral_choice"].max()
            top_ax.set_ylim(max(-1.0, lower - 0.05), min(1.0, upper + 0.08))
            top_ax.grid(False)
            sns.despine(ax=top_ax)

            bottom_ax = axes[1, col_idx]
            for metric in bottom_metrics:
                bottom_ax.plot(
                    proxy_df["bucket_id"],
                    proxy_df[metric],
                    marker="o",
                    linewidth=2.6,
                    color=COLORS[metric],
                    label=metric_labels[metric],
                )
            bottom_ax.set_xlabel("Quintile (Q1 lower values, Q5 higher values)")
            if col_idx == 0:
                bottom_ax.set_ylabel("Harmful outcome rate")
            bottom_ax.set_xticks([1, 2, 3, 4, 5])
            bottom_ax.set_ylim(0, min(1.0, proxy_df[bottom_metrics].max().max() + 0.08))
            bottom_ax.grid(False)
            sns.despine(ax=bottom_ax)
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=4,
            frameon=False,
        )
        fig.suptitle(f"`incorrect_suggestion`: harmful outcomes across all neutral confidence proxies\\n{run['run_label']}", y=1.04, fontsize=20)
        fig.tight_layout()
        return fig, subset

    display(Markdown("### 5. Friction: harmful `incorrect_suggestion` outcomes"))
    harmful_summary_rows = []
    for run in RUNS.values():
        display(Markdown(f"#### {run['run_label']}"))
        fig, harmful_df = plot_incorrect_harm_all_proxies(run)
        save_figure(fig, f"harmful_incorrect_suggestion__{run['run_key']}", subdir=HARMFUL_PLOTS_DIR)
        display(harmful_df.round(3))
        margin_df = harmful_df.loc[harmful_df["proxy"].eq("neutral_chosen_margin")].sort_values("bucket_id")
        low_row = margin_df.iloc[0]
        high_row = margin_df.iloc[-1]
        harmful_summary_rows.append(
            {
                "run_label": run["run_label"],
                "low_margin_relative_drop": low_row["mean_relative_drop_neutral_choice"],
                "high_margin_relative_drop": high_row["mean_relative_drop_neutral_choice"],
                "low_margin_adopt_suggested_incorrect_rate": low_row["adopt_bias_target_rate"],
                "high_margin_adopt_suggested_incorrect_rate": high_row["adopt_bias_target_rate"],
                "low_margin_harmful_flip_rate": low_row["harmful_flip_rate"],
                "high_margin_harmful_flip_rate": high_row["harmful_flip_rate"],
            }
        )
        plt.show()
        plt.close(fig)

    harmful_summary_df = pd.DataFrame(harmful_summary_rows)
    harmful_summary_df.to_csv(ARTIFACT_DIR / "harmful_margin_summary.csv", index=False)

    display(Markdown("### Compact harmful-friction summary"))
    display(harmful_summary_df.round(3))
    """
).strip()


ARTIFACT_CODE = dedent(
    """
    artifact_index_df = pd.DataFrame(
        {
            "artifact": sorted(str(path.relative_to(REPO_ROOT)) for path in SAVED_ARTIFACTS),
        }
    )
    artifact_index_df.to_csv(ARTIFACT_DIR / "artifact_index.csv", index=False)

    display(Markdown("## Saved artifacts"))
    display(artifact_index_df)
    """
).strip()


CAVEATS_MARKDOWN = dedent(
    """
    ## Caveats for the meeting

    - Coverage is still partial. The notebook only uses the runs explicitly listed at the top.
    - The GPT-5.4 Nano runs are sampling-only in this summary, so probe-based evidence is intentionally left out here.
    - The friction language is descriptive: higher neutral confidence is associated with less erosion of the model's original choice under bias. The notebook does not claim that confidence alone is the causal mechanism.
    """
).strip()


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(INTRO_MARKDOWN),
        new_code_cell(SETUP_CODE),
        new_code_cell(HELPERS_CODE),
        new_code_cell(INVENTORY_CODE),
        new_markdown_cell(GENERAL_MARKDOWN),
        new_code_cell(DELTA_CODE),
        new_code_cell(TRANSITION_CODE),
        new_markdown_cell(FRICTION_MARKDOWN),
        new_code_cell(FRICTION_CODE),
        new_code_cell(HARMFUL_CODE),
        new_code_cell(ARTIFACT_CODE),
        new_markdown_cell(CAVEATS_MARKDOWN),
    ]
    notebook = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
