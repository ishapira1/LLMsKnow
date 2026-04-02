from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPO_ROOT / "notebooks" / "analysis_20260322_v5.ipynb"
V3_NOTEBOOK = REPO_ROOT / "notebooks" / "analysis_20260325_v3.ipynb"


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Expected substring not found: {old!r}")
    return text.replace(old, new, 1)


def _section_two_markdown() -> str:
    return dedent(
        r"""
        ## Section 2 - Internal Knowledge

        This section uses the backfilled neutral-probe artifacts and keeps the probe notation explicit:

        - $S_N(x, y)$ means the probe trained on neutral prompts ($N$), evaluated on the prompt-response pair $(x, y)$.
        - $S_{\mathrm{incorrect\_suggestion}}(x', y)$ means the probe trained on `incorrect_suggestion` prompts, evaluated on the biased prompt-response pair $(x', y)$.
        - $\mathrm{margin}(x)=\log \pi(x)[c]-\log \pi(x)[b]$ and $\mathrm{margin}(x')=\log \pi(x')[c]-\log \pi(x')[b]$.

        The copied artifacts directly provide:
        - $S_N(x,\cdot)$ and $S_N(x',\cdot)$ from `probe_no_bias_all_templates`
        - $S_{\mathrm{incorrect\_suggestion}}(x',\cdot)$ from the original matched-template probe table

        If an optional all-templates backfill for `probe_bias_incorrect_suggestion` is present, the notebook will also populate $S_{\mathrm{incorrect\_suggestion}}(x,\cdot)$. Otherwise, the corresponding sanity-check panel is omitted entirely.

        For rank-based metrics, the notebook uses the saved probe probabilities $S(\cdot,\cdot)$ directly. For scatter plots against model log-probability margins, it uses probe logits $\operatorname{logit}(S(\cdot,\cdot))$ so the internal geometry stays on a linear score scale.
        """
    ).strip()


def _section_three_markdown() -> str:
    return dedent(
        r"""
        ## Section 3 - Summary

        This section summarizes the multiple-choice metrics using the full answer set for each question.

        - Model quantities use $\hat y_P(x)=\arg\max_y L(x,y)$, where $L(x,y)=\log P(y\mid x)$.
        - Probe quantities use the neutral probe by default: $\hat y_S(x)=\arg\max_y S_N(x,y)$.
        - For the `incorrect_suggestion` condition, the user-backed wrong answer is denoted by $w=b$.
        - The first table reports only top-1 accuracies.
        - The second table reports hidden-knowledge and non-linearity rates, with an `overall` column computed by combining all test rows across the four runs before taking the rate.
        """
    ).strip()


def _internal_loader_source() -> str:
    return dedent(
        r"""
        import json

        sns.set_style("white")

        INTERNAL_KNOWLEDGE_DIR = ARTIFACT_DIR / "internal_knowledge"
        INTERNAL_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        INTERNAL_RUN_KEYS = [
            "llama31_8b__commonsense_qa",
            "llama31_8b__arc_challenge",
            "qwen25_7b__commonsense_qa",
            "qwen25_7b__arc_challenge",
        ]
        RESULTS_ROOT = REPO_ROOT / "results" / "sycophancy_bias_probe"

        def clean_option(value: object) -> str | float:
            if pd.isna(value):
                return np.nan
            text = str(value).strip().upper()
            if text in {"", "NAN", "NONE", "NULL"}:
                return np.nan
            return text

        def lookup_prob(row: pd.Series, prefix: str, option: object) -> float:
            option = clean_option(option)
            if pd.isna(option):
                return np.nan
            column = f"{prefix}{option}"
            value = row.get(column, np.nan)
            return float(value) if pd.notna(value) else np.nan

        def score_value(row: pd.Series, prefix: str, option: object) -> float:
            option = clean_option(option)
            if pd.isna(option):
                return np.nan
            column = f"{prefix}{option}"
            value = row.get(column, np.nan)
            return float(value) if pd.notna(value) else np.nan

        def probe_logit(value: object, eps: float = 1e-6) -> float:
            if pd.isna(value):
                return np.nan
            numeric = float(value)
            if not np.isfinite(numeric):
                return np.nan
            clipped = float(np.clip(numeric, eps, 1.0 - eps))
            return float(np.log(clipped) - np.log1p(-clipped))

        def score_rank(row: pd.Series, prefix: str, option: object) -> float:
            option = clean_option(option)
            if pd.isna(option):
                return np.nan
            target_column = f"{prefix}{option}"
            if target_column not in row.index or pd.isna(row[target_column]):
                return np.nan
            target_score = float(row[target_column])
            candidate_scores = [
                float(row[column])
                for column in row.index
                if str(column).startswith(prefix) and pd.notna(row[column])
            ]
            if not candidate_scores:
                return np.nan
            return float(1 + sum(score > target_score for score in candidate_scores))

        def prefixed_option_letters(row: pd.Series, prefix: str) -> list[str]:
            letters = []
            for column in row.index:
                text = str(column)
                if not text.startswith(prefix):
                    continue
                if pd.isna(row[column]):
                    continue
                letters.append(text[len(prefix) :])
            return sorted(set(letters))

        def argmax_prefixed_value(row: pd.Series, prefix: str) -> str | float:
            best_letter = np.nan
            best_value = None
            for letter in prefixed_option_letters(row, prefix):
                column = f"{prefix}{letter}"
                value = row.get(column, np.nan)
                if pd.isna(value):
                    continue
                value = float(value)
                if best_value is None or value > best_value:
                    best_value = value
                    best_letter = letter
            return best_letter

        def response_group(response: object, correct_letter: object, bias_letter: object) -> str:
            response = clean_option(response)
            correct_letter = clean_option(correct_letter)
            bias_letter = clean_option(bias_letter)
            if pd.isna(response):
                return "missing"
            if response == correct_letter:
                return "correct"
            if response == bias_letter:
                return "b"
            return "wrong_not_b"

        def log_margin(p_correct: float, p_bias: float, eps: float = 1e-12) -> float:
            if pd.isna(p_correct) or pd.isna(p_bias):
                return np.nan
            return float(np.log(np.clip(p_correct, eps, 1.0)) - np.log(np.clip(p_bias, eps, 1.0)))

        def resolve_existing_file(
            run_name: str,
            relative_suffix: str,
            *,
            preferred_run_dir: str | Path | None = None,
            required: bool = True,
        ) -> Path | None:
            if preferred_run_dir is not None:
                preferred = Path(preferred_run_dir) / relative_suffix
                if preferred.exists():
                    return preferred.resolve()

            candidates = sorted(RESULTS_ROOT.glob(f"**/{run_name}/{relative_suffix}"))
            if candidates:
                return candidates[0].resolve()

            if required:
                raise FileNotFoundError(
                    f"Could not find {relative_suffix!r} for run {run_name!r} under {RESULTS_ROOT}."
                )
            return None

        def resolve_optional_incorrect_all_templates_path(run: dict) -> Path | None:
            run_name = str(run["run_name"])
            preferred_dir = Path(run["run_dir"])
            preferred = preferred_dir / "probes" / "backfills" / "probe_bias_incorrect_suggestion_all_templates" / "probe_scores_by_prompt.csv"
            if preferred.exists():
                return preferred.resolve()

            patterns = [
                f"**/{run_name}/probes/backfills/probe_bias_incorrect_suggestion_all_templates/probe_scores_by_prompt.csv",
                f"**/{run_name}/probes/backfills/*incorrect_suggestion*all_templates*/probe_scores_by_prompt.csv",
            ]
            for pattern in patterns:
                matches = sorted(RESULTS_ROOT.glob(pattern))
                if matches:
                    return matches[0].resolve()
            return None

        def load_probe_test_auc(run_dir: Path, probe_name: str) -> float:
            metrics_path = run_dir / "probes" / "chosen_probe" / probe_name / "metrics.json"
            if not metrics_path.exists():
                return np.nan
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            auc = payload.get("splits", {}).get("test", {}).get("auc")
            return float(auc) if auc is not None else np.nan

        def load_prompt_df(path: Path) -> pd.DataFrame:
            df = pd.read_csv(path)
            for column in ["correct_letter", "selected_choice", "probe_argmax_choice"]:
                if column in df.columns:
                    df[column] = df[column].map(clean_option)
            return df

        def rename_score_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
            rename_map = {
                column: f"{prefix}{column.split('score_', 1)[1]}"
                for column in df.columns
                if column.startswith("score_")
            }
            return df.rename(columns=rename_map)

        def build_variant_score_df(
            prompt_df: pd.DataFrame,
            *,
            template_type: str,
            prefix: str,
            join_keys: list[str],
        ) -> pd.DataFrame:
            if prompt_df.empty or "template_type" not in prompt_df.columns:
                return pd.DataFrame(columns=join_keys)
            subset = prompt_df.loc[prompt_df["template_type"].eq(template_type)].copy()
            if subset.empty:
                return pd.DataFrame(columns=join_keys)
            subset = rename_score_columns(subset, prefix)
            keep_cols = join_keys + [column for column in subset.columns if column.startswith(prefix)]
            return subset[keep_cols].drop_duplicates(subset=join_keys)

        def prepare_internal_runs() -> dict[str, dict]:
            return {
                run_key: RUNS[run_key]
                for run_key in INTERNAL_RUN_KEYS
                if run_key in RUNS and RUNS[run_key].get("probe_available", False)
            }

        def build_internal_probe_item_df(runs: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            inventory_rows = []
            item_frames = []
            shift_availability_rows = []

            for run in runs.values():
                run_dir = Path(run["run_dir"])
                neutral_backfill_path = resolve_existing_file(
                    run["run_name"],
                    "probes/backfills/probe_no_bias_all_templates/probe_scores_by_prompt.csv",
                    preferred_run_dir=run_dir,
                )
                standard_prompt_path = resolve_existing_file(
                    run["run_name"],
                    "probes/probe_scores_by_prompt.csv",
                    preferred_run_dir=run_dir,
                )
                incorrect_all_templates_path = resolve_optional_incorrect_all_templates_path(run)

                inventory_rows.append(
                    {
                        "run_key": run["run_key"],
                        "run_label": run["run_label"],
                        "model": run["model_label"],
                        "dataset": run["dataset_label"],
                        "run_dir": str(run_dir),
                        "neutral_backfill_path": str(neutral_backfill_path),
                        "standard_prompt_path": str(standard_prompt_path),
                        "incorrect_all_templates_path": (
                            str(incorrect_all_templates_path) if incorrect_all_templates_path is not None else ""
                        ),
                        "incorrect_all_templates_available": incorrect_all_templates_path is not None,
                        "neutral_probe_test_auc": load_probe_test_auc(run_dir, "probe_no_bias"),
                        "incorrect_probe_test_auc": load_probe_test_auc(run_dir, "probe_bias_incorrect_suggestion"),
                    }
                )

                paired_df = run["paired_df"].copy()
                paired_df = paired_df.loc[
                    paired_df["bias_type"].eq("incorrect_suggestion")
                    & paired_df["split"].eq("test")
                ].copy()
                if paired_df.empty:
                    continue

                for column in ["correct_letter", "incorrect_letter", "response_x", "response_xprime"]:
                    if column in paired_df.columns:
                        paired_df[column] = paired_df[column].map(clean_option)

                join_keys = [
                    column
                    for column in ["question_id", "split", "draw_idx"]
                    if column in paired_df.columns
                ]
                probability_cols = [
                    column
                    for column in paired_df.columns
                    if column.startswith("p_x_") or column.startswith("p_xprime_")
                ]

                neutral_backfill_df = load_prompt_df(neutral_backfill_path)
                neutral_backfill_df = neutral_backfill_df.loc[
                    neutral_backfill_df["split"].eq("test")
                    & neutral_backfill_df["template_type"].isin(["neutral", "incorrect_suggestion"])
                ].copy()

                standard_prompt_df = load_prompt_df(standard_prompt_path)
                standard_prompt_df = standard_prompt_df.loc[
                    standard_prompt_df["split"].eq("test")
                    & standard_prompt_df["probe_name"].eq("probe_bias_incorrect_suggestion")
                    & standard_prompt_df["template_type"].eq("incorrect_suggestion")
                ].copy()

                incorrect_all_templates_df = pd.DataFrame()
                if incorrect_all_templates_path is not None:
                    incorrect_all_templates_df = load_prompt_df(incorrect_all_templates_path)
                    if "probe_name" in incorrect_all_templates_df.columns:
                        incorrect_all_templates_df = incorrect_all_templates_df.loc[
                            incorrect_all_templates_df["probe_name"].eq("probe_bias_incorrect_suggestion")
                        ].copy()
                    incorrect_all_templates_df = incorrect_all_templates_df.loc[
                        incorrect_all_templates_df["split"].eq("test")
                        & incorrect_all_templates_df["template_type"].isin(["neutral", "incorrect_suggestion"])
                    ].copy()

                sn_x_df = build_variant_score_df(
                    neutral_backfill_df,
                    template_type="neutral",
                    prefix="sn_x_",
                    join_keys=join_keys,
                )
                sn_xprime_df = build_variant_score_df(
                    neutral_backfill_df,
                    template_type="incorrect_suggestion",
                    prefix="sn_xprime_",
                    join_keys=join_keys,
                )
                si_xprime_df = build_variant_score_df(
                    standard_prompt_df,
                    template_type="incorrect_suggestion",
                    prefix="si_xprime_",
                    join_keys=join_keys,
                )
                si_x_df = build_variant_score_df(
                    incorrect_all_templates_df,
                    template_type="neutral",
                    prefix="si_x_",
                    join_keys=join_keys,
                )

                merged = (
                    paired_df
                    .merge(sn_x_df, on=join_keys, how="inner")
                    .merge(sn_xprime_df, on=join_keys, how="inner")
                    .merge(si_xprime_df, on=join_keys, how="left")
                )
                if not si_x_df.empty:
                    merged = merged.merge(si_x_df, on=join_keys, how="left")

                if merged.empty:
                    continue

                merged["neutral_correct"] = pd.to_numeric(merged["correctness_x"], errors="coerce").eq(1)
                merged["neutral_response_group"] = merged.apply(
                    lambda row: response_group(row["response_x"], row["correct_letter"], row["incorrect_letter"]),
                    axis=1,
                )
                merged["biased_response_group"] = merged.apply(
                    lambda row: response_group(row["response_xprime"], row["correct_letter"], row["incorrect_letter"]),
                    axis=1,
                )
                merged["current_response_group"] = merged["biased_response_group"]
                merged["model_argmax_x"] = merged.apply(
                    lambda row: argmax_prefixed_value(row, "p_x_"),
                    axis=1,
                )
                merged["model_argmax_xprime"] = merged.apply(
                    lambda row: argmax_prefixed_value(row, "p_xprime_"),
                    axis=1,
                )
                merged["margin_x"] = merged.apply(
                    lambda row: log_margin(
                        lookup_prob(row, "p_x_", row["correct_letter"]),
                        lookup_prob(row, "p_x_", row["incorrect_letter"]),
                    ),
                    axis=1,
                )
                merged["margin_xprime"] = merged.apply(
                    lambda row: log_margin(
                        lookup_prob(row, "p_xprime_", row["correct_letter"]),
                        lookup_prob(row, "p_xprime_", row["incorrect_letter"]),
                    ),
                    axis=1,
                )

                merged["sn_score_correct_x"] = merged.apply(
                    lambda row: score_value(row, "sn_x_", row["correct_letter"]),
                    axis=1,
                )
                merged["sn_score_correct_xprime"] = merged.apply(
                    lambda row: score_value(row, "sn_xprime_", row["correct_letter"]),
                    axis=1,
                )
                merged["sn_score_bias_target_x"] = merged.apply(
                    lambda row: score_value(row, "sn_x_", row["incorrect_letter"]),
                    axis=1,
                )
                merged["sn_score_bias_target_xprime"] = merged.apply(
                    lambda row: score_value(row, "sn_xprime_", row["incorrect_letter"]),
                    axis=1,
                )
                merged["si_score_correct_x"] = merged.apply(
                    lambda row: score_value(row, "si_x_", row["correct_letter"]),
                    axis=1,
                )
                merged["si_score_correct_xprime"] = merged.apply(
                    lambda row: score_value(row, "si_xprime_", row["correct_letter"]),
                    axis=1,
                )
                merged["si_score_bias_target_x"] = merged.apply(
                    lambda row: score_value(row, "si_x_", row["incorrect_letter"]),
                    axis=1,
                )
                merged["si_score_bias_target_xprime"] = merged.apply(
                    lambda row: score_value(row, "si_xprime_", row["incorrect_letter"]),
                    axis=1,
                )

                merged["sn_margin_x"] = merged["sn_score_correct_x"] - merged["sn_score_bias_target_x"]
                merged["sn_margin_xprime"] = merged["sn_score_correct_xprime"] - merged["sn_score_bias_target_xprime"]
                merged["si_margin_x"] = merged["si_score_correct_x"] - merged["si_score_bias_target_x"]
                merged["si_margin_xprime"] = merged["si_score_correct_xprime"] - merged["si_score_bias_target_xprime"]
                merged["sn_logit_score_correct_x"] = merged["sn_score_correct_x"].map(probe_logit)
                merged["sn_logit_score_correct_xprime"] = merged["sn_score_correct_xprime"].map(probe_logit)
                merged["sn_logit_score_bias_target_x"] = merged["sn_score_bias_target_x"].map(probe_logit)
                merged["sn_logit_score_bias_target_xprime"] = merged["sn_score_bias_target_xprime"].map(probe_logit)
                merged["si_logit_score_correct_x"] = merged["si_score_correct_x"].map(probe_logit)
                merged["si_logit_score_correct_xprime"] = merged["si_score_correct_xprime"].map(probe_logit)
                merged["si_logit_score_bias_target_x"] = merged["si_score_bias_target_x"].map(probe_logit)
                merged["si_logit_score_bias_target_xprime"] = merged["si_score_bias_target_xprime"].map(probe_logit)
                merged["sn_logit_margin_x"] = merged["sn_logit_score_correct_x"] - merged["sn_logit_score_bias_target_x"]
                merged["sn_logit_margin_xprime"] = merged["sn_logit_score_correct_xprime"] - merged["sn_logit_score_bias_target_xprime"]
                merged["si_logit_margin_x"] = merged["si_logit_score_correct_x"] - merged["si_logit_score_bias_target_x"]
                merged["si_logit_margin_xprime"] = merged["si_logit_score_correct_xprime"] - merged["si_logit_score_bias_target_xprime"]
                merged["sn_delta_correct"] = merged["sn_score_correct_xprime"] - merged["sn_score_correct_x"]
                merged["sn_delta_bias_target"] = (
                    merged["sn_score_bias_target_xprime"] - merged["sn_score_bias_target_x"]
                )
                merged["si_minus_sn_x_correct"] = merged["si_score_correct_x"] - merged["sn_score_correct_x"]
                merged["si_minus_sn_xprime_correct"] = (
                    merged["si_score_correct_xprime"] - merged["sn_score_correct_xprime"]
                )
                merged["sn_argmax_x"] = merged.apply(
                    lambda row: argmax_prefixed_value(row, "sn_x_"),
                    axis=1,
                )
                merged["sn_argmax_xprime"] = merged.apply(
                    lambda row: argmax_prefixed_value(row, "sn_xprime_"),
                    axis=1,
                )
                merged["si_argmax_x"] = merged.apply(
                    lambda row: argmax_prefixed_value(row, "si_x_"),
                    axis=1,
                )
                merged["si_argmax_xprime"] = merged.apply(
                    lambda row: argmax_prefixed_value(row, "si_xprime_"),
                    axis=1,
                )
                merged["sn_rank_correct_x"] = merged.apply(
                    lambda row: score_rank(row, "sn_x_", row["correct_letter"]),
                    axis=1,
                )
                merged["sn_rank_correct_xprime"] = merged.apply(
                    lambda row: score_rank(row, "sn_xprime_", row["correct_letter"]),
                    axis=1,
                )
                merged["sn_prefers_c_over_b_x"] = merged["sn_margin_x"].gt(0)
                merged["sn_prefers_c_over_b_xprime"] = merged["sn_margin_xprime"].gt(0)
                sn_x_cols = [column for column in merged.columns if column.startswith("sn_x_")]
                sn_xprime_cols = [column for column in merged.columns if column.startswith("sn_xprime_")]
                si_x_cols = [column for column in merged.columns if column.startswith("si_x_")]
                si_xprime_cols = [column for column in merged.columns if column.startswith("si_xprime_")]

                item_frames.append(
                    merged.assign(
                        run_key=run["run_key"],
                        run_label=run["run_label"],
                        model=run["model_label"],
                        dataset=run["dataset_label"],
                    )[
                        [
                            "run_key",
                            "run_label",
                            "model",
                            "dataset",
                            "split",
                            "question_id",
                            "draw_idx",
                            "correct_letter",
                            "incorrect_letter",
                            "response_x",
                            "response_xprime",
                            "neutral_correct",
                            "neutral_response_group",
                            "biased_response_group",
                            "current_response_group",
                            "model_argmax_x",
                            "model_argmax_xprime",
                            "margin_x",
                            "margin_xprime",
                            "sn_score_correct_x",
                            "sn_score_correct_xprime",
                            "sn_score_bias_target_x",
                            "sn_score_bias_target_xprime",
                            "si_score_correct_x",
                            "si_score_correct_xprime",
                            "si_score_bias_target_x",
                            "si_score_bias_target_xprime",
                            "sn_margin_x",
                            "sn_margin_xprime",
                            "si_margin_x",
                            "si_margin_xprime",
                            "sn_logit_score_correct_x",
                            "sn_logit_score_correct_xprime",
                            "sn_logit_score_bias_target_x",
                            "sn_logit_score_bias_target_xprime",
                            "si_logit_score_correct_x",
                            "si_logit_score_correct_xprime",
                            "si_logit_score_bias_target_x",
                            "si_logit_score_bias_target_xprime",
                            "sn_logit_margin_x",
                            "sn_logit_margin_xprime",
                            "si_logit_margin_x",
                            "si_logit_margin_xprime",
                            "sn_delta_correct",
                            "sn_delta_bias_target",
                            "si_minus_sn_x_correct",
                            "si_minus_sn_xprime_correct",
                            "sn_prefers_c_over_b_x",
                            "sn_prefers_c_over_b_xprime",
                            "sn_rank_correct_x",
                            "sn_rank_correct_xprime",
                            "sn_argmax_x",
                            "sn_argmax_xprime",
                            "si_argmax_x",
                            "si_argmax_xprime",
                        ]
                        + probability_cols
                        + sn_x_cols
                        + sn_xprime_cols
                        + si_x_cols
                        + si_xprime_cols
                    ]
                )

                shift_availability_rows.append(
                    {
                        "run_key": run["run_key"],
                        "run_label": run["run_label"],
                        "has_si_on_x": merged["si_score_correct_x"].notna().any(),
                        "has_si_on_xprime": merged["si_score_correct_xprime"].notna().any(),
                    }
                )

            inventory_df = pd.DataFrame(inventory_rows)
            item_df = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
            shift_availability_df = pd.DataFrame(shift_availability_rows)

            if not item_df.empty:
                item_df = item_df.replace([np.inf, -np.inf], np.nan)
                item_df = item_df.dropna(
                    subset=[
                        "margin_x",
                        "margin_xprime",
                        "sn_margin_x",
                        "sn_margin_xprime",
                    ]
                ).reset_index(drop=True)

            summary_df = (
                item_df.groupby(["run_key", "run_label", "model", "dataset"], as_index=False)
                .agg(
                    n_items=("question_id", "size"),
                    n_questions=("question_id", "nunique"),
                    frac_sn_prefers_c_over_b_x=("sn_prefers_c_over_b_x", "mean"),
                    frac_sn_prefers_c_over_b_xprime=("sn_prefers_c_over_b_xprime", "mean"),
                    mean_sn_margin_x=("sn_margin_x", "mean"),
                    mean_sn_margin_xprime=("sn_margin_xprime", "mean"),
                    frac_current_response_b=("current_response_group", lambda s: float((s == "b").mean())),
                )
                .sort_values(["dataset", "model"])
                .reset_index(drop=True)
            ) if not item_df.empty else pd.DataFrame()

            return inventory_df, item_df, summary_df, shift_availability_df

        INTERNAL_RUNS = prepare_internal_runs()
        internal_probe_inventory_df, internal_probe_item_df, internal_probe_summary_df, shift_availability_df = (
            build_internal_probe_item_df(INTERNAL_RUNS)
        )

        internal_probe_inventory_df.to_csv(
            INTERNAL_KNOWLEDGE_DIR / "internal_probe_inventory__test_only.csv",
            index=False,
        )
        internal_probe_item_df.to_csv(
            INTERNAL_KNOWLEDGE_DIR / "internal_probe_item_level__incorrect_suggestion__test_only.csv",
            index=False,
        )
        internal_probe_summary_df.to_csv(
            INTERNAL_KNOWLEDGE_DIR / "internal_probe_summary__incorrect_suggestion__test_only.csv",
            index=False,
        )
        shift_availability_df.to_csv(
            INTERNAL_KNOWLEDGE_DIR / "internal_probe_shift_availability__test_only.csv",
            index=False,
        )

        display(internal_probe_inventory_df)
        display(internal_probe_summary_df.round(3))
        display(shift_availability_df)

        if internal_probe_item_df.empty:
            raise ValueError("No incorrect_suggestion / test rows were found for the internal-knowledge section.")
        """
    ).strip()


def _rank_plot_source() -> str:
    return dedent(
        r"""
        from matplotlib.lines import Line2D

        sns.set_style("white")

        INTERNAL_RANK_DIR = ARTIFACT_DIR / "internal_probe_rank"
        INTERNAL_RANK_DIR.mkdir(parents=True, exist_ok=True)
        MIN_N_PER_POINT = 5

        rank_summary_df = (
            internal_probe_item_df.assign(flip_to_b=internal_probe_item_df["current_response_group"].eq("b").astype(int))
            .dropna(subset=["sn_rank_correct_x", "margin_xprime"])
            .assign(sn_rank_correct_x=lambda df: pd.to_numeric(df["sn_rank_correct_x"], errors="coerce"))
            .dropna(subset=["sn_rank_correct_x"])
            .assign(sn_rank_correct_x=lambda df: df["sn_rank_correct_x"].astype(int))
            .groupby(["run_key", "run_label", "model", "dataset", "sn_rank_correct_x"], as_index=False)
            .agg(
                n_items=("question_id", "size"),
                flip_to_b_rate=("flip_to_b", "mean"),
                harmful_external_rate=("margin_xprime", lambda s: float((s < 0).mean())),
            )
            .query("n_items >= @MIN_N_PER_POINT")
            .sort_values(["dataset", "model", "sn_rank_correct_x"])
            .reset_index(drop=True)
        )

        rank_summary_df.to_csv(
            INTERNAL_RANK_DIR / f"neutral_probe_rank_summary__incorrect_suggestion__min_n_{MIN_N_PER_POINT}.csv",
            index=False,
        )
        display(rank_summary_df.round(3))

        def plot_internal_rank_summary(df: pd.DataFrame) -> plt.Figure:
            run_labels = df["run_label"].drop_duplicates().tolist()
            n_panels = len(run_labels)
            ncols = 2 if n_panels <= 4 else 3
            nrows = int(np.ceil(n_panels / ncols))

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6.0 * ncols, 4.9 * nrows),
                sharey=True,
            )
            axes = np.atleast_1d(axes).ravel()

            for ax, run_label in zip(axes, run_labels):
                subset = df.loc[df["run_label"].eq(run_label)].sort_values("sn_rank_correct_x").copy()

                ax.plot(
                    subset["sn_rank_correct_x"],
                    subset["harmful_external_rate"],
                    marker="o",
                    linewidth=2.6,
                    color=COLORS["incorrect_suggestion"],
                )
                for _, row in subset.iterrows():
                    ax.text(
                        row["sn_rank_correct_x"],
                        row["harmful_external_rate"],
                        f"n={int(row['n_items'])}",
                        ha="left",
                        va="bottom",
                        fontsize=10,
                    )

                ax.set_title(run_label, fontsize=20, pad=10)
                ax.set_xlabel(r"$\mathrm{rank}_{S_N}(x, c)$", fontsize=15)
                ax.set_xticks(sorted(subset["sn_rank_correct_x"].unique()))
                ax.tick_params(axis="both", labelsize=12)
                ax.grid(False)
                sns.despine(ax=ax)

            for ax in axes[len(run_labels):]:
                ax.axis("off")

            for ax in axes[::ncols]:
                ax.set_ylabel(r"$\mathbb{1}[\mathrm{margin}(x') < 0]$", fontsize=15)

            fig.suptitle(
                f"Incorrect Suggestion: Harmful External Rate vs Neutral-Probe Rank of $c$ (n >= {MIN_N_PER_POINT})",
                fontsize=24,
                y=0.995,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            return fig

        internal_rank_fig = plot_internal_rank_summary(rank_summary_df)
        internal_rank_path = save_figure(
            internal_rank_fig,
            "neutral_probe_rank_vs_harmful_external_rate__incorrect_suggestion__test_only",
            subdir=INTERNAL_RANK_DIR,
        )
        plt.show()
        plt.close(internal_rank_fig)

        print(internal_rank_path)
        """
    ).strip()


def _targeted_probability_df_source() -> str:
    return dedent(
        r"""
        # Cell 1: dataframe schema for the targeted incorrect-suggestion probability analysis

        sns.set_style("white")

        target_run_keys = [
            "llama31_8b__commonsense_qa",
            "llama31_8b__arc_challenge",
            "qwen25_7b__commonsense_qa",
            "qwen25_7b__arc_challenge",
            "gpt54nano__commonsense_qa",
        ]

        def prepare_targeted_probability_runs(target_run_keys: list[str]) -> dict[str, dict]:
            if "RUNS" in globals() and all(run_key in RUNS for run_key in target_run_keys):
                return {run_key: RUNS[run_key] for run_key in target_run_keys}

            corrected_qwen_commonsense_dir = (
                "results/sycophancy_bias_probe/"
                "Qwen_Qwen2_5_7B_Instruct/commonsense_qa/"
                "full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas"
            )

            run_specs = []
            for spec in RUN_SPECS:
                if spec["run_key"] not in target_run_keys:
                    continue
                spec = dict(spec)
                if spec["run_key"] == "qwen25_7b__commonsense_qa":
                    original_path = REPO_ROOT / spec["relative_run_dir"]
                    corrected_path = REPO_ROOT / corrected_qwen_commonsense_dir
                    if not original_path.exists() and corrected_path.exists():
                        spec["relative_run_dir"] = corrected_qwen_commonsense_dir
                run_specs.append(spec)

            return {
                spec["run_key"]: prepare_run(spec)
                for spec in run_specs
            }

        def lookup_prob(row: pd.Series, prefix: str, option: str) -> float:
            option = str(option).strip().upper()
            column = f"{prefix}{option}"
            value = row.get(column, np.nan)
            return float(value) if pd.notna(value) else np.nan

        def build_targeted_probability_df(runs: dict[str, dict]) -> pd.DataFrame:
            frames = []

            for run in runs.values():
                paired_df = run["paired_df"].copy()
                subset = paired_df.loc[
                    paired_df["bias_type"].eq("incorrect_suggestion")
                    & paired_df["correctness_x"].eq(1)
                ].copy()

                if subset.empty:
                    continue

                if "split" not in subset.columns:
                    subset["split"] = "all"

                subset["p_x_c"] = subset.apply(
                    lambda row: lookup_prob(row, "p_x_", row["correct_letter"]),
                    axis=1,
                )
                subset["p_x_b"] = subset.apply(
                    lambda row: lookup_prob(row, "p_x_", row["incorrect_letter"]),
                    axis=1,
                )
                subset["p_xprime_c"] = subset.apply(
                    lambda row: lookup_prob(row, "p_xprime_", row["correct_letter"]),
                    axis=1,
                )
                subset["p_xprime_b"] = subset.apply(
                    lambda row: lookup_prob(row, "p_xprime_", row["incorrect_letter"]),
                    axis=1,
                )

                frames.append(
                    subset[
                        [
                            "split",
                            "p_x_c",
                            "p_x_b",
                            "p_xprime_c",
                            "p_xprime_b",
                        ]
                    ].assign(
                        model=run["model_label"],
                        dataset=run["dataset_label"],
                    )[
                        [
                            "model",
                            "dataset",
                            "split",
                            "p_x_c",
                            "p_x_b",
                            "p_xprime_c",
                            "p_xprime_b",
                        ]
                    ]
                )

            targeted_probability_df = pd.concat(frames, ignore_index=True)
            targeted_probability_df = targeted_probability_df.replace([np.inf, -np.inf], np.nan)
            targeted_probability_df = targeted_probability_df.dropna(
                subset=["p_x_c", "p_x_b", "p_xprime_c", "p_xprime_b"]
            )
            targeted_probability_df = targeted_probability_df.sort_values(
                ["dataset", "model", "split"]
            ).reset_index(drop=True)
            return targeted_probability_df

        TARGETED_PROBABILITY_RUNS = prepare_targeted_probability_runs(target_run_keys)
        targeted_probability_df = build_targeted_probability_df(TARGETED_PROBABILITY_RUNS)

        targeted_probability_summary_df = (
            targeted_probability_df.groupby(["model", "dataset"], as_index=False)
            .agg(
                n_items=("p_x_c", "size"),
                mean_p_x_c=("p_x_c", "mean"),
                mean_p_xprime_c=("p_xprime_c", "mean"),
                mean_p_x_b=("p_x_b", "mean"),
                mean_p_xprime_b=("p_xprime_b", "mean"),
            )
            .sort_values(["dataset", "model"])
            .reset_index(drop=True)
        )

        display(targeted_probability_summary_df.round(3))
        targeted_probability_df.head()
        """
    ).strip()


def _external_margin_before_after_source() -> str:
    return dedent(
        r"""
        from matplotlib.lines import Line2D

        sns.set_style("white")

        EXTERNAL_PROBABILITY_DIR = ARTIFACT_DIR / "external_probability_before_vs_after_bias"
        EXTERNAL_PROBABILITY_DIR.mkdir(parents=True, exist_ok=True)

        plot_df = targeted_probability_df.copy()
        plot_df["run_label"] = plot_df["model"] + " / " + plot_df["dataset"]

        preferred_models = [
            "Llama 3.1 8B Instruct",
            "Qwen 2.5 7B Instruct",
            "GPT-5.4 Nano",
        ]
        preferred_datasets = ["CommonsenseQA", "ARC-Challenge"]

        model_order = [model for model in preferred_models if model in set(plot_df["model"])]
        model_order += [model for model in plot_df["model"].drop_duplicates().tolist() if model not in model_order]
        dataset_order = [dataset for dataset in preferred_datasets if dataset in set(plot_df["dataset"])]
        dataset_order += [dataset for dataset in plot_df["dataset"].drop_duplicates().tolist() if dataset not in dataset_order]

        def fit_line_stats(subset: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, float]:
            work = subset[[x_col, y_col]].dropna().copy()
            if len(work) < 2 or work[x_col].nunique() < 2 or work[y_col].nunique() < 2:
                return np.nan, np.nan, np.nan
            slope, intercept = np.polyfit(work[x_col], work[y_col], 1)
            corr = work[x_col].corr(work[y_col])
            r2 = float(corr ** 2) if pd.notna(corr) else np.nan
            return float(slope), float(intercept), r2

        regression_rows = []
        fig, axes = plt.subplots(
            len(model_order),
            len(dataset_order),
            figsize=(6.6 * len(dataset_order), 4.7 * len(model_order)),
            sharex=False,
            sharey=False,
            squeeze=False,
        )

        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=8, color="#73b3ab", label=r"$P(c \mid \cdot)$"),
            Line2D([0], [0], marker="^", linestyle="", markersize=8, color="#d4651a", label=r"$P(b \mid \cdot)$"),
            Line2D([0], [0], color="#73b3ab", linewidth=2.0, label=r"Linear fit for $P(c \mid \cdot)$"),
            Line2D([0], [0], color="#d4651a", linewidth=2.0, label=r"Linear fit for $P(b \mid \cdot)$"),
        ]

        for row_idx, model in enumerate(model_order):
            for col_idx, dataset in enumerate(dataset_order):
                ax = axes[row_idx, col_idx]
                subset = plot_df.loc[
                    plot_df["model"].eq(model) & plot_df["dataset"].eq(dataset)
                ].copy()
                if subset.empty:
                    ax.axis("off")
                    continue

                sns.scatterplot(
                    data=subset,
                    x="p_x_c",
                    y="p_xprime_c",
                    color="#73b3ab",
                    marker="o",
                    s=36,
                    alpha=0.55,
                    edgecolor=None,
                    legend=False,
                    ax=ax,
                )
                sns.scatterplot(
                    data=subset,
                    x="p_x_b",
                    y="p_xprime_b",
                    color="#d4651a",
                    marker="^",
                    s=42,
                    alpha=0.55,
                    edgecolor=None,
                    legend=False,
                    ax=ax,
                )

                x_values = pd.concat([subset["p_x_c"], subset["p_x_b"]], ignore_index=True).dropna()
                y_values = pd.concat([subset["p_xprime_c"], subset["p_xprime_b"]], ignore_index=True).dropna()
                x_min = float(x_values.min()) if not x_values.empty else 0.0
                x_max = float(x_values.max()) if not x_values.empty else 1.0
                y_min = float(y_values.min()) if not y_values.empty else 0.0
                y_max = float(y_values.max()) if not y_values.empty else 1.0
                x_pad = max(0.02, 0.05 * max(x_max - x_min, 0.1))
                y_pad = max(0.02, 0.05 * max(y_max - y_min, 0.1))
                x_left = max(-0.02, x_min - x_pad)
                x_right = min(1.02, x_max + x_pad)
                y_bottom = max(-0.02, y_min - y_pad)
                y_top = min(1.02, y_max + y_pad)
                diag_min = min(x_left, y_bottom)
                diag_max = max(x_right, y_top)

                ax.plot(
                    [diag_min, diag_max],
                    [diag_min, diag_max],
                    color="#4f6d7a",
                    linestyle=":",
                    linewidth=1.4,
                )

                slope_c, intercept_c, r2_c = fit_line_stats(subset, "p_x_c", "p_xprime_c")
                slope_b, intercept_b, r2_b = fit_line_stats(subset, "p_x_b", "p_xprime_b")
                regression_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "n_items": int(len(subset)),
                        "series": "correct",
                        "slope": slope_c,
                        "intercept": intercept_c,
                        "r2": r2_c,
                    }
                )
                regression_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "n_items": int(len(subset)),
                        "series": "bias_target",
                        "slope": slope_b,
                        "intercept": intercept_b,
                        "r2": r2_b,
                    }
                )

                x_line = np.array([x_left, x_right], dtype=float)
                if pd.notna(slope_c) and pd.notna(intercept_c):
                    y_line_c = slope_c * x_line + intercept_c
                    ax.plot(x_line, y_line_c, color="#73b3ab", linewidth=1.8)
                    formula_text_c = f"c: y = {slope_c:+.2f}x {intercept_c:+.2f}"
                else:
                    formula_text_c = "c: y = NA"
                if pd.notna(slope_b) and pd.notna(intercept_b):
                    y_line_b = slope_b * x_line + intercept_b
                    ax.plot(x_line, y_line_b, color="#d4651a", linewidth=1.8)
                    formula_text_b = f"b: y = {slope_b:+.2f}x {intercept_b:+.2f}"
                else:
                    formula_text_b = "b: y = NA"

                r2_c_text = f"{r2_c:.3f}" if pd.notna(r2_c) else "NA"
                r2_b_text = f"{r2_b:.3f}" if pd.notna(r2_b) else "NA"
                ax.text(
                    0.02,
                    0.98,
                    (
                        f"n = {len(subset)}\n"
                        f"{formula_text_c}\nR²(c) = {r2_c_text}\n"
                        f"{formula_text_b}\nR²(b) = {r2_b_text}"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=2.5),
                )

                if row_idx == 0:
                    ax.set_title(dataset, fontsize=20, pad=10)
                if col_idx == 0:
                    ax.text(
                        -0.28,
                        0.5,
                        model,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=16,
                    )
                ax.set_xlim(x_left, x_right)
                ax.set_ylim(y_bottom, y_top)
                ax.tick_params(axis="both", labelsize=12)
                sns.despine(ax=ax)

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
            frameon=True,
            fontsize=12,
        )

        fig.suptitle("Incorrect Suggestion: Probability of c and b Before vs After Bias", fontsize=24, y=0.995)
        fig.supxlabel(r"$P(c \mid x)$ and $P(b \mid x)$", fontsize=18)
        fig.supylabel(r"$P(c \mid x')$ and $P(b \mid x')$", fontsize=18)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.90, hspace=0.28, wspace=0.18)

        external_margin_regression_df = pd.DataFrame(regression_rows).sort_values(["model", "dataset"]).reset_index(drop=True)
        external_margin_regression_df.to_csv(
            EXTERNAL_PROBABILITY_DIR / "incorrect_suggestion_probability_regression_summary.csv",
            index=False,
        )
        display(external_margin_regression_df.round(3))

        external_margin_path = save_figure(
            fig,
            "incorrect_suggestion_probability_before_vs_after_bias__correct_and_b",
            subdir=EXTERNAL_PROBABILITY_DIR,
        )
        plt.show()
        plt.close(fig)

        print(external_margin_path)
        """
    ).strip()


def _shift_plot_source() -> str:
    return dedent(
        r"""
        from matplotlib.lines import Line2D

        sns.set_style("white")

        INTERNAL_SHIFT_DIR = ARTIFACT_DIR / "internal_probe_shift_comparison"
        INTERNAL_SHIFT_DIR.mkdir(parents=True, exist_ok=True)

        SHIFT_SPECS = [
            {
                "shift_kind": "sn_correct_shift",
                "label": r"$S_N(x', c) - S_N(x, c)$",
                "column": "sn_delta_correct",
                "required": True,
            },
            {
                "shift_kind": "sn_bias_shift",
                "label": r"$S_N(x', b) - S_N(x, b)$",
                "column": "sn_delta_bias_target",
                "required": True,
            },
            {
                "shift_kind": "si_minus_sn_x",
                "label": r"$S_{\mathrm{incorrect\_suggestion}}(x, c) - S_N(x, c)$",
                "column": "si_minus_sn_x_correct",
                "required": False,
            },
            {
                "shift_kind": "si_minus_sn_xprime",
                "label": r"$S_{\mathrm{incorrect\_suggestion}}(x', c) - S_N(x', c)$",
                "column": "si_minus_sn_xprime_correct",
                "required": True,
            },
        ]
        SHIFT_COLOR = "#d4651a"
        MEAN_LINE_COLOR = "#4f6d7a"

        active_shift_specs = []
        for spec in SHIFT_SPECS:
            if spec["required"]:
                active_shift_specs.append(spec)
                continue
            column = spec["column"]
            if column in internal_probe_item_df.columns and internal_probe_item_df[column].notna().any():
                active_shift_specs.append(spec)

        shift_frames = []
        for spec in active_shift_specs:
            column = spec["column"]
            subset = (
                internal_probe_item_df[
                    ["run_key", "run_label", "model", "dataset", column]
                ]
                .rename(columns={column: "delta_value"})
                .assign(
                    shift_kind=spec["shift_kind"],
                    shift_label=spec["label"],
                )
            )
            shift_frames.append(subset)

        shift_long_df = pd.concat(shift_frames, ignore_index=True)
        shift_summary_df = (
            shift_long_df.dropna(subset=["delta_value"])
            .groupby(
                ["run_key", "run_label", "model", "dataset", "shift_kind", "shift_label"],
                as_index=False,
            )
            .agg(
                n_items=("delta_value", "size"),
                mean_delta=("delta_value", "mean"),
                median_delta=("delta_value", "median"),
                std_delta=("delta_value", "std"),
            )
            .sort_values(["dataset", "model", "shift_kind"])
            .reset_index(drop=True)
        )

        shift_long_df.to_csv(
            INTERNAL_SHIFT_DIR / "internal_probe_shift_long__incorrect_suggestion__test_only.csv",
            index=False,
        )
        shift_summary_df.to_csv(
            INTERNAL_SHIFT_DIR / "internal_probe_shift_summary__incorrect_suggestion__test_only.csv",
            index=False,
        )
        display(shift_summary_df.round(3))

        available_values = shift_long_df["delta_value"].dropna()
        if available_values.empty:
            raise ValueError("No internal shift values were available.")

        def plot_internal_shift_histograms(
            df: pd.DataFrame,
            availability_df: pd.DataFrame,
        ) -> plt.Figure:
            run_labels = internal_probe_item_df["run_label"].drop_duplicates().tolist()
            global_min = float(available_values.min())
            global_max = float(available_values.max())
            padding = max(0.01, 0.05 * max(global_max - global_min, 0.05))
            bins = np.linspace(global_min - padding, global_max + padding, 31)

            fig, axes = plt.subplots(
                len(active_shift_specs),
                len(run_labels),
                figsize=(5.5 * len(run_labels), 3.7 * len(active_shift_specs)),
                sharex=False,
                sharey=True,
                squeeze=False,
            )

            availability_lookup = availability_df.set_index("run_label") if not availability_df.empty else pd.DataFrame()

            for col_idx, run_label in enumerate(run_labels):
                for row_idx, spec in enumerate(active_shift_specs):
                    ax = axes[row_idx, col_idx]
                    subset = df.loc[
                        df["run_label"].eq(run_label) & df["shift_kind"].eq(spec["shift_kind"])
                    ].dropna(subset=["delta_value"]).copy()

                    if col_idx == 0:
                        ax.set_ylabel("Probability", fontsize=15)
                    if row_idx == 0:
                        ax.set_title(run_label, fontsize=20, pad=10)

                    if subset.empty:
                        availability_note = "Unavailable in current artifacts."
                        if (
                            spec["shift_kind"] == "si_minus_sn_x"
                            and not availability_df.empty
                            and run_label in availability_lookup.index
                            and not bool(availability_lookup.loc[run_label, "has_si_on_x"])
                        ):
                            availability_note = (
                                "Needs an all-templates backfill for\n"
                                r"$S_{\mathrm{incorrect\_suggestion}}(x,\cdot)$"
                            )
                        ax.text(
                            0.5,
                            0.5,
                            availability_note,
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=12,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3.0),
                        )
                        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
                        ax.set_ylim(0, 1)
                        ax.set_xlabel(spec["label"], fontsize=15)
                        ax.tick_params(axis="both", labelsize=12)
                        ax.tick_params(axis="x", labelbottom=True)
                        ax.grid(False)
                        sns.despine(ax=ax)
                        continue

                    sns.histplot(
                        data=subset,
                        x="delta_value",
                        bins=bins,
                        stat="probability",
                        color=SHIFT_COLOR,
                        edgecolor="white",
                        alpha=0.90,
                        ax=ax,
                    )
                    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
                    ax.axvline(
                        subset["delta_value"].mean(),
                        color=MEAN_LINE_COLOR,
                        linestyle=":",
                        linewidth=2.0,
                    )
                    ax.text(
                        0.02,
                        0.98,
                        (
                            f"n = {len(subset)}\n"
                            f"mean = {subset['delta_value'].mean():+.3f}\n"
                            f"median = {subset['delta_value'].median():+.3f}\n"
                            f"std = {subset['delta_value'].std():.3f}"
                        ),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=11,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
                    )
                    ax.set_ylim(0, 1)
                    ax.set_xlabel(spec["label"], fontsize=15)
                    ax.tick_params(axis="both", labelsize=12)
                    ax.tick_params(axis="x", labelbottom=True)
                    ax.grid(False)
                    sns.despine(ax=ax)

            legend_handles = [
                Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Zero shift"),
                Line2D([0], [0], color=MEAN_LINE_COLOR, linestyle=":", linewidth=2.0, label="Mean shift"),
            ]
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=2,
                frameon=True,
                fontsize=12,
            )

            fig.suptitle(
                "Incorrect Suggestion: Neutral-Probe Shifts and Matched-Probe Sanity Checks",
                fontsize=24,
                y=0.995,
            )
            fig.text(0.985, 0.987, "Test split only", ha="right", va="top", fontsize=12)
            fig.tight_layout(rect=(0, 0.05, 1, 0.96))
            return fig

        internal_shift_fig = plot_internal_shift_histograms(shift_long_df, shift_availability_df)
        internal_shift_path = save_figure(
            internal_shift_fig,
            "internal_probe_shift_histograms__incorrect_suggestion__test_only",
            subdir=INTERNAL_SHIFT_DIR,
        )
        plt.show()
        plt.close(internal_shift_fig)

        print(internal_shift_path)
        """
    ).strip()


def _external_internal_source() -> str:
    return dedent(
        r"""
        from matplotlib.lines import Line2D

        sns.set_style("white")

        EXTERNAL_INTERNAL_DIR = ARTIFACT_DIR / "external_internal_knowledge"
        EXTERNAL_INTERNAL_DIR.mkdir(parents=True, exist_ok=True)

        RESPONSE_ORDER = ["correct", "b", "wrong_not_b"]
        RESPONSE_LABELS = {
            "correct": "Correct",
            "b": "b",
            "wrong_not_b": "Wrong not b",
        }
        RESPONSE_COLORS = {
            "correct": "#73b3ab",
            "b": "#d4651a",
            "wrong_not_b": "#7a7a7a",
        }
        ZERO_LINE_COLOR = "black"
        NEUTRAL_RESPONSE_ORDER = ["correct", "b", "wrong_not_b"]
        NEUTRAL_RESPONSE_LABELS = {
            "correct": "Neutral correct",
            "b": "Neutral said b",
            "wrong_not_b": "Neutral wrong, not b",
        }
        NEUTRAL_RESPONSE_COLORS = {
            "correct": "#73b3ab",
            "b": "#d4651a",
            "wrong_not_b": "#7a7a7a",
        }
        BIASED_RESPONSE_ORDER = ["correct", "b", "wrong_not_b"]
        BIASED_RESPONSE_LABELS = {
            "correct": "Biased correct",
            "b": "Biased said b",
            "wrong_not_b": "Biased said other",
        }
        BIASED_RESPONSE_MARKERS = {
            "correct": "+",
            "b": "*",
            "wrong_not_b": "_",
        }
        BIASED_RESPONSE_SIZES = {
            "correct": 70,
            "b": 95,
            "wrong_not_b": 260,
        }
        BIASED_RESPONSE_LINEWIDTHS = {
            "correct": 1.5,
            "b": 0.8,
            "wrong_not_b": 2.2,
        }

        probe_auc_lookup = internal_probe_inventory_df.set_index("run_label")[
            ["neutral_probe_test_auc", "incorrect_probe_test_auc"]
        ]

        def scatter_panel(ax, subset: pd.DataFrame, x_col: str, y_col: str) -> None:
            for neutral_group in NEUTRAL_RESPONSE_ORDER:
                for biased_group in BIASED_RESPONSE_ORDER:
                    part = subset.loc[
                        subset["neutral_response_group"].eq(neutral_group)
                        & subset["biased_response_group"].eq(biased_group)
                    ].copy()
                    if part.empty:
                        continue

                    ax.scatter(
                        part[x_col],
                        part[y_col],
                        color=NEUTRAL_RESPONSE_COLORS[neutral_group],
                        marker=BIASED_RESPONSE_MARKERS[biased_group],
                        s=BIASED_RESPONSE_SIZES[biased_group],
                        linewidths=BIASED_RESPONSE_LINEWIDTHS[biased_group],
                        alpha=0.80,
                    )

        def compute_panel_r2(subset: pd.DataFrame, x_col: str, y_col: str) -> float:
            work = subset[[x_col, y_col]].dropna().copy()
            if len(work) < 2:
                return np.nan
            if work[x_col].nunique() < 2 or work[y_col].nunique() < 2:
                return np.nan
            corr = work[x_col].corr(work[y_col])
            if pd.isna(corr):
                return np.nan
            return float(corr ** 2)

        def quadrant_fraction_text(subset: pd.DataFrame, x_col: str, y_col: str) -> str:
            work = subset[[x_col, y_col]].dropna().copy()
            if work.empty:
                return "(+,+) NA | (-,+) NA\n(-,-) NA | (+,-) NA"

            total = float(len(work))
            pp = ((work[x_col] > 0) & (work[y_col] > 0)).sum() / total
            np_ = ((work[x_col] < 0) & (work[y_col] > 0)).sum() / total
            nn = ((work[x_col] < 0) & (work[y_col] < 0)).sum() / total
            pn = ((work[x_col] > 0) & (work[y_col] < 0)).sum() / total
            return (
                f"(+,+) {pp:.1%} | (-,+) {np_:.1%}\n"
                f"(-,-) {nn:.1%} | (+,-) {pn:.1%}"
            )

        def probe_auc_for(run_label: str, probe_family: str) -> float:
            if run_label not in probe_auc_lookup.index:
                return np.nan
            if probe_family == "incorrect":
                return float(probe_auc_lookup.loc[run_label, "incorrect_probe_test_auc"])
            return float(probe_auc_lookup.loc[run_label, "neutral_probe_test_auc"])

        def plot_external_internal_scatter_all_points(
            item_df: pd.DataFrame,
            *,
            x_col: str,
            y_col: str,
            title: str,
            x_label: str,
            y_label: str,
            filename_stub: str,
            probe_family: str = "neutral",
        ) -> Path:
            run_labels = item_df["run_label"].drop_duplicates().tolist()
            n_panels = len(run_labels)
            ncols = 2 if n_panels <= 4 else 3
            nrows = int(np.ceil(n_panels / ncols))

            x_min = float(item_df[x_col].min())
            x_max = float(item_df[x_col].max())
            y_min = float(item_df[y_col].min())
            y_max = float(item_df[y_col].max())
            x_pad = max(0.1, 0.05 * max(x_max - x_min, 1.0))
            y_pad = max(0.1, 0.05 * max(y_max - y_min, 1.0))

            x_left = x_min - x_pad
            x_right = x_max + x_pad
            y_bottom = y_min - y_pad
            y_top = y_max + y_pad

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6.2 * ncols, 5.0 * nrows),
                sharex=True,
                sharey=True,
            )
            axes = np.atleast_1d(axes).ravel()

            for ax, run_label in zip(axes, run_labels):
                subset = item_df.loc[item_df["run_label"].eq(run_label)].copy()

                scatter_panel(ax, subset, x_col, y_col)

                n_b = int(subset["biased_response_group"].eq("b").sum())
                r2 = compute_panel_r2(subset, x_col, y_col)
                r2_text = f"{r2:.3f}" if pd.notna(r2) else "NA"
                probe_auc = probe_auc_for(run_label, probe_family)
                probe_auc_text = f"{probe_auc:.3f}" if pd.notna(probe_auc) else "NA"
                quad_text = quadrant_fraction_text(subset, x_col, y_col)

                ax.text(
                    0.02,
                    0.98,
                    f"n = {len(subset)}\nb = {n_b}\nR² = {r2_text}\nProbe AUC = {probe_auc_text}\n{quad_text}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.5),
                )

                ax.axvline(0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.2)
                ax.axhline(0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.2)
                ax.set_xlim(x_left, x_right)
                ax.set_ylim(y_bottom, y_top)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title(run_label, fontsize=18, pad=10)
                ax.tick_params(axis="both", labelsize=12)
                ax.grid(False)
                sns.despine(ax=ax)

            for ax in axes[len(run_labels):]:
                ax.axis("off")

            color_handles = [
                Line2D([0], [0], marker="o", linestyle="", markersize=8, color=NEUTRAL_RESPONSE_COLORS[group], label=NEUTRAL_RESPONSE_LABELS[group])
                for group in NEUTRAL_RESPONSE_ORDER
            ]
            marker_handles = [
                Line2D([0], [0], marker=BIASED_RESPONSE_MARKERS[group], linestyle="", markersize=10, color="black", label=BIASED_RESPONSE_LABELS[group])
                for group in BIASED_RESPONSE_ORDER
            ]

            fig.legend(
                handles=color_handles + marker_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=6,
                frameon=True,
                fontsize=12,
            )

            fig.suptitle(title, fontsize=24, y=0.995)
            fig.supxlabel(x_label, fontsize=21)
            fig.supylabel(y_label, fontsize=21)
            fig.text(0.985, 0.987, "Test split only", ha="right", va="top", fontsize=12)
            fig.tight_layout(rect=(0, 0.07, 1, 0.95))

            path = save_figure(fig, filename_stub, subdir=EXTERNAL_INTERNAL_DIR)
            plt.show()
            plt.close(fig)
            return path

        external_internal_paths = []

        external_internal_paths.append(
            {
                "artifact": "margin_x_vs_sn_margin_x",
                "path": str(
                    plot_external_internal_scatter_all_points(
                        internal_probe_item_df,
                        x_col="margin_x",
                        y_col="sn_logit_margin_x",
                        title="External vs Internal Knowledge Without Bias",
                        x_label=r"$\mathrm{margin}(x)=\log \pi(x)[c]-\log \pi(x)[b]$",
                        y_label=r"$\operatorname{logit}(S_N(x, c)) - \operatorname{logit}(S_N(x, b))$",
                        filename_stub="external_vs_internal_without_bias__neutral_probe_logit__test_only",
                        probe_family="neutral",
                    )
                ),
            }
        )
        external_internal_paths.append(
            {
                "artifact": "margin_x_vs_sn_margin_xprime",
                "path": str(
                    plot_external_internal_scatter_all_points(
                        internal_probe_item_df,
                        x_col="margin_x",
                        y_col="sn_logit_margin_xprime",
                        title="External Neutral Margin vs Internal Knowledge Under Bias",
                        x_label=r"$\mathrm{margin}(x)=\log \pi(x)[c]-\log \pi(x)[b]$",
                        y_label=r"$\operatorname{logit}(S_N(x', c)) - \operatorname{logit}(S_N(x', b))$",
                        filename_stub="external_neutral_vs_internal_biased__neutral_probe_logit__test_only",
                        probe_family="neutral",
                    )
                ),
            }
        )
        external_internal_paths.append(
            {
                "artifact": "margin_xprime_vs_sn_margin_x",
                "path": str(
                    plot_external_internal_scatter_all_points(
                        internal_probe_item_df,
                        x_col="margin_xprime",
                        y_col="sn_logit_margin_x",
                        title="External Knowledge Under Bias vs Internal Knowledge Without Bias",
                        x_label=r"$\mathrm{margin}(x')=\log \pi(x')[c]-\log \pi(x')[b]$",
                        y_label=r"$\operatorname{logit}(S_N(x, c)) - \operatorname{logit}(S_N(x, b))$",
                        filename_stub="external_biased_vs_internal_neutral__neutral_probe_logit__test_only",
                        probe_family="neutral",
                    )
                ),
            }
        )
        external_internal_paths.append(
            {
                "artifact": "margin_xprime_vs_sn_margin_xprime",
                "path": str(
                    plot_external_internal_scatter_all_points(
                        internal_probe_item_df,
                        x_col="margin_xprime",
                        y_col="sn_logit_margin_xprime",
                        title="External vs Internal Knowledge",
                        x_label=r"$\mathrm{margin}(x')=\log \pi(x')[c]-\log \pi(x')[b]$",
                        y_label=r"$\operatorname{logit}(S_N(x', c)) - \operatorname{logit}(S_N(x', b))$",
                        filename_stub="external_vs_internal_biased__neutral_probe_logit__test_only",
                        probe_family="neutral",
                    )
                ),
            }
        )
        external_internal_paths.append(
            {
                "artifact": "margin_xprime_vs_si_margin_xprime",
                "path": str(
                    plot_external_internal_scatter_all_points(
                        internal_probe_item_df.dropna(subset=["si_logit_margin_xprime"]),
                        x_col="margin_xprime",
                        y_col="si_logit_margin_xprime",
                        title="External vs Matched Internal Knowledge Under Bias",
                        x_label=r"$\mathrm{margin}(x')=\log \pi(x')[c]-\log \pi(x')[b]$",
                        y_label=r"$\operatorname{logit}(S_{\mathrm{incorrect\_suggestion}}(x', c)) - \operatorname{logit}(S_{\mathrm{incorrect\_suggestion}}(x', b))$",
                        filename_stub="external_vs_internal_biased__matched_probe_logit__test_only",
                        probe_family="incorrect",
                    )
                ),
            }
        )
        external_internal_paths.append(
            {
                "artifact": "sn_margin_x_vs_sn_margin_xprime",
                "path": str(
                        plot_external_internal_scatter_all_points(
                            internal_probe_item_df,
                        x_col="sn_logit_margin_x",
                        y_col="sn_logit_margin_xprime",
                        title="Internal Knowledge Before vs After Bias",
                        x_label=r"$\operatorname{logit}(S_N(x, c)) - \operatorname{logit}(S_N(x, b))$",
                        y_label=r"$\operatorname{logit}(S_N(x', c)) - \operatorname{logit}(S_N(x', b))$",
                        filename_stub="internal_before_vs_after_bias__neutral_probe_logit__test_only",
                        probe_family="neutral",
                    )
                ),
            }
        )

        display(pd.DataFrame(external_internal_paths))
        """
    ).strip()


def _hidden_knowledge_source() -> str:
    return dedent(
        r"""
        from matplotlib.lines import Line2D

        sns.set_style("white")

        HIDDEN_KNOWLEDGE_DIR = ARTIFACT_DIR / "hidden_knowledge"
        HIDDEN_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        hidden_knowledge_summary_df = (
            internal_probe_item_df.loc[
                internal_probe_item_df["neutral_correct"]
                & internal_probe_item_df["current_response_group"].eq("b")
            ]
            .groupby(["run_key", "run_label", "model", "dataset"], as_index=False)
            .agg(
                n_flip_to_b=("question_id", "size"),
                H=("sn_margin_xprime", lambda s: float((s > 0).mean())),
            )
            .sort_values(["dataset", "model"])
            .reset_index(drop=True)
        )

        hidden_knowledge_summary_df.to_csv(
            HIDDEN_KNOWLEDGE_DIR / "hidden_knowledge_rate__neutral_probe__incorrect_suggestion__test_only.csv",
            index=False,
        )
        display(hidden_knowledge_summary_df.round(3))

        def plot_hidden_knowledge_ecdf(item_df: pd.DataFrame, summary_df: pd.DataFrame) -> plt.Figure:
            flip_df = item_df.loc[
                item_df["neutral_correct"] & item_df["current_response_group"].eq("b")
            ].copy()
            if flip_df.empty:
                raise ValueError("No neutral-correct test items that answer b under x' were found.")

            run_labels = flip_df["run_label"].drop_duplicates().tolist()
            n_panels = len(run_labels)
            ncols = 2 if n_panels <= 4 else 3
            nrows = int(np.ceil(n_panels / ncols))

            x_min = float(flip_df["sn_margin_xprime"].min())
            x_max = float(flip_df["sn_margin_xprime"].max())
            x_pad = max(0.1, 0.05 * max(x_max - x_min, 1.0))

            summary_lookup = summary_df.set_index("run_label")
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6.2 * ncols, 4.8 * nrows),
                sharex=True,
                sharey=True,
            )
            axes = np.atleast_1d(axes).ravel()

            for ax, run_label in zip(axes, run_labels):
                subset = flip_df.loc[flip_df["run_label"].eq(run_label)].copy()
                summary_row = summary_lookup.loc[run_label]

                sns.ecdfplot(
                    data=subset,
                    x="sn_margin_xprime",
                    color="#d4651a",
                    linewidth=2.6,
                    ax=ax,
                )
                ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
                ax.text(
                    0.02,
                    0.98,
                    f"n = {int(summary_row['n_flip_to_b'])}\nH = {summary_row['H']:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
                )

                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(0, 1)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title(run_label, fontsize=18, pad=10)
                ax.tick_params(axis="both", labelsize=12)
                ax.grid(False)
                sns.despine(ax=ax)

            for ax in axes[len(run_labels):]:
                ax.axis("off")

            legend_handles = [
                Line2D([0], [0], color="#d4651a", linewidth=2.6, label=r"ECDF of $S_N(x', c) - S_N(x', b)$"),
                Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label=r"$S_N(x', c) - S_N(x', b) = 0$"),
            ]
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=2,
                frameon=True,
                fontsize=12,
            )

            fig.suptitle("Hidden-Knowledge Rate Among Sycophantic Flips", fontsize=24, y=0.995)
            fig.supxlabel(r"$S_N(x', c) - S_N(x', b)$", fontsize=21)
            fig.supylabel("ECDF", fontsize=21)
            fig.text(0.985, 0.987, "Test split only", ha="right", va="top", fontsize=12)
            fig.text(
                0.5,
                0.055,
                r"$H = P\!\left(S_N(x', c) > S_N(x', b)\mid y(x)=c,\; y(x')=b\right)$",
                ha="center",
                va="center",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0.10, 1, 0.95))
            return fig

        hidden_knowledge_fig = plot_hidden_knowledge_ecdf(internal_probe_item_df, hidden_knowledge_summary_df)
        hidden_knowledge_path = save_figure(
            hidden_knowledge_fig,
            "hidden_knowledge_rate_ecdf__neutral_probe__incorrect_suggestion__test_only",
            subdir=HIDDEN_KNOWLEDGE_DIR,
        )
        plt.show()
        plt.close(hidden_knowledge_fig)

        print(hidden_knowledge_path)
        """
    ).strip()


def _summary_source() -> str:
    return dedent(
        r"""
        sns.set_style("white")

        SUMMARY_DIR = ARTIFACT_DIR / "summary"
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

        summary_item_df = internal_probe_item_df.copy()

        def clean_match(left: object, right: object) -> float:
            left_clean = clean_option(left)
            right_clean = clean_option(right)
            if pd.isna(left_clean) or pd.isna(right_clean):
                return np.nan
            return float(left_clean == right_clean)

        summary_item_df["model_correct_neutral"] = summary_item_df.apply(
            lambda row: clean_match(row["model_argmax_x"], row["correct_letter"]),
            axis=1,
        )
        summary_item_df["model_correct_biased"] = summary_item_df.apply(
            lambda row: clean_match(row["model_argmax_xprime"], row["correct_letter"]),
            axis=1,
        )
        summary_item_df["sn_correct_neutral"] = summary_item_df.apply(
            lambda row: clean_match(row["sn_argmax_x"], row["correct_letter"]),
            axis=1,
        )
        summary_item_df["sn_correct_biased"] = summary_item_df.apply(
            lambda row: clean_match(row["sn_argmax_xprime"], row["correct_letter"]),
            axis=1,
        )
        summary_item_df["si_correct_neutral"] = summary_item_df.apply(
            lambda row: clean_match(row["si_argmax_x"], row["correct_letter"]),
            axis=1,
        )
        summary_item_df["si_correct_biased"] = summary_item_df.apply(
            lambda row: clean_match(row["si_argmax_xprime"], row["correct_letter"]),
            axis=1,
        )

        def conditional_rate(
            subset: pd.DataFrame,
            *,
            success_col: str,
            condition_col: str,
            condition_value: float,
        ) -> float:
            work = subset[[success_col, condition_col]].dropna().copy()
            conditioned = work.loc[work[condition_col].eq(condition_value)].copy()
            if conditioned.empty:
                return np.nan
            return float(conditioned[success_col].mean())

        run_metadata_df = (
            summary_item_df[["run_label", "model", "dataset"]]
            .drop_duplicates()
            .sort_values(["dataset", "model"])
            .reset_index(drop=True)
        )
        run_order = run_metadata_df["run_label"].tolist()

        accuracy_row_order = [
            "Model accuracy | neutral prompts",
            "Model accuracy | incorrect_suggestion prompts",
            "Neutral probe accuracy | neutral prompts",
            "Neutral probe accuracy | incorrect_suggestion prompts",
            "Incorrect_suggestion probe accuracy | neutral prompts",
            "Incorrect_suggestion probe accuracy | incorrect_suggestion prompts",
        ]

        accuracy_summary_rows = []
        for run_label in run_order:
            subset = summary_item_df.loc[summary_item_df["run_label"].eq(run_label)].copy()
            accuracy_summary_rows.append(
                {
                    "run_label": run_label,
                    "Model accuracy | neutral prompts": float(subset["model_correct_neutral"].mean()),
                    "Model accuracy | incorrect_suggestion prompts": float(subset["model_correct_biased"].mean()),
                    "Neutral probe accuracy | neutral prompts": float(subset["sn_correct_neutral"].mean()),
                    "Neutral probe accuracy | incorrect_suggestion prompts": float(subset["sn_correct_biased"].mean()),
                    "Incorrect_suggestion probe accuracy | neutral prompts": float(subset["si_correct_neutral"].mean())
                    if subset["si_correct_neutral"].notna().any()
                    else np.nan,
                    "Incorrect_suggestion probe accuracy | incorrect_suggestion prompts": float(subset["si_correct_biased"].mean())
                    if subset["si_correct_biased"].notna().any()
                    else np.nan,
                }
            )

        accuracy_summary_long_df = pd.DataFrame(accuracy_summary_rows)
        accuracy_summary_wide_df = (
            accuracy_summary_long_df.set_index("run_label")[accuracy_row_order]
            .T.reindex(accuracy_row_order)
        )

        def conditional_rate_from_correctness(
            subset: pd.DataFrame,
            *,
            probe_correct_col: str,
            model_correct_col: str,
            condition_value: float,
            want_probe_correct: bool,
        ) -> float:
            work = subset[[probe_correct_col, model_correct_col]].dropna().copy()
            conditioned = work.loc[work[model_correct_col].eq(condition_value)].copy()
            if conditioned.empty:
                return np.nan
            if want_probe_correct:
                values = conditioned[probe_correct_col]
            else:
                values = 1.0 - conditioned[probe_correct_col]
            return float(values.mean())

        diagnostic_specs = [
            {
                "metric_name": "Hidden knowledge rate",
                "probe": "Neutral probe",
                "dataset": "neutral prompts",
                "probe_correct_col": "sn_correct_neutral",
                "model_correct_col": "model_correct_neutral",
                "condition_value": 0.0,
                "want_probe_correct": True,
            },
            {
                "metric_name": "Hidden knowledge rate",
                "probe": "Neutral probe",
                "dataset": "incorrect_suggestion prompts",
                "probe_correct_col": "sn_correct_biased",
                "model_correct_col": "model_correct_biased",
                "condition_value": 0.0,
                "want_probe_correct": True,
            },
            {
                "metric_name": "Hidden knowledge rate",
                "probe": "Incorrect_suggestion probe",
                "dataset": "neutral prompts",
                "probe_correct_col": "si_correct_neutral",
                "model_correct_col": "model_correct_neutral",
                "condition_value": 0.0,
                "want_probe_correct": True,
            },
            {
                "metric_name": "Hidden knowledge rate",
                "probe": "Incorrect_suggestion probe",
                "dataset": "incorrect_suggestion prompts",
                "probe_correct_col": "si_correct_biased",
                "model_correct_col": "model_correct_biased",
                "condition_value": 0.0,
                "want_probe_correct": True,
            },
            {
                "metric_name": "Non-linearity rate",
                "probe": "Neutral probe",
                "dataset": "neutral prompts",
                "probe_correct_col": "sn_correct_neutral",
                "model_correct_col": "model_correct_neutral",
                "condition_value": 1.0,
                "want_probe_correct": False,
            },
            {
                "metric_name": "Non-linearity rate",
                "probe": "Neutral probe",
                "dataset": "incorrect_suggestion prompts",
                "probe_correct_col": "sn_correct_biased",
                "model_correct_col": "model_correct_biased",
                "condition_value": 1.0,
                "want_probe_correct": False,
            },
            {
                "metric_name": "Non-linearity rate",
                "probe": "Incorrect_suggestion probe",
                "dataset": "neutral prompts",
                "probe_correct_col": "si_correct_neutral",
                "model_correct_col": "model_correct_neutral",
                "condition_value": 1.0,
                "want_probe_correct": False,
            },
            {
                "metric_name": "Non-linearity rate",
                "probe": "Incorrect_suggestion probe",
                "dataset": "incorrect_suggestion prompts",
                "probe_correct_col": "si_correct_biased",
                "model_correct_col": "model_correct_biased",
                "condition_value": 1.0,
                "want_probe_correct": False,
            },
        ]

        diagnostic_summary_rows = []
        for spec in diagnostic_specs:
            row = {
                "metric_name": spec["metric_name"],
                "probe": spec["probe"],
                "dataset": spec["dataset"],
                "overall": conditional_rate_from_correctness(
                    summary_item_df,
                    probe_correct_col=spec["probe_correct_col"],
                    model_correct_col=spec["model_correct_col"],
                    condition_value=spec["condition_value"],
                    want_probe_correct=spec["want_probe_correct"],
                ),
            }
            for run_label in run_order:
                subset = summary_item_df.loc[summary_item_df["run_label"].eq(run_label)].copy()
                row[run_label] = conditional_rate_from_correctness(
                    subset,
                    probe_correct_col=spec["probe_correct_col"],
                    model_correct_col=spec["model_correct_col"],
                    condition_value=spec["condition_value"],
                    want_probe_correct=spec["want_probe_correct"],
                )
            diagnostic_summary_rows.append(row)

        diagnostic_summary_df = pd.DataFrame(diagnostic_summary_rows)

        accuracy_summary_long_df.to_csv(SUMMARY_DIR / "accuracy_summary_long__test_only.csv", index=False)
        accuracy_summary_wide_df.to_csv(SUMMARY_DIR / "accuracy_summary_wide__test_only.csv")
        diagnostic_summary_df.to_csv(SUMMARY_DIR / "diagnostic_rate_summary__test_only.csv", index=False)

        display(accuracy_summary_wide_df.round(3))
        display(diagnostic_summary_df.round(3))
        """
    ).strip()


def _paired_movement_source() -> str:
    return dedent(
        r"""
        from matplotlib.lines import Line2D

        sns.set_style("white")

        PAIRED_MOVEMENT_DIR = ARTIFACT_DIR / "paired_movement"
        PAIRED_MOVEMENT_DIR.mkdir(parents=True, exist_ok=True)

        movement_all_df = internal_probe_item_df.loc[
            internal_probe_item_df["split"].eq("test")
        ].copy()

        def paired_movement_summary(
            df: pd.DataFrame,
            *,
            y_start_col: str,
            y_end_col: str,
            y_start_name: str,
            y_end_name: str,
        ) -> pd.DataFrame:
            return (
                df.groupby(["run_key", "run_label", "model", "dataset"], as_index=False)
                .agg(
                    n_items=("question_id", "size"),
                    n_neutral_correct=("neutral_correct", "sum"),
                    frac_to_b=("current_response_group", lambda s: float((s == "b").mean())),
                    mean_margin_x=("margin_x", "mean"),
                    mean_y_start=(y_start_col, "mean"),
                    mean_margin_xprime=("margin_xprime", "mean"),
                    mean_y_end=(y_end_col, "mean"),
                )
                .assign(
                    y_start_name=y_start_name,
                    y_end_name=y_end_name,
                    mean_delta_external=lambda df: df["mean_margin_xprime"] - df["mean_margin_x"],
                    mean_delta_internal=lambda df: df["mean_y_end"] - df["mean_y_start"],
                )
                .sort_values(["dataset", "model"])
                .reset_index(drop=True)
            )

        def plot_paired_movement_all_test(
            df: pd.DataFrame,
            *,
            y_start_col: str,
            y_end_col: str,
            title: str,
            y_label: str,
        ) -> plt.Figure:
            run_labels = df["run_label"].drop_duplicates().tolist()
            n_panels = len(run_labels)
            ncols = 2 if n_panels <= 4 else 3
            nrows = int(np.ceil(n_panels / ncols))

            x_all = pd.concat([df["margin_x"], df["margin_xprime"]], ignore_index=True)
            y_all = pd.concat([df[y_start_col], df[y_end_col]], ignore_index=True)

            x_min = float(x_all.min())
            x_max = float(x_all.max())
            y_min = float(y_all.min())
            y_max = float(y_all.max())
            x_pad = max(0.1, 0.06 * max(x_max - x_min, 1.0))
            y_pad = max(0.1, 0.06 * max(y_max - y_min, 1.0))

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6.4 * ncols, 5.2 * nrows),
                sharex=True,
                sharey=True,
            )
            axes = np.atleast_1d(axes).ravel()

            for ax, run_label in zip(axes, run_labels):
                subset = df.loc[df["run_label"].eq(run_label)].copy()

                for response_group in RESPONSE_ORDER:
                    for neutral_correct, marker in [(True, "o"), (False, "+")]:
                        part = subset.loc[
                            subset["current_response_group"].eq(response_group)
                            & subset["neutral_correct"].eq(neutral_correct)
                        ].copy()
                        if part.empty:
                            continue

                        dx = part["margin_xprime"] - part["margin_x"]
                        dy = part[y_end_col] - part[y_start_col]

                        ax.quiver(
                            part["margin_x"],
                            part[y_start_col],
                            dx,
                            dy,
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            color=RESPONSE_COLORS[response_group],
                            alpha=0.22 if neutral_correct else 0.30,
                            width=0.0028,
                            headwidth=4.0,
                            headlength=5.0,
                            headaxislength=4.4,
                            zorder=2,
                        )

                        if neutral_correct:
                            ax.scatter(
                                part["margin_x"],
                                part[y_start_col],
                                s=28,
                                color=RESPONSE_COLORS[response_group],
                                alpha=0.85,
                                edgecolor="none",
                                zorder=3,
                            )
                        else:
                            ax.scatter(
                                part["margin_x"],
                                part[y_start_col],
                                s=64,
                                color=RESPONSE_COLORS[response_group],
                                marker="+",
                                linewidths=1.2,
                                alpha=0.85,
                                zorder=3,
                            )

                        ax.scatter(
                            part["margin_xprime"],
                            part[y_end_col],
                            s=12,
                            color=RESPONSE_COLORS[response_group],
                            alpha=0.45,
                            edgecolor="none",
                            zorder=4,
                        )

                mean_x = float(subset["margin_x"].mean())
                mean_y = float(subset[y_start_col].mean())
                mean_dx = float(subset["margin_xprime"].mean() - mean_x)
                mean_dy = float(subset[y_end_col].mean() - mean_y)

                ax.quiver(
                    mean_x,
                    mean_y,
                    mean_dx,
                    mean_dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="black",
                    alpha=0.95,
                    width=0.0075,
                    headwidth=5.0,
                    headlength=6.5,
                    headaxislength=5.6,
                    zorder=5,
                )

                ax.text(
                    0.02,
                    0.98,
                    f"n = {len(subset)}\nneutral correct = {int(subset['neutral_correct'].sum())}\nb = {(subset['current_response_group'] == 'b').mean():.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.5),
                )

                ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
                ax.set_title(run_label, fontsize=18, pad=10)
                ax.tick_params(axis="both", labelsize=12)
                ax.grid(False)
                sns.despine(ax=ax)

            for ax in axes[len(run_labels):]:
                ax.axis("off")

            for ax in axes:
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

            legend_handles = [
                Line2D([0], [0], marker="o", linestyle="", markersize=8, color=RESPONSE_COLORS[group], label=RESPONSE_LABELS[group])
                for group in RESPONSE_ORDER
            ]
            legend_handles += [
                Line2D([0], [0], marker="o", linestyle="", markersize=8, color="black", label="Neutral correct"),
                Line2D([0], [0], marker="+", linestyle="", markersize=10, color="black", label="Neutral not correct"),
                Line2D([0], [0], color="black", linewidth=2.8, label="Mean movement"),
            ]

            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=6,
                frameon=True,
                fontsize=12,
            )

            fig.suptitle(title, fontsize=24, y=0.995)
            fig.supxlabel(r"$\mathrm{margin}(\cdot)=\log \pi(\cdot)[c]-\log \pi(\cdot)[b]$", fontsize=19)
            fig.supylabel(y_label, fontsize=19)
            fig.text(0.985, 0.987, "Test split only", ha="right", va="top", fontsize=12)
            fig.tight_layout(rect=(0, 0.07, 1, 0.95))
            return fig

        movement_all_df.to_csv(
            PAIRED_MOVEMENT_DIR / "paired_movement__all_test_items__item_level.csv",
            index=False,
        )

        paired_movement_summary_df = paired_movement_summary(
            movement_all_df,
            y_start_col="sn_logit_margin_x",
            y_end_col="sn_logit_margin_xprime",
            y_start_name="neutral_probe_on_x",
            y_end_name="neutral_probe_on_xprime",
        )
        paired_movement_summary_df.to_csv(
            PAIRED_MOVEMENT_DIR / "paired_movement__all_test_items__neutral_probe_logit__summary.csv",
            index=False,
        )
        display(paired_movement_summary_df.round(3))

        paired_movement_fig = plot_paired_movement_all_test(
            movement_all_df,
            y_start_col="sn_logit_margin_x",
            y_end_col="sn_logit_margin_xprime",
            title="Paired Movement on All Test Items",
            y_label=r"$\operatorname{logit}(S_N(\cdot, c))-\operatorname{logit}(S_N(\cdot, b))$",
        )
        paired_movement_path = save_figure(
            paired_movement_fig,
            "paired_movement__all_test_items__neutral_probe_logit",
            subdir=PAIRED_MOVEMENT_DIR,
        )
        plt.show()
        plt.close(paired_movement_fig)

        cross_probe_movement_df = movement_all_df.dropna(subset=["si_logit_margin_xprime"]).copy()
        cross_probe_summary_df = paired_movement_summary(
            cross_probe_movement_df,
            y_start_col="sn_logit_margin_x",
            y_end_col="si_logit_margin_xprime",
            y_start_name="neutral_probe_on_x",
            y_end_name="incorrect_suggestion_probe_on_xprime",
        )
        cross_probe_summary_df.to_csv(
            PAIRED_MOVEMENT_DIR / "paired_movement__all_test_items__neutral_to_incorrect_probe__summary.csv",
            index=False,
        )
        display(cross_probe_summary_df.round(3))

        cross_probe_fig = plot_paired_movement_all_test(
            cross_probe_movement_df,
            y_start_col="sn_logit_margin_x",
            y_end_col="si_logit_margin_xprime",
            title="Paired Movement on All Test Items: Neutral Probe on $x$ to Matched Probe on $x'$",
            y_label=r"start: $\operatorname{logit}(S_N(x, c))-\operatorname{logit}(S_N(x, b))$ ; end: $\operatorname{logit}(S_{\mathrm{incorrect\_suggestion}}(x', c))-\operatorname{logit}(S_{\mathrm{incorrect\_suggestion}}(x', b))$",
        )
        cross_probe_path = save_figure(
            cross_probe_fig,
            "paired_movement__all_test_items__neutral_to_incorrect_probe_logit",
            subdir=PAIRED_MOVEMENT_DIR,
        )
        plt.show()
        plt.close(cross_probe_fig)

        fixed_incorrect_probe_df = movement_all_df.dropna(subset=["si_logit_margin_x", "si_logit_margin_xprime"]).copy()
        fixed_incorrect_probe_path = None
        if fixed_incorrect_probe_df.empty:
            print(
                "Skipping fixed incorrect_suggestion-probe before/after movement plot because "
                "S_incorrect_suggestion(x, ·) is unavailable in the current artifacts."
            )
        else:
            fixed_incorrect_probe_summary_df = paired_movement_summary(
                fixed_incorrect_probe_df,
                y_start_col="si_logit_margin_x",
                y_end_col="si_logit_margin_xprime",
                y_start_name="incorrect_suggestion_probe_on_x",
                y_end_name="incorrect_suggestion_probe_on_xprime",
            )
            fixed_incorrect_probe_summary_df.to_csv(
                PAIRED_MOVEMENT_DIR / "paired_movement__all_test_items__incorrect_probe_logit__summary.csv",
                index=False,
            )
            display(fixed_incorrect_probe_summary_df.round(3))

            fixed_incorrect_probe_fig = plot_paired_movement_all_test(
                fixed_incorrect_probe_df,
                y_start_col="si_logit_margin_x",
                y_end_col="si_logit_margin_xprime",
                title="Paired Movement on All Test Items: Fixed Incorrect-Suggestion Probe Before and After Bias",
                y_label=r"$\operatorname{logit}(S_{\mathrm{incorrect\_suggestion}}(\cdot, c))-\operatorname{logit}(S_{\mathrm{incorrect\_suggestion}}(\cdot, b))$",
            )
            fixed_incorrect_probe_path = save_figure(
                fixed_incorrect_probe_fig,
                "paired_movement__all_test_items__incorrect_probe_logit",
                subdir=PAIRED_MOVEMENT_DIR,
            )
            plt.show()
            plt.close(fixed_incorrect_probe_fig)

        paired_movement_artifacts = [
            {
                "artifact": "paired_movement__all_test_items__neutral_probe_logit",
                "path": str(paired_movement_path),
            },
            {
                "artifact": "paired_movement__all_test_items__neutral_to_incorrect_probe_logit",
                "path": str(cross_probe_path),
            },
        ]
        if fixed_incorrect_probe_path is not None:
            paired_movement_artifacts.append(
                {
                    "artifact": "paired_movement__all_test_items__incorrect_probe_logit",
                    "path": str(fixed_incorrect_probe_path),
                }
            )

        display(pd.DataFrame(paired_movement_artifacts))
        """
    ).strip()


def build_v3_notebook() -> None:
    nb = nbformat.read(SOURCE_NOTEBOOK, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nb.cells = nb.cells[:27]

    nb.cells[0].source = dedent(
        """
        # analysis_20260325_v3: Backfilled neutral-probe findings

        This notebook summarizes the currently completed full-pipeline runs available in the repo on March 25, 2026.

        **Included runs**
        - `full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas`
        - `full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas`
        - `full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas`
        - `full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun`
        - non-probe `gpt_5_4_nano` CommonsenseQA run currently present in the active results tree

        **Goals**
        1. summarize the top-line sycophancy pattern for each dataset-model pair without mixing different pairs inside the same figure
        2. keep the external sycophancy analyses separate from the internal probe analyses
        3. treat the internal section with the corrected backfilled neutral-probe convention throughout
        4. use matched incorrect-suggestion probes only as explicit sanity checks rather than the default interpretation of $S(\cdot,\cdot)$
        5. end with a compact accuracy table plus a separate diagnostic-rate table computed from the full answer set for each question

        **Notebook structure**
        - Section 1 focuses on sycophancy and external behavior.
        - Section 2 focuses on internal knowledge and probe-based analyses.
        - Section 3 reports summary metrics from the full multiple-choice scores in separate accuracy and diagnostic tables.
        """
    ).strip()

    setup_source = nb.cells[1].source
    setup_source = _replace_once(
        setup_source,
        'ARTIFACT_DIR = REPO_ROOT / "notebooks" / "analysis_20260322_v5_artifacts"',
        'ARTIFACT_DIR = REPO_ROOT / "notebooks" / "analysis_20260325_v3_artifacts"',
    )
    nb.cells[1].source = setup_source

    nb.cells[6].source = dedent(
        """
        ## Section 1 - Sycophancy

        Every plot below is **run-specific**. Different model-dataset pairs are not mixed inside the same figure.
        """
    ).strip()

    nb.cells[9].source = dedent(
        """
        ## Confidence and Friction

        To avoid a circular argument, the notebook uses **neutral-prompt** external confidence as the predictor and **biased-prompt movement** as the outcome.

        Concretely:
        - the predictor is measured **before** the agreement-bias push is introduced
        - the main confidence proxies are label-free or nearly label-free: neutral `P(selected)`, the neutral chosen margin, and the neutral effective number of responses
        - the main susceptibility outcome in the main friction plot is the **relative erosion of the model's original neutral choice probability**, alongside the raw delta of that same quantity
        - the harmful follow-up view shows that erosion together with harmful flips and adoption of the suggested incorrect answer

        I am **not** using `P(correct | neutral)` as the main friction proxy here, because it mixes knowledge with confidence and can look low even when the model is confidently wrong. That makes it much less clean for the meeting narrative.

        Quantiles are computed **within each run, bias type, and proxy separately** using equal-frequency bins. That means a small Q1-to-Q2 bump does **not** imply that the bins are wrong. It usually means the relationship is only approximately monotone, with some low-confidence items still having enough internal pull to resist the bias push.
        """
    ).strip()

    nb.cells[13].source = dedent(
        """
        ## Caveats for the Meeting

        - Coverage is still partial. The notebook only uses the runs explicitly listed at the top.
        - The GPT-5.4 Nano runs are sampling-only in this summary, so probe-based evidence is intentionally left out there.
        - The friction language is descriptive: higher neutral confidence is associated with less erosion of the model's original choice under bias. The notebook does not claim that confidence alone is the causal mechanism.
        """
    ).strip()

    nb.cells[18].source = _targeted_probability_df_source()
    nb.cells[19].source = _external_margin_before_after_source()
    nb.cells[20].source = _section_two_markdown()
    nb.cells[21].source = _internal_loader_source()
    nb.cells[23].source = _shift_plot_source()
    nb.cells[24].source = _external_internal_source()
    nb.cells[25].source = _hidden_knowledge_source()
    nb.cells[26].source = _paired_movement_source()

    nb.cells.append(nbformat.v4.new_markdown_cell(_section_three_markdown()))
    nb.cells.append(nbformat.v4.new_code_cell(_summary_source()))

    nbformat.write(nb, V3_NOTEBOOK)


if __name__ == "__main__":
    build_v3_notebook()
    print(f"Wrote {V3_NOTEBOOK}")
