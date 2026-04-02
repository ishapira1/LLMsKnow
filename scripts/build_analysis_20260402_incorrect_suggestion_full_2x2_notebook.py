from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "analysis_20260402_incorrect_suggestion_full_2x2.ipynb"


def _intro_markdown() -> str:
    return dedent(
        """
        # analysis_20260402_incorrect_suggestion_full_2x2

        This notebook is a focused analysis of the full incorrect-suggestion 2x2 probe readout matrix on the main Llama 3.1 8B and Qwen 2.5 7B runs.

        **Goal**
        - answer the preservation question with the frozen neutral probe
        - answer the re-encoding question with the incorrect-suggestion probe
        - show the direct `trained_on` / `evaluated_on` comparison for the incorrect-suggestion condition

        **Probe notation**
        - `S_N(x, ·)` = neutral probe on the neutral prompt
        - `S_N(x', ·)` = neutral probe on the incorrect-suggestion prompt
        - `S_I(x, ·)` = incorrect-suggestion probe on the neutral prompt
        - `S_I(x', ·)` = incorrect-suggestion probe on the incorrect-suggestion prompt

        **Primary metrics**
        - `rank1_correct`: whether the probe ranks the correct answer first
        - `prefers_c_over_b`: whether the correct answer beats the suggested wrong answer
        - `margin_c_minus_b`: `S(·, c) - S(·, b)`
        - `truth_margin`: `S(·, c) - max_{y != c} S(·, y)`

        The notebook writes tables to `notebooks/analysis_20260402_incorrect_suggestion_full_2x2_artifacts/tables` and plots to `notebooks/analysis_20260402_incorrect_suggestion_full_2x2_artifacts/plots`.
        """
    ).strip()


def _setup_source() -> str:
    return dedent(
        """
        from pathlib import Path
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        seaborn = sns
        seaborn.set_style("white")

        REPO_ROOT = Path.cwd()
        if not (REPO_ROOT / "src" / "llmssycoph").exists():
            for candidate in [Path.cwd(), *Path.cwd().parents]:
                src_dir = candidate / "src"
                if (src_dir / "llmssycoph").exists():
                    REPO_ROOT = candidate
                    break
            else:
                raise FileNotFoundError("Could not locate repo root containing src/llmssycoph.")

        SRC_DIR = REPO_ROOT / "src"
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        from llmssycoph.analysis import load_analysis_context
        from llmssycoph.analysis.dataframes import build_paired_external_df

        RESULTS_ROOT = REPO_ROOT / "results" / "sycophancy_bias_probe"
        ARTIFACT_DIR = REPO_ROOT / "notebooks" / "analysis_20260402_incorrect_suggestion_full_2x2_artifacts"
        TABLE_DIR = ARTIFACT_DIR / "tables"
        PLOT_DIR = ARTIFACT_DIR / "plots"
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        NEUTRAL_COLOR = "#73b3ab"
        INCORRECT_COLOR = "#d4651a"
        SUPPORT_COLOR = "#4c4c4c"
        PALETTE = {
            "neutral": NEUTRAL_COLOR,
            "incorrect_suggestion": INCORRECT_COLOR,
            "preservation": NEUTRAL_COLOR,
            "reencoding": INCORRECT_COLOR,
            "recovered_only": SUPPORT_COLOR,
        }

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 160)
        """
    ).strip()


def _loader_source() -> str:
    return dedent(
        r"""
        RUN_SPECS = [
            {
                "run_key": "llama31_8b__commonsense_qa",
                "run_name": "full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas",
                "relative_run_dir": "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/commonsense_qa/full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas",
                "model_label": "Llama 3.1 8B Instruct",
                "dataset_label": "CommonsenseQA",
            },
            {
                "run_key": "llama31_8b__arc_challenge",
                "run_name": "full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas",
                "relative_run_dir": "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/arc_challenge/full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas",
                "model_label": "Llama 3.1 8B Instruct",
                "dataset_label": "ARC-Challenge",
            },
            {
                "run_key": "qwen25_7b__commonsense_qa",
                "run_name": "full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas",
                "relative_run_dir": "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/commonsense_qa/full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas",
                "model_label": "Qwen 2.5 7B Instruct",
                "dataset_label": "CommonsenseQA",
            },
            {
                "run_key": "qwen25_7b__arc_challenge",
                "run_name": "full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun",
                "relative_run_dir": "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/arc_challenge/full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun",
                "model_label": "Qwen 2.5 7B Instruct",
                "dataset_label": "ARC-Challenge",
            },
        ]

        SCORE_LETTERS = list("ABCDE")


        def clean_option(value: object) -> str | float:
            if pd.isna(value):
                return np.nan
            text = str(value).strip().upper()
            if text in {"", "NAN", "NONE", "NULL"}:
                return np.nan
            return text


        def resolve_run_dir(spec: dict) -> Path:
            preferred = (REPO_ROOT / spec["relative_run_dir"]).resolve()
            if preferred.exists() and (preferred / "run_config.json").exists():
                return preferred

            matches = sorted(path.parent for path in RESULTS_ROOT.glob(f"**/{spec['run_name']}/run_config.json"))
            if len(matches) == 1:
                return matches[0].resolve()
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple run directories matched {spec['run_name']!r}: "
                    + ", ".join(str(path) for path in matches)
                )
            raise FileNotFoundError(f"Could not resolve run {spec['run_name']!r} under {RESULTS_ROOT}.")


        def resolve_existing_file(
            run_name: str,
            relative_suffix: str,
            *,
            preferred_run_dir: Path | None = None,
            required: bool = True,
        ) -> Path | None:
            if preferred_run_dir is not None:
                preferred = preferred_run_dir / relative_suffix
                if preferred.exists():
                    return preferred.resolve()

            matches = sorted(RESULTS_ROOT.glob(f"**/{run_name}/{relative_suffix}"))
            if matches:
                return matches[0].resolve()

            if required:
                raise FileNotFoundError(
                    f"Could not find {relative_suffix!r} for run {run_name!r} under {RESULTS_ROOT}."
                )
            return None


        def resolve_optional_incorrect_all_templates_path(run_name: str, run_dir: Path) -> Path | None:
            preferred = run_dir / "probes" / "backfills" / "probe_bias_incorrect_suggestion_all_templates" / "probe_scores_by_prompt.csv"
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


        def load_prompt_df(path: Path | None) -> pd.DataFrame:
            if path is None or not path.exists():
                return pd.DataFrame()
            df = pd.read_csv(path)
            for column in ["correct_letter", "incorrect_letter", "selected_choice", "probe_argmax_choice"]:
                if column in df.columns:
                    df[column] = df[column].map(clean_option)
            return df


        def rename_score_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
            rename_map = {f"score_{letter}": f"{prefix}{letter}" for letter in SCORE_LETTERS if f"score_{letter}" in df.columns}
            return df.rename(columns=rename_map)


        def build_variant_score_df(
            prompt_df: pd.DataFrame,
            *,
            template_type: str,
            prefix: str,
            join_keys: list[str],
            probe_name: str | None = None,
        ) -> pd.DataFrame:
            if prompt_df.empty or "template_type" not in prompt_df.columns:
                return pd.DataFrame(columns=join_keys)
            subset = prompt_df.loc[prompt_df["template_type"].astype(str).eq(template_type)].copy()
            if probe_name is not None and "probe_name" in subset.columns:
                subset = subset.loc[subset["probe_name"].astype(str).eq(probe_name)].copy()
            if subset.empty:
                return pd.DataFrame(columns=join_keys)
            subset = rename_score_columns(subset, prefix)
            score_cols = [f"{prefix}{letter}" for letter in SCORE_LETTERS if f"{prefix}{letter}" in subset.columns]
            keep_cols = [column for column in join_keys if column in subset.columns] + score_cols
            return subset[keep_cols].drop_duplicates(subset=join_keys)


        def ensure_prefixed_score_columns(df: pd.DataFrame, prefix: str) -> None:
            for letter in SCORE_LETTERS:
                column = f"{prefix}{letter}"
                if column not in df.columns:
                    df[column] = np.nan


        def score_value(row: pd.Series, prefix: str, option: object) -> float:
            option = clean_option(option)
            if pd.isna(option):
                return np.nan
            value = row.get(f"{prefix}{option}", np.nan)
            return float(value) if pd.notna(value) else np.nan


        def argmax_prefixed_value(row: pd.Series, prefix: str) -> str | float:
            best_letter = np.nan
            best_value = None
            for letter in SCORE_LETTERS:
                value = row.get(f"{prefix}{letter}", np.nan)
                if pd.isna(value):
                    continue
                numeric = float(value)
                if best_value is None or numeric > best_value:
                    best_value = numeric
                    best_letter = letter
            return best_letter


        def best_other_score(row: pd.Series, prefix: str, correct_letter: object) -> float:
            correct_letter = clean_option(correct_letter)
            values = []
            for letter in SCORE_LETTERS:
                if letter == correct_letter:
                    continue
                value = row.get(f"{prefix}{letter}", np.nan)
                if pd.notna(value):
                    values.append(float(value))
            return max(values) if values else np.nan


        def summarize_prefix_columns(df: pd.DataFrame, prefix: str, *, correct_col: str = "correct_letter", bias_col: str = "incorrect_letter") -> None:
            ensure_prefixed_score_columns(df, prefix)
            df[f"{prefix}available"] = df[[f"{prefix}{letter}" for letter in SCORE_LETTERS]].notna().any(axis=1)
            df[f"{prefix}score_correct"] = df.apply(lambda row: score_value(row, prefix, row[correct_col]), axis=1)
            df[f"{prefix}score_bias_target"] = df.apply(lambda row: score_value(row, prefix, row[bias_col]), axis=1)
            df[f"{prefix}margin_c_minus_b"] = df[f"{prefix}score_correct"] - df[f"{prefix}score_bias_target"]
            df[f"{prefix}truth_margin"] = df.apply(
                lambda row: row[f"{prefix}score_correct"] - best_other_score(row, prefix, row[correct_col]),
                axis=1,
            )
            df[f"{prefix}argmax_choice"] = df.apply(lambda row: argmax_prefixed_value(row, prefix), axis=1)
            df[f"{prefix}argmax_is_correct"] = df[f"{prefix}argmax_choice"].eq(df[correct_col])
            df[f"{prefix}prefers_c_over_b"] = df[f"{prefix}margin_c_minus_b"].gt(0)


        def load_run_item_df(spec: dict) -> tuple[dict, pd.DataFrame]:
            run_dir = resolve_run_dir(spec)
            ctx = load_analysis_context(run_dir)
            paired_df = build_paired_external_df(ctx)
            paired_df = paired_df.loc[
                paired_df["bias_type"].astype(str).eq("incorrect_suggestion")
                & paired_df["split"].astype(str).eq("test")
            ].copy()
            if paired_df.empty:
                raise ValueError(f"No incorrect_suggestion/test rows found for {spec['run_name']}.")

            for column in ["correct_letter", "incorrect_letter", "response_x", "response_xprime"]:
                if column in paired_df.columns:
                    paired_df[column] = paired_df[column].map(clean_option)

            join_keys = [column for column in ["question_id", "split", "draw_idx"] if column in paired_df.columns]

            neutral_backfill_path = resolve_existing_file(
                spec["run_name"],
                "probes/backfills/probe_no_bias_all_templates/probe_scores_by_prompt.csv",
                preferred_run_dir=run_dir,
                required=True,
            )
            standard_prompt_path = resolve_existing_file(
                spec["run_name"],
                "probes/probe_scores_by_prompt.csv",
                preferred_run_dir=run_dir,
                required=True,
            )
            incorrect_all_templates_path = resolve_optional_incorrect_all_templates_path(spec["run_name"], run_dir)

            neutral_backfill_df = load_prompt_df(neutral_backfill_path)
            standard_prompt_df = load_prompt_df(standard_prompt_path)
            incorrect_all_templates_df = load_prompt_df(incorrect_all_templates_path)

            sn_x_df = build_variant_score_df(
                neutral_backfill_df,
                template_type="neutral",
                prefix="sn_x_",
                join_keys=join_keys,
                probe_name="probe_no_bias",
            )
            sn_xprime_df = build_variant_score_df(
                neutral_backfill_df,
                template_type="incorrect_suggestion",
                prefix="sn_xprime_",
                join_keys=join_keys,
                probe_name="probe_no_bias",
            )
            si_xprime_df = build_variant_score_df(
                standard_prompt_df,
                template_type="incorrect_suggestion",
                prefix="si_xprime_",
                join_keys=join_keys,
                probe_name="probe_bias_incorrect_suggestion",
            )
            si_x_df = build_variant_score_df(
                incorrect_all_templates_df,
                template_type="neutral",
                prefix="si_x_",
                join_keys=join_keys,
                probe_name="probe_bias_incorrect_suggestion",
            )

            merged = (
                paired_df
                .merge(sn_x_df, on=join_keys, how="left")
                .merge(sn_xprime_df, on=join_keys, how="left")
                .merge(si_xprime_df, on=join_keys, how="left")
                .merge(si_x_df, on=join_keys, how="left")
            )

            for prefix in ["sn_x_", "sn_xprime_", "si_x_", "si_xprime_"]:
                summarize_prefix_columns(merged, prefix)

            merged["response_changed"] = merged["response_x"].ne(merged["response_xprime"])
            merged["adopts_bias_target"] = merged["response_xprime"].eq(merged["incorrect_letter"])
            merged["recovered_only_rank1"] = (~merged["sn_xprime_argmax_is_correct"].fillna(False)) & merged["si_xprime_argmax_is_correct"].fillna(False)
            merged["recovered_only_prefers_c_over_b"] = (
                (~merged["sn_xprime_prefers_c_over_b"].fillna(False))
                & merged["si_xprime_prefers_c_over_b"].fillna(False)
            )
            merged["neutral_probe_loses_rank1_under_bias"] = merged["sn_x_argmax_is_correct"].fillna(False) & (~merged["sn_xprime_argmax_is_correct"].fillna(False))
            merged["neutral_probe_loses_c_over_b_under_bias"] = merged["sn_x_prefers_c_over_b"].fillna(False) & (~merged["sn_xprime_prefers_c_over_b"].fillna(False))

            merged.insert(0, "run_key", spec["run_key"])
            merged.insert(1, "run_label", f"{spec['model_label']} / {spec['dataset_label']}")
            merged.insert(2, "model_label", spec["model_label"])
            merged.insert(3, "dataset_label", spec["dataset_label"])
            merged.insert(4, "resolved_run_dir", str(run_dir))

            inventory_row = {
                "run_key": spec["run_key"],
                "run_label": f"{spec['model_label']} / {spec['dataset_label']}",
                "model_label": spec["model_label"],
                "dataset_label": spec["dataset_label"],
                "resolved_run_dir": str(run_dir),
                "neutral_backfill_path": str(neutral_backfill_path),
                "standard_prompt_path": str(standard_prompt_path),
                "incorrect_all_templates_path": str(incorrect_all_templates_path) if incorrect_all_templates_path is not None else "",
                "has_si_x_backfill": incorrect_all_templates_path is not None,
                "n_test_items": int(len(merged)),
            }
            return inventory_row, merged


        inventory_rows = []
        item_frames = []
        for spec in RUN_SPECS:
            inventory_row, run_item_df = load_run_item_df(spec)
            inventory_rows.append(inventory_row)
            item_frames.append(run_item_df)

        inventory_df = pd.DataFrame(inventory_rows).sort_values(["dataset_label", "model_label"]).reset_index(drop=True)
        item_df = pd.concat(item_frames, ignore_index=True).sort_values(["dataset_label", "model_label", "question_id", "draw_idx"]).reset_index(drop=True)

        inventory_df.to_csv(TABLE_DIR / "01_inventory.csv", index=False)
        item_df.to_csv(TABLE_DIR / "02_item_level_full_2x2.csv", index=False)

        display(inventory_df)
        print(f"Loaded {len(item_df):,} incorrect-suggestion test rows across {len(inventory_df)} runs.")
        """
    ).strip()


def _cross_eval_source() -> str:
    return dedent(
        r"""
        CROSS_EVAL_SPECS = [
            ("neutral", "neutral", "sn_x_"),
            ("neutral", "incorrect_suggestion", "sn_xprime_"),
            ("incorrect_suggestion", "neutral", "si_x_"),
            ("incorrect_suggestion", "incorrect_suggestion", "si_xprime_"),
        ]


        def build_cross_eval_long(df: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for trained_on, evaluated_on, prefix in CROSS_EVAL_SPECS:
                available_col = f"{prefix}available"
                rows.append(
                    df[
                        [
                            "run_key",
                            "run_label",
                            "model_label",
                            "dataset_label",
                            "question_id",
                            "draw_idx",
                        ]
                    ].assign(
                        trained_on=trained_on,
                        evaluated_on=evaluated_on,
                        available=df[available_col].astype(bool),
                        rank1_correct=df[f"{prefix}argmax_is_correct"],
                        prefers_c_over_b=df[f"{prefix}prefers_c_over_b"],
                        margin_c_minus_b=df[f"{prefix}margin_c_minus_b"],
                        truth_margin=df[f"{prefix}truth_margin"],
                    )
                )
            return pd.concat(rows, ignore_index=True)


        cross_eval_long_df = build_cross_eval_long(item_df)
        cross_eval_long_df.to_csv(TABLE_DIR / "03_cross_eval_long.csv", index=False)


        def summarize_cross_eval(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
            records = []
            grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
            for group_key, subset in grouped:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                base = {column: value for column, value in zip(group_cols, group_key)}
                for trained_on, evaluated_on in [(a, b) for a, b, _ in CROSS_EVAL_SPECS]:
                    block = subset.loc[
                        subset["trained_on"].eq(trained_on)
                        & subset["evaluated_on"].eq(evaluated_on)
                    ].copy()
                    available_block = block.loc[block["available"]].copy()
                    records.append(
                        {
                            **base,
                            "trained_on": trained_on,
                            "evaluated_on": evaluated_on,
                            "n_pairs": int(len(block)),
                            "available_rows": int(block["available"].sum()),
                            "availability_rate": float(block["available"].mean()) if len(block) else np.nan,
                            "rank1_correct": float(available_block["rank1_correct"].mean()) if not available_block.empty else np.nan,
                            "prefers_c_over_b": float(available_block["prefers_c_over_b"].mean()) if not available_block.empty else np.nan,
                            "mean_margin_c_minus_b": float(available_block["margin_c_minus_b"].mean()) if not available_block.empty else np.nan,
                            "mean_truth_margin": float(available_block["truth_margin"].mean()) if not available_block.empty else np.nan,
                        }
                    )
            return pd.DataFrame(records)


        cross_eval_overall_df = summarize_cross_eval(cross_eval_long_df, [])
        cross_eval_by_run_df = summarize_cross_eval(cross_eval_long_df, ["run_key", "run_label", "model_label", "dataset_label"])

        cross_eval_overall_df.to_csv(TABLE_DIR / "04_cross_eval_summary_overall.csv", index=False)
        cross_eval_by_run_df.to_csv(TABLE_DIR / "05_cross_eval_summary_by_run.csv", index=False)

        rank1_matrix = cross_eval_overall_df.pivot(index="trained_on", columns="evaluated_on", values="rank1_correct")
        margin_matrix = cross_eval_overall_df.pivot(index="trained_on", columns="evaluated_on", values="mean_margin_c_minus_b")

        display(cross_eval_overall_df.round(4))
        display(rank1_matrix.round(4))
        display(margin_matrix.round(4))
        """
    ).strip()


def _preservation_source() -> str:
    return dedent(
        r"""
        def conditional_mean(frame: pd.DataFrame, numerator_col: str, condition: pd.Series) -> float:
            subset = frame.loc[condition].copy()
            if subset.empty:
                return np.nan
            return float(subset[numerator_col].mean())


        preservation_rows = []
        grouped = item_df.groupby(["run_key", "run_label", "model_label", "dataset_label"], dropna=False)
        for group_key, subset in grouped:
            run_key, run_label, model_label, dataset_label = group_key
            preservation_rows.append(
                {
                    "run_key": run_key,
                    "run_label": run_label,
                    "model_label": model_label,
                    "dataset_label": dataset_label,
                    "n_items": int(len(subset)),
                    "si_x_available_rate": float(subset["si_x_available"].mean()),
                    "sn_rank1_on_x": float(subset["sn_x_argmax_is_correct"].mean()),
                    "sn_rank1_on_xprime": float(subset["sn_xprime_argmax_is_correct"].mean()),
                    "si_rank1_on_x": float(subset.loc[subset["si_x_available"], "si_x_argmax_is_correct"].mean()) if subset["si_x_available"].any() else np.nan,
                    "si_rank1_on_xprime": float(subset["si_xprime_argmax_is_correct"].mean()),
                    "sn_prefers_c_over_b_on_x": float(subset["sn_x_prefers_c_over_b"].mean()),
                    "sn_prefers_c_over_b_on_xprime": float(subset["sn_xprime_prefers_c_over_b"].mean()),
                    "si_prefers_c_over_b_on_x": float(subset.loc[subset["si_x_available"], "si_x_prefers_c_over_b"].mean()) if subset["si_x_available"].any() else np.nan,
                    "si_prefers_c_over_b_on_xprime": float(subset["si_xprime_prefers_c_over_b"].mean()),
                    "neutral_probe_rank1_survival": conditional_mean(subset, "sn_xprime_argmax_is_correct", subset["sn_x_argmax_is_correct"].fillna(False)),
                    "neutral_probe_c_over_b_survival": conditional_mean(subset, "sn_xprime_prefers_c_over_b", subset["sn_x_prefers_c_over_b"].fillna(False)),
                    "reencoding_rank1_when_neutral_probe_fails": conditional_mean(subset, "si_xprime_argmax_is_correct", ~subset["sn_xprime_argmax_is_correct"].fillna(False)),
                    "reencoding_c_over_b_when_neutral_probe_fails": conditional_mean(subset, "si_xprime_prefers_c_over_b", ~subset["sn_xprime_prefers_c_over_b"].fillna(False)),
                    "recovered_only_rank1": float(subset["recovered_only_rank1"].mean()),
                    "recovered_only_prefers_c_over_b": float(subset["recovered_only_prefers_c_over_b"].mean()),
                    "neutral_probe_margin_shift": float((subset["sn_xprime_margin_c_minus_b"] - subset["sn_x_margin_c_minus_b"]).mean()),
                    "incorrect_probe_margin_shift": float((subset["si_xprime_margin_c_minus_b"] - subset["si_x_margin_c_minus_b"]).mean()) if subset["si_x_available"].any() else np.nan,
                    "response_changed_rate": float(subset["response_changed"].mean()),
                    "adopts_bias_target_rate": float(subset["adopts_bias_target"].mean()),
                }
            )

        preservation_df = pd.DataFrame(preservation_rows).sort_values(["dataset_label", "model_label"]).reset_index(drop=True)
        preservation_df.to_csv(TABLE_DIR / "06_preservation_reencoding_summary_by_run.csv", index=False)

        overall_row = {
            "run_key": "overall",
            "run_label": "Overall",
            "model_label": "Overall",
            "dataset_label": "Overall",
            "n_items": int(len(item_df)),
            "si_x_available_rate": float(item_df["si_x_available"].mean()),
            "sn_rank1_on_x": float(item_df["sn_x_argmax_is_correct"].mean()),
            "sn_rank1_on_xprime": float(item_df["sn_xprime_argmax_is_correct"].mean()),
            "si_rank1_on_x": float(item_df.loc[item_df["si_x_available"], "si_x_argmax_is_correct"].mean()) if item_df["si_x_available"].any() else np.nan,
            "si_rank1_on_xprime": float(item_df["si_xprime_argmax_is_correct"].mean()),
            "sn_prefers_c_over_b_on_x": float(item_df["sn_x_prefers_c_over_b"].mean()),
            "sn_prefers_c_over_b_on_xprime": float(item_df["sn_xprime_prefers_c_over_b"].mean()),
            "si_prefers_c_over_b_on_x": float(item_df.loc[item_df["si_x_available"], "si_x_prefers_c_over_b"].mean()) if item_df["si_x_available"].any() else np.nan,
            "si_prefers_c_over_b_on_xprime": float(item_df["si_xprime_prefers_c_over_b"].mean()),
            "neutral_probe_rank1_survival": conditional_mean(item_df, "sn_xprime_argmax_is_correct", item_df["sn_x_argmax_is_correct"].fillna(False)),
            "neutral_probe_c_over_b_survival": conditional_mean(item_df, "sn_xprime_prefers_c_over_b", item_df["sn_x_prefers_c_over_b"].fillna(False)),
            "reencoding_rank1_when_neutral_probe_fails": conditional_mean(item_df, "si_xprime_argmax_is_correct", ~item_df["sn_xprime_argmax_is_correct"].fillna(False)),
            "reencoding_c_over_b_when_neutral_probe_fails": conditional_mean(item_df, "si_xprime_prefers_c_over_b", ~item_df["sn_xprime_prefers_c_over_b"].fillna(False)),
            "recovered_only_rank1": float(item_df["recovered_only_rank1"].mean()),
            "recovered_only_prefers_c_over_b": float(item_df["recovered_only_prefers_c_over_b"].mean()),
            "neutral_probe_margin_shift": float((item_df["sn_xprime_margin_c_minus_b"] - item_df["sn_x_margin_c_minus_b"]).mean()),
            "incorrect_probe_margin_shift": float((item_df["si_xprime_margin_c_minus_b"] - item_df["si_x_margin_c_minus_b"]).mean()) if item_df["si_x_available"].any() else np.nan,
            "response_changed_rate": float(item_df["response_changed"].mean()),
            "adopts_bias_target_rate": float(item_df["adopts_bias_target"].mean()),
        }
        preservation_overall_df = pd.DataFrame([overall_row])
        preservation_overall_df.to_csv(TABLE_DIR / "07_preservation_reencoding_summary_overall.csv", index=False)

        display(preservation_overall_df.round(4))
        display(preservation_df.round(4))
        """
    ).strip()


def _plots_source() -> str:
    return dedent(
        r"""
        def finalize_axis(ax: plt.Axes, *, xlabel: str, ylabel: str, title: str) -> None:
            ax.set_xlabel(xlabel, fontsize=15)
            ax.set_ylabel(ylabel, fontsize=15)
            ax.set_title(title, fontsize=20, pad=10)
            ax.tick_params(axis="both", labelsize=12)
            sns.despine(ax=ax)


        plot_cross_eval_df = cross_eval_by_run_df.loc[
            cross_eval_by_run_df["evaluated_on"].isin(["neutral", "incorrect_suggestion"])
        ].copy()
        plot_cross_eval_df["trained_on_label"] = plot_cross_eval_df["trained_on"].map(
            {
                "neutral": "Train on neutral",
                "incorrect_suggestion": "Train on incorrect_suggestion",
            }
        )
        plot_cross_eval_df["evaluated_on_label"] = plot_cross_eval_df["evaluated_on"].map(
            {
                "neutral": "Eval on neutral",
                "incorrect_suggestion": "Eval on incorrect_suggestion",
            }
        )

        g = sns.catplot(
            data=plot_cross_eval_df,
            kind="bar",
            x="evaluated_on_label",
            y="rank1_correct",
            hue="trained_on_label",
            col="run_label",
            col_wrap=2,
            palette=[NEUTRAL_COLOR, INCORRECT_COLOR],
            height=4.7,
            aspect=1.1,
            legend=False,
        )
        for ax in g.axes.flatten():
            finalize_axis(ax, xlabel="", ylabel="Probe Rank-1 Correct", title=ax.get_title().replace("run_label = ", ""))
            ax.set_ylim(0, 1.02)
        handles, labels = g.axes.flatten()[0].get_legend_handles_labels()
        g.fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=True)
        g.fig.suptitle("Incorrect-Suggestion 2x2: Rank-1 Correct by Train/Eval Condition", fontsize=24, y=1.02)
        g.fig.tight_layout(rect=(0, 0.04, 1, 0.98))
        rank1_plot_path = PLOT_DIR / "01_rank1_correct_train_on_eval_on_by_run.pdf"
        g.fig.savefig(rank1_plot_path, bbox_inches="tight")
        plt.show()
        plt.close(g.fig)

        shift_plot_df = preservation_df[
            [
                "run_label",
                "neutral_probe_margin_shift",
                "incorrect_probe_margin_shift",
            ]
        ].melt(
            id_vars="run_label",
            var_name="series",
            value_name="delta_margin",
        )
        shift_plot_df["series_label"] = shift_plot_df["series"].map(
            {
                "neutral_probe_margin_shift": "Neutral probe: x' - x",
                "incorrect_probe_margin_shift": "Incorrect probe: x' - x",
            }
        )

        fig, ax = plt.subplots(figsize=(12, 5.8))
        sns.barplot(
            data=shift_plot_df,
            x="run_label",
            y="delta_margin",
            hue="series_label",
            palette=[NEUTRAL_COLOR, INCORRECT_COLOR],
            ax=ax,
        )
        ax.axhline(0, color=SUPPORT_COLOR, linewidth=1.2, alpha=0.8)
        finalize_axis(
            ax,
            xlabel="",
            ylabel="Mean Δ margin(c - b)",
            title="Incorrect-Suggestion 2x2: Mean Margin Shift Within Each Probe Family",
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, title="")
        fig.tight_layout()
        margin_shift_plot_path = PLOT_DIR / "02_margin_shift_by_probe_family.pdf"
        fig.savefig(margin_shift_plot_path, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        recovery_plot_df = preservation_df[
            [
                "run_label",
                "neutral_probe_rank1_survival",
                "reencoding_rank1_when_neutral_probe_fails",
                "recovered_only_rank1",
            ]
        ].melt(
            id_vars="run_label",
            var_name="metric",
            value_name="rate",
        )
        recovery_plot_df["metric_label"] = recovery_plot_df["metric"].map(
            {
                "neutral_probe_rank1_survival": "Preservation",
                "reencoding_rank1_when_neutral_probe_fails": "Re-encoding when neutral fails",
                "recovered_only_rank1": "Recovered only in incorrect probe",
            }
        )

        fig, ax = plt.subplots(figsize=(12, 5.8))
        sns.barplot(
            data=recovery_plot_df,
            x="run_label",
            y="rate",
            hue="metric_label",
            palette=[NEUTRAL_COLOR, INCORRECT_COLOR, SUPPORT_COLOR],
            ax=ax,
        )
        finalize_axis(
            ax,
            xlabel="",
            ylabel="Rate",
            title="Incorrect-Suggestion 2x2: Preservation vs Re-encoding",
        )
        ax.set_ylim(0, 1.02)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=True, title="")
        fig.tight_layout()
        recovery_plot_path = PLOT_DIR / "03_preservation_vs_reencoding_rates.pdf"
        fig.savefig(recovery_plot_path, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        display(
            pd.DataFrame(
                [
                    {"artifact": "rank1_correct_train_on_eval_on_by_run", "path": str(rank1_plot_path)},
                    {"artifact": "margin_shift_by_probe_family", "path": str(margin_shift_plot_path)},
                    {"artifact": "preservation_vs_reencoding_rates", "path": str(recovery_plot_path)},
                ]
            )
        )
        """
    ).strip()


def build_notebook() -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(_intro_markdown()),
        nbformat.v4.new_code_cell(_setup_source()),
        nbformat.v4.new_markdown_cell(
            "## 1. Load the full incorrect-suggestion 2x2\n\nThis cell resolves the four main runs, loads the neutral and incorrect-suggestion prompt-level probe tables, and constructs a single item-level dataframe with `S_N(x, ·)`, `S_N(x', ·)`, `S_I(x, ·)`, and `S_I(x', ·)`."
        ),
        nbformat.v4.new_code_cell(_loader_source()),
        nbformat.v4.new_markdown_cell(
            "## 2. Direct Train-On / Eval-On Tables\n\nThis is the explicit cross-evaluation view we wanted: train on neutral vs incorrect-suggestion, evaluated on neutral vs incorrect-suggestion, summarized with prompt-level metrics that directly match the preservation and re-encoding story."
        ),
        nbformat.v4.new_code_cell(_cross_eval_source()),
        nbformat.v4.new_markdown_cell(
            "## 3. Preservation vs Re-encoding Summary\n\nThese summaries separate two questions: does the neutral readout survive pressure, and if it does not, can the incorrect-suggestion readout recover the truth?"
        ),
        nbformat.v4.new_code_cell(_preservation_source()),
        nbformat.v4.new_markdown_cell(
            "## 4. Focused Plots\n\nThe plots below emphasize rank-1 correctness, within-probe margin shifts, and recovery rates. They use a fixed, high-contrast palette and place legends below the plots for easier presentation use."
        ),
        nbformat.v4.new_code_cell(_plots_source()),
    ]
    nbformat.write(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
    print(f"Wrote {NOTEBOOK_PATH}")
