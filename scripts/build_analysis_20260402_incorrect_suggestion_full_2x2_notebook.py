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


def _simplex_source() -> str:
    return dedent(
        r"""
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D

        SQRT3_OVER_2 = np.sqrt(3) / 2
        SIMPLEX_VERTICES = {
            "other": np.array([0.0, 0.0]),
            "bias": np.array([1.0, 0.0]),
            "correct": np.array([0.5, SQRT3_OVER_2]),
        }


        def simplex_to_xy(p_correct: object, p_bias: object) -> tuple[object, object]:
            p_correct = pd.to_numeric(p_correct, errors="coerce")
            p_bias = pd.to_numeric(p_bias, errors="coerce")
            return p_bias + 0.5 * p_correct, SQRT3_OVER_2 * p_correct


        def barycentric_to_xy(p_correct: float, p_bias: float, p_other: float) -> np.ndarray:
            return np.array([p_bias + 0.5 * p_correct, SQRT3_OVER_2 * p_correct], dtype=float)


        def simplex_xy_mask(x: np.ndarray, y: np.ndarray, tol: float = 1e-9) -> np.ndarray:
            return (
                (x >= -tol)
                & (x <= 1.0 + tol)
                & (y >= -tol)
                & (y <= SQRT3_OVER_2 + tol)
                & (y <= np.sqrt(3) * x + tol)
                & (y <= np.sqrt(3) * (1.0 - x) + tol)
            )


        def simplex_segment_for_constant(component: str, value: float) -> np.ndarray:
            value = float(value)
            if component == "correct":
                return np.vstack(
                    [
                        barycentric_to_xy(value, 0.0, 1.0 - value),
                        barycentric_to_xy(value, 1.0 - value, 0.0),
                    ]
                )
            if component == "bias":
                return np.vstack(
                    [
                        barycentric_to_xy(0.0, value, 1.0 - value),
                        barycentric_to_xy(1.0 - value, value, 0.0),
                    ]
                )
            return np.vstack(
                [
                    barycentric_to_xy(0.0, 1.0 - value, value),
                    barycentric_to_xy(1.0 - value, 0.0, value),
                ]
            )


        def draw_simplex_frame(ax: plt.Axes, grid_values: tuple[float, ...] = (0.25, 0.5, 0.75)) -> None:
            boundary = np.vstack(
                [
                    SIMPLEX_VERTICES["other"],
                    SIMPLEX_VERTICES["bias"],
                    SIMPLEX_VERTICES["correct"],
                    SIMPLEX_VERTICES["other"],
                ]
            )
            ax.plot(boundary[:, 0], boundary[:, 1], color=SUPPORT_COLOR, linewidth=1.6, zorder=0)

            for value in grid_values:
                for component in ("correct", "bias", "other"):
                    segment = simplex_segment_for_constant(component, value)
                    ax.plot(
                        segment[:, 0],
                        segment[:, 1],
                        color="#d8d8d8",
                        linewidth=0.8,
                        alpha=0.8,
                        zorder=0,
                    )

            ax.text(
                SIMPLEX_VERTICES["correct"][0],
                SIMPLEX_VERTICES["correct"][1] + 0.05,
                "P(correct)",
                ha="center",
                va="bottom",
                fontsize=15,
            )
            ax.text(
                SIMPLEX_VERTICES["other"][0] - 0.03,
                SIMPLEX_VERTICES["other"][1] - 0.05,
                "P(other)",
                ha="right",
                va="top",
                fontsize=15,
            )
            ax.text(
                SIMPLEX_VERTICES["bias"][0] + 0.03,
                SIMPLEX_VERTICES["bias"][1] - 0.05,
                "P(biased wrong)",
                ha="left",
                va="top",
                fontsize=15,
            )
            ax.set_xlim(-0.14, 1.14)
            ax.set_ylim(-0.14, SQRT3_OVER_2 + 0.14)
            ax.set_aspect("equal")
            ax.axis("off")


        def build_simplex_shift_df(df: pd.DataFrame) -> pd.DataFrame:
            working = df.copy()
            working["p_c_before"] = pd.to_numeric(working["p_correct_x"], errors="coerce")
            working["p_c_after"] = pd.to_numeric(working["p_correct_xprime"], errors="coerce")
            working["p_b_before"] = working.apply(
                lambda row: score_value(row, "p_x_", row["incorrect_letter"]),
                axis=1,
            )
            working["p_b_after"] = working.apply(
                lambda row: score_value(row, "p_xprime_", row["incorrect_letter"]),
                axis=1,
            )
            working["p_other_before"] = 1.0 - working["p_c_before"] - working["p_b_before"]
            working["p_other_after"] = 1.0 - working["p_c_after"] - working["p_b_after"]

            probability_columns = [
                "p_c_before",
                "p_b_before",
                "p_other_before",
                "p_c_after",
                "p_b_after",
                "p_other_after",
            ]
            for column in probability_columns:
                working[column] = pd.to_numeric(working[column], errors="coerce").clip(lower=0.0, upper=1.0)

            working = working.dropna(subset=probability_columns).copy()
            before_total = working[["p_c_before", "p_b_before", "p_other_before"]].sum(axis=1)
            after_total = working[["p_c_after", "p_b_after", "p_other_after"]].sum(axis=1)
            working = working.loc[before_total.gt(0) & after_total.gt(0)].copy()

            working[["p_c_before", "p_b_before", "p_other_before"]] = working[
                ["p_c_before", "p_b_before", "p_other_before"]
            ].div(before_total.loc[working.index], axis=0)
            working[["p_c_after", "p_b_after", "p_other_after"]] = working[
                ["p_c_after", "p_b_after", "p_other_after"]
            ].div(after_total.loc[working.index], axis=0)

            working["x_before"], working["y_before"] = simplex_to_xy(working["p_c_before"], working["p_b_before"])
            working["x_after"], working["y_after"] = simplex_to_xy(working["p_c_after"], working["p_b_after"])
            working["delta_p_correct"] = working["p_c_after"] - working["p_c_before"]
            working["delta_p_bias"] = working["p_b_after"] - working["p_b_before"]
            working["delta_p_other"] = working["p_other_after"] - working["p_other_before"]
            return working[
                [
                    "run_key",
                    "run_label",
                    "model_label",
                    "dataset_label",
                    "question_id",
                    "draw_idx",
                    "correct_letter",
                    "incorrect_letter",
                    "p_c_before",
                    "p_b_before",
                    "p_other_before",
                    "p_c_after",
                    "p_b_after",
                    "p_other_after",
                    "x_before",
                    "y_before",
                    "x_after",
                    "y_after",
                    "delta_p_correct",
                    "delta_p_bias",
                    "delta_p_other",
                ]
            ].reset_index(drop=True)


        def build_smoothed_flow_df(
            subset: pd.DataFrame,
            *,
            bandwidth: float = 0.075,
            grid_nx: int = 17,
            grid_ny: int = 15,
        ) -> pd.DataFrame:
            if subset.empty:
                return pd.DataFrame()

            x_before = subset["x_before"].to_numpy(dtype=float)
            y_before = subset["y_before"].to_numpy(dtype=float)
            dx = (subset["x_after"] - subset["x_before"]).to_numpy(dtype=float)
            dy = (subset["y_after"] - subset["y_before"]).to_numpy(dtype=float)

            x_grid = np.linspace(0.05, 0.95, grid_nx)
            y_grid = np.linspace(0.04, SQRT3_OVER_2 - 0.04, grid_ny)

            rows = []
            for x0 in x_grid:
                for y0 in y_grid:
                    if not simplex_xy_mask(np.array([x0]), np.array([y0]))[0]:
                        continue
                    dist2 = np.square(x_before - x0) + np.square(y_before - y0)
                    weights = np.exp(-0.5 * dist2 / (bandwidth ** 2))
                    total_weight = float(weights.sum())
                    if total_weight <= 1e-12:
                        continue
                    effective_n = float((total_weight ** 2) / np.square(weights).sum())
                    mean_dx = float(np.dot(weights, dx) / total_weight)
                    mean_dy = float(np.dot(weights, dy) / total_weight)
                    rows.append(
                        {
                            "x": x0,
                            "y": y0,
                            "dx": mean_dx,
                            "dy": mean_dy,
                            "magnitude": float(np.hypot(mean_dx, mean_dy)),
                            "effective_n": effective_n,
                            "weight_mass": total_weight,
                        }
                    )

            flow_df = pd.DataFrame(rows)
            if flow_df.empty:
                return flow_df

            min_effective_n = max(8.0, 0.015 * len(subset))
            magnitude_cutoff = float(flow_df["magnitude"].quantile(0.35))
            return flow_df.loc[
                flow_df["effective_n"].ge(min_effective_n)
                & flow_df["magnitude"].ge(magnitude_cutoff)
            ].reset_index(drop=True)


        def build_smoothed_start_magnitude_df(
            subset: pd.DataFrame,
            *,
            bandwidth: float = 0.08,
            grid_nx: int = 45,
            grid_ny: int = 38,
        ) -> pd.DataFrame:
            if subset.empty:
                return pd.DataFrame()

            x_before = subset["x_before"].to_numpy(dtype=float)
            y_before = subset["y_before"].to_numpy(dtype=float)
            magnitude_xy = np.hypot(
                (subset["x_after"] - subset["x_before"]).to_numpy(dtype=float),
                (subset["y_after"] - subset["y_before"]).to_numpy(dtype=float),
            )
            magnitude_prob = np.sqrt(
                np.square(subset["delta_p_correct"].to_numpy(dtype=float))
                + np.square(subset["delta_p_bias"].to_numpy(dtype=float))
                + np.square(subset["delta_p_other"].to_numpy(dtype=float))
            )

            x_grid = np.linspace(0.0, 1.0, grid_nx)
            y_grid = np.linspace(0.0, SQRT3_OVER_2, grid_ny)

            rows = []
            for x0 in x_grid:
                for y0 in y_grid:
                    if not simplex_xy_mask(np.array([x0]), np.array([y0]))[0]:
                        continue
                    dist2 = np.square(x_before - x0) + np.square(y_before - y0)
                    weights = np.exp(-0.5 * dist2 / (bandwidth ** 2))
                    total_weight = float(weights.sum())
                    if total_weight <= 1e-12:
                        continue
                    effective_n = float((total_weight ** 2) / np.square(weights).sum())
                    rows.append(
                        {
                            "x": x0,
                            "y": y0,
                            "mean_magnitude_xy": float(np.dot(weights, magnitude_xy) / total_weight),
                            "mean_magnitude_prob": float(np.dot(weights, magnitude_prob) / total_weight),
                            "effective_n": effective_n,
                            "weight_mass": total_weight,
                        }
                    )
            return pd.DataFrame(rows)


        def draw_simplex_background_magnitude(
            ax: plt.Axes,
            magnitude_df: pd.DataFrame,
            *,
            value_col: str,
            norm: Normalize,
            cmap: str = "YlOrBr",
        ) -> None:
            if magnitude_df.empty:
                return

            ax.scatter(
                magnitude_df["x"],
                magnitude_df["y"],
                c=magnitude_df[value_col],
                cmap=cmap,
                norm=norm,
                s=58,
                marker="s",
                linewidths=0,
                alpha=0.9,
                zorder=0,
                rasterized=True,
            )


        simplex_df = build_simplex_shift_df(item_df)
        simplex_df.to_csv(TABLE_DIR / "08_simplex_probability_shift_points.csv", index=False)

        model_order = list(dict.fromkeys(spec["model_label"] for spec in RUN_SPECS))
        dataset_order = list(dict.fromkeys(spec["dataset_label"] for spec in RUN_SPECS))

        fig, axes = plt.subplots(
            len(model_order),
            len(dataset_order),
            figsize=(14.5, 13.5),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_2d(axes)

        for row_idx, model_label in enumerate(model_order):
            for col_idx, dataset_label in enumerate(dataset_order):
                ax = axes[row_idx, col_idx]
                draw_simplex_frame(ax)
                subset = simplex_df.loc[
                    simplex_df["model_label"].eq(model_label)
                    & simplex_df["dataset_label"].eq(dataset_label)
                ].copy()

                if subset.empty:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=15)
                    continue

                segments = [
                    ((x_before, y_before), (x_after, y_after))
                    for x_before, y_before, x_after, y_after in zip(
                        subset["x_before"],
                        subset["y_before"],
                        subset["x_after"],
                        subset["y_after"],
                    )
                ]
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=SUPPORT_COLOR,
                        linewidths=0.55,
                        alpha=0.09,
                        zorder=1,
                    )
                )
                ax.scatter(
                    subset["x_before"],
                    subset["y_before"],
                    s=18,
                    color=NEUTRAL_COLOR,
                    alpha=0.28,
                    edgecolors="none",
                    zorder=2,
                    rasterized=True,
                )
                ax.scatter(
                    subset["x_after"],
                    subset["y_after"],
                    s=18,
                    color=INCORRECT_COLOR,
                    alpha=0.28,
                    edgecolors="none",
                    zorder=3,
                    rasterized=True,
                )
                ax.text(
                    0.03,
                    0.97,
                    f"n = {len(subset):,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=12,
                    color=SUPPORT_COLOR,
                )
                if row_idx == 0:
                    ax.set_title(dataset_label, fontsize=20, pad=18)
                if col_idx == 0:
                    ax.text(
                        -0.24,
                        0.5,
                        model_label,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=18,
                    )

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=NEUTRAL_COLOR,
                markeredgecolor="none",
                markersize=8,
                label="Before incorrect_suggestion",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=INCORRECT_COLOR,
                markeredgecolor="none",
                markersize=8,
                label="After incorrect_suggestion",
            ),
            Line2D(
                [0, 1],
                [0, 0],
                color=SUPPORT_COLOR,
                linewidth=1.4,
                alpha=0.35,
                label="Per-item shift",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.06),
            ncol=3,
            frameon=True,
            fontsize=12,
        )
        fig.suptitle(
            "Incorrect-Suggestion Shift in the P(correct) / P(biased wrong) / P(other) Simplex",
            fontsize=24,
            y=0.965,
        )
        fig.text(
            0.5,
            0.93,
            "Each neutral-prompt point is connected to its paired incorrect-suggestion point for the same item.",
            ha="center",
            va="top",
            fontsize=14,
        )
        fig.subplots_adjust(left=0.14, right=0.98, top=0.89, bottom=0.13, wspace=0.08, hspace=0.12)
        simplex_plot_path = PLOT_DIR / "04_simplex_probability_shift_2x2.pdf"
        fig.savefig(simplex_plot_path, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        fig, axes = plt.subplots(
            len(model_order),
            len(dataset_order),
            figsize=(14.5, 13.5),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_2d(axes)

        for row_idx, model_label in enumerate(model_order):
            for col_idx, dataset_label in enumerate(dataset_order):
                ax = axes[row_idx, col_idx]
                draw_simplex_frame(ax)
                subset = simplex_df.loc[
                    simplex_df["model_label"].eq(model_label)
                    & simplex_df["dataset_label"].eq(dataset_label)
                ].copy()

                if subset.empty:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=15)
                    continue

                flow_df = build_smoothed_flow_df(subset)
                centroid_before = np.array([subset["x_before"].mean(), subset["y_before"].mean()], dtype=float)
                centroid_after = np.array([subset["x_after"].mean(), subset["y_after"].mean()], dtype=float)

                ax.scatter(
                    subset["x_before"],
                    subset["y_before"],
                    s=8,
                    color=NEUTRAL_COLOR,
                    alpha=0.08,
                    edgecolors="none",
                    zorder=1,
                    rasterized=True,
                )
                ax.scatter(
                    subset["x_after"],
                    subset["y_after"],
                    s=8,
                    color=INCORRECT_COLOR,
                    alpha=0.08,
                    edgecolors="none",
                    zorder=1,
                    rasterized=True,
                )

                if not flow_df.empty:
                    width_scale = np.clip(flow_df["effective_n"].to_numpy(dtype=float), 8.0, 40.0)
                    ax.quiver(
                        flow_df["x"],
                        flow_df["y"],
                        flow_df["dx"],
                        flow_df["dy"],
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        width=0.0045,
                        headwidth=4.8,
                        headlength=6.0,
                        headaxislength=5.4,
                        color=SUPPORT_COLOR,
                        alpha=0.72,
                        zorder=3,
                    )
                    ax.scatter(
                        flow_df["x"],
                        flow_df["y"],
                        s=width_scale * 1.8,
                        color=SUPPORT_COLOR,
                        alpha=0.16,
                        edgecolors="none",
                        zorder=2,
                    )

                ax.plot(
                    [centroid_before[0], centroid_after[0]],
                    [centroid_before[1], centroid_after[1]],
                    color=INCORRECT_COLOR,
                    linewidth=2.4,
                    alpha=0.9,
                    zorder=4,
                )
                ax.scatter(
                    [centroid_before[0]],
                    [centroid_before[1]],
                    s=90,
                    color=NEUTRAL_COLOR,
                    edgecolors="white",
                    linewidths=0.9,
                    zorder=5,
                )
                ax.scatter(
                    [centroid_after[0]],
                    [centroid_after[1]],
                    s=90,
                    color=INCORRECT_COLOR,
                    edgecolors="white",
                    linewidths=0.9,
                    zorder=5,
                )
                ax.text(
                    0.03,
                    0.97,
                    f"n = {len(subset):,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=12,
                    color=SUPPORT_COLOR,
                )
                if row_idx == 0:
                    ax.set_title(dataset_label, fontsize=20, pad=18)
                if col_idx == 0:
                    ax.text(
                        -0.24,
                        0.5,
                        model_label,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=18,
                    )

        smooth_legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=NEUTRAL_COLOR,
                markeredgecolor="none",
                markersize=8,
                label="Raw before cloud",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=INCORRECT_COLOR,
                markeredgecolor="none",
                markersize=8,
                label="Raw after cloud",
            ),
            Line2D(
                [0, 1],
                [0, 0],
                color=SUPPORT_COLOR,
                linewidth=2.0,
                alpha=0.72,
                label="Kernel-smoothed local flow",
            ),
            Line2D(
                [0, 1],
                [0, 0],
                color=INCORRECT_COLOR,
                linewidth=2.4,
                alpha=0.9,
                label="Panel mean shift",
            ),
        ]
        fig.legend(
            handles=smooth_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.06),
            ncol=4,
            frameon=True,
            fontsize=12,
        )
        fig.suptitle(
            "Incorrect-Suggestion Simplex: Smoothed Movement Field",
            fontsize=24,
            y=0.965,
        )
        fig.text(
            0.5,
            0.93,
            "Gray arrows show locally averaged movement from the neutral cloud toward the biased cloud.",
            ha="center",
            va="top",
            fontsize=14,
        )
        fig.subplots_adjust(left=0.14, right=0.98, top=0.89, bottom=0.13, wspace=0.08, hspace=0.12)
        simplex_smooth_plot_path = PLOT_DIR / "05_simplex_probability_shift_smoothed_2x2.pdf"
        fig.savefig(simplex_smooth_plot_path, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        magnitude_frames = []
        for model_label in model_order:
            for dataset_label in dataset_order:
                subset = simplex_df.loc[
                    simplex_df["model_label"].eq(model_label)
                    & simplex_df["dataset_label"].eq(dataset_label)
                ].copy()
                magnitude_df = build_smoothed_start_magnitude_df(subset)
                if magnitude_df.empty:
                    continue
                magnitude_df["model_label"] = model_label
                magnitude_df["dataset_label"] = dataset_label
                magnitude_frames.append(magnitude_df)

        start_magnitude_df = pd.concat(magnitude_frames, ignore_index=True) if magnitude_frames else pd.DataFrame()
        start_magnitude_df.to_csv(TABLE_DIR / "09_simplex_start_location_magnitude_surface.csv", index=False)

        magnitude_norm = Normalize(
            vmin=float(start_magnitude_df["mean_magnitude_xy"].min()) if not start_magnitude_df.empty else 0.0,
            vmax=float(start_magnitude_df["mean_magnitude_xy"].max()) if not start_magnitude_df.empty else 1.0,
        )

        fig, axes = plt.subplots(
            len(model_order),
            len(dataset_order),
            figsize=(14.5, 13.5),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_2d(axes)
        cmap = plt.get_cmap("YlOrBr")

        for row_idx, model_label in enumerate(model_order):
            for col_idx, dataset_label in enumerate(dataset_order):
                ax = axes[row_idx, col_idx]
                subset = simplex_df.loc[
                    simplex_df["model_label"].eq(model_label)
                    & simplex_df["dataset_label"].eq(dataset_label)
                ].copy()
                magnitude_df = start_magnitude_df.loc[
                    start_magnitude_df["model_label"].eq(model_label)
                    & start_magnitude_df["dataset_label"].eq(dataset_label)
                ].copy()

                draw_simplex_background_magnitude(
                    ax,
                    magnitude_df,
                    value_col="mean_magnitude_xy",
                    norm=magnitude_norm,
                    cmap="YlOrBr",
                )
                draw_simplex_frame(ax)

                if subset.empty:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=15)
                    continue

                flow_df = build_smoothed_flow_df(subset)
                if not flow_df.empty:
                    ax.quiver(
                        flow_df["x"],
                        flow_df["y"],
                        flow_df["dx"],
                        flow_df["dy"],
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        width=0.0042,
                        headwidth=4.8,
                        headlength=6.0,
                        headaxislength=5.4,
                        color=SUPPORT_COLOR,
                        alpha=0.82,
                        zorder=3,
                    )

                ax.scatter(
                    subset["x_before"],
                    subset["y_before"],
                    s=7,
                    color="white",
                    alpha=0.14,
                    edgecolors="none",
                    zorder=2,
                    rasterized=True,
                )
                ax.text(
                    0.03,
                    0.97,
                    f"n = {len(subset):,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=12,
                    color=SUPPORT_COLOR,
                )
                if row_idx == 0:
                    ax.set_title(dataset_label, fontsize=20, pad=18)
                if col_idx == 0:
                    ax.text(
                        -0.24,
                        0.5,
                        model_label,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=18,
                    )

            scalar_mappable = plt.cm.ScalarMappable(norm=magnitude_norm, cmap=cmap)
            scalar_mappable.set_array([])

        cbar = fig.colorbar(
            scalar_mappable,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.045,
            pad=0.08,
        )
        cbar.set_label("Expected movement magnitude from this starting region", fontsize=15)
        cbar.ax.tick_params(labelsize=12)

        start_mag_legend_handles = [
            Line2D(
                [0, 1],
                [0, 0],
                color=SUPPORT_COLOR,
                linewidth=2.0,
                alpha=0.82,
                label="Smoothed direction of movement",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="none",
                markersize=7,
                alpha=0.5,
                label="Starting-point cloud",
            ),
        ]
        fig.legend(
            handles=start_mag_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.04),
            ncol=2,
            frameon=True,
            fontsize=12,
        )
        fig.suptitle(
            "Incorrect-Suggestion Simplex: How Far the Model Moves Depends on Where It Starts",
            fontsize=24,
            y=0.975,
        )
        fig.text(
            0.5,
            0.945,
            "Color shows the expected size of the shift conditional on the neutral starting region; arrows show the average direction.",
            ha="center",
            va="top",
            fontsize=14,
        )
        fig.subplots_adjust(left=0.14, right=0.98, top=0.89, bottom=0.12, wspace=0.08, hspace=0.12)
        simplex_start_magnitude_plot_path = PLOT_DIR / "06_simplex_start_location_magnitude_2x2.pdf"
        fig.savefig(simplex_start_magnitude_plot_path, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        display(
            pd.DataFrame(
                [
                    {
                        "artifact": "simplex_probability_shift_2x2",
                        "path": str(simplex_plot_path),
                    },
                    {
                        "artifact": "simplex_probability_shift_smoothed_2x2",
                        "path": str(simplex_smooth_plot_path),
                    },
                    {
                        "artifact": "simplex_start_location_magnitude_2x2",
                        "path": str(simplex_start_magnitude_plot_path),
                    },
                ]
            )
        )
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
            "## 4. Probability Simplex Shift\n\nThis simplex view collapses each item into three masses: `c = P(correct)`, `b = P(biased wrong answer)`, and `other = 1 - c - b`. Teal points are the neutral prompt and orange points are the paired incorrect-suggestion prompt; faint gray segments show the movement induced by the bias."
        ),
        nbformat.v4.new_code_cell(_simplex_source()),
        nbformat.v4.new_markdown_cell(
            "## 5. Focused Plots\n\nThe plots below emphasize rank-1 correctness, within-probe margin shifts, and recovery rates. They use a fixed, high-contrast palette and place legends below the plots for easier presentation use."
        ),
        nbformat.v4.new_code_cell(_plots_source()),
    ]
    nbformat.write(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
    print(f"Wrote {NOTEBOOK_PATH}")
