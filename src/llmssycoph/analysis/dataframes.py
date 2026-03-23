from __future__ import annotations

import json
from typing import Callable

import pandas as pd

from .core import AnalysisContext


MC_OPTION_COLUMNS = [f"P({letter})" for letter in "ABCDE"]
PROBE_SCORE_COLUMNS_PREFIX = "score_"


def _cached_frame(ctx: AnalysisContext, key: str, builder: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    cached = ctx.cache.get(key)
    if cached is None:
        cached = builder()
        ctx.cache[key] = cached
    return cached.copy(deep=True)


def _probe_training_template_type_from_name(probe_name: object) -> str:
    text = str(probe_name or "").strip()
    if not text:
        return ""
    if text == "probe_no_bias":
        return "neutral"
    if text.startswith("probe_bias_"):
        return text[len("probe_bias_") :]
    return ""


def _classify_probe_pairing_semantics(
    probe_name_x: object,
    probe_name_xprime: object,
    training_template_x: object,
    training_template_xprime: object,
    bias_type: object,
) -> str:
    probe_name_x = str(probe_name_x or "").strip()
    probe_name_xprime = str(probe_name_xprime or "").strip()
    training_template_x = str(training_template_x or "").strip()
    training_template_xprime = str(training_template_xprime or "").strip()
    bias_type = str(bias_type or "").strip()

    if probe_name_x and probe_name_x == probe_name_xprime:
        return "same_probe_family"
    if training_template_x == "neutral" and training_template_xprime == bias_type and bias_type:
        return "neutral_on_x__matched_template_on_xprime"
    if training_template_x == "neutral" and training_template_xprime:
        return "neutral_on_x__other_probe_on_xprime"
    return "cross_family_other"


def build_sampled_responses_df(ctx: AnalysisContext) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        df = ctx.sampled_responses.copy()
        numeric_columns = ["P(correct)", "P(selected)", *MC_OPTION_COLUMNS]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "response" in df.columns:
            df["response"] = df["response"].astype(str).str.strip().str.upper()
        return df

    return _cached_frame(ctx, "sampled_responses_df", _builder)


def build_neutral_sampled_responses_df(ctx: AnalysisContext) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        df = build_sampled_responses_df(ctx)
        return df.loc[df["template_type"].astype(str).eq("neutral")].reset_index(drop=True)

    return _cached_frame(ctx, "neutral_sampled_responses_df", _builder)


def build_candidate_probability_long_df(ctx: AnalysisContext) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        df = build_sampled_responses_df(ctx)
        id_columns = [
            column
            for column in [
                "question_id",
                "prompt_id",
                "template_type",
                "split",
                "draw_idx",
                "dataset",
                "response",
                "correct_letter",
                "incorrect_letter",
            ]
            if column in df.columns
        ]
        long_df = df.melt(
            id_vars=id_columns,
            value_vars=[column for column in MC_OPTION_COLUMNS if column in df.columns],
            var_name="option_column",
            value_name="p_option",
        )
        long_df["candidate_option"] = long_df["option_column"].str.extract(r"P\((.)\)")
        long_df["is_chosen_option"] = long_df["candidate_option"].eq(long_df.get("response"))
        if "correct_letter" in long_df.columns:
            long_df["is_correct_option"] = long_df["candidate_option"].eq(long_df["correct_letter"])
        if "incorrect_letter" in long_df.columns:
            long_df["is_incorrect_option"] = long_df["candidate_option"].eq(long_df["incorrect_letter"])
        return long_df.reset_index(drop=True)

    return _cached_frame(ctx, "candidate_probability_long_df", _builder)


def build_paired_external_df(ctx: AnalysisContext) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        df = build_sampled_responses_df(ctx)
        neutral_df = df.loc[df["template_type"].astype(str).eq("neutral")].copy()
        neutral_columns = {
            "response": "response_x",
            "P(selected)": "p_selected_x",
            "P(correct)": "p_correct_x",
            "correctness": "correctness_x",
        }
        for letter in "ABCDE":
            neutral_columns[f"P({letter})"] = f"p_x_{letter}"
        neutral_df = neutral_df.rename(columns=neutral_columns)

        paired_frames = []
        join_keys = [column for column in ["question_id", "split", "draw_idx"] if column in df.columns]
        meta_columns = [column for column in ["dataset", "correct_letter", "incorrect_letter"] if column in neutral_df.columns]
        neutral_keep = join_keys + meta_columns + list(neutral_columns.values())
        neutral_keep = [column for column in neutral_keep if column in neutral_df.columns]
        neutral_df = neutral_df[neutral_keep].drop_duplicates(subset=join_keys)

        for bias_type in sorted(
            {
                str(value)
                for value in df["template_type"].dropna().tolist()
                if str(value) and str(value) != "neutral"
            }
        ):
            bias_df = df.loc[df["template_type"].astype(str).eq(bias_type)].copy()
            bias_columns = {
                "response": "response_xprime",
                "P(selected)": "p_selected_xprime",
                "P(correct)": "p_correct_xprime",
                "correctness": "correctness_xprime",
            }
            for letter in "ABCDE":
                bias_columns[f"P({letter})"] = f"p_xprime_{letter}"
            bias_df = bias_df.rename(columns=bias_columns)
            bias_keep = join_keys + list(bias_columns.values())
            if "incorrect_letter" in bias_df.columns:
                bias_keep.append("incorrect_letter")
            if "correct_letter" in bias_df.columns:
                bias_keep.append("correct_letter")
            bias_keep = [column for column in bias_keep if column in bias_df.columns]
            merged = neutral_df.merge(
                bias_df[bias_keep].drop_duplicates(subset=join_keys),
                on=join_keys,
                how="inner",
                suffixes=("", "_bias"),
            )
            merged["bias_type"] = bias_type
            paired_frames.append(merged)

        if not paired_frames:
            return pd.DataFrame()
        return pd.concat(paired_frames, ignore_index=True)

    return _cached_frame(ctx, "paired_external_df", _builder)


def build_probe_scores_df(ctx: AnalysisContext) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        df = ctx.probe_scores_by_prompt.copy()
        if df.empty:
            return df
        score_columns = [
            column
            for column in df.columns
            if column.startswith(PROBE_SCORE_COLUMNS_PREFIX) or column.startswith("probe_score_")
        ]
        for column in score_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if "probe_training_template_type" not in df.columns:
            df["probe_training_template_type"] = df.get("probe_name", pd.Series(dtype=str)).map(
                _probe_training_template_type_from_name
            )
        if "probe_evaluated_on_template_type" not in df.columns:
            df["probe_evaluated_on_template_type"] = (
                df.get("template_type", pd.Series(dtype=str)).fillna("").astype(str)
            )
        if "probe_is_neutral_family" not in df.columns:
            df["probe_is_neutral_family"] = df["probe_training_template_type"].astype(str).eq("neutral")
        if "probe_matches_evaluated_template" not in df.columns:
            df["probe_matches_evaluated_template"] = (
                df["probe_training_template_type"].astype(str)
                == df["probe_evaluated_on_template_type"].astype(str)
            ) & df["probe_training_template_type"].astype(str).ne("")
        if "selected_choice" in df.columns:
            df["selected_choice"] = df["selected_choice"].astype(str).str.strip().str.upper()
        return df

    return _cached_frame(ctx, "probe_scores_df", _builder)


def build_probe_option_long_df(ctx: AnalysisContext) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        probe_df = build_probe_scores_df(ctx)
        if probe_df.empty:
            return pd.DataFrame()

        score_columns = [f"{PROBE_SCORE_COLUMNS_PREFIX}{letter}" for letter in "ABCDE" if f"{PROBE_SCORE_COLUMNS_PREFIX}{letter}" in probe_df.columns]
        if not score_columns:
            return pd.DataFrame()

        id_columns = [
            column
            for column in [
                "model_name",
                "probe_name",
                "split",
                "question_id",
                "prompt_id",
                "dataset",
                "template_type",
                "probe_training_template_type",
                "probe_evaluated_on_template_type",
                "probe_is_neutral_family",
                "probe_matches_evaluated_template",
                "draw_idx",
                "source_record_id",
                "correct_letter",
                "selected_choice",
                "probe_argmax_choice",
                "probe_score_correct_choice",
                "probe_score_selected_choice",
                "correct_choice_probability",
                "selected_choice_probability",
            ]
            if column in probe_df.columns
        ]
        long_df = probe_df.melt(
            id_vars=id_columns,
            value_vars=score_columns,
            var_name="score_column",
            value_name="probe_score",
        )
        long_df["candidate_option"] = long_df["score_column"].str.replace(PROBE_SCORE_COLUMNS_PREFIX, "", regex=False)
        long_df["is_correct_option"] = long_df["candidate_option"].eq(long_df.get("correct_letter"))
        long_df["is_selected_option"] = long_df["candidate_option"].eq(long_df.get("selected_choice"))
        long_df["is_probe_argmax_option"] = long_df["candidate_option"].eq(long_df.get("probe_argmax_choice"))
        return long_df.reset_index(drop=True)

    return _cached_frame(ctx, "probe_option_long_df", _builder)


def build_chosen_probe_summary_df(ctx: AnalysisContext) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        manifest_path = ctx.run_dir / "probes" / "chosen_probe" / "manifest.json"
        if not manifest_path.exists():
            return pd.DataFrame()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        probe_rows: list[dict[str, object]] = []
        for probe_name, payload in sorted(manifest.get("probes", {}).items()):
            row: dict[str, object] = {
                "probe_name": probe_name,
                "template_type": "neutral" if probe_name == "probe_no_bias" else probe_name.replace("probe_bias_", ""),
                "chosen_layer": payload.get("chosen_layer"),
                "best_dev_auc": payload.get("best_dev_auc"),
                "probe_construction": payload.get("probe_construction"),
                "probe_example_weighting": payload.get("probe_example_weighting"),
            }
            metrics_path = ctx.run_dir / "probes" / "chosen_probe" / probe_name / "metrics.json"
            metadata_path = ctx.run_dir / "probes" / "chosen_probe" / probe_name / "metadata.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for split_name in ("train", "val", "test"):
                    split_payload = metrics.get("splits", {}).get(split_name, {})
                    row[f"{split_name}_auc"] = split_payload.get("auc")
                    row[f"{split_name}_balanced_accuracy"] = split_payload.get("balanced_accuracy")
                    row[f"{split_name}_accuracy"] = split_payload.get("accuracy")
                    row[f"{split_name}_true_label_accuracy"] = split_payload.get("true_label_accuracy")
                    row[f"{split_name}_false_label_accuracy"] = split_payload.get("false_label_accuracy")
                    row[f"{split_name}_n_total"] = split_payload.get("n_total")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                row["label_positive_meaning"] = metadata.get("label_schema", {}).get("positive_meaning")
                row["label_negative_meaning"] = metadata.get("label_schema", {}).get("negative_meaning")
                row["token_pooling_rule"] = metadata.get("feature_source", {}).get("token_position")
                row["selection_metric"] = metadata.get("selection", {}).get("selection_metric")
                row["selection_split"] = metadata.get("selection", {}).get("selection_split")
            probe_rows.append(row)

        return pd.DataFrame(probe_rows)

    return _cached_frame(ctx, "chosen_probe_summary_df", _builder)


def build_all_probe_layer_metrics_df(ctx: AnalysisContext) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        group_manifest_path = ctx.run_dir / "probes" / "all_probes" / "manifest.json"
        if not group_manifest_path.exists():
            return pd.DataFrame()

        group_manifest = json.loads(group_manifest_path.read_text(encoding="utf-8"))
        rows: list[dict[str, object]] = []
        for probe_name in sorted(group_manifest.get("probes", {}).keys()):
            family_manifest_path = ctx.run_dir / "probes" / "all_probes" / probe_name / "manifest.json"
            if not family_manifest_path.exists():
                continue
            payload = json.loads(family_manifest_path.read_text(encoding="utf-8"))
            best_layer = payload.get("best_layer")
            best_dev_auc = payload.get("best_dev_auc")
            template_type = payload.get("template_type") or ("neutral" if probe_name == "probe_no_bias" else probe_name.replace("probe_bias_", ""))
            for layer_str, layer_payload in sorted(payload.get("layers", {}).items(), key=lambda item: int(item[0])):
                metrics_path = ctx.run_dir / "probes" / "all_probes" / probe_name / f"layer_{int(layer_str):03d}" / "metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for split_name in ("train", "val", "test"):
                    split_payload = metrics.get("splits", {}).get(split_name, {})
                    rows.append(
                        {
                            "probe_name": probe_name,
                            "template_type": template_type,
                            "layer": int(layer_str),
                            "split": split_name,
                            "auc": split_payload.get("auc"),
                            "accuracy": split_payload.get("accuracy"),
                            "balanced_accuracy": split_payload.get("balanced_accuracy"),
                            "true_label_accuracy": split_payload.get("true_label_accuracy"),
                            "false_label_accuracy": split_payload.get("false_label_accuracy"),
                            "n_total": split_payload.get("n_total"),
                            "best_layer": best_layer,
                            "best_dev_auc": best_dev_auc,
                            "selection_val_auc": layer_payload.get("selection_val_auc"),
                        }
                    )

        return pd.DataFrame(rows)

    return _cached_frame(ctx, "all_probe_layer_metrics_df", _builder)


def build_paired_probe_df(ctx: AnalysisContext) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        probe_df = build_probe_scores_df(ctx)
        if probe_df.empty or "template_type" not in probe_df.columns:
            return pd.DataFrame()

        neutral_df = probe_df.loc[probe_df["template_type"].astype(str).eq("neutral")].copy()
        neutral_df = neutral_df.rename(
            columns={
                "probe_name": "probe_name_x",
                "probe_training_template_type": "probe_training_template_type_x",
                "probe_evaluated_on_template_type": "probe_evaluated_on_template_type_x",
                "probe_is_neutral_family": "probe_is_neutral_family_x",
                "probe_matches_evaluated_template": "probe_matches_evaluated_template_x",
                "selected_choice": "selected_choice_x",
                "probe_score_correct_choice": "probe_score_correct_choice_x",
                "probe_score_selected_choice": "probe_score_selected_choice_x",
                "probe_argmax_choice": "probe_argmax_choice_x",
            }
        )
        score_columns = [column for column in probe_df.columns if column.startswith(PROBE_SCORE_COLUMNS_PREFIX)]
        for column in score_columns:
            neutral_df = neutral_df.rename(columns={column: f"{column}_x"})

        paired_frames = []
        join_keys = [column for column in ["question_id", "split", "draw_idx"] if column in probe_df.columns]
        neutral_keep = [
            column
            for column in neutral_df.columns
            if column in join_keys
            or column.endswith("_x")
            or column in {"correct_letter", "template_type"}
        ]
        neutral_df = neutral_df[neutral_keep].drop_duplicates(subset=join_keys)

        for bias_type in sorted(
            {
                str(value)
                for value in probe_df["template_type"].dropna().tolist()
                if str(value) and str(value) != "neutral"
            }
        ):
            bias_df = probe_df.loc[probe_df["template_type"].astype(str).eq(bias_type)].copy()
            bias_df = bias_df.rename(
                columns={
                    "probe_name": "probe_name_xprime",
                    "probe_training_template_type": "probe_training_template_type_xprime",
                    "probe_evaluated_on_template_type": "probe_evaluated_on_template_type_xprime",
                    "probe_is_neutral_family": "probe_is_neutral_family_xprime",
                    "probe_matches_evaluated_template": "probe_matches_evaluated_template_xprime",
                    "selected_choice": "selected_choice_xprime",
                    "probe_score_correct_choice": "probe_score_correct_choice_xprime",
                    "probe_score_selected_choice": "probe_score_selected_choice_xprime",
                    "probe_argmax_choice": "probe_argmax_choice_xprime",
                }
            )
            for column in score_columns:
                bias_df = bias_df.rename(columns={column: f"{column}_xprime"})

            bias_keep = [
                column
                for column in bias_df.columns
                if column in join_keys
                or column.endswith("_xprime")
                or column in {"template_type", "correct_letter", "incorrect_letter"}
            ]
            merged = neutral_df.merge(
                bias_df[bias_keep].drop_duplicates(subset=join_keys),
                on=join_keys,
                how="inner",
                suffixes=("", "_bias"),
            )
            merged["bias_type"] = bias_type
            merged["same_probe_name_across_conditions"] = merged["probe_name_x"].eq(merged["probe_name_xprime"])
            merged["same_probe_training_template_across_conditions"] = merged[
                "probe_training_template_type_x"
            ].eq(merged["probe_training_template_type_xprime"])
            merged["probe_pairing_semantics"] = merged.apply(
                lambda row: _classify_probe_pairing_semantics(
                    probe_name_x=row.get("probe_name_x"),
                    probe_name_xprime=row.get("probe_name_xprime"),
                    training_template_x=row.get("probe_training_template_type_x"),
                    training_template_xprime=row.get("probe_training_template_type_xprime"),
                    bias_type=row.get("bias_type"),
                ),
                axis=1,
            )
            paired_frames.append(merged)

        if not paired_frames:
            return pd.DataFrame()
        return pd.concat(paired_frames, ignore_index=True)

    return _cached_frame(ctx, "paired_probe_df", _builder)
