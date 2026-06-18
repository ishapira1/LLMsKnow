from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "analysis_20260615_activation_movement_stability.ipynb"


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    nb.cells = [
        md(
            """
            # Activation Movement Stability Analysis

            This notebook analyzes whether the chosen probes stay stable when we change prompt family, rephrase the question stem, or move from a weak prompt family to its strong version.

            The central distinction:

            - **Activation movement**: how far the hidden state moves in representation space.
            - **Probe-direction movement**: how much of that movement is parallel to the probe weight vector.
            - **Probe score/logit movement**: whether the probe's predicted correctness score actually changes.
            - **Breakage**: boundary flips, low source-target score correlation, or large probe-logit movement, especially under paraphrase.

            The main movement artifacts use the source prompt's forced response. For strict-MC runs, that means the selected answer letter for the source prompt. The optional mini displacement artifacts separately compare the correct choice and the endorsed wrong choice.
            """
        ),
        code(
            """
            from __future__ import annotations

            import json
            import math
            from pathlib import Path
            from typing import Any

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns

            seaborn = sns
            seaborn.set_style("white")

            REPO_ROOT = Path.cwd()
            if not (REPO_ROOT / "src" / "llmssycoph").exists():
                for candidate in [Path.cwd(), *Path.cwd().parents]:
                    if (candidate / "src" / "llmssycoph").exists():
                        REPO_ROOT = candidate
                        break
                else:
                    raise FileNotFoundError("Could not locate repo root containing src/llmssycoph.")

            RESULTS_ROOT = REPO_ROOT / "results" / "sycophancy_bias_probe"
            TARGET_RUN_DIR = (
                RESULTS_ROOT
                / "meta_llama_Llama_3_1_8B_Instruct"
                / "arc_challenge"
                / "full_arc_challenge_llama31_8b_allfamilies_paraphrase_20260614_seas__fresh__20260615T023915.693470Z_23025458_1037844_6ded20f2"
            )
            ARTIFACT_DIR = (
                REPO_ROOT
                / "notebooks"
                / "analysis_20260615_activation_movement_stability_artifacts"
                / TARGET_RUN_DIR.name
            )
            TABLE_DIR = ARTIFACT_DIR / "tables"
            PLOT_DIR = ARTIFACT_DIR / "plots"
            TABLE_DIR.mkdir(parents=True, exist_ok=True)
            PLOT_DIR.mkdir(parents=True, exist_ok=True)

            BASE_COLOR = "#73b3ab"
            CONTRAST_COLOR = "#d4651a"
            SUPPORT_COLOR = "#4f5d75"
            MUTED_COLOR = "#8a8a8a"
            PALETTE = {
                "prompt_family": BASE_COLOR,
                "paraphrase": CONTRAST_COLOR,
                "base_to_strong": CONTRAST_COLOR,
                "strong_to_base": SUPPORT_COLOR,
                "same_strength_or_other": BASE_COLOR,
                "other_family": SUPPORT_COLOR,
            }

            pd.set_option("display.max_columns", 200)
            pd.set_option("display.width", 180)

            def pretty_model_name(model: str) -> str:
                text = str(model or "")
                replacements = {
                    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
                    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
                    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B",
                }
                return replacements.get(text, text.replace("_", "/"))

            def pretty_dataset_name(dataset: str) -> str:
                text = str(dataset or "")
                return {
                    "arc_challenge": "ARC-Challenge",
                    "commonsense_qa": "CommonsenseQA",
                    "aqua_mc": "AQuA",
                    "truthful_qa_mc": "TruthfulQA MC",
                }.get(text, text)

            def style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
                ax.set_title(title, fontsize=20, pad=16)
                ax.set_xlabel(xlabel, fontsize=15)
                ax.set_ylabel(ylabel, fontsize=15)
                ax.tick_params(axis="both", labelsize=12)
                legend = ax.get_legend()
                if legend is not None:
                    legend.set_title(legend.get_title().get_text(), prop={"size": 12})
                    for text in legend.get_texts():
                        text.set_fontsize(12)
                    legend.set_bbox_to_anchor((0.5, -0.22))
                    legend._loc = 9
                    legend.get_frame().set_alpha(0.95)

            def save_figure(fig, name: str) -> Path:
                path = PLOT_DIR / f"{name}.png"
                fig.tight_layout()
                fig.savefig(path, dpi=180, bbox_inches="tight")
                return path

            def safe_corr(left: pd.Series, right: pd.Series) -> float:
                values = pd.DataFrame({
                    "left": pd.to_numeric(left, errors="coerce"),
                    "right": pd.to_numeric(right, errors="coerce"),
                }).dropna()
                if len(values) < 3:
                    return float("nan")
                if values["left"].nunique() < 2 or values["right"].nunique() < 2:
                    return float("nan")
                return float(values["left"].corr(values["right"]))
            """
        ),
        md(
            """
            ## Load Movement Artifacts

            This notebook is intentionally scoped to a single run:

            - model: `meta-llama/Llama-3.1-8B-Instruct`
            - dataset: `arc_challenge`
            - run family: full all-families + paraphrase

            It does **not** scan every run under `results/sycophancy_bias_probe`.

            Expected files per chosen probe:

            - `movement_rows.csv`
            - `movement_summary.csv`
            - `movement_coverage.json`
            - `metrics.json`
            """
        ),
        code(
            """
            def find_run_dir_from_probe_artifact(path: Path) -> Path:
                # movement_rows.csv -> probe dir -> chosen_probe -> probes -> run dir
                return path.parents[3]

            def load_run_config(run_dir: Path) -> dict[str, Any]:
                config_path = run_dir / "run_config.json"
                if not config_path.exists():
                    return {}
                return json.loads(config_path.read_text(encoding="utf-8"))

            def artifact_metadata(path: Path) -> dict[str, Any]:
                run_dir = find_run_dir_from_probe_artifact(path)
                config = load_run_config(run_dir)
                try:
                    rel_parts = run_dir.relative_to(RESULTS_ROOT).parts
                except ValueError:
                    rel_parts = run_dir.parts
                model_slug = rel_parts[0] if len(rel_parts) >= 1 else ""
                dataset_slug = rel_parts[1] if len(rel_parts) >= 2 else ""
                model = str(config.get("model", "") or model_slug)
                dataset = str(config.get("dataset_name", "") or dataset_slug)
                return {
                    "run_dir": str(run_dir.resolve()),
                    "run_name": run_dir.name,
                    "model": model,
                    "model_label": pretty_model_name(model),
                    "dataset_name": dataset,
                    "dataset_label": pretty_dataset_name(dataset),
                    "probe_name": path.parent.parent.parent.name,
                    "movement_rows_path": str(path.resolve()),
                    "movement_summary_path": str((path.parent / "all_summary.csv").resolve()),
                    "movement_coverage_path": str((path.parent / "coverage.json").resolve()),
                    "metrics_path": str((path.parent.parent.parent / "metrics.json").resolve()),
                }

            REQUIRED_RUN_FILES = [
                TARGET_RUN_DIR / "meta" / "status.json",
                TARGET_RUN_DIR / "meta" / "run_config.json",
                TARGET_RUN_DIR / "meta" / "run_summary.json",
                TARGET_RUN_DIR / "probes" / "chosen" / "manifest.json",
            ]
            missing_required = [str(path) for path in REQUIRED_RUN_FILES if not path.exists()]
            if missing_required:
                raise FileNotFoundError(
                    "The targeted all-families ARC run is not fully present locally. Missing:\\n"
                    + "\\n".join(missing_required)
                )

            movement_path = TARGET_RUN_DIR / "query" / "chosen_probe_movement_items.jsonl"
            if not movement_path.exists():
                raise FileNotFoundError(f"Missing query movement items file: {movement_path}")
            movement_df = pd.DataFrame(read_jsonl(movement_path))
            if movement_df.empty:
                raise RuntimeError("The run query movement table exists, but it is empty.")
            movement_df["run_dir"] = str(TARGET_RUN_DIR.resolve())
            movement_df["run_name"] = TARGET_RUN_DIR.name
            movement_df.head()
            """
        ),
        code(
            """
            NUMERIC_COLUMNS = [
                "probe_layer",
                "draw_idx",
                "cosine_similarity",
                "delta_l2_sq",
                "parallel_l2_sq",
                "orthogonal_l2_sq",
                "parallel_fraction_sq",
                "orthogonal_fraction_sq",
                "probe_score_source",
                "probe_score_target",
                "delta_probe_score",
                "probe_logit_source",
                "probe_logit_target",
                "delta_probe_logit",
            ]
            for column in NUMERIC_COLUMNS:
                if column in movement_df.columns:
                    movement_df[column] = pd.to_numeric(movement_df[column], errors="coerce")

            def bool_series(series: pd.Series) -> pd.Series:
                return (
                    series.astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": True, "1": True, "false": False, "0": False})
                )

            if "forced_response_is_correct" in movement_df.columns:
                movement_df["forced_response_is_correct_bool"] = bool_series(movement_df["forced_response_is_correct"])
            else:
                movement_df["forced_response_is_correct_bool"] = pd.NA

            movement_df["abs_delta_probe_score"] = movement_df["delta_probe_score"].abs()
            movement_df["abs_delta_probe_logit"] = movement_df["delta_probe_logit"].abs()
            movement_df["delta_l2"] = np.sqrt(movement_df["delta_l2_sq"].clip(lower=0))
            movement_df["source_probe_positive"] = movement_df["probe_logit_source"] >= 0
            movement_df["target_probe_positive"] = movement_df["probe_logit_target"] >= 0
            movement_df["boundary_flip"] = (
                movement_df["probe_logit_source"].notna()
                & movement_df["probe_logit_target"].notna()
                & movement_df["source_probe_positive"].ne(movement_df["target_probe_positive"])
            )

            def base_family(family: object) -> str:
                text = str(family or "")
                return text[:-7] if text.endswith("_strong") else text

            def strength_level(family: object) -> str:
                return "strong" if str(family or "").endswith("_strong") else "base"

            def strength_transition(row: pd.Series) -> str:
                if row.get("target_change_kind") == "paraphrase":
                    return "paraphrase"
                source = str(row.get("source_template_type") or "")
                target = str(row.get("target_template_type") or "")
                if base_family(source) == base_family(target) and strength_level(source) != strength_level(target):
                    return f"{strength_level(source)}_to_{strength_level(target)}"
                if source == target:
                    return "same_family"
                return "other_family"

            movement_df["source_base_family"] = movement_df["source_template_type"].map(base_family)
            movement_df["target_base_family"] = movement_df["target_template_type"].map(base_family)
            movement_df["source_strength"] = movement_df["source_template_type"].map(strength_level)
            movement_df["target_strength"] = movement_df["target_template_type"].map(strength_level)
            movement_df["strength_transition"] = movement_df.apply(strength_transition, axis=1)
            movement_df["transition_label"] = (
                movement_df["source_template_type"].astype(str)
                + " -> "
                + movement_df["target_template_type"].astype(str)
            )

            # This is not the full all-candidate probe loss. It is a useful per-row proxy:
            # BCE for the same forced answer under source and target prompts.
            eps = 1e-6
            y = movement_df["forced_response_is_correct_bool"].astype("float")
            p_source = movement_df["probe_score_source"].clip(eps, 1 - eps)
            p_target = movement_df["probe_score_target"].clip(eps, 1 - eps)
            movement_df["forced_response_probe_bce_source"] = -(y * np.log(p_source) + (1 - y) * np.log(1 - p_source))
            movement_df["forced_response_probe_bce_target"] = -(y * np.log(p_target) + (1 - y) * np.log(1 - p_target))
            movement_df["delta_forced_response_probe_bce"] = (
                movement_df["forced_response_probe_bce_target"] - movement_df["forced_response_probe_bce_source"]
            )
            movement_df["abs_delta_forced_response_probe_bce"] = movement_df["delta_forced_response_probe_bce"].abs()

            movement_df.to_csv(TABLE_DIR / "movement_rows_enriched.csv", index=False)
            movement_df.shape
            """
        ),
        md(
            """
            ## Artifact Inventory

            This table answers: for each completed run/probe, how many questions and target prompt-family movements are available?
            """
        ),
        code(
            """
            inventory = (
                movement_df.groupby(
                    [
                        "model_label",
                        "dataset_label",
                        "run_name",
                        "probe_name",
                        "probe_training_template_type",
                        "source_template_type",
                    ],
                    dropna=False,
                )
                .agg(
                    rows=("question_id", "size"),
                    questions=("question_id", "nunique"),
                    target_changes=("target_template_type", "nunique"),
                    target_change_kinds=("target_change_kind", lambda values: ", ".join(sorted(set(map(str, values))))),
                    target_families=("target_template_type", lambda values: ", ".join(sorted(set(map(str, values))))),
                    movement_rows_path=("movement_rows_path", "first"),
                )
                .reset_index()
                .sort_values(["dataset_label", "model_label", "probe_name"])
            )
            inventory.to_csv(TABLE_DIR / "artifact_inventory.csv", index=False)
            inventory
            """
        ),
        code(
            """
            coverage_rows = []
            for path_text in sorted(movement_df["movement_coverage_path"].dropna().unique()):
                path = Path(path_text)
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                meta = artifact_metadata(path.parent / "movement_rows.csv")
                exclusion_counts = payload.get("exclusion_counts", {}) or {}
                row = {
                    **{key: meta[key] for key in ["model_label", "dataset_label", "run_name", "probe_name"]},
                    "source_record_count": payload.get("source_record_count"),
                    "expected_comparisons_upper_bound": payload.get("expected_comparisons_upper_bound"),
                    "computed_row_count": payload.get("computed_row_count"),
                    "summary_row_count": payload.get("summary_row_count"),
                    "paraphrase_artifact_path": payload.get("paraphrase_artifact_path", ""),
                }
                for reason, count in exclusion_counts.items():
                    row[f"excluded__{reason}"] = count
                coverage_rows.append(row)

            coverage_df = pd.DataFrame(coverage_rows)
            coverage_df.to_csv(TABLE_DIR / "movement_coverage_summary.csv", index=False)
            coverage_df
            """
        ),
        md(
            """
            ## Master Movement Summary

            This table is the core readout. It summarizes each source-family to target-family move for each probe.

            Especially important columns:

            - `mean_delta_l2_sq`: total hidden-state movement.
            - `mean_orthogonal_fraction_sq`: fraction of movement invisible to the probe direction.
            - `mean_parallel_fraction_sq`: fraction of movement aligned with the probe direction.
            - `mean_abs_delta_probe_logit`: probe-visible movement in linear score space.
            - `boundary_flip_rate`: how often the probe crosses its decision boundary.
            - `score_corr` / `logit_corr`: whether item ranking is preserved across the prompt change.
            - `mean_delta_forced_response_probe_bce`: proxy loss movement for the forced answer only.
            """
        ),
        code(
            """
            GROUP_COLUMNS = [
                "model_label",
                "dataset_label",
                "run_name",
                "probe_name",
                "probe_training_template_type",
                "source_template_type",
                "target_change_kind",
                "target_template_type",
                "strength_transition",
            ]

            def summarize_movement_group(group: pd.DataFrame) -> pd.Series:
                return pd.Series(
                    {
                        "n_rows": int(len(group)),
                        "n_questions": int(group["question_id"].nunique()),
                        "mean_delta_l2_sq": group["delta_l2_sq"].mean(),
                        "median_delta_l2_sq": group["delta_l2_sq"].median(),
                        "mean_delta_l2": group["delta_l2"].mean(),
                        "mean_cosine_similarity": group["cosine_similarity"].mean(),
                        "mean_parallel_fraction_sq": group["parallel_fraction_sq"].mean(),
                        "mean_orthogonal_fraction_sq": group["orthogonal_fraction_sq"].mean(),
                        "mean_delta_probe_score": group["delta_probe_score"].mean(),
                        "mean_abs_delta_probe_score": group["abs_delta_probe_score"].mean(),
                        "p95_abs_delta_probe_score": group["abs_delta_probe_score"].quantile(0.95),
                        "mean_delta_probe_logit": group["delta_probe_logit"].mean(),
                        "mean_abs_delta_probe_logit": group["abs_delta_probe_logit"].mean(),
                        "p95_abs_delta_probe_logit": group["abs_delta_probe_logit"].quantile(0.95),
                        "boundary_flip_rate": group["boundary_flip"].mean(),
                        "score_corr": safe_corr(group["probe_score_source"], group["probe_score_target"]),
                        "logit_corr": safe_corr(group["probe_logit_source"], group["probe_logit_target"]),
                        "mean_probe_score_source": group["probe_score_source"].mean(),
                        "mean_probe_score_target": group["probe_score_target"].mean(),
                        "mean_probe_logit_source": group["probe_logit_source"].mean(),
                        "mean_probe_logit_target": group["probe_logit_target"].mean(),
                        "mean_delta_forced_response_probe_bce": group["delta_forced_response_probe_bce"].mean(),
                        "mean_abs_delta_forced_response_probe_bce": group[
                            "abs_delta_forced_response_probe_bce"
                        ].mean(),
                    }
                )

            movement_summary = (
                movement_df.groupby(GROUP_COLUMNS, dropna=False)
                .apply(summarize_movement_group)
                .reset_index()
                .sort_values(
                    [
                        "dataset_label",
                        "model_label",
                        "probe_training_template_type",
                        "source_template_type",
                        "target_change_kind",
                        "target_template_type",
                    ]
                )
            )
            movement_summary.to_csv(TABLE_DIR / "movement_master_summary.csv", index=False)
            movement_summary
            """
        ),
        md(
            """
            ## Prompt-Family Transition Matrices

            These heatmaps show where the probe is stable or unstable across source and target prompt families.

            For the full all-families run, weak-to-strong cells should appear directly in the matrix.
            """
        ),
        code(
            """
            def pick_default_run(summary: pd.DataFrame) -> str:
                counts = summary.groupby("run_name")["n_rows"].sum().sort_values(ascending=False)
                return str(counts.index[0])

            DEFAULT_RUN_NAME = pick_default_run(movement_summary)
            DEFAULT_DATASET = str(movement_summary.loc[movement_summary["run_name"].eq(DEFAULT_RUN_NAME), "dataset_label"].iloc[0])
            DEFAULT_MODEL = str(movement_summary.loc[movement_summary["run_name"].eq(DEFAULT_RUN_NAME), "model_label"].iloc[0])
            print("Default heatmap run:", DEFAULT_MODEL, DEFAULT_DATASET, DEFAULT_RUN_NAME)

            def plot_transition_heatmap(
                summary: pd.DataFrame,
                *,
                run_name: str,
                metric: str,
                title: str,
                output_name: str,
                cmap: str = "viridis",
                fmt: str = ".2f",
            ) -> Path | None:
                subset = summary.loc[summary["run_name"].eq(run_name)].copy()
                if subset.empty or metric not in subset.columns:
                    return None
                pivot = subset.pivot_table(
                    index="source_template_type",
                    columns="target_template_type",
                    values=metric,
                    aggfunc="mean",
                )
                if pivot.empty:
                    return None
                fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(pivot.columns)), max(5, 0.8 * len(pivot.index))))
                sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, linewidths=0.6, linecolor="white", ax=ax)
                style_axis(ax, title, "Target prompt family", "Source prompt family")
                ax.tick_params(axis="x", rotation=35)
                ax.tick_params(axis="y", rotation=0)
                path = save_figure(fig, output_name)
                plt.show()
                return path

            heatmap_paths = []
            heatmap_specs = [
                ("mean_abs_delta_probe_logit", "Mean absolute probe-logit movement", "transition_heatmap_abs_logit", "mako"),
                ("boundary_flip_rate", "Probe boundary flip rate", "transition_heatmap_flip_rate", "rocket_r"),
                ("mean_delta_l2_sq", "Mean hidden-state movement", "transition_heatmap_delta_l2_sq", "mako"),
                ("mean_parallel_fraction_sq", "Mean probe-parallel movement fraction", "transition_heatmap_parallel_fraction", "rocket_r"),
                ("score_corr", "Source-target probe score correlation", "transition_heatmap_score_corr", "crest"),
            ]
            for metric, title, filename, cmap in heatmap_specs:
                path = plot_transition_heatmap(
                    movement_summary,
                    run_name=DEFAULT_RUN_NAME,
                    metric=metric,
                    title=title,
                    output_name=f"{filename}__{DEFAULT_RUN_NAME}",
                    cmap=cmap,
                    fmt=".2f",
                )
                if path is not None:
                    heatmap_paths.append(path)

            heatmap_paths
            """
        ),
        md(
            """
            ## Prompt-Family Movement vs Probe-Visible Movement

            A stable probe can tolerate large hidden-state movement if the movement is mostly orthogonal to the probe direction and score/logit rankings remain stable.
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(10, 7))
            plot_df = movement_summary.copy()
            sns.scatterplot(
                data=plot_df,
                x="mean_delta_l2_sq",
                y="mean_abs_delta_probe_logit",
                hue="target_change_kind",
                style="strength_transition",
                palette=PALETTE,
                s=95,
                ax=ax,
            )
            style_axis(
                ax,
                "Activation Movement vs Probe-Logit Movement",
                "Mean hidden-state movement (delta L2 squared)",
                "Mean absolute probe-logit movement",
            )
            scatter_path = save_figure(fig, "movement_vs_probe_logit_movement")
            plt.show()
            scatter_path
            """
        ),
        code(
            """
            transition_rollup = (
                movement_summary.groupby(
                    ["model_label", "dataset_label", "target_change_kind", "strength_transition"],
                    dropna=False,
                )
                .agg(
                    groups=("n_rows", "size"),
                    mean_delta_l2_sq=("mean_delta_l2_sq", "mean"),
                    mean_abs_delta_probe_logit=("mean_abs_delta_probe_logit", "mean"),
                    mean_abs_delta_probe_score=("mean_abs_delta_probe_score", "mean"),
                    mean_boundary_flip_rate=("boundary_flip_rate", "mean"),
                    mean_score_corr=("score_corr", "mean"),
                )
                .reset_index()
                .sort_values(["dataset_label", "model_label", "target_change_kind", "strength_transition"])
            )
            transition_rollup.to_csv(TABLE_DIR / "transition_rollup_summary.csv", index=False)
            transition_rollup
            """
        ),
        code(
            """
            if not transition_rollup.empty:
                fig, ax = plt.subplots(figsize=(11, 6))
                sns.barplot(
                    data=transition_rollup,
                    x="strength_transition",
                    y="mean_abs_delta_probe_logit",
                    hue="model_label",
                    palette=[BASE_COLOR, CONTRAST_COLOR, SUPPORT_COLOR, MUTED_COLOR],
                    ax=ax,
                )
                style_axis(
                    ax,
                    "Probe-Logit Movement by Transition Type",
                    "Transition type",
                    "Mean absolute probe-logit movement",
                )
                ax.tick_params(axis="x", rotation=25)
                bar_path = save_figure(fig, "probe_logit_movement_by_transition_type")
                plt.show()
                bar_path
            """
        ),
        md(
            """
            ## Weak-to-Strong Analysis

            This table isolates moves such as `incorrect_suggestion -> incorrect_suggestion_strong`.

            If the table is empty, the loaded artifacts do not include both weak and strong versions yet.
            """
        ),
        code(
            """
            weak_strong_summary = movement_summary.loc[
                movement_summary["strength_transition"].isin(["base_to_strong", "strong_to_base"])
            ].copy()
            weak_strong_summary = weak_strong_summary.sort_values(
                ["mean_abs_delta_probe_logit", "boundary_flip_rate"],
                ascending=False,
            )
            weak_strong_summary.to_csv(TABLE_DIR / "weak_strong_movement_summary.csv", index=False)
            if weak_strong_summary.empty:
                print("No weak-to-strong movement rows found in the currently completed artifacts.")
            weak_strong_summary
            """
        ),
        md(
            """
            ## Paraphrase Analysis

            Same prompt family, same answer, paraphrased question stem.

            This should be the easiest stability test. Large logit shifts or boundary flips here are strong evidence that the probe is brittle to wording.
            """
        ),
        code(
            """
            paraphrase_summary = movement_summary.loc[
                movement_summary["target_change_kind"].eq("paraphrase")
            ].copy()
            paraphrase_summary = paraphrase_summary.sort_values(
                ["boundary_flip_rate", "mean_abs_delta_probe_logit"],
                ascending=False,
            )
            paraphrase_summary.to_csv(TABLE_DIR / "paraphrase_movement_summary.csv", index=False)
            if paraphrase_summary.empty:
                print("No paraphrase movement rows found. Run with --paraphrase_artifact_path to populate this section.")
            paraphrase_summary
            """
        ),
        md(
            """
            ## Breakage Flags

            These thresholds are intentionally simple starting points. Tune them after inspecting distributions.

            A row is suspicious if it has large mean absolute logit movement, high boundary flip rate, low score correlation, or bad paraphrase behavior.
            """
        ),
        code(
            """
            LOGIT_SHIFT_WARN = 0.50
            SCORE_SHIFT_WARN = 0.10
            FLIP_RATE_WARN = 0.05
            SCORE_CORR_WARN = 0.90
            PARAPHRASE_FLIP_WARN = 0.02

            stability_flags = movement_summary.copy()
            stability_flags["flag_large_logit_shift"] = stability_flags["mean_abs_delta_probe_logit"] >= LOGIT_SHIFT_WARN
            stability_flags["flag_large_score_shift"] = stability_flags["mean_abs_delta_probe_score"] >= SCORE_SHIFT_WARN
            stability_flags["flag_boundary_flips"] = stability_flags["boundary_flip_rate"] >= FLIP_RATE_WARN
            stability_flags["flag_low_score_corr"] = stability_flags["score_corr"].lt(SCORE_CORR_WARN)
            stability_flags["flag_paraphrase_break"] = (
                stability_flags["target_change_kind"].eq("paraphrase")
                & stability_flags["boundary_flip_rate"].ge(PARAPHRASE_FLIP_WARN)
            )
            flag_columns = [column for column in stability_flags.columns if column.startswith("flag_")]
            stability_flags["breakage_score"] = stability_flags[flag_columns].sum(axis=1)
            stability_flags = stability_flags.sort_values(
                ["breakage_score", "boundary_flip_rate", "mean_abs_delta_probe_logit"],
                ascending=False,
            )
            stability_flags.to_csv(TABLE_DIR / "stability_breakage_flags.csv", index=False)
            stability_flags.head(40)
            """
        ),
        md(
            """
            ## Worst Question-Level Moves

            These rows find individual questions where the probe shifts most across any target family.

            This is useful for manual inspection: average stability can hide a small group of fragile examples.
            """
        ),
        code(
            """
            QUESTION_GROUP_COLUMNS = [
                "model_label",
                "dataset_label",
                "run_name",
                "probe_name",
                "probe_training_template_type",
                "source_template_type",
                "split",
                "question_id",
                "draw_idx",
            ]

            worst_indices = (
                movement_df.sort_values("abs_delta_probe_logit", ascending=False)
                .groupby(QUESTION_GROUP_COLUMNS, dropna=False)
                .head(1)
                .index
            )
            worst_question_columns = [
                *QUESTION_GROUP_COLUMNS,
                "target_change_kind",
                "target_template_type",
                "transition_label",
                "forced_response",
                "forced_response_is_correct_bool",
                "question",
                "probe_score_source",
                "probe_score_target",
                "delta_probe_score",
                "probe_logit_source",
                "probe_logit_target",
                "delta_probe_logit",
                "abs_delta_probe_logit",
                "boundary_flip",
                "delta_l2_sq",
                "parallel_fraction_sq",
                "orthogonal_fraction_sq",
            ]
            worst_question_columns = [column for column in worst_question_columns if column in movement_df.columns]
            worst_question_moves = movement_df.loc[worst_indices, worst_question_columns].sort_values(
                "abs_delta_probe_logit",
                ascending=False,
            )
            worst_question_moves.to_csv(TABLE_DIR / "worst_question_level_moves.csv", index=False)
            worst_question_moves.head(50)
            """
        ),
        md(
            """
            ## Probe Score and Logit Movement Distributions

            These item-level distributions show whether a condition has a long tail of unstable movements.
            """
        ),
        code(
            """
            distribution_df = movement_df.copy()
            max_rows = 20000
            if len(distribution_df) > max_rows:
                distribution_df = distribution_df.sample(max_rows, random_state=5)

            fig, ax = plt.subplots(figsize=(11, 6))
            sns.boxplot(
                data=distribution_df,
                x="target_change_kind",
                y="abs_delta_probe_logit",
                hue="source_template_type",
                palette="Set2",
                fliersize=1.5,
                ax=ax,
            )
            style_axis(
                ax,
                "Item-Level Absolute Probe-Logit Movement",
                "Target change kind",
                "Absolute probe-logit movement",
            )
            box_path = save_figure(fig, "item_level_abs_probe_logit_distribution")
            plt.show()
            box_path
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(11, 6))
            sns.histplot(
                data=distribution_df,
                x="delta_probe_logit",
                hue="target_change_kind",
                palette=PALETTE,
                bins=60,
                element="step",
                stat="density",
                common_norm=False,
                ax=ax,
            )
            style_axis(
                ax,
                "Signed Probe-Logit Movement",
                "Target logit minus source logit",
                "Density",
            )
            hist_path = save_figure(fig, "signed_probe_logit_movement_distribution")
            plt.show()
            hist_path
            """
        ),
        md(
            """
            ## Cross-Family Probe Metrics

            Movement tells us how the representation shifts for paired items. Cross-family metrics ask a related held-out question: if we evaluate a chosen probe on another family, does its classification quality hold up?

            The saved metrics currently include accuracy, balanced accuracy, AUC, and confusion counts. They do not include full probe log-loss.
            """
        ),
        code(
            """
            def find_run_dir_from_metrics(path: Path) -> Path:
                # metrics.json -> probe dir -> families -> chosen -> probes -> run dir
                return path.parents[4]

            metric_rows = []
            for metrics_path in sorted(RESULTS_ROOT.glob("**/probes/chosen/families/*/metrics.json")):
                run_dir = find_run_dir_from_metrics(metrics_path)
                config = load_run_config(run_dir)
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                probe_name = metrics_path.parent.name
                metadata_path = metrics_path.parent / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                probe_family = str(metadata.get("template_type", "") or "")
                model = str(config.get("model", "") or "")
                dataset = str(config.get("dataset_name", "") or "")

                def append_metric_row(eval_kind: str, evaluated_template_type: str, block: dict[str, Any]) -> None:
                    metric_rows.append(
                        {
                            "model": model,
                            "model_label": pretty_model_name(model),
                            "dataset_name": dataset,
                            "dataset_label": pretty_dataset_name(dataset),
                            "run_name": run_dir.name,
                            "probe_name": probe_name,
                            "probe_training_template_type": probe_family,
                            "eval_kind": eval_kind,
                            "evaluated_template_type": evaluated_template_type,
                            "accuracy": block.get("accuracy"),
                            "balanced_accuracy": block.get("balanced_accuracy"),
                            "auc": block.get("auc"),
                            "positive_rate": block.get("positive_rate"),
                            "predicted_positive_rate": block.get("predicted_positive_rate"),
                            "n_total": block.get("n_total"),
                            "n_label_1": block.get("n_label_1"),
                            "n_label_0": block.get("n_label_0"),
                            "tp": block.get("tp"),
                            "tn": block.get("tn"),
                            "fp": block.get("fp"),
                            "fn": block.get("fn"),
                            "metrics_path": str(metrics_path.resolve()),
                        }
                    )

                own_test = (payload.get("splits", {}) or {}).get("test", {})
                if own_test:
                    append_metric_row("own_family", probe_family, own_test)
                cross_family = payload.get("cross_family", {}) or {}
                for template_type, block in (cross_family.get("by_template_type", {}) or {}).items():
                    append_metric_row("cross_family", str(template_type), block or {})

            probe_metric_df = pd.DataFrame(metric_rows)
            for column in ["accuracy", "balanced_accuracy", "auc", "positive_rate", "predicted_positive_rate", "n_total"]:
                if column in probe_metric_df.columns:
                    probe_metric_df[column] = pd.to_numeric(probe_metric_df[column], errors="coerce")
            probe_metric_df.to_csv(TABLE_DIR / "chosen_probe_cross_family_metrics.csv", index=False)
            probe_metric_df.head(50)
            """
        ),
        code(
            """
            if not probe_metric_df.empty:
                metric_run = probe_metric_df.groupby("run_name")["n_total"].sum().sort_values(ascending=False).index[0]
                metric_subset = probe_metric_df.loc[probe_metric_df["run_name"].eq(metric_run)].copy()
                pivot = metric_subset.pivot_table(
                    index="probe_training_template_type",
                    columns="evaluated_template_type",
                    values="auc",
                    aggfunc="mean",
                )
                fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(pivot.columns)), max(5, 0.8 * len(pivot.index))))
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="crest", linewidths=0.6, linecolor="white", ax=ax)
                style_axis(ax, "Cross-Family Probe AUC", "Evaluated prompt family", "Probe training family")
                ax.tick_params(axis="x", rotation=35)
                ax.tick_params(axis="y", rotation=0)
                auc_heatmap_path = save_figure(fig, f"cross_family_auc_heatmap__{metric_run}")
                plt.show()
                auc_heatmap_path
            """
        ),
        md(
            """
            ## Optional Mini Displacement Artifacts

            The main movement artifacts use the source selected answer. The mini displacement script instead compares two explicit candidate roles:

            - `correct_choice`
            - `endorsed_wrong_choice`

            It is especially useful for the neutral probe under `incorrect_suggestion` versus `model_congruent_suggestion`.
            """
        ),
        code(
            """
            def find_run_dir_upwards(path: Path) -> Path | None:
                for parent in path.parents:
                    if (parent / "run_config.json").exists():
                        return parent
                return None

            displacement_paths = sorted(RESULTS_ROOT.glob("**/analysis/probe_displacement_decomposition*/pairwise_probe_displacement.csv"))
            displacement_frames = []
            for path in displacement_paths:
                run_dir = find_run_dir_upwards(path)
                if run_dir is None:
                    continue
                config = load_run_config(run_dir)
                frame = pd.read_csv(path)
                if frame.empty:
                    continue
                frame["run_dir"] = str(run_dir.resolve())
                frame["run_name"] = run_dir.name
                frame["model"] = str(config.get("model", "") or "")
                frame["model_label"] = frame["model"].map(pretty_model_name)
                frame["dataset_name"] = str(config.get("dataset_name", "") or "")
                frame["dataset_label"] = frame["dataset_name"].map(pretty_dataset_name)
                frame["pairwise_probe_displacement_path"] = str(path.resolve())
                displacement_frames.append(frame)

            displacement_df = pd.concat(displacement_frames, ignore_index=True) if displacement_frames else pd.DataFrame()
            if displacement_df.empty:
                print("No pairwise_probe_displacement.csv files found.")
            else:
                displacement_df.to_csv(TABLE_DIR / "pairwise_probe_displacement_rows_enriched.csv", index=False)
            displacement_df.head()
            """
        ),
        code(
            """
            if not displacement_df.empty:
                for column in [
                    "score_shift_linear",
                    "delta_l2",
                    "parallel_l2",
                    "orthogonal_l2",
                    "orthogonal_fraction",
                    "parallel_fraction",
                ]:
                    displacement_df[column] = pd.to_numeric(displacement_df[column], errors="coerce")

                displacement_summary = (
                    displacement_df.groupby(
                        ["model_label", "dataset_label", "run_name", "condition", "candidate_role"],
                        dropna=False,
                    )
                    .agg(
                        n_rows=("question_id", "size"),
                        n_questions=("question_id", "nunique"),
                        mean_delta_l2=("delta_l2", "mean"),
                        mean_parallel_fraction=("parallel_fraction", "mean"),
                        mean_orthogonal_fraction=("orthogonal_fraction", "mean"),
                        mean_score_shift_linear=("score_shift_linear", "mean"),
                        mean_abs_score_shift_linear=("score_shift_linear", lambda values: pd.to_numeric(values, errors="coerce").abs().mean()),
                    )
                    .reset_index()
                    .sort_values(["run_name", "condition", "candidate_role"])
                )
                displacement_summary.to_csv(TABLE_DIR / "pairwise_probe_displacement_summary.csv", index=False)
                display(displacement_summary)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=displacement_summary,
                    x="condition",
                    y="mean_abs_score_shift_linear",
                    hue="candidate_role",
                    palette=[BASE_COLOR, CONTRAST_COLOR],
                    ax=ax,
                )
                style_axis(
                    ax,
                    "Mini Displacement: Probe-Direction Movement",
                    "Condition",
                    "Mean absolute linear probe shift",
                )
                ax.tick_params(axis="x", rotation=20)
                mini_path = save_figure(fig, "mini_displacement_abs_linear_probe_shift")
                plt.show()
                mini_path
            """
        ),
        md(
            """
            ## What to Look At First

            A practical reading order:

            1. Start with `artifact_inventory.csv` to confirm which runs and families are present.
            2. Use the transition heatmaps for `mean_abs_delta_probe_logit`, `boundary_flip_rate`, and `score_corr`.
            3. Check `paraphrase_movement_summary.csv`; paraphrase instability is the clearest probe brittleness signal.
            4. Check `weak_strong_movement_summary.csv`; weak-to-strong instability means prompt confidence changes the probe-visible representation.
            5. Inspect `stability_breakage_flags.csv` and `worst_question_level_moves.csv` for where the probe breaks.
            6. Compare against `chosen_probe_cross_family_metrics.csv`; if AUC collapses where movement looks large, the probe is not portable across that family.

            The current notebook also computes a forced-answer BCE proxy, but that is not a full all-candidate probe loss. A true whole-probe-loss movement table would need all candidate scores under each source-target prompt pair.
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbformat.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
