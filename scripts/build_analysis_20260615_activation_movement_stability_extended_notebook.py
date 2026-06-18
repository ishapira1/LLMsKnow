from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "analysis_20260615_activation_movement_stability_extended.ipynb"


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    nb.cells = [
        md(
            """
            # Activation Movement Stability, Extended: Latest Experiment Only

            This notebook intentionally analyzes exactly one experiment:

            `full_arc_challenge_llama31_8b_allfamilies_paraphrase_20260614_seas__fresh__20260615T023915.693470Z_23025458_1037844_6ded20f2`

            The hard boundary is part of the notebook: every loaded table is asserted to live under that run directory. The run contains chosen-probe `movement_rows.jsonl` files for neutral, weak, strong, random, and paraphrase movements. It does **not** currently contain all-letter `probe_candidate_scores.csv`, so K/margin analyses that require forced scores for every answer letter are explicitly marked unavailable rather than mixed in from older runs.
            """
        ),
        md("## 1. Setup"),
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
            from scipy import stats

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

            RUN_NAME = "full_arc_challenge_llama31_8b_allfamilies_paraphrase_20260614_seas__fresh__20260615T023915.693470Z_23025458_1037844_6ded20f2"
            RUN_DIR = (
                REPO_ROOT
                / "results"
                / "sycophancy_bias_probe"
                / "meta_llama_Llama_3_1_8B_Instruct"
                / "arc_challenge"
                / RUN_NAME
            ).resolve()

            if not RUN_DIR.exists():
                raise FileNotFoundError(RUN_DIR)

            ARTIFACT_DIR = REPO_ROOT / "notebooks" / "analysis_20260615_activation_movement_stability_extended_latest_only_artifacts"
            TABLE_DIR = ARTIFACT_DIR / "tables"
            PLOT_DIR = ARTIFACT_DIR / "plots"
            TABLE_DIR.mkdir(parents=True, exist_ok=True)
            PLOT_DIR.mkdir(parents=True, exist_ok=True)

            BASE_COLOR = "#73b3ab"
            CONTRAST_COLOR = "#d4651a"
            SUPPORT_COLOR = "#4f5d75"
            MUTED_COLOR = "#8a8a8a"
            STRONG_COLOR = "#a33d00"
            GRID_COLOR = "#d9d9d9"

            PALETTE = {
                "neutral": MUTED_COLOR,
                "paraphrase": SUPPORT_COLOR,
                "suggest_correct": BASE_COLOR,
                "suggest_correct_strong": "#2f7f77",
                "incorrect_suggestion": CONTRAST_COLOR,
                "incorrect_suggestion_strong": STRONG_COLOR,
                "doubt_correct": CONTRAST_COLOR,
                "doubt_correct_strong": STRONG_COLOR,
                "suggest_random": "#9c6ade",
                "suggest_random_strong": "#5f3dc4",
            }

            pd.set_option("display.max_columns", 240)
            pd.set_option("display.width", 220)

            def assert_latest_path(path: Path) -> Path:
                path = path.resolve()
                if not path.is_relative_to(RUN_DIR):
                    raise AssertionError(f"Refusing to load non-latest artifact: {path}")
                return path

            def load_json(path: Path) -> dict[str, Any]:
                path = assert_latest_path(path)
                return json.loads(path.read_text(encoding="utf-8"))

            def read_jsonl(path: Path) -> list[dict[str, Any]]:
                path = assert_latest_path(path)
                with path.open("r", encoding="utf-8") as handle:
                    return [json.loads(line) for line in handle if line.strip()]

            def style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
                ax.set_title(title, fontsize=20, pad=16)
                ax.set_xlabel(xlabel, fontsize=15)
                ax.set_ylabel(ylabel, fontsize=15)
                ax.tick_params(axis="both", labelsize=12)
                legend = ax.get_legend()
                if legend is not None:
                    legend.set_bbox_to_anchor((0.5, -0.24))
                    legend._loc = 9
                    legend.get_frame().set_alpha(0.95)
                    for text in legend.get_texts():
                        text.set_fontsize(12)

            def save_figure(fig, name: str) -> Path:
                path = PLOT_DIR / f"{name}.png"
                fig.tight_layout()
                fig.savefig(path, dpi=180, bbox_inches="tight")
                return path

            def logit(values: pd.Series | np.ndarray, eps: float = 1e-6) -> pd.Series:
                series = pd.Series(values).astype(float).clip(eps, 1 - eps)
                return np.log(series / (1 - series))

            def binary_loss(score: pd.Series | np.ndarray, y: pd.Series | np.ndarray, eps: float = 1e-6) -> pd.Series:
                s = pd.Series(score).astype(float).clip(eps, 1 - eps)
                label = pd.Series(y).astype(float)
                return -(label * np.log(s) + (1 - label) * np.log(1 - s))

            def cluster_bootstrap_mean(
                df: pd.DataFrame,
                value_col: str,
                cluster_cols: list[str],
                *,
                n_boot: int = 2000,
                seed: int = 5,
            ) -> pd.Series:
                data = df[[*cluster_cols, value_col]].dropna().copy()
                if data.empty:
                    return pd.Series({"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0, "n_clusters": 0})
                cluster_means = data.groupby(cluster_cols, dropna=False)[value_col].mean().to_numpy(dtype=float)
                rng = np.random.default_rng(seed)
                draws = rng.choice(cluster_means, size=(n_boot, len(cluster_means)), replace=True).mean(axis=1)
                return pd.Series(
                    {
                        "mean": float(cluster_means.mean()),
                        "ci_low": float(np.quantile(draws, 0.025)),
                        "ci_high": float(np.quantile(draws, 0.975)),
                        "n": int(len(data)),
                        "n_clusters": int(len(cluster_means)),
                    }
                )

            def spearman_test(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float]:
                sub = df[[x_col, y_col]].dropna()
                if len(sub) < 3 or sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
                    return {"n": int(len(sub)), "rho": np.nan, "p": np.nan}
                result = stats.spearmanr(sub[x_col], sub[y_col])
                return {"n": int(len(sub)), "rho": float(result.statistic), "p": float(result.pvalue)}

            def wilcoxon_pair(
                df: pd.DataFrame,
                value_col: str,
                a: str,
                b: str,
                *,
                family_col: str = "target_template_type",
                index_cols: list[str] | None = None,
            ) -> dict[str, float]:
                if index_cols is None:
                    index_cols = ["probe_name", "question_id", "draw_idx"]
                pivot = df.pivot_table(index=index_cols, columns=family_col, values=value_col, aggfunc="mean")
                if a not in pivot.columns or b not in pivot.columns:
                    return {"n": 0, "mean_a": np.nan, "mean_b": np.nan, "mean_diff_b_minus_a": np.nan, "p": np.nan}
                paired = pivot[[a, b]].dropna()
                if paired.empty:
                    return {"n": 0, "mean_a": np.nan, "mean_b": np.nan, "mean_diff_b_minus_a": np.nan, "p": np.nan}
                diff = paired[b] - paired[a]
                p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(paired[b], paired[a]).pvalue)
                return {
                    "n": int(len(paired)),
                    "mean_a": float(paired[a].mean()),
                    "mean_b": float(paired[b].mean()),
                    "mean_diff_b_minus_a": float(diff.mean()),
                    "p": p,
                }

            def binomial_sign_test(values: pd.Series, *, alternative: str = "two-sided") -> dict[str, float]:
                clean = pd.to_numeric(values, errors="coerce").dropna()
                clean = clean.loc[~np.isclose(clean, 0.0)]
                if clean.empty:
                    return {"n": 0, "n_positive": 0, "positive_rate": np.nan, "p": np.nan}
                n_positive = int((clean > 0).sum())
                result = stats.binomtest(n_positive, n=len(clean), p=0.5, alternative=alternative)
                return {"n": int(len(clean)), "n_positive": n_positive, "positive_rate": float(n_positive / len(clean)), "p": float(result.pvalue)}
            """
        ),
        md("## 2. Load Only The Latest Experiment"),
        code(
            """
            run_config = load_json(RUN_DIR / "run_config.json")
            run_summary = load_json(RUN_DIR / "run_summary.json")

            print("RUN_DIR:", RUN_DIR)
            print("model:", run_config.get("model"))
            print("dataset:", run_config.get("dataset_name"))
            print("bias_types:", run_config.get("bias_types"))
            print("generated_at_utc:", run_summary.get("generated_at_utc"))
            """
        ),
        code(
            """
            sampling_records = read_jsonl(RUN_DIR / "logs" / "sampling_records.jsonl")
            sampling_df = pd.DataFrame(sampling_records)
            sampling_df["record_id_str"] = sampling_df["record_id"].astype(str)
            sampling_df["question_cluster"] = sampling_df["question_id"].astype(str)
            sampling_df["correct_letter"] = sampling_df["correct_letter"].astype(str).str.upper()
            sampling_df["incorrect_letter"] = sampling_df["incorrect_letter"].astype(str).str.upper()
            sampling_df["committed_answer"] = sampling_df["committed_answer"].astype(str).str.upper()
            sampling_df["choice_probability_correct"] = pd.to_numeric(sampling_df["choice_probability_correct"], errors="coerce")
            sampling_df["choice_probability_selected"] = pd.to_numeric(sampling_df["choice_probability_selected"], errors="coerce")

            def choice_prob(row: pd.Series, letter_col: str) -> float:
                probs = row.get("choice_probabilities")
                letter = row.get(letter_col)
                if isinstance(probs, dict) and isinstance(letter, str) and letter:
                    return float(probs.get(letter, np.nan))
                return np.nan

            sampling_df["choice_probability_incorrect"] = sampling_df.apply(lambda row: choice_prob(row, "incorrect_letter"), axis=1)
            sampling_df["output_margin_c_minus_b"] = sampling_df["choice_probability_correct"] - sampling_df["choice_probability_incorrect"]
            sampling_df["output_logit_correct"] = logit(sampling_df["choice_probability_correct"]).to_numpy()
            sampling_df["output_logit_incorrect"] = logit(sampling_df["choice_probability_incorrect"]).to_numpy()
            sampling_df["output_logit_margin_c_minus_b"] = sampling_df["output_logit_correct"] - sampling_df["output_logit_incorrect"]

            sampling_inventory = (
                sampling_df.groupby(["split", "template_type"], dropna=False)
                .agg(
                    prompt_rows=("record_id", "size"),
                    questions=("question_id", "nunique"),
                    accuracy=("correctness", "mean"),
                    mean_p_correct=("choice_probability_correct", "mean"),
                    mean_p_selected=("choice_probability_selected", "mean"),
                    mean_output_margin_c_minus_b=("output_margin_c_minus_b", "mean"),
                )
                .reset_index()
                .sort_values(["split", "template_type"])
            )
            sampling_inventory.to_csv(TABLE_DIR / "latest_sampling_inventory.csv", index=False)
            sampling_inventory
            """
        ),
        code(
            """
            movement_path = RUN_DIR / "query" / "chosen_probe_movement_items.jsonl"
            if not movement_path.exists():
                raise FileNotFoundError("No query/chosen_probe_movement_items.jsonl file found.")
            assert_latest_path(movement_path)
            movement_df = pd.DataFrame(read_jsonl(movement_path))
            movement_df["source_path"] = str(movement_path)
            movement_df["probe_dir"] = movement_df["probe_name"].astype(str)
            movement_df["target_record_id_str"] = movement_df["target_record_id"].astype(str)
            movement_df["source_record_id_str"] = movement_df["source_record_id"].astype(str)
            movement_df["question_cluster"] = movement_df["question_id"].astype(str)
            movement_df["is_paraphrase"] = movement_df["target_change_kind"].eq("paraphrase")
            movement_df["is_prompt_family_move"] = movement_df["target_change_kind"].eq("prompt_family")
            movement_df["is_strong_target"] = movement_df["target_template_type"].astype(str).str.endswith("_strong")
            movement_df["is_bias_probe"] = movement_df["probe_name"].astype(str).str.startswith("probe_bias_")
            movement_df["probe_source_matches_training"] = movement_df["source_template_type"].eq(movement_df["probe_training_template_type"])

            for col in [
                "probe_score_source",
                "probe_score_target",
                "delta_probe_score",
                "probe_logit_source",
                "probe_logit_target",
                "delta_probe_logit",
                "delta_l2_sq",
                "parallel_l2_sq",
                "orthogonal_l2_sq",
                "parallel_fraction_sq",
                "orthogonal_fraction_sq",
                "cosine_similarity",
            ]:
                movement_df[col] = pd.to_numeric(movement_df[col], errors="coerce")

            movement_df["label"] = movement_df["forced_response_is_correct"].astype(float)
            movement_df["probe_loss_source"] = binary_loss(movement_df["probe_score_source"], movement_df["label"]).to_numpy()
            movement_df["probe_loss_target"] = binary_loss(movement_df["probe_score_target"], movement_df["label"]).to_numpy()
            movement_df["delta_probe_loss"] = movement_df["probe_loss_target"] - movement_df["probe_loss_source"]
            movement_df["score_crosses_half"] = movement_df["probe_score_source"].ge(0.5) != movement_df["probe_score_target"].ge(0.5)
            movement_df["signed_parallel_l2"] = np.sign(movement_df["delta_probe_logit"]) * np.sqrt(movement_df["parallel_l2_sq"].clip(lower=0))

            source_cols = [
                "record_id_str",
                "choice_probability_correct",
                "choice_probability_incorrect",
                "output_margin_c_minus_b",
                "output_logit_margin_c_minus_b",
                "correctness",
                "committed_answer",
            ]
            target_cols = source_cols.copy()
            source_meta = sampling_df[source_cols].rename(columns={c: f"source_{c}" for c in source_cols if c != "record_id_str"})
            target_meta = sampling_df[target_cols].rename(columns={c: f"target_{c}" for c in target_cols if c != "record_id_str"})
            movement_df = movement_df.merge(source_meta, left_on="source_record_id_str", right_on="record_id_str", how="left").drop(columns=["record_id_str"])
            movement_df = movement_df.merge(target_meta, left_on="target_record_id_str", right_on="record_id_str", how="left").drop(columns=["record_id_str"])
            movement_df["delta_output_margin_c_minus_b"] = movement_df["target_output_margin_c_minus_b"] - movement_df["source_output_margin_c_minus_b"]
            movement_df["delta_output_logit_margin_c_minus_b"] = movement_df["target_output_logit_margin_c_minus_b"] - movement_df["source_output_logit_margin_c_minus_b"]
            movement_df["answer_changed"] = movement_df["target_committed_answer"].notna() & movement_df["source_committed_answer"].notna() & movement_df["target_committed_answer"].ne(movement_df["source_committed_answer"])

            loaded_roots = set(Path(path).resolve().is_relative_to(RUN_DIR) for path in movement_df["source_path"].dropna().unique())
            assert loaded_roots == {True}, loaded_roots

            movement_df.to_csv(TABLE_DIR / "latest_chosen_probe_movement_rows.csv", index=False)
            print(movement_df.shape)
            movement_df.head()
            """
        ),
        md("## 3. Latest-Run Coverage"),
        code(
            """
            movement_inventory = (
                movement_df.groupby(["probe_name", "probe_training_template_type", "source_template_type", "target_change_kind", "target_template_type"], dropna=False)
                .agg(
                    rows=("question_id", "size"),
                    questions=("question_id", "nunique"),
                    finite_rows=("delta_probe_logit", lambda values: int(np.isfinite(values).sum())),
                    source_path=("source_path", "first"),
                )
                .reset_index()
                .sort_values(["probe_name", "target_change_kind", "target_template_type"])
            )
            movement_inventory.to_csv(TABLE_DIR / "latest_movement_inventory.csv", index=False)
            movement_inventory
            """
        ),
        code(
            """
            candidate_score_paths = sorted(RUN_DIR.glob("**/probe_candidate_scores.csv"))
            candidate_artifacts = pd.DataFrame(
                [{"path": str(assert_latest_path(path)), "relative_path": str(path.relative_to(RUN_DIR))} for path in candidate_score_paths]
            )
            if candidate_artifacts.empty:
                candidate_artifacts = pd.DataFrame(
                    [
                        {
                            "path": "",
                            "relative_path": "",
                            "status": "not_available_in_latest_run",
                            "implication": "All-letter probe margins and K_pairwise cannot be recomputed for this run without a fresh candidate-score extraction.",
                        }
                    ]
                )
            candidate_artifacts.to_csv(TABLE_DIR / "latest_candidate_score_artifacts.csv", index=False)
            candidate_artifacts
            """
        ),
        md(
            """
            ## 4. Paraphrase Gate

            This gate uses the selected-response probe readout. The stricter all-letter margin gate remains unavailable unless candidate-score rows are generated for this exact run.
            """
        ),
        code(
            """
            paraphrase_rows = movement_df.loc[movement_df["is_paraphrase"]].copy()
            gate_parts = []
            for probe_name, group in paraphrase_rows.groupby("probe_name"):
                row = {"probe_name": probe_name, "target_template_type": "neutral_paraphrase"}
                row.update(cluster_bootstrap_mean(group, "delta_probe_score", ["question_id"]).add_prefix("delta_score_").to_dict())
                row.update(cluster_bootstrap_mean(group, "delta_probe_logit", ["question_id"]).add_prefix("delta_logit_").to_dict())
                row.update(cluster_bootstrap_mean(group, "delta_probe_loss", ["question_id"]).add_prefix("delta_loss_").to_dict())
                row["score_cross_half_rate"] = float(group["score_crosses_half"].mean())
                row["mean_abs_delta_probe_logit"] = float(group["delta_probe_logit"].abs().mean())
                gate_parts.append(row)

            paraphrase_gate = pd.DataFrame(gate_parts).sort_values("probe_name")
            paraphrase_gate.to_csv(TABLE_DIR / "latest_paraphrase_gate.csv", index=False)
            paraphrase_gate
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(9, 5))
            plot_df = paraphrase_gate.sort_values("delta_logit_mean")
            sns.barplot(data=plot_df, x="delta_logit_mean", y="probe_name", color=BASE_COLOR, ax=ax)
            ax.axvline(0, color="black", linewidth=1)
            style_axis(ax, "Paraphrase Movement By Chosen Probe", "Mean delta probe logit", "Probe")
            save_figure(fig, "latest_paraphrase_delta_logit_by_probe")
            plt.show()
            """
        ),
        md("## 5. Control Ladder For The Neutral Probe"),
        code(
            """
            CONFLICT_RANK = {
                "neutral": 0,
                "suggest_correct": 1,
                "doubt_correct": 2,
                "incorrect_suggestion": 2,
                "suggest_random": 2,
                "suggest_correct_strong": 3,
                "doubt_correct_strong": 3,
                "incorrect_suggestion_strong": 3,
                "suggest_random_strong": 3,
            }
            FAMILY_ROLE = {
                "neutral": "paraphrase",
                "suggest_correct": "support_correct",
                "suggest_correct_strong": "support_correct_strong",
                "doubt_correct": "conflict",
                "incorrect_suggestion": "conflict",
                "suggest_random": "conflict_random",
                "doubt_correct_strong": "conflict_strong",
                "incorrect_suggestion_strong": "conflict_strong",
                "suggest_random_strong": "conflict_random_strong",
            }

            neutral_probe = movement_df.loc[movement_df["probe_name"].eq("probe_no_bias")].copy()
            neutral_probe["target_conflict_rank"] = neutral_probe["target_template_type"].map(CONFLICT_RANK).fillna(2).astype(int)
            neutral_probe["target_role"] = neutral_probe["target_template_type"].map(FAMILY_ROLE).fillna("other")
            neutral_probe["target_order"] = neutral_probe["target_template_type"].map(
                {
                    "neutral": 0,
                    "suggest_correct": 1,
                    "doubt_correct": 2,
                    "incorrect_suggestion": 3,
                    "suggest_random": 4,
                    "suggest_correct_strong": 5,
                    "doubt_correct_strong": 6,
                    "incorrect_suggestion_strong": 7,
                    "suggest_random_strong": 8,
                }
            )

            ladder_summary = (
                neutral_probe.groupby(["target_template_type", "target_role", "target_conflict_rank", "target_order"], dropna=False)
                .apply(lambda g: pd.concat(
                    [
                        cluster_bootstrap_mean(g, "delta_probe_score", ["question_id"]).add_prefix("delta_score_"),
                        cluster_bootstrap_mean(g, "delta_probe_logit", ["question_id"]).add_prefix("delta_logit_"),
                        cluster_bootstrap_mean(g, "delta_probe_loss", ["question_id"]).add_prefix("delta_loss_"),
                        cluster_bootstrap_mean(g, "delta_l2_sq", ["question_id"]).add_prefix("delta_l2_sq_"),
                    ]
                ))
                .reset_index()
                .sort_values("target_order")
            )
            ladder_summary.to_csv(TABLE_DIR / "latest_neutral_probe_control_ladder_summary.csv", index=False)

            ladder_tests = []
            prompt_only = neutral_probe.loc[neutral_probe["is_prompt_family_move"]].copy()
            ladder_tests.append({"test": "spearman_delta_logit_vs_conflict_rank", **spearman_test(prompt_only, "target_conflict_rank", "delta_probe_logit")})
            ladder_tests.append({"test": "spearman_delta_loss_vs_conflict_rank", **spearman_test(prompt_only, "target_conflict_rank", "delta_probe_loss")})
            ladder_tests.append({"test": "wilcoxon_suggest_correct_vs_incorrect_suggestion_delta_logit", **wilcoxon_pair(prompt_only, "delta_probe_logit", "suggest_correct", "incorrect_suggestion", index_cols=["question_id", "draw_idx"])})
            ladder_tests.append({"test": "wilcoxon_suggest_correct_vs_doubt_correct_delta_logit", **wilcoxon_pair(prompt_only, "delta_probe_logit", "suggest_correct", "doubt_correct", index_cols=["question_id", "draw_idx"])})
            ladder_tests_df = pd.DataFrame(ladder_tests)
            ladder_tests_df.to_csv(TABLE_DIR / "latest_neutral_probe_control_ladder_tests.csv", index=False)

            display(ladder_summary)
            ladder_tests_df
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(11, 5.5))
            sns.pointplot(
                data=ladder_summary,
                x="target_template_type",
                y="delta_logit_mean",
                hue="target_role",
                palette={
                    "paraphrase": SUPPORT_COLOR,
                    "support_correct": BASE_COLOR,
                    "support_correct_strong": "#2f7f77",
                    "conflict": CONTRAST_COLOR,
                    "conflict_random": "#9c6ade",
                    "conflict_strong": STRONG_COLOR,
                    "conflict_random_strong": "#5f3dc4",
                },
                errorbar=None,
                ax=ax,
            )
            for _, row in ladder_summary.iterrows():
                ax.plot([row["target_template_type"], row["target_template_type"]], [row["delta_logit_ci_low"], row["delta_logit_ci_high"]], color="black", linewidth=1)
            ax.axhline(0, color="black", linewidth=1)
            ax.tick_params(axis="x", rotation=35)
            style_axis(ax, "Neutral Probe: Selected-Answer Logit Movement", "Target family", "Mean delta probe logit")
            save_figure(fig, "latest_neutral_probe_ladder_delta_logit")
            plt.show()
            """
        ),
        md("## 6. Probe Loss Movement"),
        code(
            """
            loss_summary = (
                movement_df.groupby(["probe_name", "probe_training_template_type", "target_template_type"], dropna=False)
                .apply(lambda g: pd.concat(
                    [
                        cluster_bootstrap_mean(g, "delta_probe_loss", ["question_id"]).add_prefix("delta_loss_"),
                        cluster_bootstrap_mean(g, "delta_probe_logit", ["question_id"]).add_prefix("delta_logit_"),
                    ]
                ))
                .reset_index()
                .sort_values(["probe_name", "target_template_type"])
            )
            loss_summary.to_csv(TABLE_DIR / "latest_probe_loss_movement_summary.csv", index=False)
            loss_summary
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(11, 5.5))
            plot_df = loss_summary.loc[loss_summary["probe_name"].eq("probe_no_bias")].copy()
            plot_df["target_order"] = plot_df["target_template_type"].map(neutral_probe.set_index("target_template_type")["target_order"].to_dict())
            plot_df = plot_df.sort_values("target_order")
            sns.barplot(
                data=plot_df,
                x="target_template_type",
                y="delta_loss_mean",
                hue=plot_df["target_template_type"].map(FAMILY_ROLE),
                dodge=False,
                palette={
                    "paraphrase": SUPPORT_COLOR,
                    "support_correct": BASE_COLOR,
                    "support_correct_strong": "#2f7f77",
                    "conflict": CONTRAST_COLOR,
                    "conflict_random": "#9c6ade",
                    "conflict_strong": STRONG_COLOR,
                    "conflict_random_strong": "#5f3dc4",
                },
                ax=ax,
            )
            ax.axhline(0, color="black", linewidth=1)
            ax.tick_params(axis="x", rotation=35)
            style_axis(ax, "Neutral Probe: Selected-Answer Probe Loss Movement", "Target family", "Mean delta probe loss")
            save_figure(fig, "latest_neutral_probe_delta_loss")
            plt.show()
            """
        ),
        md("## 7. Vector Movement And Parallel Component"),
        code(
            """
            vector_summary = (
                movement_df.groupby(["probe_name", "target_template_type"], dropna=False)
                .agg(
                    rows=("question_id", "size"),
                    questions=("question_id", "nunique"),
                    mean_cosine=("cosine_similarity", "mean"),
                    mean_delta_l2_sq=("delta_l2_sq", "mean"),
                    mean_parallel_fraction_sq=("parallel_fraction_sq", "mean"),
                    mean_orthogonal_fraction_sq=("orthogonal_fraction_sq", "mean"),
                    mean_signed_parallel_l2=("signed_parallel_l2", "mean"),
                    positive_logit_delta_rate=("delta_probe_logit", lambda values: float((values > 0).mean())),
                )
                .reset_index()
                .sort_values(["probe_name", "target_template_type"])
            )
            vector_summary.to_csv(TABLE_DIR / "latest_vector_movement_summary.csv", index=False)
            vector_summary
            """
        ),
        code(
            """
            sign_tests = []
            for (probe_name, target_template), group in movement_df.groupby(["probe_name", "target_template_type"]):
                sign_tests.append(
                    {
                        "probe_name": probe_name,
                        "target_template_type": target_template,
                        "quantity": "delta_probe_logit",
                        **binomial_sign_test(group["delta_probe_logit"]),
                    }
                )
                sign_tests.append(
                    {
                        "probe_name": probe_name,
                        "target_template_type": target_template,
                        "quantity": "signed_parallel_l2",
                        **binomial_sign_test(group["signed_parallel_l2"]),
                    }
                )
            sign_tests_df = pd.DataFrame(sign_tests).sort_values(["probe_name", "target_template_type", "quantity"])
            sign_tests_df.to_csv(TABLE_DIR / "latest_sign_consistency_tests.csv", index=False)
            sign_tests_df
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            plot_df = vector_summary.loc[vector_summary["probe_name"].eq("probe_no_bias")].copy()
            plot_df["target_order"] = plot_df["target_template_type"].map(neutral_probe.set_index("target_template_type")["target_order"].to_dict())
            plot_df = plot_df.sort_values("target_order")
            sns.barplot(data=plot_df, x="target_template_type", y="mean_delta_l2_sq", color=BASE_COLOR, ax=axes[0])
            sns.barplot(data=plot_df, x="target_template_type", y="mean_parallel_fraction_sq", color=CONTRAST_COLOR, ax=axes[1])
            for ax in axes:
                ax.tick_params(axis="x", rotation=35)
            style_axis(axes[0], "Neutral Probe: L2 Movement", "Target family", "Mean delta L2 squared")
            style_axis(axes[1], "Neutral Probe: Parallel Fraction", "Target family", "Mean parallel fraction squared")
            save_figure(fig, "latest_neutral_probe_vector_movement")
            plt.show()
            """
        ),
        md("## 8. Source-By-Target Heatmaps Across Chosen Probes"),
        code(
            """
            heatmap_summary = (
                movement_df.groupby(["source_template_type", "target_template_type"], dropna=False)
                .agg(
                    mean_delta_probe_logit=("delta_probe_logit", "mean"),
                    mean_delta_probe_score=("delta_probe_score", "mean"),
                    mean_delta_probe_loss=("delta_probe_loss", "mean"),
                    mean_abs_delta_probe_logit=("delta_probe_logit", lambda values: float(values.abs().mean())),
                    score_cross_half_rate=("score_crosses_half", "mean"),
                    mean_delta_l2_sq=("delta_l2_sq", "mean"),
                    rows=("question_id", "size"),
                    questions=("question_id", "nunique"),
                )
                .reset_index()
            )
            heatmap_summary.to_csv(TABLE_DIR / "latest_source_target_heatmap_summary.csv", index=False)
            heatmap_summary
            """
        ),
        code(
            """
            source_order = [
                "neutral",
                "suggest_correct",
                "suggest_correct_strong",
                "doubt_correct",
                "doubt_correct_strong",
                "incorrect_suggestion",
                "incorrect_suggestion_strong",
                "suggest_random",
                "suggest_random_strong",
            ]
            target_order = source_order
            for value_col, title, name in [
                ("mean_delta_probe_logit", "Mean Delta Probe Logit", "latest_heatmap_delta_probe_logit"),
                ("mean_delta_probe_loss", "Mean Delta Probe Loss", "latest_heatmap_delta_probe_loss"),
                ("score_cross_half_rate", "Probe Score Crosses 0.5 Rate", "latest_heatmap_score_cross_half"),
            ]:
                matrix = heatmap_summary.pivot(index="source_template_type", columns="target_template_type", values=value_col)
                matrix = matrix.reindex(index=[x for x in source_order if x in matrix.index], columns=[x for x in target_order if x in matrix.columns])
                fig, ax = plt.subplots(figsize=(10.5, 7))
                cmap = "vlag" if "delta" in value_col else "mako"
                center = 0 if "delta" in value_col else None
                sns.heatmap(matrix, cmap=cmap, center=center, annot=True, fmt=".2f", linewidths=0.5, linecolor="white", ax=ax)
                style_axis(ax, title, "Target family", "Probe/source family")
                ax.tick_params(axis="x", rotation=35)
                ax.tick_params(axis="y", rotation=0)
                save_figure(fig, name)
                plt.show()
            """
        ),
        md("## 9. Weak-To-Strong Dose Response"),
        code(
            """
            STRONG_PAIRS = {
                "suggest_correct": "suggest_correct_strong",
                "doubt_correct": "doubt_correct_strong",
                "incorrect_suggestion": "incorrect_suggestion_strong",
                "suggest_random": "suggest_random_strong",
            }

            dose_rows = []
            for weak, strong in STRONG_PAIRS.items():
                sub = neutral_probe.loc[neutral_probe["target_template_type"].isin([weak, strong])].copy()
                pivot = sub.pivot_table(index=["question_id", "draw_idx"], columns="target_template_type", values=["delta_probe_logit", "delta_probe_loss", "delta_l2_sq", "signed_parallel_l2"], aggfunc="mean")
                if weak not in pivot["delta_probe_logit"].columns or strong not in pivot["delta_probe_logit"].columns:
                    continue
                paired = pd.DataFrame(
                    {
                        "weak": weak,
                        "strong": strong,
                        "delta_logit_strong_minus_weak": pivot["delta_probe_logit"][strong] - pivot["delta_probe_logit"][weak],
                        "delta_loss_strong_minus_weak": pivot["delta_probe_loss"][strong] - pivot["delta_probe_loss"][weak],
                        "delta_l2_sq_strong_minus_weak": pivot["delta_l2_sq"][strong] - pivot["delta_l2_sq"][weak],
                        "signed_parallel_l2_strong_minus_weak": pivot["signed_parallel_l2"][strong] - pivot["signed_parallel_l2"][weak],
                    }
                ).dropna()
                for value_col in [
                    "delta_logit_strong_minus_weak",
                    "delta_loss_strong_minus_weak",
                    "delta_l2_sq_strong_minus_weak",
                    "signed_parallel_l2_strong_minus_weak",
                ]:
                    values = paired[value_col]
                    p = 1.0 if np.allclose(values, 0) else float(stats.wilcoxon(values).pvalue)
                    dose_rows.append(
                        {
                            "weak": weak,
                            "strong": strong,
                            "quantity": value_col,
                            "n": int(len(values)),
                            "mean": float(values.mean()),
                            "median": float(values.median()),
                            "p_wilcoxon_vs_zero": p,
                        }
                    )

            weak_strong_dose = pd.DataFrame(dose_rows)
            weak_strong_dose.to_csv(TABLE_DIR / "latest_weak_to_strong_dose_response.csv", index=False)
            weak_strong_dose
            """
        ),
        md("## 10. Friction Link To Output Commitment"),
        code(
            """
            conflict_targets = [
                "doubt_correct",
                "incorrect_suggestion",
                "suggest_random",
                "doubt_correct_strong",
                "incorrect_suggestion_strong",
                "suggest_random_strong",
            ]
            friction_rows = neutral_probe.loc[
                neutral_probe["target_template_type"].isin(conflict_targets)
                & neutral_probe["source_output_margin_c_minus_b"].notna()
            ].copy()

            friction_tests = []
            for target_template, group in friction_rows.groupby("target_template_type"):
                friction_tests.append(
                    {
                        "target_template_type": target_template,
                        "x": "source_output_margin_c_minus_b",
                        "y": "delta_probe_logit",
                        **spearman_test(group, "source_output_margin_c_minus_b", "delta_probe_logit"),
                    }
                )
                friction_tests.append(
                    {
                        "target_template_type": target_template,
                        "x": "source_output_logit_margin_c_minus_b",
                        "y": "delta_probe_logit",
                        **spearman_test(group, "source_output_logit_margin_c_minus_b", "delta_probe_logit"),
                    }
                )
                friction_tests.append(
                    {
                        "target_template_type": target_template,
                        "x": "source_output_margin_c_minus_b",
                        "y": "delta_output_margin_c_minus_b",
                        **spearman_test(group, "source_output_margin_c_minus_b", "delta_output_margin_c_minus_b"),
                    }
                )

            friction_tests_df = pd.DataFrame(friction_tests)
            friction_tests_df.to_csv(TABLE_DIR / "latest_friction_commitment_link.csv", index=False)
            friction_tests_df
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_df = friction_rows.loc[friction_rows["target_template_type"].isin(["incorrect_suggestion", "incorrect_suggestion_strong"])].copy()
            sns.scatterplot(
                data=plot_df,
                x="source_output_margin_c_minus_b",
                y="delta_probe_logit",
                hue="target_template_type",
                palette={"incorrect_suggestion": CONTRAST_COLOR, "incorrect_suggestion_strong": STRONG_COLOR},
                alpha=0.5,
                s=28,
                ax=ax,
            )
            ax.axhline(0, color="black", linewidth=1)
            ax.axvline(0, color="black", linewidth=1)
            style_axis(ax, "Neutral Probe Movement vs Neutral Output Commitment", "Neutral P(c) - P(b)", "Delta probe logit")
            save_figure(fig, "latest_friction_scatter_incorrect_suggestion")
            plt.show()
            """
        ),
        md("## 11. Worst-Case Items"),
        code(
            """
            worst_items = (
                neutral_probe.assign(abs_delta_probe_logit=neutral_probe["delta_probe_logit"].abs())
                .sort_values("abs_delta_probe_logit", ascending=False)
                .groupby(["question_id", "draw_idx"], as_index=False)
                .first()
            )
            worst_items = worst_items[
                [
                    "question_id",
                    "draw_idx",
                    "source_example_id",
                    "target_template_type",
                    "delta_probe_logit",
                    "delta_probe_score",
                    "delta_probe_loss",
                    "abs_delta_probe_logit",
                    "delta_l2_sq",
                    "parallel_fraction_sq",
                    "source_output_margin_c_minus_b",
                    "delta_output_margin_c_minus_b",
                    "score_crosses_half",
                    "answer_changed",
                ]
            ].sort_values("abs_delta_probe_logit", ascending=False)

            tail_summary = pd.DataFrame(
                [
                    {
                        "tail_threshold_abs_delta_logit": threshold,
                        "tail_fraction": float(worst_items["abs_delta_probe_logit"].ge(threshold).mean()),
                        "n_tail": int(worst_items["abs_delta_probe_logit"].ge(threshold).sum()),
                        "n_questions": int(len(worst_items)),
                    }
                    for threshold in [0.5, 1.0, 2.0, 3.0]
                ]
            )
            worst_by_wrapper = (
                worst_items.groupby("target_template_type")
                .agg(
                    n_worst=("question_id", "size"),
                    mean_abs_delta_probe_logit=("abs_delta_probe_logit", "mean"),
                    score_cross_half_rate=("score_crosses_half", "mean"),
                    answer_changed_rate=("answer_changed", "mean"),
                )
                .reset_index()
                .sort_values("n_worst", ascending=False)
            )

            worst_items.to_csv(TABLE_DIR / "latest_worst_case_question_items.csv", index=False)
            tail_summary.to_csv(TABLE_DIR / "latest_worst_case_tail_summary.csv", index=False)
            worst_by_wrapper.to_csv(TABLE_DIR / "latest_worst_case_by_wrapper.csv", index=False)
            display(tail_summary)
            worst_by_wrapper
            """
        ),
        md("## 12. Verdict Table"),
        code(
            """
            def mean_or_nan(df: pd.DataFrame, col: str) -> float:
                return float(df[col].mean()) if len(df) else np.nan

            verdict_rows = []
            for probe_name, group in movement_df.groupby("probe_name"):
                paraphrase = group.loc[group["is_paraphrase"]]
                supportive = group.loc[group["target_template_type"].isin(["suggest_correct"])]
                supportive_strong = group.loc[group["target_template_type"].isin(["suggest_correct_strong"])]
                conflict = group.loc[group["target_template_type"].isin(["doubt_correct", "incorrect_suggestion", "suggest_random"])]
                conflict_strong = group.loc[group["target_template_type"].isin(["doubt_correct_strong", "incorrect_suggestion_strong", "suggest_random_strong"])]
                probe_tail = (
                    group.loc[group["target_change_kind"].eq("prompt_family")]
                    .groupby(["question_id", "draw_idx"])["delta_probe_logit"]
                    .apply(lambda values: float(values.abs().max()))
                )
                verdict_rows.append(
                    {
                        "run_name": RUN_NAME,
                        "probe_name": probe_name,
                        "probe_training_template_type": group["probe_training_template_type"].iloc[0],
                        "paraphrase_delta_logit": mean_or_nan(paraphrase, "delta_probe_logit"),
                        "paraphrase_abs_delta_logit": float(paraphrase["delta_probe_logit"].abs().mean()) if len(paraphrase) else np.nan,
                        "supportive_delta_logit": mean_or_nan(supportive, "delta_probe_logit"),
                        "supportive_strong_delta_logit": mean_or_nan(supportive_strong, "delta_probe_logit"),
                        "conflict_delta_logit": mean_or_nan(conflict, "delta_probe_logit"),
                        "conflict_strong_delta_logit": mean_or_nan(conflict_strong, "delta_probe_logit"),
                        "conflict_delta_loss": mean_or_nan(conflict, "delta_probe_loss"),
                        "conflict_strong_delta_loss": mean_or_nan(conflict_strong, "delta_probe_loss"),
                        "conflict_mean_l2_sq": mean_or_nan(conflict, "delta_l2_sq"),
                        "strong_conflict_mean_l2_sq": mean_or_nan(conflict_strong, "delta_l2_sq"),
                        "worst_case_tail_abs_logit_ge_2": float(probe_tail.ge(2.0).mean()) if len(probe_tail) else np.nan,
                        "score_cross_half_rate_conflict": float(conflict["score_crosses_half"].mean()) if len(conflict) else np.nan,
                        "candidate_score_K_available": bool(len(candidate_score_paths)),
                    }
                )

            verdict_table = pd.DataFrame(verdict_rows).sort_values("probe_name")
            verdict_table["claim3_selected_readout"] = np.where(
                verdict_table["paraphrase_abs_delta_logit"].le(0.5),
                "paraphrase_reasonably_stable_selected_readout",
                "paraphrase_moves_selected_readout",
            )
            verdict_table["claim2_K_readout"] = np.where(
                verdict_table["candidate_score_K_available"],
                "available",
                "not_available_without_latest_run_candidate_scores",
            )
            verdict_table["interpretation_caveat"] = (
                "Selected-response probe degradation means the chosen readout changes under wrapper pressure; "
                "without all-letter candidate scores it is not a K/margin proof, and it does not prove the information is absent."
            )
            verdict_table.to_csv(TABLE_DIR / "latest_verdict_table.csv", index=False)
            verdict_table
            """
        ),
        md("## 13. Artifact Index"),
        code(
            """
            artifact_index = pd.DataFrame(
                {
                    "artifact": sorted(str(path.relative_to(ARTIFACT_DIR)) for path in ARTIFACT_DIR.glob("**/*") if path.is_file())
                }
            )
            artifact_index.to_csv(TABLE_DIR / "latest_artifact_index.csv", index=False)
            artifact_index
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
