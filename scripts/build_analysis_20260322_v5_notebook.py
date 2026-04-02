from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
V4_NOTEBOOK = REPO_ROOT / "notebooks" / "analysis_20260322_v4.ipynb"
V5_NOTEBOOK = REPO_ROOT / "notebooks" / "analysis_20260322_v5.ipynb"


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

        The currently copied artifacts directly provide:
        - $S_N(x,\cdot)$ and $S_N(x',\cdot)$ from `probe_no_bias_all_templates`
        - $S_{\mathrm{incorrect\_suggestion}}(x',\cdot)$ from the original matched-template probe table

        If an optional all-templates backfill for `probe_bias_incorrect_suggestion` is present, the notebook will also populate $S_{\mathrm{incorrect\_suggestion}}(x,\cdot)$; otherwise the corresponding sanity-check panel is marked unavailable.
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

                merged["sn_margin_x"] = merged["sn_score_correct_x"] - merged["sn_score_bias_target_x"]
                merged["sn_margin_xprime"] = merged["sn_score_correct_xprime"] - merged["sn_score_bias_target_xprime"]
                merged["sn_delta_correct"] = merged["sn_score_correct_xprime"] - merged["sn_score_correct_x"]
                merged["sn_delta_bias_target"] = (
                    merged["sn_score_bias_target_xprime"] - merged["sn_score_bias_target_x"]
                )
                merged["si_minus_sn_x_correct"] = merged["si_score_correct_x"] - merged["sn_score_correct_x"]
                merged["si_minus_sn_xprime_correct"] = (
                    merged["si_score_correct_xprime"] - merged["sn_score_correct_xprime"]
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
                merged["current_response_group"] = np.select(
                    [
                        merged["response_xprime"].eq(merged["correct_letter"]),
                        merged["response_xprime"].eq(merged["incorrect_letter"]),
                    ],
                    [
                        "correct",
                        "b",
                    ],
                    default="wrong_not_b",
                )

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
                            "current_response_group",
                            "margin_x",
                            "margin_xprime",
                            "sn_score_correct_x",
                            "sn_score_correct_xprime",
                            "sn_score_bias_target_x",
                            "sn_score_bias_target_xprime",
                            "si_score_correct_x",
                            "si_score_correct_xprime",
                            "sn_margin_x",
                            "sn_margin_xprime",
                            "sn_delta_correct",
                            "sn_delta_bias_target",
                            "si_minus_sn_x_correct",
                            "si_minus_sn_xprime_correct",
                            "sn_prefers_c_over_b_x",
                            "sn_prefers_c_over_b_xprime",
                            "sn_rank_correct_x",
                            "sn_rank_correct_xprime",
                        ]
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
                "color": "#d4651a",
                "required": True,
            },
            {
                "shift_kind": "sn_bias_shift",
                "label": r"$S_N(x', b) - S_N(x, b)$",
                "column": "sn_delta_bias_target",
                "color": "#73b3ab",
                "required": True,
            },
            {
                "shift_kind": "si_minus_sn_x",
                "label": r"$S_{\mathrm{incorrect\_suggestion}}(x, c) - S_N(x, c)$",
                "column": "si_minus_sn_x_correct",
                "color": "#7a8793",
                "required": False,
            },
            {
                "shift_kind": "si_minus_sn_xprime",
                "label": r"$S_{\mathrm{incorrect\_suggestion}}(x', c) - S_N(x', c)$",
                "column": "si_minus_sn_xprime_correct",
                "color": "#4f6d7a",
                "required": True,
            },
        ]

        shift_frames = []
        for spec in SHIFT_SPECS:
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
                len(SHIFT_SPECS),
                len(run_labels),
                figsize=(5.5 * len(run_labels), 14.0),
                sharex=True,
                sharey=True,
                squeeze=False,
            )

            availability_lookup = availability_df.set_index("run_label") if not availability_df.empty else pd.DataFrame()

            for col_idx, run_label in enumerate(run_labels):
                for row_idx, spec in enumerate(SHIFT_SPECS):
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
                        ax.grid(False)
                        sns.despine(ax=ax)
                        continue

                    sns.histplot(
                        data=subset,
                        x="delta_value",
                        bins=bins,
                        stat="probability",
                        color=spec["color"],
                        edgecolor="white",
                        alpha=0.90,
                        ax=ax,
                    )
                    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
                    ax.axvline(
                        subset["delta_value"].mean(),
                        color="#4f6d7a",
                        linestyle=":",
                        linewidth=2.0,
                    )
                    ax.text(
                        0.02,
                        0.98,
                        f"n = {len(subset)}\nmean = {subset['delta_value'].mean():+.3f}\nmedian = {subset['delta_value'].median():+.3f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=11,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
                    )
                    ax.set_ylim(0, 1)
                    ax.set_xlabel(spec["label"], fontsize=15)
                    ax.tick_params(axis="both", labelsize=12)
                    ax.grid(False)
                    sns.despine(ax=ax)

            legend_handles = [
                Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Zero shift"),
                Line2D([0], [0], color="#4f6d7a", linestyle=":", linewidth=2.0, label="Mean shift"),
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
        from matplotlib.patches import Rectangle

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
        HIDDEN_REGION_COLOR = "#f6d7bf"
        ZERO_LINE_COLOR = "black"

        def scatter_panel(ax, subset: pd.DataFrame, x_col: str, y_col: str) -> None:
            for response_group in RESPONSE_ORDER:
                for neutral_correct, marker in [(True, "o"), (False, "+")]:
                    part = subset.loc[
                        subset["current_response_group"].eq(response_group)
                        & subset["neutral_correct"].eq(neutral_correct)
                    ].copy()
                    if part.empty:
                        continue

                    sns.scatterplot(
                        data=part,
                        x=x_col,
                        y=y_col,
                        color=RESPONSE_COLORS[response_group],
                        marker=marker,
                        s=34 if marker == "o" else 72,
                        linewidth=1.4 if marker == "+" else 0,
                        alpha=0.80,
                        ax=ax,
                        legend=False,
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

        def plot_external_internal_scatter_all_points(
            item_df: pd.DataFrame,
            *,
            x_col: str,
            y_col: str,
            title: str,
            x_label: str,
            y_label: str,
            filename_stub: str,
            annotate_hidden_region: bool = False,
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

            for panel_idx, (ax, run_label) in enumerate(zip(axes, run_labels)):
                subset = item_df.loc[item_df["run_label"].eq(run_label)].copy()

                if x_left < 0 and y_top > 0:
                    ax.add_patch(
                        Rectangle(
                            (x_left, 0),
                            0 - x_left,
                            y_top,
                            facecolor=HIDDEN_REGION_COLOR,
                            edgecolor="none",
                            alpha=0.25,
                            zorder=0,
                        )
                    )

                scatter_panel(ax, subset, x_col, y_col)

                n_b = int(subset["current_response_group"].eq("b").sum())
                r2 = compute_panel_r2(subset, x_col, y_col)
                r2_text = f"{r2:.3f}" if pd.notna(r2) else "NA"
                quad_text = quadrant_fraction_text(subset, x_col, y_col)

                ax.text(
                    0.02,
                    0.98,
                    f"n = {len(subset)}\nb = {n_b}\nR² = {r2_text}\n{quad_text}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.5),
                )

                if annotate_hidden_region and panel_idx == 0 and x_left < 0 and y_top > 0:
                    ax.text(
                        x_left + 0.07 * (0 - x_left),
                        0 + 0.93 * y_top,
                        "x < 0, y > 0:\noutput favors b,\nprobe favors c",
                        ha="left",
                        va="top",
                        fontsize=11,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=3.0),
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
                Line2D([0], [0], marker="o", linestyle="", markersize=8, color=RESPONSE_COLORS[group], label=RESPONSE_LABELS[group])
                for group in RESPONSE_ORDER
            ]
            marker_handles = [
                Line2D([0], [0], marker="o", linestyle="", markersize=8, color="black", label="Neutral correct"),
                Line2D([0], [0], marker="+", linestyle="", markersize=10, color="black", label="Neutral not correct"),
            ]
            hidden_handle = Rectangle(
                (0, 0),
                1,
                1,
                facecolor=HIDDEN_REGION_COLOR,
                edgecolor="none",
                alpha=0.25,
                label="x < 0, y > 0",
            )

            fig.legend(
                handles=color_handles + marker_handles + [hidden_handle],
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
                        y_col="sn_margin_x",
                        title="External vs Internal Knowledge Without Bias",
                        x_label=r"$\mathrm{margin}(x)=\log \pi(x)[c]-\log \pi(x)[b]$",
                        y_label=r"$S_N(x, c) - S_N(x, b)$",
                        filename_stub="external_vs_internal_without_bias__neutral_probe__test_only",
                        annotate_hidden_region=True,
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
                        y_col="sn_margin_xprime",
                        title="External Neutral Margin vs Internal Knowledge Under Bias",
                        x_label=r"$\mathrm{margin}(x)=\log \pi(x)[c]-\log \pi(x)[b]$",
                        y_label=r"$S_N(x', c) - S_N(x', b)$",
                        filename_stub="external_neutral_vs_internal_biased__neutral_probe__test_only",
                        annotate_hidden_region=True,
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
                        y_col="sn_margin_x",
                        title="External Knowledge Under Bias vs Internal Knowledge Without Bias",
                        x_label=r"$\mathrm{margin}(x')=\log \pi(x')[c]-\log \pi(x')[b]$",
                        y_label=r"$S_N(x, c) - S_N(x, b)$",
                        filename_stub="external_biased_vs_internal_neutral__neutral_probe__test_only",
                        annotate_hidden_region=True,
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
                        y_col="sn_margin_xprime",
                        title="External vs Internal Knowledge",
                        x_label=r"$\mathrm{margin}(x')=\log \pi(x')[c]-\log \pi(x')[b]$",
                        y_label=r"$S_N(x', c) - S_N(x', b)$",
                        filename_stub="external_vs_internal_biased__neutral_probe__test_only",
                        annotate_hidden_region=True,
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
                        x_col="sn_margin_x",
                        y_col="sn_margin_xprime",
                        title="Internal Knowledge Before vs After Bias",
                        x_label=r"$S_N(x, c) - S_N(x, b)$",
                        y_label=r"$S_N(x', c) - S_N(x', b)$",
                        filename_stub="internal_before_vs_after_bias__neutral_probe__test_only",
                        annotate_hidden_region=False,
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
                "H = among test items that are correct under x and answer the bias-backed wrong option b under x', the fraction with S_N(x', c) > S_N(x', b).",
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

        movement_all_summary_df = (
            movement_all_df.groupby(["run_key", "run_label", "model", "dataset"], as_index=False)
            .agg(
                n_items=("question_id", "size"),
                n_neutral_correct=("neutral_correct", "sum"),
                frac_to_b=("current_response_group", lambda s: float((s == "b").mean())),
                mean_margin_x=("margin_x", "mean"),
                mean_sn_margin_x=("sn_margin_x", "mean"),
                mean_margin_xprime=("margin_xprime", "mean"),
                mean_sn_margin_xprime=("sn_margin_xprime", "mean"),
            )
            .assign(
                mean_delta_external=lambda df: df["mean_margin_xprime"] - df["mean_margin_x"],
                mean_delta_internal=lambda df: df["mean_sn_margin_xprime"] - df["mean_sn_margin_x"],
            )
            .sort_values(["dataset", "model"])
            .reset_index(drop=True)
        )

        movement_all_df.to_csv(
            PAIRED_MOVEMENT_DIR / "paired_movement__all_test_items__item_level.csv",
            index=False,
        )
        movement_all_summary_df.to_csv(
            PAIRED_MOVEMENT_DIR / "paired_movement__all_test_items__summary.csv",
            index=False,
        )
        display(movement_all_summary_df.round(3))

        def plot_paired_movement_all_test(df: pd.DataFrame) -> plt.Figure:
            run_labels = df["run_label"].drop_duplicates().tolist()
            n_panels = len(run_labels)
            ncols = 2 if n_panels <= 4 else 3
            nrows = int(np.ceil(n_panels / ncols))

            x_all = pd.concat([df["margin_x"], df["margin_xprime"]], ignore_index=True)
            y_all = pd.concat([df["sn_margin_x"], df["sn_margin_xprime"]], ignore_index=True)

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
                        dy = part["sn_margin_xprime"] - part["sn_margin_x"]

                        ax.quiver(
                            part["margin_x"],
                            part["sn_margin_x"],
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
                                part["sn_margin_x"],
                                s=28,
                                color=RESPONSE_COLORS[response_group],
                                alpha=0.85,
                                edgecolor="none",
                                zorder=3,
                            )
                        else:
                            ax.scatter(
                                part["margin_x"],
                                part["sn_margin_x"],
                                s=64,
                                color=RESPONSE_COLORS[response_group],
                                marker="+",
                                linewidths=1.2,
                                alpha=0.85,
                                zorder=3,
                            )

                        ax.scatter(
                            part["margin_xprime"],
                            part["sn_margin_xprime"],
                            s=12,
                            color=RESPONSE_COLORS[response_group],
                            alpha=0.45,
                            edgecolor="none",
                            zorder=4,
                        )

                mean_x = float(subset["margin_x"].mean())
                mean_y = float(subset["sn_margin_x"].mean())
                mean_dx = float(subset["margin_xprime"].mean() - mean_x)
                mean_dy = float(subset["sn_margin_xprime"].mean() - mean_y)

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

            fig.suptitle("Paired Movement on All Test Items", fontsize=24, y=0.995)
            fig.supxlabel(r"$\mathrm{margin}(\cdot)=\log \pi(\cdot)[c]-\log \pi(\cdot)[b]$", fontsize=19)
            fig.supylabel(r"$S_N(\cdot, c)-S_N(\cdot, b)$", fontsize=19)
            fig.text(0.985, 0.987, "Test split only", ha="right", va="top", fontsize=12)
            fig.tight_layout(rect=(0, 0.07, 1, 0.95))
            return fig

        paired_movement_fig = plot_paired_movement_all_test(movement_all_df)
        paired_movement_path = save_figure(
            paired_movement_fig,
            "paired_movement__all_test_items__neutral_probe",
            subdir=PAIRED_MOVEMENT_DIR,
        )
        plt.show()
        plt.close(paired_movement_fig)

        print(paired_movement_path)
        """
    ).strip()


def build_v5_notebook() -> None:
    nb = nbformat.read(V4_NOTEBOOK, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nb.cells = nb.cells[:20]

    nb.cells[0].source = dedent(
        """
        # analysis_20260322_v5: Backfilled neutral-probe findings

        This notebook summarizes the currently completed full-pipeline runs available in the repo on March 22, 2026 for the upcoming research meeting.

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

        **Notebook structure**
        - Section 1 focuses on sycophancy and external behavior.
        - Section 2 focuses on internal knowledge and probe-based analyses.
        """
    ).strip()

    setup_source = nb.cells[1].source
    setup_source = _replace_once(
        setup_source,
        'ARTIFACT_DIR = REPO_ROOT / "notebooks" / "analysis_20260322_v4_artifacts"',
        'ARTIFACT_DIR = REPO_ROOT / "notebooks" / "analysis_20260322_v5_artifacts"',
    )
    nb.cells[1].source = setup_source

    nb.cells[6].source = dedent(
        """
        ## Section 1 - SYCPAHNYC

        Every plot below is **run-specific**. Different model-dataset pairs are not mixed inside the same figure.
        """
    ).strip()

    nb.cells.append(nbformat.v4.new_markdown_cell(_section_two_markdown()))
    nb.cells.append(nbformat.v4.new_code_cell(_internal_loader_source()))
    nb.cells.append(nbformat.v4.new_code_cell(_rank_plot_source()))
    nb.cells.append(nbformat.v4.new_code_cell(_shift_plot_source()))
    nb.cells.append(nbformat.v4.new_code_cell(_external_internal_source()))
    nb.cells.append(nbformat.v4.new_code_cell(_hidden_knowledge_source()))
    nb.cells.append(nbformat.v4.new_code_cell(_paired_movement_source()))

    nbformat.write(nb, V5_NOTEBOOK)


if __name__ == "__main__":
    build_v5_notebook()
    print(f"Wrote {V5_NOTEBOOK}")
