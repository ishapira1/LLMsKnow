from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
V3_NOTEBOOK = REPO_ROOT / "notebooks" / "analysis_20260322_v3.ipynb"
V4_NOTEBOOK = REPO_ROOT / "notebooks" / "analysis_20260322_v4.ipynb"


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Expected substring not found: {old!r}")
    return text.replace(old, new, 1)


def build_v4_notebook() -> None:
    nb = nbformat.read(V3_NOTEBOOK, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nb.cells[0].source = dedent(
        """
        # analysis_20260322_v4: Neutral-probe rescored findings

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

        **Probe convention for the custom v4 section**
        - `s(x, a)` always means the saved neutral chosen probe (`probe_no_bias`) applied to candidate `a`
        - `s(x', a)` means that same neutral probe evaluated on the biased prompt `x'`
        - the v4 custom probe plots below do **not** use matched bias-template probe families for `x'`
        """
    ).strip()

    setup_source = nb.cells[1].source
    setup_source = _replace_once(
        setup_source,
        'ARTIFACT_DIR = REPO_ROOT / "notebooks" / "analysis_20260322_artifacts"',
        'ARTIFACT_DIR = REPO_ROOT / "notebooks" / "analysis_20260322_v4_artifacts"',
    )
    setup_source = _replace_once(
        setup_source,
        '"relative_run_dir": "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas",',
        '"relative_run_dir": "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/commonsense_qa/full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas",',
    )
    nb.cells[1].source = setup_source

    nb.cells = nb.cells[:21]

    nb.cells.append(
        nbformat.v4.new_markdown_cell(
            dedent(
                """
                ## Neutral-probe rescored internal analyses

                The cells below rebuild the internal `s(x, a)` / `s(x', a)` quantities from the raw sampling records using the saved neutral chosen probe only.

                Concretely:
                - the classifier is always `probes/chosen_probe/probe_no_bias/model.pkl`
                - the layer is always the neutral chosen layer from `probes/chosen_probe/probe_no_bias/metadata.json`
                - the same neutral probe is evaluated on both `neutral` and `incorrect_suggestion` prompt variants
                - this custom rescored section focuses on probe-backed runs and the `test` split
                """
            ).strip()
        )
    )

    nb.cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                import gc
                import json
                import pickle
                import sys
                import warnings

                from tqdm.auto import tqdm

                if str(REPO_ROOT / "src") not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT / "src"))

                from llmssycoph.grading.probe_data import build_choice_candidate_records
                from llmssycoph.llm.loading import load_model_and_tokenizer
                from llmssycoph.probes.score import score_records_with_probe
                from llmssycoph.saving_manager import build_mc_probe_scores_by_prompt_df

                warnings.filterwarnings(
                    "ignore",
                    message="Trying to unpickle estimator LogisticRegression",
                )

                NEUTRAL_PROBE_RESCORING_DIR = ARTIFACT_DIR / "neutral_probe_rescoring"
                NEUTRAL_PROBE_RESCORING_DIR.mkdir(parents=True, exist_ok=True)

                NEUTRAL_PROBE_TARGET_KEYS = [
                    "llama31_8b__commonsense_qa",
                    "llama31_8b__arc_challenge",
                    "qwen25_7b__commonsense_qa",
                    "qwen25_7b__arc_challenge",
                ]

                _MODEL_RUNTIME_CACHE = {
                    "model_name": None,
                    "model": None,
                    "tokenizer": None,
                }

                MODEL_PATH_OVERRIDES = {
                    # Optional local overrides for machines that do not have the
                    # original Hugging Face cache mount from the training runs.
                    # Example:
                    # "meta-llama/Llama-3.1-8B-Instruct": "/abs/path/to/local/Llama-3.1-8B-Instruct",
                    # "Qwen/Qwen2.5-7B-Instruct": "/abs/path/to/local/Qwen2.5-7B-Instruct",
                }

                def load_jsonl_records(path: Path) -> list[dict]:
                    rows = []
                    with path.open() as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            rows.append(json.loads(line))
                    return rows

                def clean_option(value: object) -> str | float:
                    if pd.isna(value):
                        return np.nan
                    text = str(value).strip().upper()
                    if text in {"", "NAN", "NONE", "NULL"}:
                        return np.nan
                    return text

                def load_run_config(run_dir: Path) -> dict:
                    return json.loads((run_dir / "run_config.json").read_text())

                def unload_cached_model() -> None:
                    model = _MODEL_RUNTIME_CACHE.get("model")
                    tokenizer = _MODEL_RUNTIME_CACHE.get("tokenizer")
                    if model is not None:
                        del model
                    if tokenizer is not None:
                        del tokenizer
                    _MODEL_RUNTIME_CACHE["model_name"] = None
                    _MODEL_RUNTIME_CACHE["model"] = None
                    _MODEL_RUNTIME_CACHE["tokenizer"] = None
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                def resolve_model_identifier_and_cache(run_cfg: dict) -> tuple[str, str | None]:
                    model_name = str(run_cfg.get("model", "") or "").strip()
                    if not model_name:
                        raise ValueError("Run config is missing a model name.")

                    override_path = MODEL_PATH_OVERRIDES.get(model_name)
                    if override_path:
                        override_path = Path(str(override_path)).expanduser().resolve()
                        if not override_path.exists():
                            raise FileNotFoundError(
                                f"MODEL_PATH_OVERRIDES[{model_name!r}] points to a missing path: {override_path}"
                            )
                        return str(override_path), None

                    hf_cache_dir = run_cfg.get("hf_cache_dir")
                    if hf_cache_dir:
                        hf_cache_path = Path(str(hf_cache_dir)).expanduser()
                        if hf_cache_path.exists():
                            return model_name, str(hf_cache_path)

                    raise FileNotFoundError(
                        "Neutral-probe rescoring needs local Hugging Face weights for "
                        f"{model_name!r}, but the original run cache is not available on this machine. "
                        f"Expected hf_cache_dir={hf_cache_dir!r}. "
                        "Set MODEL_PATH_OVERRIDES[model_name] to a local model directory, "
                        "or run this notebook on a machine with the original cache mounted."
                    )

                def get_model_bundle_for_run(run_dir: Path):
                    run_cfg = load_run_config(run_dir)
                    backend = str(run_cfg.get("model_backend", "huggingface") or "huggingface")
                    if backend != "huggingface":
                        raise ValueError(
                            f"Neutral-probe rescoring currently supports only huggingface runs, got {backend!r}."
                        )

                    model_name = str(run_cfg.get("model", "") or "").strip()
                    if not model_name:
                        raise ValueError(f"Run config is missing a model name: {run_dir}")

                    resolved_model_name, resolved_hf_cache_dir = resolve_model_identifier_and_cache(run_cfg)

                    if _MODEL_RUNTIME_CACHE["model_name"] != resolved_model_name:
                        unload_cached_model()
                        resolved_device = str(
                            run_cfg.get("resolved_device")
                            or run_cfg.get("device")
                            or "cpu"
                        ).strip()
                        if resolved_device == "auto":
                            resolved_device = "cpu"
                        model, tokenizer = load_model_and_tokenizer(
                            model_name=resolved_model_name,
                            device=resolved_device,
                            device_map_auto=bool(run_cfg.get("device_map_auto", False)),
                            hf_cache_dir=resolved_hf_cache_dir,
                        )
                        _MODEL_RUNTIME_CACHE["model_name"] = resolved_model_name
                        _MODEL_RUNTIME_CACHE["model"] = model
                        _MODEL_RUNTIME_CACHE["tokenizer"] = tokenizer

                    return _MODEL_RUNTIME_CACHE["model"], _MODEL_RUNTIME_CACHE["tokenizer"]

                def load_neutral_probe_bundle(run_dir: Path) -> tuple[object, int, dict]:
                    probe_dir = run_dir / "probes" / "chosen_probe" / "probe_no_bias"
                    metadata = json.loads((probe_dir / "metadata.json").read_text())
                    with (probe_dir / "model.pkl").open("rb") as handle:
                        clf = pickle.load(handle)
                    return clf, int(metadata["layer"]), metadata

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

                def best_other_score(row: pd.Series, prefix: str, correct_letter: object) -> float:
                    correct_letter = clean_option(correct_letter)
                    if pd.isna(correct_letter):
                        return np.nan
                    values = []
                    for letter in "ABCDE":
                        if letter == correct_letter:
                            continue
                        column = f"{prefix}{letter}"
                        if column in row.index and pd.notna(row[column]):
                            values.append(float(row[column]))
                    return float(max(values)) if values else np.nan

                def score_rank(row: pd.Series, prefix: str, option: object) -> float:
                    option = clean_option(option)
                    if pd.isna(option):
                        return np.nan
                    target_column = f"{prefix}{option}"
                    if target_column not in row.index or pd.isna(row[target_column]):
                        return np.nan
                    target_score = float(row[target_column])
                    candidate_scores = []
                    for letter in "ABCDE":
                        column = f"{prefix}{letter}"
                        if column in row.index and pd.notna(row[column]):
                            candidate_scores.append(float(row[column]))
                    return float(1 + sum(score > target_score for score in candidate_scores))

                def log_margin(p_correct: float, p_bias: float, eps: float = 1e-12) -> float:
                    if pd.isna(p_correct) or pd.isna(p_bias):
                        return np.nan
                    return float(np.log(np.clip(p_correct, eps, 1.0)) - np.log(np.clip(p_bias, eps, 1.0)))

                def prepare_neutral_probe_runs() -> dict[str, dict]:
                    runs = {}
                    for run_key in NEUTRAL_PROBE_TARGET_KEYS:
                        if run_key not in RUNS:
                            continue
                        run = RUNS[run_key]
                        if not run.get("probe_available", False):
                            continue
                        runs[run_key] = run
                    return runs

                def rescore_run_with_neutral_probe(run: dict) -> tuple[pd.DataFrame, dict]:
                    run_dir = Path(run["run_dir"])
                    cache_path = (
                        NEUTRAL_PROBE_RESCORING_DIR
                        / f"{run['run_key']}__probe_no_bias_rescored__test_only__neutral_and_incorrect_suggestion.csv"
                    )

                    inventory_row = {
                        "run_key": run["run_key"],
                        "run_label": run["run_label"],
                        "model": run["model_label"],
                        "dataset": run["dataset_label"],
                        "cache_path": str(cache_path),
                        "cache_exists": cache_path.exists(),
                    }

                    if cache_path.exists():
                        prompt_df = pd.read_csv(cache_path)
                    else:
                        records = load_jsonl_records(run_dir / "logs" / "sampling_records.jsonl")
                        records = [
                            record
                            for record in records
                            if str(record.get("split", "") or "") == "test"
                            and str(record.get("template_type", "") or "") in {"neutral", "incorrect_suggestion"}
                            and bool(record.get("usable_for_metrics", True))
                        ]
                        candidate_records = build_choice_candidate_records(
                            records,
                            probe_name="probe_no_bias_rescored",
                            example_weighting="model_probability",
                        )
                        if not candidate_records:
                            prompt_df = pd.DataFrame()
                        else:
                            model, tokenizer = get_model_bundle_for_run(run_dir)
                            clf, layer, metadata = load_neutral_probe_bundle(run_dir)
                            score_records_with_probe(
                                model=model,
                                tokenizer=tokenizer,
                                records=candidate_records,
                                clf=clf,
                                layer=layer,
                                score_key="probe_score",
                                desc=f"{run['run_label']} neutral probe rescoring",
                            )
                            prompt_df = build_mc_probe_scores_by_prompt_df(pd.DataFrame(candidate_records))
                            prompt_df["rescored_with_probe_name"] = "probe_no_bias"
                            prompt_df["rescored_with_layer"] = int(layer)
                            prompt_df["rescored_probe_template_type"] = str(metadata.get("template_type", "neutral"))
                            prompt_df.to_csv(cache_path, index=False)

                    inventory_row["n_prompt_rows"] = int(len(prompt_df))
                    return prompt_df, inventory_row

                def build_neutral_probe_item_df(runs: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                    inventory_rows = []
                    prompt_frames = []
                    item_frames = []

                    for run in runs.values():
                        prompt_df, inventory_row = rescore_run_with_neutral_probe(run)
                        inventory_rows.append(inventory_row)

                        if prompt_df.empty:
                            continue

                        prompt_df = prompt_df.assign(
                            run_key=run["run_key"],
                            run_label=run["run_label"],
                            model=run["model_label"],
                            dataset=run["dataset_label"],
                        )
                        prompt_frames.append(prompt_df)

                        join_keys = [
                            column
                            for column in ["question_id", "split", "draw_idx"]
                            if column in prompt_df.columns and column in run["paired_df"].columns
                        ]

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

                        neutral_prompt_df = prompt_df.loc[prompt_df["template_type"].eq("neutral")].copy()
                        neutral_prompt_df = neutral_prompt_df.rename(
                            columns={
                                "selected_choice": "probe_selected_choice_x",
                                "probe_argmax_choice": "probe_argmax_choice_x",
                                "probe_argmax_score": "probe_argmax_score_x",
                                "probe_score_correct_choice": "probe_score_correct_choice_x_saved",
                                "probe_score_selected_choice": "probe_score_selected_choice_x_saved",
                            }
                        )
                        for letter in "ABCDE":
                            score_column = f"score_{letter}"
                            if score_column in neutral_prompt_df.columns:
                                neutral_prompt_df = neutral_prompt_df.rename(columns={score_column: f"score_x_{letter}"})

                        biased_prompt_df = prompt_df.loc[prompt_df["template_type"].eq("incorrect_suggestion")].copy()
                        biased_prompt_df = biased_prompt_df.rename(
                            columns={
                                "selected_choice": "probe_selected_choice_xprime",
                                "probe_argmax_choice": "probe_argmax_choice_xprime",
                                "probe_argmax_score": "probe_argmax_score_xprime",
                                "probe_score_correct_choice": "probe_score_correct_choice_xprime_saved",
                                "probe_score_selected_choice": "probe_score_selected_choice_xprime_saved",
                            }
                        )
                        for letter in "ABCDE":
                            score_column = f"score_{letter}"
                            if score_column in biased_prompt_df.columns:
                                biased_prompt_df = biased_prompt_df.rename(columns={score_column: f"score_xprime_{letter}"})

                        neutral_keep = join_keys + [
                            column
                            for column in neutral_prompt_df.columns
                            if column.startswith("score_x_")
                            or column in {
                                "probe_selected_choice_x",
                                "probe_argmax_choice_x",
                                "probe_argmax_score_x",
                                "probe_score_correct_choice_x_saved",
                                "probe_score_selected_choice_x_saved",
                            }
                        ]
                        biased_keep = join_keys + [
                            column
                            for column in biased_prompt_df.columns
                            if column.startswith("score_xprime_")
                            or column in {
                                "probe_selected_choice_xprime",
                                "probe_argmax_choice_xprime",
                                "probe_argmax_score_xprime",
                                "probe_score_correct_choice_xprime_saved",
                                "probe_score_selected_choice_xprime_saved",
                            }
                        ]

                        merged = (
                            paired_df
                            .merge(
                                neutral_prompt_df[neutral_keep].drop_duplicates(subset=join_keys),
                                on=join_keys,
                                how="inner",
                            )
                            .merge(
                                biased_prompt_df[biased_keep].drop_duplicates(subset=join_keys),
                                on=join_keys,
                                how="inner",
                            )
                        )
                        if merged.empty:
                            continue

                        merged["neutral_correct"] = merged["correctness_x"].eq(1)
                        merged["external_margin_x"] = merged.apply(
                            lambda row: log_margin(
                                lookup_prob(row, "p_x_", row["correct_letter"]),
                                lookup_prob(row, "p_x_", row["incorrect_letter"]),
                            ),
                            axis=1,
                        )
                        merged["external_margin_xprime"] = merged.apply(
                            lambda row: log_margin(
                                lookup_prob(row, "p_xprime_", row["correct_letter"]),
                                lookup_prob(row, "p_xprime_", row["incorrect_letter"]),
                            ),
                            axis=1,
                        )

                        merged["probe_score_correct_x"] = merged.apply(
                            lambda row: score_value(row, "score_x_", row["correct_letter"]),
                            axis=1,
                        )
                        merged["probe_score_correct_xprime"] = merged.apply(
                            lambda row: score_value(row, "score_xprime_", row["correct_letter"]),
                            axis=1,
                        )
                        merged["probe_score_bias_target_x"] = merged.apply(
                            lambda row: score_value(row, "score_x_", row["incorrect_letter"]),
                            axis=1,
                        )
                        merged["probe_score_bias_target_xprime"] = merged.apply(
                            lambda row: score_value(row, "score_xprime_", row["incorrect_letter"]),
                            axis=1,
                        )
                        merged["probe_best_other_x"] = merged.apply(
                            lambda row: best_other_score(row, "score_x_", row["correct_letter"]),
                            axis=1,
                        )
                        merged["probe_best_other_xprime"] = merged.apply(
                            lambda row: best_other_score(row, "score_xprime_", row["correct_letter"]),
                            axis=1,
                        )
                        merged["rank_probe_x_correct"] = merged.apply(
                            lambda row: score_rank(row, "score_x_", row["correct_letter"]),
                            axis=1,
                        )
                        merged["rank_probe_xprime_correct"] = merged.apply(
                            lambda row: score_rank(row, "score_xprime_", row["correct_letter"]),
                            axis=1,
                        )
                        merged["m_probe_truth_x"] = merged["probe_score_correct_x"] - merged["probe_best_other_x"]
                        merged["m_probe_truth_xprime"] = (
                            merged["probe_score_correct_xprime"] - merged["probe_best_other_xprime"]
                        )
                        merged["m_probe_bias_x"] = (
                            merged["probe_score_correct_x"] - merged["probe_score_bias_target_x"]
                        )
                        merged["m_probe_bias_xprime"] = (
                            merged["probe_score_correct_xprime"] - merged["probe_score_bias_target_xprime"]
                        )
                        merged["delta_probe_score_correct"] = (
                            merged["probe_score_correct_xprime"] - merged["probe_score_correct_x"]
                        )
                        merged["delta_probe_score_bias_target"] = (
                            merged["probe_score_bias_target_xprime"] - merged["probe_score_bias_target_x"]
                        )
                        merged["probe_prefers_correct_over_target_x"] = (
                            merged["probe_score_correct_x"] > merged["probe_score_bias_target_x"]
                        )
                        merged["probe_prefers_correct_over_target_xprime"] = (
                            merged["probe_score_correct_xprime"] > merged["probe_score_bias_target_xprime"]
                        )
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
                        merged = merged.assign(
                            run_key=run["run_key"],
                            run_label=run["run_label"],
                            model=run["model_label"],
                            dataset=run["dataset_label"],
                        )

                        item_frames.append(
                            merged[
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
                                    "external_margin_x",
                                    "external_margin_xprime",
                                    "probe_score_correct_x",
                                    "probe_score_correct_xprime",
                                    "probe_score_bias_target_x",
                                    "probe_score_bias_target_xprime",
                                    "m_probe_truth_x",
                                    "m_probe_truth_xprime",
                                    "m_probe_bias_x",
                                    "m_probe_bias_xprime",
                                    "delta_probe_score_correct",
                                    "delta_probe_score_bias_target",
                                    "probe_prefers_correct_over_target_x",
                                    "probe_prefers_correct_over_target_xprime",
                                    "rank_probe_x_correct",
                                    "rank_probe_xprime_correct",
                                ]
                            ]
                        )

                    inventory_df = pd.DataFrame(inventory_rows)
                    prompt_df = pd.concat(prompt_frames, ignore_index=True) if prompt_frames else pd.DataFrame()
                    item_df = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()

                    if not item_df.empty:
                        item_df = item_df.replace([np.inf, -np.inf], np.nan)
                        item_df = item_df.dropna(
                            subset=[
                                "external_margin_x",
                                "external_margin_xprime",
                                "m_probe_bias_x",
                                "m_probe_bias_xprime",
                            ]
                        ).reset_index(drop=True)
                    return inventory_df, prompt_df, item_df

                def build_neutral_probe_summary(item_df: pd.DataFrame) -> pd.DataFrame:
                    if item_df.empty:
                        return pd.DataFrame()
                    return (
                        item_df.groupby(["run_key", "run_label", "model", "dataset"], as_index=False)
                        .agg(
                            n_items=("question_id", "size"),
                            n_questions=("question_id", "nunique"),
                            frac_probe_prefers_c_over_b_x=("probe_prefers_correct_over_target_x", "mean"),
                            frac_probe_prefers_c_over_b_xprime=("probe_prefers_correct_over_target_xprime", "mean"),
                            mean_m_probe_bias_x=("m_probe_bias_x", "mean"),
                            mean_m_probe_bias_xprime=("m_probe_bias_xprime", "mean"),
                        )
                        .assign(
                            delta_frac_prefers_c_over_b=lambda df: (
                                df["frac_probe_prefers_c_over_b_xprime"] - df["frac_probe_prefers_c_over_b_x"]
                            )
                        )
                        .sort_values(["dataset", "model"])
                        .reset_index(drop=True)
                    )

                NEUTRAL_PROBE_RUNS = prepare_neutral_probe_runs()
                neutral_probe_inventory_df, neutral_probe_prompt_df, neutral_probe_item_df = build_neutral_probe_item_df(
                    NEUTRAL_PROBE_RUNS
                )
                neutral_probe_summary_df = build_neutral_probe_summary(neutral_probe_item_df)

                neutral_probe_inventory_df.to_csv(
                    NEUTRAL_PROBE_RESCORING_DIR / "neutral_probe_rescoring_inventory__test_only.csv",
                    index=False,
                )
                neutral_probe_prompt_df.to_csv(
                    NEUTRAL_PROBE_RESCORING_DIR / "neutral_probe_scores_by_prompt__test_only__neutral_and_incorrect_suggestion.csv",
                    index=False,
                )
                neutral_probe_item_df.to_csv(
                    NEUTRAL_PROBE_RESCORING_DIR / "neutral_probe_item_level__incorrect_suggestion__test_only.csv",
                    index=False,
                )
                neutral_probe_summary_df.to_csv(
                    NEUTRAL_PROBE_RESCORING_DIR / "neutral_probe_summary__incorrect_suggestion__test_only.csv",
                    index=False,
                )

                display(neutral_probe_inventory_df)
                display(neutral_probe_summary_df.round(3))

                if neutral_probe_item_df.empty:
                    raise ValueError("Neutral-probe rescoring produced no test rows.")

                unload_cached_model()
                """
            ).strip()
        )
    )

    nb.cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch

                sns.set_style("white")

                NEUTRAL_PROBE_SHIFT_DIR = ARTIFACT_DIR / "neutral_probe_shift_comparison"
                NEUTRAL_PROBE_SHIFT_DIR.mkdir(parents=True, exist_ok=True)

                CORRECT_SHIFT_COLOR = "#d4651a"
                BIAS_SHIFT_COLOR = "#73b3ab"
                REFERENCE_LINE_COLOR = "#4f6d7a"

                shift_long_df = pd.concat(
                    [
                        neutral_probe_item_df.assign(
                            shift_kind="correct",
                            shift_label=r"$s_N(x', c) - s_N(x, c)$",
                            delta_value=neutral_probe_item_df["delta_probe_score_correct"],
                        )[
                            ["run_key", "run_label", "model", "dataset", "shift_kind", "shift_label", "delta_value"]
                        ],
                        neutral_probe_item_df.assign(
                            shift_kind="bias_target",
                            shift_label=r"$s_N(x', b) - s_N(x, b)$",
                            delta_value=neutral_probe_item_df["delta_probe_score_bias_target"],
                        )[
                            ["run_key", "run_label", "model", "dataset", "shift_kind", "shift_label", "delta_value"]
                        ],
                    ],
                    ignore_index=True,
                ).dropna(subset=["delta_value"])

                shift_summary_df = (
                    shift_long_df.groupby(
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
                    NEUTRAL_PROBE_SHIFT_DIR / "neutral_probe_shift_comparison__test_only__item_level.csv",
                    index=False,
                )
                shift_summary_df.to_csv(
                    NEUTRAL_PROBE_SHIFT_DIR / "neutral_probe_shift_comparison__test_only__summary.csv",
                    index=False,
                )

                display(shift_summary_df.round(3))

                def plot_neutral_probe_shift_histograms(df: pd.DataFrame) -> plt.Figure:
                    run_labels = df["run_label"].drop_duplicates().tolist()
                    shift_order = ["correct", "bias_target"]
                    shift_colors = {
                        "correct": CORRECT_SHIFT_COLOR,
                        "bias_target": BIAS_SHIFT_COLOR,
                    }
                    shift_titles = {
                        "correct": r"$s_N(x', c) - s_N(x, c)$",
                        "bias_target": r"$s_N(x', b) - s_N(x, b)$",
                    }

                    global_min = float(df["delta_value"].min())
                    global_max = float(df["delta_value"].max())
                    padding = max(0.01, 0.05 * max(global_max - global_min, 0.05))
                    bins = np.linspace(global_min - padding, global_max + padding, 31)

                    fig, axes = plt.subplots(
                        len(shift_order),
                        len(run_labels),
                        figsize=(5.4 * len(run_labels), 8.8),
                        sharex=True,
                        sharey=True,
                        squeeze=False,
                    )

                    for col_idx, run_label in enumerate(run_labels):
                        for row_idx, shift_kind in enumerate(shift_order):
                            ax = axes[row_idx, col_idx]
                            subset = df.loc[
                                df["run_label"].eq(run_label) & df["shift_kind"].eq(shift_kind)
                            ].copy()

                            sns.histplot(
                                data=subset,
                                x="delta_value",
                                bins=bins,
                                stat="probability",
                                color=shift_colors[shift_kind],
                                edgecolor="white",
                                alpha=0.90,
                                ax=ax,
                            )
                            ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
                            ax.axvline(
                                subset["delta_value"].mean(),
                                color=REFERENCE_LINE_COLOR,
                                linestyle=":",
                                linewidth=2.0,
                            )

                            if row_idx == 0:
                                ax.set_title(run_label, fontsize=20, pad=10)

                            ax.text(
                                0.02,
                                0.98,
                                f"n = {len(subset)}\\nmean = {subset['delta_value'].mean():+.3f}\\nmedian = {subset['delta_value'].median():+.3f}",
                                transform=ax.transAxes,
                                ha="left",
                                va="top",
                                fontsize=11,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
                            )

                            ax.set_ylim(0, 1)
                            ax.set_xlabel(shift_titles[shift_kind], fontsize=15)
                            ax.tick_params(axis="both", labelsize=12)
                            ax.grid(False)
                            sns.despine(ax=ax)

                    for col_idx in range(len(run_labels)):
                        axes[0, col_idx].set_ylabel("Probability", fontsize=15)
                        axes[1, col_idx].set_ylabel("Probability", fontsize=15)

                    legend_handles = [
                        Patch(facecolor=CORRECT_SHIFT_COLOR, edgecolor="white", label=r"Histogram of $s_N(x', c) - s_N(x, c)$"),
                        Patch(facecolor=BIAS_SHIFT_COLOR, edgecolor="white", label=r"Histogram of $s_N(x', b) - s_N(x, b)$"),
                        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Zero shift"),
                        Line2D([0], [0], color=REFERENCE_LINE_COLOR, linestyle=":", linewidth=2.0, label="Mean shift"),
                    ]
                    fig.legend(
                        handles=legend_handles,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.02),
                        ncol=4,
                        frameon=True,
                        fontsize=12,
                    )

                    fig.suptitle("Neutral-Probe Score Shifts Under Incorrect Suggestion", fontsize=24, y=0.995)
                    fig.text(0.985, 0.987, "Test split only", ha="right", va="top", fontsize=12)
                    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
                    return fig

                neutral_probe_shift_fig = plot_neutral_probe_shift_histograms(shift_long_df)
                neutral_probe_shift_path = save_figure(
                    neutral_probe_shift_fig,
                    "neutral_probe_shift_comparison__incorrect_suggestion__test_only",
                    subdir=NEUTRAL_PROBE_SHIFT_DIR,
                )
                plt.show()
                plt.close(neutral_probe_shift_fig)

                print(neutral_probe_shift_path)
                """
            ).strip()
        )
    )

    nb.cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                from matplotlib.lines import Line2D
                from matplotlib.patches import Rectangle

                sns.set_style("white")

                NEUTRAL_HIDDEN_KNOWLEDGE_DIR = ARTIFACT_DIR / "neutral_probe_hidden_knowledge"
                NEUTRAL_HIDDEN_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

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
                ECDF_COLOR = "#d4651a"

                hidden_knowledge_summary_df = (
                    neutral_probe_item_df.loc[
                        neutral_probe_item_df["neutral_correct"]
                        & neutral_probe_item_df["current_response_group"].eq("b")
                    ]
                    .groupby(["run_key", "run_label", "model", "dataset"], as_index=False)
                    .agg(
                        n_flip_to_b=("question_id", "size"),
                        H=("m_probe_bias_xprime", lambda s: float((s > 0).mean())),
                    )
                    .sort_values(["dataset", "model"])
                    .reset_index(drop=True)
                )

                hidden_knowledge_summary_df.to_csv(
                    NEUTRAL_HIDDEN_KNOWLEDGE_DIR / "hidden_knowledge_rate__neutral_probe__incorrect_suggestion__test_only.csv",
                    index=False,
                )
                display(hidden_knowledge_summary_df.round(3))

                def scatter_panel(ax, subset, x_col, y_col):
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

                def compute_panel_r2(subset, x_col, y_col):
                    work = subset[[x_col, y_col]].dropna().copy()
                    if len(work) < 2:
                        return np.nan
                    if work[x_col].nunique() < 2 or work[y_col].nunique() < 2:
                        return np.nan
                    corr = work[x_col].corr(work[y_col])
                    if pd.isna(corr):
                        return np.nan
                    return float(corr ** 2)

                def plot_hidden_knowledge_scatter(item_df: pd.DataFrame) -> plt.Figure:
                    run_labels = item_df["run_label"].drop_duplicates().tolist()
                    n_panels = len(run_labels)
                    ncols = 2 if n_panels <= 4 else 3
                    nrows = int(np.ceil(n_panels / ncols))

                    x_col = "external_margin_xprime"
                    y_col = "m_probe_bias_xprime"

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
                        ax.text(
                            0.02,
                            0.98,
                            f"n = {len(subset)}\\nb = {n_b}\\nR² = {r2_text}",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=11,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
                        )

                        if panel_idx == 0 and x_left < 0 and y_top > 0:
                            ax.text(
                                x_left + 0.07 * (0 - x_left),
                                0 + 0.93 * y_top,
                                "x < 0, y > 0:\\noutput favors b,\\nprobe favors c",
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

                    fig.suptitle("External vs Internal Knowledge", fontsize=24, y=0.995)
                    fig.supxlabel(r"$\\log \\pi(x')[c] - \\log \\pi(x')[b]$", fontsize=21)
                    fig.supylabel(r"$s_N(x', c) - s_N(x', b)$", fontsize=21)
                    fig.text(0.985, 0.987, "Test split only", ha="right", va="top", fontsize=12)
                    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
                    return fig

                def plot_hidden_knowledge_ecdf(item_df: pd.DataFrame, summary_df: pd.DataFrame) -> plt.Figure:
                    flip_df = item_df.loc[
                        item_df["neutral_correct"] & item_df["current_response_group"].eq("b")
                    ].copy()

                    run_labels = flip_df["run_label"].drop_duplicates().tolist()
                    n_panels = len(run_labels)
                    ncols = 2 if n_panels <= 4 else 3
                    nrows = int(np.ceil(n_panels / ncols))

                    x_min = float(flip_df["m_probe_bias_xprime"].min())
                    x_max = float(flip_df["m_probe_bias_xprime"].max())
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
                            x="m_probe_bias_xprime",
                            color=ECDF_COLOR,
                            linewidth=2.6,
                            ax=ax,
                        )
                        ax.axvline(0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.2)
                        ax.text(
                            0.02,
                            0.98,
                            f"n = {int(summary_row['n_flip_to_b'])}\\nH = {summary_row['H']:.3f}",
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
                        Line2D([0], [0], color=ECDF_COLOR, linewidth=2.6, label=r"ECDF of $s_N(x', c) - s_N(x', b)$"),
                        Line2D([0], [0], color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.2, label=r"$s_N(x', c) - s_N(x', b) = 0$"),
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
                    fig.supxlabel(r"$s_N(x', c) - s_N(x', b)$", fontsize=21)
                    fig.supylabel("ECDF", fontsize=21)
                    fig.text(0.985, 0.987, "Test split only", ha="right", va="top", fontsize=12)
                    fig.text(
                        0.5,
                        0.055,
                        "H = among test items that are correct under x and answer the bias-backed wrong option b under x', the fraction with s_N(x', c) > s_N(x', b).",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
                    return fig

                hidden_scatter_fig = plot_hidden_knowledge_scatter(neutral_probe_item_df)
                hidden_scatter_path = save_figure(
                    hidden_scatter_fig,
                    "neutral_probe_hidden_knowledge_scatter__incorrect_suggestion__test_only",
                    subdir=NEUTRAL_HIDDEN_KNOWLEDGE_DIR,
                )
                plt.show()
                plt.close(hidden_scatter_fig)

                hidden_ecdf_fig = plot_hidden_knowledge_ecdf(neutral_probe_item_df, hidden_knowledge_summary_df)
                hidden_ecdf_path = save_figure(
                    hidden_ecdf_fig,
                    "neutral_probe_hidden_knowledge_ecdf__incorrect_suggestion__flip_to_b__test_only",
                    subdir=NEUTRAL_HIDDEN_KNOWLEDGE_DIR,
                )
                plt.show()
                plt.close(hidden_ecdf_fig)

                print(hidden_scatter_path)
                print(hidden_ecdf_path)
                """
            ).strip()
        )
    )

    nb.cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                from matplotlib.lines import Line2D

                sns.set_style("white")

                NEUTRAL_PAIRED_MOVEMENT_DIR = ARTIFACT_DIR / "neutral_probe_paired_movement"
                NEUTRAL_PAIRED_MOVEMENT_DIR.mkdir(parents=True, exist_ok=True)

                movement_all_df = neutral_probe_item_df.loc[
                    neutral_probe_item_df["split"].eq("test")
                ].copy()

                movement_all_summary_df = (
                    movement_all_df.groupby(["run_key", "run_label", "model", "dataset"], as_index=False)
                    .agg(
                        n_items=("question_id", "size"),
                        n_neutral_correct=("neutral_correct", "sum"),
                        frac_to_b=("current_response_group", lambda s: float((s == "b").mean())),
                        mean_m_ext_x=("external_margin_x", "mean"),
                        mean_m_int_x=("m_probe_bias_x", "mean"),
                        mean_m_ext_xprime=("external_margin_xprime", "mean"),
                        mean_m_int_xprime=("m_probe_bias_xprime", "mean"),
                    )
                    .assign(
                        mean_delta_ext=lambda df: df["mean_m_ext_xprime"] - df["mean_m_ext_x"],
                        mean_delta_int=lambda df: df["mean_m_int_xprime"] - df["mean_m_int_x"],
                    )
                    .sort_values(["dataset", "model"])
                    .reset_index(drop=True)
                )

                movement_all_df.to_csv(
                    NEUTRAL_PAIRED_MOVEMENT_DIR / "neutral_probe_paired_movement__all_test_items__item_level.csv",
                    index=False,
                )
                movement_all_summary_df.to_csv(
                    NEUTRAL_PAIRED_MOVEMENT_DIR / "neutral_probe_paired_movement__all_test_items__summary.csv",
                    index=False,
                )
                display(movement_all_summary_df.round(3))

                def plot_neutral_probe_paired_movement_all_test(df: pd.DataFrame) -> plt.Figure:
                    run_labels = df["run_label"].drop_duplicates().tolist()
                    n_panels = len(run_labels)
                    ncols = 2 if n_panels <= 4 else 3
                    nrows = int(np.ceil(n_panels / ncols))

                    x_all = pd.concat([df["external_margin_x"], df["external_margin_xprime"]], ignore_index=True)
                    y_all = pd.concat([df["m_probe_bias_x"], df["m_probe_bias_xprime"]], ignore_index=True)

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

                                dx = part["external_margin_xprime"] - part["external_margin_x"]
                                dy = part["m_probe_bias_xprime"] - part["m_probe_bias_x"]

                                ax.quiver(
                                    part["external_margin_x"],
                                    part["m_probe_bias_x"],
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
                                        part["external_margin_x"],
                                        part["m_probe_bias_x"],
                                        s=28,
                                        color=RESPONSE_COLORS[response_group],
                                        alpha=0.85,
                                        edgecolor="none",
                                        zorder=3,
                                    )
                                else:
                                    ax.scatter(
                                        part["external_margin_x"],
                                        part["m_probe_bias_x"],
                                        s=64,
                                        color=RESPONSE_COLORS[response_group],
                                        marker="+",
                                        linewidths=1.2,
                                        alpha=0.85,
                                        zorder=3,
                                    )

                                ax.scatter(
                                    part["external_margin_xprime"],
                                    part["m_probe_bias_xprime"],
                                    s=12,
                                    color=RESPONSE_COLORS[response_group],
                                    alpha=0.45,
                                    edgecolor="none",
                                    zorder=4,
                                )

                        mean_x = float(subset["external_margin_x"].mean())
                        mean_y = float(subset["m_probe_bias_x"].mean())
                        mean_dx = float(subset["external_margin_xprime"].mean() - mean_x)
                        mean_dy = float(subset["m_probe_bias_xprime"].mean() - mean_y)

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
                            f"n = {len(subset)}\\nneutral correct = {int(subset['neutral_correct'].sum())}\\nb = {(subset['current_response_group'] == 'b').mean():.2f}",
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
                    fig.supxlabel(r"$m^{\\mathrm{ext}}(\\cdot)=\\log \\pi(\\cdot)[c]-\\log \\pi(\\cdot)[b]$", fontsize=19)
                    fig.supylabel(r"$m^{\\mathrm{int}}_N(\\cdot)=s_N(\\cdot,c)-s_N(\\cdot,b)$", fontsize=19)
                    fig.text(0.985, 0.987, "Test split only", ha="right", va="top", fontsize=12)
                    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
                    return fig

                paired_movement_fig = plot_neutral_probe_paired_movement_all_test(movement_all_df)
                paired_movement_path = save_figure(
                    paired_movement_fig,
                    "neutral_probe_paired_movement__all_test_items",
                    subdir=NEUTRAL_PAIRED_MOVEMENT_DIR,
                )
                plt.show()
                plt.close(paired_movement_fig)

                print(paired_movement_path)
                """
            ).strip()
        )
    )

    nb.cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                from matplotlib.lines import Line2D
                from matplotlib.patches import Rectangle

                sns.set_style("white")

                NEUTRAL_EXTERNAL_INTERNAL_DIR = ARTIFACT_DIR / "neutral_probe_external_internal"
                NEUTRAL_EXTERNAL_INTERNAL_DIR.mkdir(parents=True, exist_ok=True)

                def plot_external_internal_scatter_all_points(
                    item_df: pd.DataFrame,
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

                        ax.text(
                            0.02,
                            0.98,
                            f"n = {len(subset)}\\nb = {n_b}\\nR² = {r2_text}",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=11,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
                        )

                        if annotate_hidden_region and panel_idx == 0 and x_left < 0 and y_top > 0:
                            ax.text(
                                x_left + 0.07 * (0 - x_left),
                                0 + 0.93 * y_top,
                                "x < 0, y > 0:\\noutput favors b,\\nprobe favors c",
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

                    path = save_figure(fig, filename_stub, subdir=NEUTRAL_EXTERNAL_INTERNAL_DIR)
                    plt.show()
                    plt.close(fig)
                    return path

                neutral_external_internal_path = plot_external_internal_scatter_all_points(
                    neutral_probe_item_df,
                    x_col="external_margin_x",
                    y_col="m_probe_bias_x",
                    title="External vs Internal Knowledge Without Bias",
                    x_label=r"$\\log \\pi(x)[c] - \\log \\pi(x)[b]$",
                    y_label=r"$s_N(x, c) - s_N(x, b)$",
                    filename_stub="external_vs_internal_knowledge_without_bias__neutral_probe__all_test_points",
                )

                biased_external_internal_path = plot_external_internal_scatter_all_points(
                    neutral_probe_item_df,
                    x_col="external_margin_xprime",
                    y_col="m_probe_bias_xprime",
                    title="External vs Internal Knowledge",
                    x_label=r"$\\log \\pi(x')[c] - \\log \\pi(x')[b]$",
                    y_label=r"$s_N(x', c) - s_N(x', b)$",
                    filename_stub="external_vs_internal_knowledge__neutral_probe__biased_prompt__all_test_points",
                    annotate_hidden_region=True,
                )

                mixed_external_internal_path = plot_external_internal_scatter_all_points(
                    neutral_probe_item_df,
                    x_col="external_margin_xprime",
                    y_col="m_probe_bias_x",
                    title="External Knowledge Under Bias vs Internal Knowledge Without Bias",
                    x_label=r"$\\log \\pi(x')[c] - \\log \\pi(x')[b]$",
                    y_label=r"$s_N(x, c) - s_N(x, b)$",
                    filename_stub="external_biased_vs_internal_neutral__neutral_probe__all_test_points",
                )

                internal_before_after_path = plot_external_internal_scatter_all_points(
                    neutral_probe_item_df,
                    x_col="m_probe_bias_x",
                    y_col="m_probe_bias_xprime",
                    title="Internal Knowledge Before vs After Bias",
                    x_label=r"$s_N(x, c) - s_N(x, b)$",
                    y_label=r"$s_N(x', c) - s_N(x', b)$",
                    filename_stub="internal_knowledge_before_vs_after_bias__neutral_probe__all_test_points",
                )

                pd.DataFrame(
                    {
                        "artifact": [
                            "neutral_external_internal_without_bias",
                            "neutral_external_internal_biased",
                            "neutral_external_biased_vs_internal_neutral",
                            "neutral_internal_before_after_bias",
                        ],
                        "path": [
                            str(neutral_external_internal_path),
                            str(biased_external_internal_path),
                            str(mixed_external_internal_path),
                            str(internal_before_after_path),
                        ],
                    }
                )
                """
            ).strip()
        )
    )

    nbformat.write(nb, V4_NOTEBOOK)


if __name__ == "__main__":
    build_v4_notebook()
