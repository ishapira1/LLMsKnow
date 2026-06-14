from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "export_claim3_presentation_bundle.py"
REAL_LONG_PATH = (
    ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "analysis_exports"
    / "claim3_model_probe_train_eval_breakdown_main_runs"
    / "claim3_model_probe_train_eval_breakdown_main_runs_long.csv"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("export_claim3_presentation_bundle", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_long_rows():
    return [
        {
            "run_id": "run_a",
            "run_name": "run_a",
            "run_dir": "/tmp/run_a",
            "model_name": "Model A",
            "dataset": "arc_challenge",
            "split": "test",
            "eval_on": "neutral",
            "top1": 0.4,
            "pairwise_k": 0.5,
            "auc": 0.6,
            "n_prompts": 10,
            "n_candidate_rows": 40,
            "source_kinds": "[]",
            "row_kind": "model",
            "probe_name": None,
            "probe_family": None,
            "trained_on": "model",
            "available": True,
        },
        {
            "run_id": "run_b",
            "run_name": "run_b",
            "run_dir": "/tmp/run_b",
            "model_name": "Model B",
            "dataset": "arc_challenge",
            "split": "test",
            "eval_on": "neutral",
            "top1": 0.8,
            "pairwise_k": 0.7,
            "auc": 0.9,
            "n_prompts": 30,
            "n_candidate_rows": 120,
            "source_kinds": "[]",
            "row_kind": "model",
            "probe_name": None,
            "probe_family": None,
            "trained_on": "model",
            "available": True,
        },
        {
            "run_id": "run_b",
            "run_name": "run_b",
            "run_dir": "/tmp/run_b",
            "model_name": "Model B",
            "dataset": "arc_challenge",
            "split": "test",
            "eval_on": "model_congruent_suggestion",
            "top1": 0.9,
            "pairwise_k": 0.95,
            "auc": 0.97,
            "n_prompts": 20,
            "n_candidate_rows": 80,
            "source_kinds": "[]",
            "row_kind": "model",
            "probe_name": None,
            "probe_family": None,
            "trained_on": "model",
            "available": True,
        },
        {
            "run_id": "run_a",
            "run_name": None,
            "run_dir": None,
            "model_name": "Model A",
            "dataset": "arc_challenge",
            "split": "test",
            "eval_on": "neutral",
            "top1": 0.2,
            "pairwise_k": 0.25,
            "auc": 0.3,
            "n_prompts": 5,
            "n_candidate_rows": 20,
            "source_kinds": "[]",
            "row_kind": "probe",
            "probe_name": "probe_no_bias",
            "probe_family": "neutral_trained",
            "trained_on": "neutral",
            "available": True,
        },
        {
            "run_id": "run_b",
            "run_name": None,
            "run_dir": None,
            "model_name": "Model B",
            "dataset": "arc_challenge",
            "split": "test",
            "eval_on": "neutral",
            "top1": 0.6,
            "pairwise_k": 0.65,
            "auc": 0.7,
            "n_prompts": 15,
            "n_candidate_rows": 60,
            "source_kinds": "[]",
            "row_kind": "probe",
            "probe_name": "probe_no_bias",
            "probe_family": "neutral_trained",
            "trained_on": "neutral",
            "available": True,
        },
        {
            "run_id": "run_b",
            "run_name": None,
            "run_dir": None,
            "model_name": "Model B",
            "dataset": "arc_challenge",
            "split": "test",
            "eval_on": "model_congruent_suggestion",
            "top1": 0.55,
            "pairwise_k": 0.58,
            "auc": 0.63,
            "n_prompts": 12,
            "n_candidate_rows": 48,
            "source_kinds": "[]",
            "row_kind": "probe",
            "probe_name": "probe_no_bias",
            "probe_family": "neutral_trained",
            "trained_on": "neutral",
            "available": True,
        },
        {
            "run_id": "run_a",
            "run_name": None,
            "run_dir": None,
            "model_name": "Model A",
            "dataset": "arc_challenge",
            "split": "test",
            "eval_on": "incorrect_suggestion",
            "top1": 0.1,
            "pairwise_k": 0.2,
            "auc": 0.25,
            "n_prompts": 8,
            "n_candidate_rows": 32,
            "source_kinds": "[]",
            "row_kind": "probe",
            "probe_name": "probe_bias_incorrect_suggestion",
            "probe_family": "incorrect_suggestion_trained",
            "trained_on": "incorrect_suggestion",
            "available": True,
        },
        {
            "run_id": "run_b",
            "run_name": None,
            "run_dir": None,
            "model_name": "Model B",
            "dataset": "arc_challenge",
            "split": "test",
            "eval_on": "incorrect_suggestion",
            "top1": 0.3,
            "pairwise_k": 0.35,
            "auc": 0.4,
            "n_prompts": 16,
            "n_candidate_rows": 64,
            "source_kinds": "[]",
            "row_kind": "probe",
            "probe_name": "probe_bias_incorrect_suggestion",
            "probe_family": "incorrect_suggestion_trained",
            "trained_on": "incorrect_suggestion",
            "available": True,
        },
    ]


def test_presentation_bundle_builds_selector_options_and_all_views():
    module = _load_script_module()
    df = module.pd.DataFrame(_fixture_long_rows())

    bundle = module.build_bundle_from_long_df(df, source_long_path="fixture.csv", split="test")

    assert bundle["selector_options"]["models"] == ["All", "Model A", "Model B"]
    assert bundle["selector_options"]["datasets"] == ["All", "arc_challenge"]
    assert bundle["selector_options"]["metrics"] == [
        "model_top1",
        "model_pairwise_k",
        "model_auc",
        "probe_top1",
        "probe_pairwise_k",
        "probe_auc",
    ]
    assert len(bundle["views"]) == 12
    assert module.build_view_key("All", "arc_challenge", "equal_weight") in bundle["views"]
    assert module.build_view_key("All", "arc_challenge", "prompt_weighted") in bundle["views"]


def test_presentation_bundle_uses_semantic_grid_weighting_and_missing_flags():
    module = _load_script_module()
    df = module.pd.DataFrame(_fixture_long_rows())

    bundle = module.build_bundle_from_long_df(df, source_long_path="fixture.csv", split="test")
    equal_view = bundle["views"][module.build_view_key("All", "arc_challenge", "equal_weight")]
    weighted_view = bundle["views"][module.build_view_key("All", "arc_challenge", "prompt_weighted")]

    assert len(equal_view["rows"]) == len(module.PROBE_TRAIN_ON_ORDER) * len(module.EVAL_ON_ORDER)

    neutral_row_equal = next(
        row for row in equal_view["rows"] if row["probe_train_on"] == "neutral" and row["eval_on"] == "neutral"
    )
    assert abs(neutral_row_equal["model"]["metrics"]["top1"] - 0.6) < 1e-9
    assert abs(neutral_row_equal["probe"]["metrics"]["top1"] - 0.4) < 1e-9

    neutral_row_weighted = next(
        row for row in weighted_view["rows"] if row["probe_train_on"] == "neutral" and row["eval_on"] == "neutral"
    )
    assert abs(neutral_row_weighted["model"]["metrics"]["top1"] - 0.7) < 1e-9
    assert abs(neutral_row_weighted["probe"]["metrics"]["top1"] - 0.5) < 1e-9

    missing_probe_row = next(
        row
        for row in equal_view["rows"]
        if row["probe_train_on"] == "incorrect_suggestion" and row["eval_on"] == "suggest_correct"
    )
    assert missing_probe_row["probe"]["available"] is False
    assert missing_probe_row["probe"]["metrics"]["top1"] is None
    assert missing_probe_row["probe"]["runs_contributing"] == 0
    assert missing_probe_row["probe"]["runs_selected"] == 2

    partial_model_row = next(
        row
        for row in equal_view["rows"]
        if row["probe_train_on"] == "neutral" and row["eval_on"] == "model_congruent_suggestion"
    )
    assert partial_model_row["model"]["available"] is True
    assert partial_model_row["model"]["runs_contributing"] == 1
    assert partial_model_row["model"]["runs_selected"] == 2
    assert partial_model_row["probe"]["runs_contributing"] == 1


def test_real_packaged_scope_bundle_keeps_semantic_rows_for_every_view():
    module = _load_script_module()
    df = module.load_long_breakdown(REAL_LONG_PATH, split="test")
    bundle = module.build_bundle_from_long_df(df, source_long_path=str(REAL_LONG_PATH), split="test")

    assert bundle["selector_options"]["models"] == [
        "All",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    assert bundle["selector_options"]["datasets"] == ["All", "arc_challenge", "commonsense_qa"]
    for view in bundle["views"].values():
        assert len(view["rows"]) == len(module.PROBE_TRAIN_ON_ORDER) * len(module.EVAL_ON_ORDER)
