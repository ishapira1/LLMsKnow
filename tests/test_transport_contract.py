from __future__ import annotations

import math

import pandas as pd

from llmssycoph.analysis.transport import (
    build_self_commitment_comparison_df,
    build_incorrect_suggestion_transport_df,
    summarize_transport_by_subset,
)


def _build_model_fixture() -> pd.DataFrame:
    rows = []

    q1_probs_neutral = {"A": 0.90, "B": 0.10}
    q1_probs_biased = {"A": 0.70, "B": 0.30}
    for choice_id in ["A", "B"]:
        rows.append(
            {
                "run_id": "demo_run",
                "model_name": "demo_model",
                "dataset": "demo_dataset",
                "split": "test",
                "question_id": "q1",
                "draw_idx": 0,
                "question_uid": "demo_run::test::q1::0",
                "choice_id": choice_id,
                "correct_choice": "A",
                "is_correct": choice_id == "A",
                "prob_neutral": q1_probs_neutral[choice_id],
                "prob_incorrect_suggestion": q1_probs_biased[choice_id],
                "endorsed_choice_incorrect_suggestion": "B",
            }
        )

    q2_probs_neutral = {"A": 0.60, "B": 0.10, "C": 0.20, "D": 0.10}
    q2_probs_biased = {"A": 0.50, "B": 0.20, "C": 0.15, "D": 0.15}
    for choice_id in ["A", "B", "C", "D"]:
        rows.append(
            {
                "run_id": "demo_run",
                "model_name": "demo_model",
                "dataset": "demo_dataset",
                "split": "test",
                "question_id": "q2",
                "draw_idx": 0,
                "question_uid": "demo_run::test::q2::0",
                "choice_id": choice_id,
                "correct_choice": "A",
                "is_correct": choice_id == "A",
                "prob_neutral": q2_probs_neutral[choice_id],
                "prob_incorrect_suggestion": q2_probs_biased[choice_id],
                "endorsed_choice_incorrect_suggestion": "B",
            }
        )

    return pd.DataFrame(rows)


def _build_probe_fixture() -> pd.DataFrame:
    rows = []

    q1_scores_neutral = {"A": 9.0, "B": 1.0}
    q1_scores_biased = {"A": 7.0, "B": 3.0}
    for choice_id in ["A", "B"]:
        rows.append(
            {
                "question_uid": "demo_run::test::q1::0",
                "choice_id": choice_id,
                "probe_family": "neutral_trained",
                "score_neutral": q1_scores_neutral[choice_id],
                "score_incorrect_suggestion": q1_scores_biased[choice_id],
            }
        )

    q2_scores_neutral = {"A": 6.0, "B": 1.0, "C": 2.0, "D": 1.0}
    q2_scores_biased = {"A": 5.0, "B": 2.0, "C": 1.5, "D": 1.5}
    for choice_id in ["A", "B", "C", "D"]:
        rows.append(
            {
                "question_uid": "demo_run::test::q2::0",
                "choice_id": choice_id,
                "probe_family": "neutral_trained",
                "score_neutral": q2_scores_neutral[choice_id],
                "score_incorrect_suggestion": q2_scores_biased[choice_id],
            }
        )

    return pd.DataFrame(rows)


def _build_self_commitment_model_fixture() -> pd.DataFrame:
    rows = []

    fixtures = {
        "q1": {
            "choices": ["A", "B", "C"],
            "correct": "A",
            "endorsed_incorrect": "B",
            "endorsed_congruent": "A",
            "p0": {"A": 0.60, "B": 0.10, "C": 0.30},
            "p1": {"A": 0.45, "B": 0.35, "C": 0.20},
            "pc": {"A": 0.75, "B": 0.05, "C": 0.20},
        },
        "q2": {
            "choices": ["A", "B", "C", "D"],
            "correct": "A",
            "endorsed_incorrect": "B",
            "endorsed_congruent": "C",
            "p0": {"A": 0.35, "B": 0.10, "C": 0.45, "D": 0.10},
            "p1": {"A": 0.25, "B": 0.25, "C": 0.35, "D": 0.15},
            "pc": {"A": 0.25, "B": 0.10, "C": 0.55, "D": 0.10},
        },
        "q3": {
            "choices": ["A", "B", "C", "D"],
            "correct": "A",
            "endorsed_incorrect": "B",
            "endorsed_congruent": "D",
            "p0": {"A": 0.20, "B": 0.10, "C": 0.30, "D": 0.40},
            "p1": {"A": 0.10, "B": 0.35, "C": 0.20, "D": 0.35},
            "pc": {"A": 0.15, "B": 0.08, "C": 0.22, "D": 0.55},
        },
        "q4": {
            "choices": ["A", "B", "C"],
            "correct": "A",
            "endorsed_incorrect": "B",
            "endorsed_congruent": "B",
            "p0": {"A": 0.20, "B": 0.50, "C": 0.30},
            "p1": {"A": 0.15, "B": 0.60, "C": 0.25},
            "pc": {"A": 0.18, "B": 0.58, "C": 0.24},
        },
    }

    for question_id, spec in fixtures.items():
        for choice_id in spec["choices"]:
            rows.append(
                {
                    "run_id": "demo_run",
                    "model_name": "demo_model",
                    "dataset": "demo_dataset",
                    "split": "test",
                    "question_id": question_id,
                    "draw_idx": 0,
                    "question_uid": f"demo_run::test::{question_id}::0",
                    "choice_id": choice_id,
                    "correct_choice": spec["correct"],
                    "prob_neutral": spec["p0"][choice_id],
                    "prob_incorrect_suggestion": spec["p1"][choice_id],
                    "prob_model_congruent_suggestion": spec["pc"][choice_id],
                    "endorsed_choice_incorrect_suggestion": spec["endorsed_incorrect"],
                    "endorsed_choice_model_congruent_suggestion": spec["endorsed_congruent"],
                }
            )

    return pd.DataFrame(rows)


def _build_self_commitment_probe_fixture() -> pd.DataFrame:
    rows = []

    fixtures = {
        "q1": {
            "choices": ["A", "B", "C"],
            "s0": {"A": 6.0, "B": 1.0, "C": 4.0},
            "s1": {"A": 4.5, "B": 3.5, "C": 2.0},
            "sc": {"A": 7.0, "B": 0.8, "C": 2.2},
        },
        "q2": {
            "choices": ["A", "B", "C", "D"],
            "s0": {"A": 5.0, "B": 1.0, "C": 4.0, "D": 1.0},
            "s1": {"A": 3.0, "B": 3.0, "C": 3.5, "D": 1.5},
            "sc": {"A": 2.5, "B": 1.0, "C": 5.5, "D": 1.0},
        },
        "q3": {
            "choices": ["A", "B", "C", "D"],
            "s0": {"A": 3.0, "B": 1.0, "C": 4.0, "D": 5.0},
            "s1": {"A": 1.5, "B": 4.5, "C": 2.0, "D": 3.5},
            "sc": {"A": 1.8, "B": 0.8, "C": 2.2, "D": 5.2},
        },
    }

    for question_id, spec in fixtures.items():
        for choice_id in spec["choices"]:
            rows.append(
                {
                    "question_uid": f"demo_run::test::{question_id}::0",
                    "choice_id": choice_id,
                    "probe_family": "neutral_trained",
                    "score_neutral": spec["s0"][choice_id],
                    "score_incorrect_suggestion": spec["s1"][choice_id],
                    "score_model_congruent_suggestion": spec["sc"][choice_id],
                }
            )

    return pd.DataFrame(rows)


def test_transport_metrics_capture_targeted_shift_residuals_and_probe_closure():
    transport_df = build_incorrect_suggestion_transport_df(
        _build_model_fixture(),
        probe_wide_df=_build_probe_fixture(),
        framing="incorrect_suggestion",
        probe_family="neutral_trained",
    )

    assert len(transport_df) == 2

    q1 = transport_df.loc[transport_df["question_id"].eq("q1")].iloc[0]
    assert abs(float(q1["alpha_cb"]) - 0.4) < 1e-9
    assert abs(float(q1["tv"]) - 0.2) < 1e-9
    assert abs(float(q1["targeted_ratio_tv"]) - 2.0) < 1e-9
    assert abs(float(q1["directional_transport_share"]) - 1.0) < 1e-9
    assert abs(float(q1["residual_l1"])) < 1e-9
    assert q1["neutral_top_choice"] == "A"
    assert q1["biased_top_choice"] == "A"
    assert bool(q1["stay_correct"]) is True
    assert abs(float(q1["probe_closed_gap_closure"]) - 0.4) < 1e-9
    assert abs(float(q1["tilt_l1_error"])) < 1e-9
    assert abs(float(q1["tilt_alpha_fit_gap"])) < 1e-9

    q2 = transport_df.loc[transport_df["question_id"].eq("q2")].iloc[0]
    assert abs(float(q2["alpha_cb"]) - 0.2) < 1e-9
    assert abs(float(q2["tv"]) - 0.15) < 1e-9
    assert abs(float(q2["directional_transport_share"]) - (2.0 / 3.0)) < 1e-9
    assert abs(float(q2["residual_l1"]) - 0.1) < 1e-9
    assert q2["best_other_wrong_neutral_choice"] == "C"
    assert abs(float(q2["delta_best_other_wrong_neutral"]) + 0.05) < 1e-9
    assert bool(q2["b_becomes_top_wrong"]) is True
    assert abs(float(q2["probe_closed_gap_closure"]) - 0.2) < 1e-9
    assert float(q2["tilt_l1_error"]) > 0.0
    assert float(q2["tilt_alpha_fit_gap"]) > 0.0


def test_transport_subset_summary_matches_fixture_means():
    transport_df = build_incorrect_suggestion_transport_df(
        _build_model_fixture(),
        probe_wide_df=_build_probe_fixture(),
        framing="incorrect_suggestion",
        probe_family="neutral_trained",
    )
    summary_df = summarize_transport_by_subset(transport_df, subsets=["all", "no_flip", "stay_correct", "flip_c_to_b"])

    assert set(summary_df["subset"]) == {"all", "no_flip", "stay_correct"}

    overall = summary_df.loc[summary_df["subset"].eq("all")].iloc[0]
    assert overall["n_questions"] == 2
    assert abs(float(overall["mean_alpha_cb"]) - 0.3) < 1e-9
    assert abs(float(overall["mean_tv"]) - 0.175) < 1e-9
    assert abs(float(overall["mean_delta_b"]) - 0.15) < 1e-9
    assert abs(float(overall["mean_probe_closed_gap_closure"]) - 0.3) < 1e-9
    assert math.isnan(float(overall["flip_c_to_b_rate"])) or abs(float(overall["flip_c_to_b_rate"])) < 1e-9

    no_flip = summary_df.loc[summary_df["subset"].eq("no_flip")].iloc[0]
    assert no_flip["n_questions"] == 2
    assert abs(float(no_flip["answer_flip_rate"])) < 1e-9

    stay_correct = summary_df.loc[summary_df["subset"].eq("stay_correct")].iloc[0]
    assert stay_correct["n_questions"] == 2
    assert abs(float(stay_correct["stay_correct_rate"]) - 1.0) < 1e-9


def test_self_commitment_comparison_tracks_c_vs_d_probe_signal_and_congruent_control():
    comparison_df = build_self_commitment_comparison_df(
        _build_self_commitment_model_fixture(),
        probe_wide_df=_build_self_commitment_probe_fixture(),
        framing="incorrect_suggestion",
        congruent_framing="model_congruent_suggestion",
        probe_family="neutral_trained",
    )

    assert len(comparison_df) == 4
    assert comparison_df["self_margin_to_b_quartile_run"].notna().all()

    q1 = comparison_df.loc[comparison_df["question_id"].eq("q1")].iloc[0]
    assert q1["neutral_top_group"] == "c_top"
    assert bool(q1["included_in_c_vs_d"]) is True
    assert abs(float(q1["self_margin_to_b"]) - 0.50) < 1e-9
    assert abs(float(q1["delta_b"]) - 0.25) < 1e-9
    assert abs(float(q1["self_to_b_gap_closure"]) - 0.40) < 1e-9
    assert bool(q1["congruent_prompt_available"]) is True
    assert bool(q1["congruent_endorses_self_choice"]) is True
    assert abs(float(q1["delta_prompt_endorsed_target_congruent"]) - 0.15) < 1e-9

    q2 = comparison_df.loc[comparison_df["question_id"].eq("q2")].iloc[0]
    assert q2["neutral_top_group"] == "d_top"
    assert bool(q2["neutral_top_is_other_wrong"]) is True
    assert q2["neutral_top_choice"] == "C"
    assert abs(float(q2["self_margin_to_b"]) - 0.35) < 1e-9
    assert abs(float(q2["self_to_b_gap_closure"]) - 0.25) < 1e-9
    assert bool(q2["probe_prefers_correct_to_self_neutral"]) is True
    assert bool(q2["congruent_endorses_self_choice"]) is True
    assert abs(float(q2["delta_prompt_endorsed_target_congruent"]) - 0.10) < 1e-9

    q3 = comparison_df.loc[comparison_df["question_id"].eq("q3")].iloc[0]
    assert q3["neutral_top_group"] == "d_top"
    assert q3["neutral_top_choice"] == "D"
    assert bool(q3["probe_prefers_correct_to_self_neutral"]) is False
    assert bool(q3["answer_changed"]) is True
    assert bool(q3["flip_self_to_b"]) is True
    assert abs(float(q3["self_to_b_gap_closure"]) - 0.30) < 1e-9

    q4 = comparison_df.loc[comparison_df["question_id"].eq("q4")].iloc[0]
    assert q4["neutral_top_group"] == "b_top"
    assert bool(q4["included_in_c_vs_d"]) is False
    assert abs(float(q4["self_to_b_gap_closure"])) < 1e-9
