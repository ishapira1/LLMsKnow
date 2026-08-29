from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from llmssycoph.analysis.confidence_resistance import (
    add_robust_scales,
    derive_item_metrics,
    design_matrix,
    fit_huber,
    fit_ols,
    geometry_null,
    holm_adjust,
    primary_results,
    run_analysis,
)
from llmssycoph.confidence_resistance import (
    DATASETS,
    ExperimentPaths,
    audit_measurement_coverage,
    audit_gpt_pilot,
    build_endorsement_manifests,
    canonical_token_text,
    gpt_request_body,
    frozen_analysis_spec,
    normalized_probabilities_from_log_scores,
    parse_openai_batch_output,
    prepare_experiment,
    ranked_choices,
    read_jsonl,
    select_reserve_extension,
    write_json,
    write_jsonl,
)


def _source_row(dataset: str, index: int, choices: int) -> dict:
    letters = "ABCDE"[:choices]
    return {
        "base": {
            "dataset": dataset,
            "question": f"Question {dataset} {index}?",
            "correct_letter": letters[index % choices],
            "letters": letters,
            "answers_list": [f"answer {index}-{option}" for option in range(choices)],
            "source_dataset": f"source/{dataset}",
            "source_split": "train",
            "source_example_id": f"{dataset}-{index}",
        }
    }


def _record(
    *,
    condition: str,
    scores: dict[str, float],
    target: str | None = None,
    target_rank: str | None = None,
) -> dict:
    letters = "ABCD"
    probabilities = normalized_probabilities_from_log_scores(scores, letters)
    return {
        "custom_id": f"row-{condition}-{target or 'none'}",
        "model_key": "llama",
        "model": "test/model",
        "model_revision": "abc",
        "dataset": "arc_challenge",
        "analysis_split": "discovery",
        "question_id": "arc:q1",
        "source_example_id": "q1",
        "question": "Question?",
        "answers": ["one", "two", "three", "four"],
        "letters": letters,
        "correct_letter": "A",
        "option_permutation": [0, 1, 2, 3],
        "condition": condition,
        "target_letter": target,
        "target_rank": target_rank,
        "choice_log_scores": scores,
        "choice_probabilities": probabilities,
        "qc_complete_choice_scores": True,
        "score_source": "test",
    }


class ConfidenceResistanceTests(unittest.TestCase):
    def test_logsumexp_normalization_survives_extreme_scores(self) -> None:
        probabilities = normalized_probabilities_from_log_scores(
            {"A": -10000.0, "B": -10001.0, "C": -11000.0}, "ABC"
        )
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities["A"], probabilities["B"])
        self.assertGreater(probabilities["B"], 0.0)
        self.assertAlmostEqual(
            math.log(probabilities["A"] / probabilities["B"]), 1.0
        )

    def test_rank_and_metric_definitions_use_model_top_not_gold(self) -> None:
        neutral = _record(
            condition="neutral",
            scores={"A": -3.0, "B": -0.1, "C": -0.6, "D": -2.0},
        )
        post = _record(
            condition="unsupported_endorsement",
            scores={"A": -3.0, "B": -1.2, "C": -0.2, "D": -2.1},
            target="C",
            target_rank="rank_2",
        )
        item = derive_item_metrics(neutral, post)
        self.assertEqual(item["neutral_top"], "B")
        self.assertEqual(item["neutral_runner_up"], "C")
        self.assertAlmostEqual(item["c0"], 0.5)
        self.assertAlmostEqual(item["q0"], 0.0)
        self.assertAlmostEqual(item["delta"], 1.5)
        self.assertGreater(item["probability_gap_movement"], 0)

    def test_prepare_is_balanced_frozen_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data" / "sycophancy-eval"
            data.mkdir(parents=True)
            for dataset, choices in (("arc_challenge", 4), ("commonsense_qa", 5)):
                write_jsonl(
                    data / f"{dataset}.jsonl",
                    [_source_row(dataset, index, choices) for index in range(20)],
                )
            paths = ExperimentPaths(root / "results")
            first = prepare_experiment(
                paths,
                root,
                discovery_per_dataset=4,
                confirmation_per_dataset=8,
                reserve_per_dataset=4,
                pilot_per_dataset=2,
            )
            second = prepare_experiment(
                paths,
                root,
                discovery_per_dataset=4,
                confirmation_per_dataset=8,
                reserve_per_dataset=4,
                pilot_per_dataset=2,
            )
            self.assertEqual(first, second)
            questions = read_jsonl(paths.questions)
            self.assertEqual(len(questions), 32)
            source_splits = {}
            for row in questions:
                source_splits.setdefault((row["dataset"], row["source_example_id"]), set()).add(
                    row["analysis_split"]
                )
            self.assertTrue(all(len(splits) == 1 for splits in source_splits.values()))
            for dataset, choices in (("arc_challenge", 4), ("commonsense_qa", 5)):
                discovery = [
                    row
                    for row in questions
                    if row["dataset"] == dataset and row["analysis_split"] == "discovery"
                ]
                positions = [row["letters"].index(row["correct_letter"]) for row in discovery]
                self.assertEqual(sorted(positions), list(range(choices))[: len(discovery)])

    def test_gpt_request_uses_equal_bias_and_top_twenty(self) -> None:
        task = {
            "letters": "ABCDE",
            "messages": [{"role": "user", "content": "Question\nAnswer:"}],
        }
        body = gpt_request_body(
            task,
            biased=True,
            token_ids={letter: index for index, letter in enumerate("ABCDE", start=1)},
        )
        self.assertEqual(body["top_logprobs"], 20)
        self.assertEqual(len(body["logit_bias"]), 5)
        self.assertEqual(set(body["logit_bias"].values()), {100})
        self.assertEqual(canonical_token_text("A"), " A")

    def test_openai_parser_marks_missing_letters_as_censored_not_zero(self) -> None:
        task = {
            "custom_id": "gpt-q1",
            "letters": "ABCD",
            "model_key": "gpt",
            "model": "gpt-5.4-nano-2026-03-17",
            "model_revision": "gpt-5.4-nano-2026-03-17",
            "dataset": "arc_challenge",
            "analysis_split": "discovery",
            "question_id": "q1",
            "condition": "neutral",
        }
        candidates = [
            {"token": " A", "logprob": -0.1},
            {"token": " B", "logprob": -1.0},
            {"token": " C", "logprob": -2.0},
        ]
        output = {
            "custom_id": "gpt-q1",
            "response": {
                "status_code": 200,
                "body": {
                    "id": "request",
                    "model": "gpt-5.4-nano-2026-03-17",
                    "system_fingerprint": "fp_test",
                    "choices": [
                        {
                            "logprobs": {
                                "content": [
                                    {
                                        "token": " A",
                                        "logprob": -0.1,
                                        "top_logprobs": candidates,
                                    }
                                ]
                            }
                        }
                    ],
                },
            },
        }
        parsed = parse_openai_batch_output(
            [task],
            [output],
            biased=False,
            batch_id="batch",
            token_ids_by_letters={"ABCD": {letter: index for index, letter in enumerate("ABCD", start=1)}},
        )
        self.assertEqual(len(parsed), 1)
        self.assertFalse(parsed[0]["qc_complete_choice_scores"])
        self.assertNotIn("D", parsed[0]["choice_log_scores"])
        self.assertEqual(parsed[0]["choice_probabilities"], {})
        self.assertEqual(parsed[0]["choice_log_score_bounds"]["D"], [None, -2.0])
        lower, upper = parsed[0]["choice_probability_bounds"]["D"]
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_target_manifest_uses_neutral_second_third_and_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ExperimentPaths(Path(directory))
            for dataset in DATASETS:
                question = {
                    "dataset": dataset,
                    "analysis_split": "discovery",
                    "question_id": f"{dataset}:q1",
                    "source_example_id": "q1",
                    "question": "Question?",
                    "answers": ["one", "two", "three", "four", "five"][: 4 if dataset == "arc_challenge" else 5],
                    "letters": "ABCD" if dataset == "arc_challenge" else "ABCDE",
                    "correct_letter": "A",
                    "option_permutation": list(range(4 if dataset == "arc_challenge" else 5)),
                    "pilot": False,
                }
                existing = read_jsonl(paths.questions)
                write_jsonl(paths.questions, [*existing, question])
                letters = question["letters"]
                scores = {letter: -float(index) for index, letter in enumerate(letters)}
                neutral = {
                    **question,
                    "custom_id": f"neutral-{dataset}",
                    "model_key": "llama",
                    "model": "test",
                    "model_revision": "rev",
                    "condition": "neutral",
                    "choice_log_scores": scores,
                    "choice_probabilities": normalized_probabilities_from_log_scores(scores, letters),
                    "qc_complete_choice_scores": True,
                }
                write_jsonl(paths.neutral_records("llama", dataset), [neutral])
            build_endorsement_manifests(paths, model_key="llama")
            arc = read_jsonl(paths.endorsed_manifest("llama", "arc_challenge"))
            targets = {row["target_rank"]: row["target_letter"] for row in arc}
            self.assertEqual(targets, {"rank_2": "B", "rank_3": "C", "rank_last": "D"})

    def test_reserve_extension_uses_common_qc_complete_prefix_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ExperimentPaths(Path(directory))
            questions = []
            for dataset in DATASETS:
                for index in range(2):
                    questions.append(
                        {
                            "dataset": dataset,
                            "analysis_split": "reserve",
                            "split_index": index,
                            "question_id": f"{dataset}:reserve-{index}",
                            "source_example_id": f"reserve-{index}",
                        }
                    )
                for model_key in ("llama", "qwen", "gpt"):
                    main_neutral = [
                        {
                            "question_id": f"{dataset}:initial",
                            "analysis_split": "confirmation",
                            "qc_complete_choice_scores": True,
                        }
                    ]
                    reserve_neutral = [
                        {
                            "question_id": f"{dataset}:reserve-{index}",
                            "analysis_split": "reserve",
                            "qc_complete_choice_scores": True,
                        }
                        for index in range(2)
                    ]
                    main_endorsed = [
                        {
                            "question_id": f"{dataset}:initial",
                            "analysis_split": "confirmation",
                            "target_rank": rank,
                            "qc_complete_choice_scores": True,
                        }
                        for rank in ("rank_2", "rank_3", "rank_last")
                    ]
                    reserve_endorsed = [
                        {
                            "question_id": f"{dataset}:reserve-{index}",
                            "analysis_split": "reserve",
                            "target_rank": rank,
                            "qc_complete_choice_scores": True,
                        }
                        for index in range(2)
                        for rank in ("rank_2", "rank_3", "rank_last")
                    ]
                    write_jsonl(paths.neutral_records(model_key, dataset), main_neutral)
                    write_jsonl(paths.endorsed_records(model_key, dataset), main_endorsed)
                    write_jsonl(paths.reserve_neutral_records(model_key, dataset), reserve_neutral)
                    write_jsonl(paths.reserve_endorsed_records(model_key, dataset), reserve_endorsed)
            write_jsonl(paths.questions, questions)
            with patch("llmssycoph.confidence_resistance.MIN_CONFIRMATION_CELL", 2):
                selection = select_reserve_extension(paths)
                audit = audit_measurement_coverage(paths)
            self.assertEqual(selection["manifest"]["rows"], 2)
            self.assertTrue(audit["passed"])
            promoted = read_jsonl(paths.reserve_selection)
            self.assertTrue(
                all(row["question_id"].endswith("reserve-0") for row in promoted)
            )

    def test_huber_recovers_negative_confidence_effect_with_outlier(self) -> None:
        rows = []
        rng = np.random.default_rng(3)
        for dataset in DATASETS:
            for question in range(80):
                c0 = 0.1 + question / 20
                for rank_index, target_rank in enumerate(("rank_2", "rank_3", "rank_last")):
                    delta = 3.0 - 0.8 * c0 + 0.2 * rank_index + rng.normal(0, 0.05)
                    rows.append(
                        {
                            "model_key": "llama",
                            "dataset": dataset,
                            "question_id": f"{dataset}:{question}",
                            "target_rank": target_rank,
                            "target_letter": "BCD"[rank_index],
                            "c0": c0,
                            "q0": float(rank_index),
                            "delta": delta,
                        }
                    )
        rows[-1]["delta"] = 1000.0
        frame = add_robust_scales(pd.DataFrame(rows))
        x, names = design_matrix(frame)
        beta_huber = fit_huber(x, frame["delta"].to_numpy(dtype=float))
        beta_ols = fit_ols(x, frame["delta"].to_numpy(dtype=float))
        confidence_index = names.index("c0_robust_z")
        self.assertLess(beta_huber[confidence_index], 0)
        self.assertLess(abs(beta_huber[confidence_index] + 0.8 * 2.0), 0.3)
        self.assertGreater(abs(beta_ols[confidence_index] - beta_huber[confidence_index]), 0.1)

    def test_holm_adjustment_is_monotone(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.20)

    def test_primary_gate_requires_negative_effect_in_every_model(self) -> None:
        rows = []
        rng = np.random.default_rng(11)
        for model_key in ("llama", "qwen", "gpt"):
            for dataset in DATASETS:
                for question in range(45):
                    c0 = 0.2 + question / 12
                    for rank_index, target_rank in enumerate(("rank_2", "rank_3", "rank_last")):
                        rows.append(
                            {
                                "model_key": model_key,
                                "dataset": dataset,
                                "question_id": f"{dataset}:{question}",
                                "target_rank": target_rank,
                                "target_letter": "BCD"[rank_index],
                                "c0": c0,
                                "q0": float(rank_index),
                                "delta": 4.0 - 0.9 * c0 + 0.15 * rank_index + rng.normal(0, 0.04),
                            }
                        )
        frame = add_robust_scales(pd.DataFrame(rows))
        _, result = primary_results(frame, replicates=100, seed=9)
        self.assertTrue(result["universal_logit_susceptibility_supported"])
        self.assertTrue(
            all(test["holm_ci_high"] < 0 for test in result["model_tests"].values())
        )
        frame.loc[frame["model_key"].eq("qwen"), "delta"] += (
            1.8 * frame.loc[frame["model_key"].eq("qwen"), "c0"]
        )
        _, heterogeneous = primary_results(frame, replicates=100, seed=9)
        self.assertFalse(heterogeneous["universal_logit_susceptibility_supported"])

    def test_constant_logit_update_can_create_probability_resistance(self) -> None:
        confidences = np.linspace(0.1, 6.0, 60)
        movements = []
        deltas = []
        for index, confidence in enumerate(confidences):
            neutral = _record(
                condition="neutral",
                scores={"A": float(confidence), "B": 0.0, "C": -4.0, "D": -5.0},
            )
            post = _record(
                condition="unsupported_endorsement",
                scores={"A": float(confidence), "B": 1.0, "C": -4.0, "D": -5.0},
                target="B",
                target_rank="rank_2",
            )
            neutral["question_id"] = post["question_id"] = f"arc:q{index}"
            item = derive_item_metrics(neutral, post)
            movements.append(item["probability_gap_movement"])
            deltas.append(item["delta"])
        self.assertLess(float(spearmanr(confidences, movements).statistic), -0.8)
        self.assertLess(max(deltas) - min(deltas), 1e-10)

    def test_null_effect_does_not_pass_universal_gate(self) -> None:
        rows = []
        for model_key in ("llama", "qwen", "gpt"):
            for dataset in DATASETS:
                for question in range(45):
                    c0 = 0.2 + question / 12
                    for rank_index, target_rank in enumerate(("rank_2", "rank_3", "rank_last")):
                        rows.append(
                            {
                                "model_key": model_key,
                                "dataset": dataset,
                                "question_id": f"{dataset}:{question}",
                                "target_rank": target_rank,
                                "target_letter": "BCD"[rank_index],
                                "c0": c0,
                                "q0": float(rank_index),
                                "delta": 2.0 + 0.1 * rank_index + 0.03 * ((question % 5) - 2),
                            }
                        )
        frame = add_robust_scales(pd.DataFrame(rows))
        _, result = primary_results(frame, replicates=100, seed=10)
        self.assertFalse(result["universal_logit_susceptibility_supported"])

    def test_geometry_null_reports_probability_and_flip_behavior(self) -> None:
        rows = []
        for model_key in ("llama", "qwen", "gpt"):
            for dataset in DATASETS:
                for question in range(20):
                    confidence = 0.1 + question / 5
                    for rank_index, rank in enumerate(("rank_2", "rank_3", "rank_last")):
                        target = "BCD"[rank_index]
                        neutral_scores = {"A": confidence, "B": 0.0, "C": -1.0, "D": -2.0}
                        post_scores = dict(neutral_scores)
                        post_scores[target] += 1.0 + 0.1 * rank_index
                        neutral = _record(condition="neutral", scores=neutral_scores)
                        post = _record(
                            condition="unsupported_endorsement",
                            scores=post_scores,
                            target=target,
                            target_rank=rank,
                        )
                        neutral.update(
                            model_key=model_key,
                            dataset=dataset,
                            question_id=f"{dataset}:q{question}",
                        )
                        post.update(
                            model_key=model_key,
                            dataset=dataset,
                            question_id=f"{dataset}:q{question}",
                        )
                        rows.append(derive_item_metrics(neutral, post))
        items, null = geometry_null(pd.DataFrame(rows), permutations=20, seed=22)
        self.assertIn("constant_update_target_became_top", items)
        self.assertIn("target_flip_null_mean", null)
        self.assertIn("top1_change_null_mean", null)

    def test_full_discovery_analysis_writes_all_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ExperimentPaths(Path(directory) / "experiment")
            write_json(paths.analysis_spec, frozen_analysis_spec())
            digest = __import__("hashlib").sha256(paths.analysis_spec.read_bytes()).hexdigest()
            write_json(paths.config, {"analysis_spec": {"sha256": digest}})
            write_json(paths.pilot_audit, {"passed": True})
            for model_key in ("llama", "qwen", "gpt"):
                for dataset in DATASETS:
                    neutral_rows = []
                    endorsed_rows = []
                    letters = "ABCD" if dataset == "arc_challenge" else "ABCDE"
                    for question in range(12):
                        confidence = 0.2 + question / 5
                        scores = {letter: -2.0 - index for index, letter in enumerate(letters)}
                        scores["A"] = confidence
                        scores["B"] = 0.0
                        neutral = {
                            **_record(condition="neutral", scores={key: scores[key] for key in "ABCD"}),
                            "custom_id": f"{model_key}-{dataset}-{question}-neutral",
                            "model_key": model_key,
                            "model": f"test/{model_key}",
                            "model_revision": "revision",
                            "dataset": dataset,
                            "question_id": f"{dataset}:q{question}",
                            "letters": letters,
                            "answers": [f"answer {letter}" for letter in letters],
                            "choice_log_scores": scores,
                            "choice_probabilities": normalized_probabilities_from_log_scores(scores, letters),
                        }
                        neutral_rows.append(neutral)
                        order = ranked_choices(scores, letters)
                        targets = {"rank_2": order[1], "rank_3": order[2], "rank_last": order[-1]}
                        for rank_index, target_rank in enumerate(("rank_2", "rank_3", "rank_last")):
                            target = targets[target_rank]
                            post_scores = dict(scores)
                            post_scores[target] += 3.0 - 0.7 * confidence + 0.1 * rank_index
                            endorsed_rows.append(
                                {
                                    **neutral,
                                    "custom_id": f"{model_key}-{dataset}-{question}-{target_rank}",
                                    "condition": "unsupported_endorsement",
                                    "target_letter": target,
                                    "target_rank": target_rank,
                                    "choice_log_scores": post_scores,
                                    "choice_probabilities": normalized_probabilities_from_log_scores(post_scores, letters),
                                }
                            )
                    write_jsonl(paths.neutral_records(model_key, dataset), neutral_rows)
                    write_jsonl(paths.endorsed_records(model_key, dataset), endorsed_rows)
            output = Path(directory) / "analysis"
            result = run_analysis(
                paths,
                output,
                scope="discovery",
                bootstrap_replicates=50,
                permutation_replicates=20,
                seed=4,
            )
            self.assertTrue(
                all(
                    model["coefficient"] < 0
                    for model in result["primary"]["model_tests"].values()
                )
            )
            for filename in (
                "confidence_coefficient_forest.png",
                "confidence_log_space_panel.png",
                "confidence_probability_space_panel.png",
                "confidence_robustness_multiverse.png",
                "qc_sample_counts.csv",
                "analysis_manifest.json",
            ):
                self.assertTrue((output / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
