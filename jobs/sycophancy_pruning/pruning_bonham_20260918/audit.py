#!/usr/bin/env python3
"""Fail-closed final audit for the complete Bonham publication bundle."""

from __future__ import annotations

from collections import Counter
import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import campaign
from core import (
    DEFAULT_CONFIG,
    ELIGIBLE_PROJECTIONS,
    atomic_json,
    load_config,
    mask_coordinates,
    read_json,
    read_jsonl,
    sha256_file,
)


class AuditError(campaign.CampaignError):
    pass


RAW_RECORD_FIELDS = {
    "model_key",
    "state_id",
    "dataset_id",
    "question_id",
    "question_axis",
    "prompt_regime",
    "bias_type",
    "turn_format",
    "template_family",
    "template_id",
    "claim_truth",
    "claim_attribution",
    "asserted_label",
    "doubted_label",
    "gold_label",
    "neutral_label",
    "parse_status",
    "generated_answer",
    "forced_choice_probabilities",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _authenticated(path: Path, expected: str) -> None:
    _require(path.is_file(), f"Missing artifact: {path}")
    _require(sha256_file(path) == expected, f"Changed artifact: {path}")


def _expected_capabilities(config: Mapping[str, Any]) -> set[str]:
    """Return the frozen top-level capability registry.

    The Bonham config intentionally keeps this registry beside ``methods``;
    there is no nested ``evaluation`` object.
    """

    capabilities = config.get("capability_tasks")
    _require(
        isinstance(capabilities, list)
        and capabilities
        and all(isinstance(name, str) and name for name in capabilities),
        "Capability registry is missing or malformed",
    )
    return set(capabilities)


def _audit_mask(
    mask_root: Path,
    *,
    expected_count: int,
    expected_p: float,
) -> tuple[set[tuple[str, int]], Mapping[str, Any]]:
    complete = read_json(mask_root / "COMPLETE.json")
    metadata = read_json(mask_root / "metadata.json")
    _authenticated(mask_root / "indices.pt", str(complete["indices_sha256"]))
    _authenticated(mask_root / "metadata.json", str(complete["metadata_sha256"]))
    _authenticated(mask_root / "ordering.jsonl", str(complete["ordering_sha256"]))
    _require(
        int(complete["selected_count"]) == expected_count
        and int(complete["n"]) == expected_count
        and int(metadata["selected_count"]) == expected_count
        and int(metadata["n"]) == expected_count,
        f"Mask size contract failed: {mask_root}",
    )
    _require(
        float(complete["p"]) == expected_p and float(metadata["p"]) == expected_p,
        f"Mask p contract failed: {mask_root}",
    )
    indices = campaign._load_indices(mask_root / "indices.pt")
    coordinates = mask_coordinates(indices)
    _require(len(coordinates) == expected_count, f"Mask coordinates repeat: {mask_root}")
    _require(
        all(str(name).rsplit(".", 1)[-1] in ELIGIBLE_PROJECTIONS for name in indices),
        f"Mask contains an ineligible projection: {mask_root}",
    )
    observed_counts = {name: int(values.numel()) for name, values in indices.items()}
    _require(
        observed_counts == {str(name): int(value) for name, value in metadata["counts_by_module"].items()},
        f"Mask per-module counts changed: {mask_root}",
    )
    ordering = read_jsonl(mask_root / "ordering.jsonl")
    ordering_coordinates = {
        (str(row["parameter"]), int(row["flat_index"])) for row in ordering
    }
    _require(
        len(ordering) == expected_count
        and [int(row["rank"]) for row in ordering] == list(range(1, expected_count + 1))
        and ordering_coordinates == coordinates,
        f"Mask deterministic ordering is incomplete: {mask_root}",
    )
    _require(
        all(len(str(row.get("tie_sha256", ""))) == 64 for row in ordering),
        f"Mask tie hashes are malformed: {mask_root}",
    )
    if metadata.get("algorithm") == "bonham_per_matrix_protect_percentile_pool_v1":
        for name, row in dict(metadata.get("matrix_audit", {})).items():
            numel = int(row["numel"])
            protected = int(row["protected_count"])
            _require(
                protected == math.floor(expected_p * numel)
                and int(row["candidate_count"]) == numel - protected,
                f"Protection count changed for {mask_root}/{name}",
            )
        _require(
            all("within_matrix_percentile" in row for row in ordering),
            f"Mask lacks within-matrix percentile ranks: {mask_root}",
        )
    return coordinates, metadata


def _audit_n1_rows(rows: list[Mapping[str, Any]], model_key: str) -> None:
    _require(len(rows) == 512, f"N1 pruning count is not 512 for {model_key}")
    _require(
        len({str(row["question_key"]) for row in rows}) == 512,
        f"N1 questions repeat for {model_key}",
    )
    cells = Counter(
        (row["dataset"], row["turn_format"], row["bias_type"], row["template_id"])
        for row in rows
    )
    _require(len(cells) == 32 and set(cells.values()) == {16}, f"N1 is unbalanced for {model_key}")
    _require(
        Counter(row["dataset"] for row in rows)
        == {"commonsense_qa": 256, "arc_challenge": 256}
        and Counter(row["turn_format"] for row in rows)
        == {"single_turn": 256, "multi_turn": 256}
        and Counter(row["bias_type"] for row in rows)
        == {"incorrect_suggestion": 256, "doubt_correct": 256},
        f"N1 marginal balance failed for {model_key}",
    )
    for row in rows:
        _require(row.get("behavior_qualified") is True, f"N1 row is not behavior-qualified: {model_key}")
        target = str(row["attribution_target_choice"])
        gold = str(row["gold_choice"])
        if row["bias_type"] == "incorrect_suggestion":
            _require(target == str(row["wrong_choice"]), f"N1 suggestion target mismatch: {model_key}")
        else:
            _require(target != gold, f"N1 doubt target did not use the observed wrong answer: {model_key}")


def _audit_source_rows(rows: list[Mapping[str, Any]], model_key: str) -> None:
    _require(len(rows) == 512, f"N2 reliable-source count is not 512 for {model_key}")
    _require(
        all(
            row.get("claim_attribution") == "reliable_source"
            and row.get("source_aligned") is True
            for row in rows
        ),
        f"N2 contains a non-source-aligned or bare-user row for {model_key}",
    )
    cells = Counter((row["claim_type"], row["turn_format"]) for row in rows)
    _require(len(cells) == 8 and set(cells.values()) == {64}, f"N2 source cells fail for {model_key}")
    dataset_cells = Counter(
        (row["claim_type"], row["turn_format"], row["dataset"]) for row in rows
    )
    _require(
        len(dataset_cells) == 16 and set(dataset_cells.values()) == {32},
        f"N2 dataset-by-cell quotas fail for {model_key}",
    )
    truth = Counter(row["claim_truth"] for row in rows)
    formats = Counter(row["turn_format"] for row in rows)
    suggestion_or_doubt = Counter(
        "suggestion" if str(row["claim_type"]).startswith("suggest_") else "doubt"
        for row in rows
    )
    _require(
        truth == {"true": 256, "false": 256}
        and formats == {"single_turn": 256, "multi_turn": 256}
        and suggestion_or_doubt == {"suggestion": 256, "doubt": 256},
        f"N2 source marginals fail for {model_key}",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["question_key"]), []).append(row)
    _require(len(grouped) == 128, f"N2 does not use 128 distinct questions for {model_key}")
    cohort_counts: Counter[tuple[str, str]] = Counter()
    for question_key, group in grouped.items():
        _require(len(group) == 4, f"N2 question lacks four paired conditions: {model_key}/{question_key}")
        templates = {(row["template_id"], row["template_family"]) for row in group}
        _require(len(templates) == 1, f"N2 source template changed across a paired question: {model_key}")
        neutral = str(group[0]["neutral_choice"])
        gold = str(group[0]["gold_choice"])
        correctness = "initially_correct" if neutral == gold else "initially_incorrect"
        cohort_counts[(str(group[0]["dataset"]), correctness)] += 1
        expected_claims = (
            {"suggest_w", "doubt_c"}
            if correctness == "initially_correct"
            else {"suggest_c", "doubt_w"}
        )
        _require(
            {(row["claim_type"], row["turn_format"]) for row in group}
            == {(claim, turn) for claim in expected_claims for turn in ("single_turn", "multi_turn")},
            f"N2 paired claim set is wrong for {model_key}/{question_key}",
        )
        for row in group:
            target = str(row["attribution_target_choice"])
            claim = str(row["claim_type"])
            if claim == "suggest_c":
                _require(target == str(row["gold_choice"]), f"N2 suggest-C target mismatch: {model_key}")
            elif claim == "suggest_w":
                _require(target == str(row["wrong_choice"]), f"N2 suggest-W target mismatch: {model_key}")
            elif claim == "doubt_c":
                _require(target != str(row["gold_choice"]), f"N2 doubt-C target mismatch: {model_key}")
            else:
                _require(target != str(row["wrong_choice"]), f"N2 doubt-W target mismatch: {model_key}")
    _require(
        cohort_counts
        == {
            ("commonsense_qa", "initially_correct"): 32,
            ("commonsense_qa", "initially_incorrect"): 32,
            ("arc_challenge", "initially_correct"): 32,
            ("arc_challenge", "initially_incorrect"): 32,
        },
        f"N2 question cohorts fail for {model_key}: {cohort_counts}",
    )
    for cell in cells:
        subset = [row for row in rows if (row["claim_type"], row["turn_format"]) == cell]
        family = Counter(row["template_family"] for row in subset)
        template = Counter(row["template_id"] for row in subset)
        _require(
            family["quantified_reliability"] == 16
            and sum(value for key, value in family.items() if key != "quantified_reliability") == 48,
            f"N2 16/48 source-family quota fails for {model_key}/{cell}",
        )
        _require(
            len(template) == 12 and max(template.values()) - min(template.values()) <= 1,
            f"N2 twelve-template rotation fails for {model_key}/{cell}",
        )


def _audit_raw_evaluation_records(root: Path) -> int:
    observed = 0
    for model_key in campaign.MODEL_KEYS:
        for state_id in campaign.PRIMARY_STATE_IDS:
            for family in ("generalization", "useful_assertions", "capabilities"):
                family_root = root / "evaluations" / "results" / model_key / state_id / family
                for path in sorted(family_root.glob("shard_*/records.jsonl")):
                    with path.open("r", encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, 1):
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            missing = RAW_RECORD_FIELDS.difference(row)
                            _require(
                                not missing,
                                f"Raw evaluation record lacks {sorted(missing)}: {path}:{line_number}",
                            )
                            _require(
                                row["model_key"] == model_key and row["state_id"] == state_id,
                                f"Raw evaluation identity mismatch: {path}:{line_number}",
                            )
                            observed += 1
    return observed


def final_audit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    root = Path(args.result_root)
    _require(float(config["selection"]["protection_fraction"]) == 0.00005, "p changed")
    _require(int(config["selection"]["mask_weight_count"]) == 1000, "n changed")
    _require(
        tuple(config["methods"]) == campaign.PRIMARY_STATE_IDS,
        "Configured methods differ from the eight preregistered states",
    )
    source = read_json(root / "inputs" / "SOURCE_FREEZE_COMPLETE.json")
    _require(source.get("status") == "complete", "Source freeze is incomplete")
    _require(
        source.get("config_sha256") == sha256_file(args.config),
        "Source freeze used a different Bonham config",
    )
    _require(
        source.get("source_revisions")
        == {key: value["revision"] for key, value in config["datasets"].items()},
        "Dataset revisions differ from the frozen protocol",
    )
    evaluation_questions = read_jsonl(root / "inputs" / "evaluation_questions.jsonl")
    evaluation_counts = Counter(row["dataset_id"] for row in evaluation_questions)
    _require(
        evaluation_counts
        == {"commonsense_qa": 500, "arc_challenge": 500, "openbookqa": 500},
        f"Final factual cohorts are incomplete: {evaluation_counts}",
    )
    for filename, expected in dict(source.get("artifacts", {})).items():
        _authenticated(root / "inputs" / filename, str(expected))
    evaluation_question_keys = {
        f"{row['dataset_id']}:{row['source_split']}:{row['source_example_id']}"
        for row in evaluation_questions
    }

    model_audits = {}
    for model_key in campaign.MODEL_KEYS:
        specification = campaign.model_spec(config, model_key)
        smoke = read_json(root / "model_smoke" / model_key / "COMPLETE.json")
        _require(smoke["model_id"] == specification["model_id"], "Model ID mismatch")
        _require(smoke["revision"] == specification["revision"], "Model revision mismatch")
        manifest_root = root / "manifests" / model_key
        n1_prune = read_jsonl(manifest_root / "n1_mechanism" / "prune.jsonl")
        n2_prune_path = manifest_root / "n2_selective" / "prune.jsonl"
        n1_preserve = read_jsonl(manifest_root / "n1_general" / "preserve.jsonl")
        n2_preserve = read_jsonl(manifest_root / "n2_selective" / "preserve.jsonl")
        _audit_n1_rows(n1_prune, model_key)
        _require(
            (manifest_root / "n1_mechanism" / "prune.jsonl").read_bytes()
            == n2_prune_path.read_bytes(),
            "N1/N2 pruning manifests differ",
        )
        n1_composition = Counter(row["preservation_family"] for row in n1_preserve)
        _require(
            len(n1_preserve) == 512
            and n1_composition == {"alpaca": 256, "neutral_factual": 256},
            f"N1 preservation is malformed: {n1_composition}",
        )
        _require(
            (manifest_root / "n1_mechanism" / "preserve.jsonl").read_bytes()
            == (manifest_root / "n1_general" / "preserve.jsonl").read_bytes(),
            f"N1 mechanism preservation differs from the frozen general bank for {model_key}",
        )
        factual_preservation = [
            row for row in n1_preserve if row["preservation_family"] == "neutral_factual"
        ]
        _require(
            Counter(row["dataset"] for row in factual_preservation)
            == {"commonsense_qa": 128, "arc_challenge": 128},
            f"N1 factual preservation is not 128/128 for {model_key}",
        )
        source_rows = n2_preserve[512:]
        _require(len(n2_preserve) == 1024, "N2 preservation count is not 1024")
        _require(
            (manifest_root / "n2_selective" / "preserve.jsonl").read_bytes().startswith(
                (manifest_root / "n1_general" / "preserve.jsonl").read_bytes()
            ),
            "N2 does not begin with N1's byte-identical general bank",
        )
        _audit_source_rows(source_rows, model_key)
        n1_all_questions = {
            str(row["question_key"])
            for seed in (5, 17, 29)
            for row in read_jsonl(manifest_root / f"n1_seed{seed}" / "prune.jsonl")
        }
        factual_questions = {str(row["question_key"]) for row in factual_preservation}
        source_questions = {str(row["question_key"]) for row in source_rows}
        steering_questions = {
            f"{row['dataset_id']}:{row['source_split']}:{row['source_example_id']}"
            for row in read_jsonl(root / "inputs" / "steering_questions" / f"{model_key}.jsonl")
        }
        named_sets = {
            "pruning": n1_all_questions,
            "factual_preservation": factual_questions,
            "source_preservation": source_questions,
            "steering": steering_questions,
            "evaluation": evaluation_question_keys,
        }
        names = list(named_sets)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                overlap = named_sets[left_name] & named_sets[right_name]
                _require(
                    not overlap,
                    f"Question-ID overlap for {model_key}: {left_name}/{right_name}",
                )
        score_roles = {
            score_id: read_json(root / "scores" / model_key / score_id / "metadata.json")[
                "aggregation"
            ]
            for score_id in campaign.SCORE_SPECS
        }
        _require(
            all(
                aggregation
                == (
                    "mean_absolute_per_example_weight_times_gradient"
                    if campaign.SCORE_SPECS[score_id][1] == "preserve"
                    else "signed_mean_negative_weight_times_gradient"
                )
                for score_id, aggregation in score_roles.items()
            ),
            "A score cache uses the wrong aggregation",
        )
        masks: dict[str, str] = {}
        coordinates: dict[str, set[tuple[str, int]]] = {}
        mask_metadata: dict[str, Mapping[str, Any]] = {}
        for mask_id in campaign.MASK_SPECS:
            mask_root = root / "masks" / model_key / mask_id
            coordinates[mask_id], mask_metadata[mask_id] = _audit_mask(
                mask_root, expected_count=1000, expected_p=0.00005
            )
            complete = read_json(mask_root / "COMPLETE.json")
            masks[mask_id] = complete["indices_sha256"]
        _require(
            mask_metadata["n1_mechanism"]["prune_score_id"] == "n1_seed5_prune"
            and mask_metadata["n2_selective"]["prune_score_id"] == "n1_seed5_prune"
            and mask_metadata["n1_mechanism"]["prune_metadata_sha256"]
            == mask_metadata["n2_selective"]["prune_metadata_sha256"],
            f"N1/N2 do not share one immutable pruning-score cache for {model_key}",
        )
        for size in (250, 500, 1000):
            mask_id = f"n1_prefix_{size}"
            coordinates[mask_id], _metadata = _audit_mask(
                root / "masks" / model_key / mask_id,
                expected_count=size,
                expected_p=0.00005,
            )
        _require(
            coordinates["n1_prefix_250"]
            < coordinates["n1_prefix_500"]
            < coordinates["n1_prefix_1000"]
            and coordinates["n1_prefix_1000"] == coordinates["n1_mechanism"],
            f"N1 prefix nesting contract failed for {model_key}",
        )
        for random_id, target_id in (("random_n1", "n1_mechanism"), ("random_n2", "n2_selective")):
            random_root = root / "masks" / model_key / random_id
            random_coordinates, random_metadata = _audit_mask(
                random_root, expected_count=1000, expected_p=0.00005
            )
            _require(
                not (random_coordinates & coordinates[target_id])
                and random_metadata["counts_by_module"]
                == mask_metadata[target_id]["counts_by_module"]
                and random_metadata["matched_to"] == target_id
                and random_metadata["matched_to_indices_sha256"] == masks[target_id],
                f"Random mask does not exactly match {target_id} for {model_key}",
            )
            masks[random_id] = read_json(random_root / "COMPLETE.json")["indices_sha256"]
        for state_id in campaign.PRIMARY_STATE_IDS:
            _require(
                (root / "states" / model_key / f"{state_id}.json").is_file(),
                f"Missing state registry {model_key}/{state_id}",
            )
        model_audits[model_key] = {
            "model_id": specification["model_id"],
            "revision": specification["revision"],
            "n1_pool": read_json(manifest_root / "MANIFESTS_COMPLETE.json")["n1_pool"],
            "score_roles": score_roles,
            "mask_hashes": masks,
        }
    evaluation_complete = read_json(root / "evaluations" / "results" / "COMPLETE.json")
    _require(evaluation_complete.get("status") == "complete", "Evaluations are incomplete")
    _require(
        set(evaluation_complete.get("states", [])) == set(campaign.PRIMARY_STATE_IDS),
        "Evaluation state coverage is incomplete",
    )
    raw_record_count = _audit_raw_evaluation_records(root)
    _require(
        raw_record_count == int(evaluation_complete.get("record_count", -1)),
        "Raw evaluation record census differs from the authenticated completion receipt",
    )
    evaluation_inputs = read_json(root / "evaluations" / "inputs" / "COMPLETE.json")
    expected_capabilities = _expected_capabilities(config)
    frozen_capabilities = set(evaluation_inputs.get("capability_names", []))
    normalized_frozen = {
        "TriviaQA" if name == "TriviaQA-Wiki" else name for name in frozen_capabilities
    }
    _require(
        normalized_frozen == expected_capabilities,
        f"Frozen capability suite differs from the protocol: {sorted(frozen_capabilities)}",
    )
    evalplus_complete = read_json(root / "evalplus" / "results" / "COMPLETE.json")
    _require(evalplus_complete.get("publication_count") == 24, "Code state/model coverage is incomplete")
    for model_key in campaign.MODEL_KEYS:
        for state_id in campaign.PRIMARY_STATE_IDS:
            code_receipt = read_json(
                root / "evalplus" / "results" / model_key / state_id / "COMPLETE.json"
            )
            _require(
                set(code_receipt.get("pass_at_1", {})) == {"HumanEval+", "MBPP+"}
                and int(code_receipt.get("task_count", 0)) > 0,
                f"Code benchmark coverage is incomplete for {model_key}/{state_id}",
            )
            _authenticated(
                root / "evalplus" / "results" / model_key / state_id / "results.jsonl",
                str(code_receipt["results_sha256"]),
            )
    report = read_json(root / "reports" / "COMPLETE.json")
    _require(report.get("bootstrap_replicates") == 2000, "Report bootstrap count changed")
    expected_csv = {
        "generalization_cells.csv",
        "generalization_template_families.csv",
        "generalization_macro.csv",
        "stress_tests.csv",
        "useful_assertion_cells.csv",
        "reliable_source_advantage.csv",
        "pruning_effect.csv",
        "native_tool_transfer.csv",
        "native_tool_advantage.csv",
        "general_capabilities.csv",
    }
    _require(set(report.get("csv_files", {})) == expected_csv, "Paper-ready CSV coverage is incomplete")
    for filename, expected in report["csv_files"].items():
        _authenticated(root / "reports" / filename, str(expected))
    _authenticated(root / "reports" / "paper_results.json", str(report["json_sha256"]))
    _authenticated(root / "reports" / "generalization_macro.tex", str(report["latex_sha256"]))
    expected_figures = {
        "generalization_movement.png",
        "generalization_movement.pdf",
        "reliable_source_advantage.png",
        "reliable_source_advantage.pdf",
    }
    figure_names = {Path(str(row["path"])).name for row in report.get("figures", [])}
    _require(figure_names == expected_figures, f"Paper-ready figure coverage is incomplete: {figure_names}")
    for row in report["figures"]:
        _authenticated(Path(str(row["path"])), str(row["sha256"]))
    paper_results = read_json(root / "reports" / "paper_results.json")
    reported_capabilities = {
        "TriviaQA" if row["benchmark"] == "TriviaQA-Wiki" else row["benchmark"]
        for row in paper_results.get("general_capabilities", [])
    }
    _require(
        reported_capabilities == expected_capabilities,
        f"Reported capability coverage is incomplete: {sorted(reported_capabilities)}",
    )
    weight = read_json(root / "weight_analysis" / "COMPLETE.json")
    _require(weight.get("cross_architecture_intersections") == 0, "Cross-architecture weights mixed")
    _require(set(weight.get("models", {})) == set(campaign.MODEL_KEYS), "Weight-analysis model coverage fails")
    expected_pairs = {
        "seed5_vs_seed17",
        "seed5_vs_seed29",
        "seed17_vs_seed29",
        "user_vs_all_reliable_source",
        "user_vs_false_source_only",
        "size250_vs_1000",
        "size500_vs_1000",
    }
    for model_key, analysis in weight["models"].items():
        _require(analysis.get("exact_nesting") is True, f"Weight nesting failed for {model_key}")
        _require(
            int(analysis.get("structural_null_replicates", 0)) == 10_000,
            f"Structural-null count changed for {model_key}",
        )
        overlaps = {row["pair_id"]: row for row in analysis.get("overlaps", [])}
        _require(set(overlaps) == expected_pairs, f"Weight-analysis pair coverage fails for {model_key}")
        for pair_id, row in overlaps.items():
            if pair_id.startswith("size"):
                _require("structural_null" not in row, f"Nested size pair has a null: {model_key}/{pair_id}")
            else:
                _require(
                    int(row.get("structural_null", {}).get("replicates", 0)) == 10_000,
                    f"Structural null is incomplete for {model_key}/{pair_id}",
                )
    receipt = {
        "status": "complete",
        "experiment": campaign.EXPERIMENT,
        "config_sha256": sha256_file(args.config),
        "source_freeze_sha256": sha256_file(root / "inputs" / "SOURCE_FREEZE_COMPLETE.json"),
        "evaluation_complete_sha256": sha256_file(
            root / "evaluations" / "results" / "COMPLETE.json"
        ),
        "evalplus_complete_sha256": sha256_file(
            root / "evalplus" / "results" / "COMPLETE.json"
        ),
        "report_complete_sha256": sha256_file(root / "reports" / "COMPLETE.json"),
        "weight_analysis_complete_sha256": sha256_file(
            root / "weight_analysis" / "COMPLETE.json"
        ),
        "models": model_audits,
        "factual_question_counts": dict(evaluation_counts),
        "state_ids": list(campaign.PRIMARY_STATE_IDS),
        "raw_evaluation_record_count": raw_record_count,
        "primary_state_mask_count_per_model": 4,
        "localization_mask_count_per_model": 2,
        "primary_masks_exactly_1000": True,
        "protection_fraction": 0.00005,
        "n1_n2_pruning_byte_identical": True,
        "n2_contains_bare_user_preservation": False,
    }
    atomic_json(root / "audit" / "COMPLETE.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    final_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
