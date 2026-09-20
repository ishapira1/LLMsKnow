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
import evaluations
import prepare_capability_sources
import reporting
from core import (
    DEFAULT_CONFIG,
    ELIGIBLE_PROJECTIONS,
    REASONING_BACKED_REGISTRY,
    atomic_json,
    canonical_shard_directories,
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
SCORE_EXPECTED_COUNTS = {
    "n1_seed5_prune": 512,
    "n1_seed17_prune": 512,
    "n1_seed29_prune": 512,
    "general_preserve": 512,
    "selective_preserve": 1024,
    "source_all_prune": 512,
    "source_false_prune": 256,
}

BEHAVIORAL_EVALUATION_FAMILIES = {
    "generalization",
    "useful_assertions",
    "source_attribution",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _authenticated(path: Path, expected: str) -> None:
    _require(path.is_file(), f"Missing artifact: {path}")
    _require(sha256_file(path) == expected, f"Changed artifact: {path}")


def _audit_raw_probability_payload(
    row: Mapping[str, Any],
    *,
    family: str,
    location: str,
) -> None:
    """Fail closed on malformed or internally inconsistent option scores."""

    raw = row.get("forced_choice_probabilities")
    _require(
        isinstance(raw, Mapping),
        f"Forced-choice probabilities are not a mapping: {location}",
    )
    probabilities = {}
    for label, value in raw.items():
        try:
            probabilities[str(label)] = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise AuditError(
                f"Forced-choice probability is not numeric: {location}"
            ) from error
    _require(
        probabilities == row.get("choice_probabilities"),
        f"Forced-choice probabilities differ from the scored probabilities: {location}",
    )
    _require(
        row.get("generated_answer") == row.get("raw_output"),
        f"Generated-answer field differs from the authenticated raw output: {location}",
    )
    if family in BEHAVIORAL_EVALUATION_FAMILIES:
        _require(
            len(probabilities) >= 2,
            f"Behavioral record lacks forced-choice probabilities: {location}",
        )
    if not probabilities:
        return
    _require(
        len(probabilities) >= 2
        and all(label.strip() for label in probabilities)
        and all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in probabilities.values()
        )
        and math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6),
        f"Forced-choice probabilities are invalid or not normalized: {location}",
    )
    gold = str(row.get("gold_label") or "").strip()
    if family in BEHAVIORAL_EVALUATION_FAMILIES:
        _require(
            gold in probabilities,
            f"Behavioral record's gold label is absent from forced-choice probabilities: {location}",
        )


def _audit_score_cache(
    root: Path,
    *,
    model_key: str,
    specification: Mapping[str, Any],
    score_id: str,
) -> Mapping[str, Any]:
    score_root = root / "scores" / model_key / score_id
    complete = read_json(score_root / "COMPLETE.json")
    identity = read_json(score_root / "identity.json")
    metadata = read_json(score_root / "metadata.json")
    _require(complete.get("status") == "complete", f"Score cache is incomplete: {score_root}")
    _authenticated(score_root / "identity.json", str(complete["identity_sha256"]))
    _authenticated(score_root / "metadata.json", str(complete["metadata_sha256"]))
    _require(
        metadata.get("identity_sha256") == complete["identity_sha256"],
        f"Score identity chain changed: {score_root}",
    )
    manifest, expected_role, _seed = campaign._score_manifest(root, model_key, score_id)
    expected_aggregation = (
        "mean_absolute_per_example_weight_times_gradient"
        if expected_role == "preserve"
        else "signed_mean_negative_weight_times_gradient"
    )
    expected_count = int(SCORE_EXPECTED_COUNTS[score_id])
    manifest_count = sum(
        1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    _require(
        identity.get("experiment") == campaign.EXPERIMENT
        and identity.get("model_key") == model_key
        and identity.get("model_id") == specification["model_id"]
        and identity.get("model_revision") == specification["revision"]
        and identity.get("score_id") == score_id
        and identity.get("role") == expected_role,
        f"Score identity differs from the frozen protocol: {score_root}",
    )
    _require(
        Path(str(identity.get("manifest", ""))).resolve() == manifest.resolve()
        and identity.get("manifest_sha256") == sha256_file(manifest)
        and int(identity.get("num_examples", -1)) == expected_count
        and manifest_count == expected_count,
        f"Score manifest identity/count changed: {score_root}",
    )
    _require(
        identity.get("aggregation") == expected_aggregation
        and identity.get("attribution") == "delta_i=-w_i*dL_dw_i"
        and identity.get("loss") == "completion_nll"
        and identity.get("precision") == "fp32_accumulation"
        and tuple(identity.get("eligible_projections", ())) == ELIGIBLE_PROJECTIONS
        and identity.get("implementation_sha256") == campaign._score_implementation_sha256()
        and identity.get("implementation_scope")
        == "attribution_code_and_vendored_scoring_dependencies",
        f"Score definition differs from the frozen protocol: {score_root}",
    )
    for key, value in identity.items():
        _require(metadata.get(key) == value, f"Score metadata changed identity field {key}: {score_root}")
    tensors = dict(metadata.get("tensors", {}))
    tensor_hashes = {str(key): str(value) for key, value in dict(complete.get("tensor_hashes", {})).items()}
    _require(
        tensors
        and int(complete.get("tensor_count", -1)) == len(tensors)
        and set(tensor_hashes) == set(tensors),
        f"Score tensor inventory is incomplete: {score_root}",
    )
    eligible_numel = 0
    for name, row in tensors.items():
        filename = str(row.get("file", ""))
        shape = tuple(int(value) for value in row.get("shape", ()))
        numel = int(row.get("numel", -1))
        expected_hash = str(row.get("sha256", ""))
        _require(
            filename
            and Path(filename).name == filename
            and str(name).rsplit(".", 1)[-1] in ELIGIBLE_PROJECTIONS
            and str(row.get("projection")) == str(name).rsplit(".", 1)[-1]
            and shape
            and math.prod(shape) == numel
            and numel > 0
            and int(row.get("block", -1)) >= 0
            and tensor_hashes[str(name)] == expected_hash,
            f"Score tensor metadata is malformed: {score_root}/{name}",
        )
        _authenticated(score_root / filename, expected_hash)
        eligible_numel += numel
    _require(
        int(metadata.get("eligible_numel", -1)) == eligible_numel,
        f"Score eligible-parameter count changed: {score_root}",
    )
    return metadata


def _audit_capability_sources(root: Path) -> str:
    source_root = root / "sources" / "capabilities"
    complete_path = source_root / "COMPLETE.json"
    complete = read_json(complete_path)
    _require(
        complete.get("status") == "complete"
        and complete.get("experiment") == campaign.EXPERIMENT,
        "Capability-source freeze is incomplete or belongs to another experiment",
    )
    sources = dict(complete.get("sources", {}))
    _require(
        set(sources) == set(prepare_capability_sources.SPECS),
        "Capability-source registry differs from the frozen protocol",
    )
    for name, (repo_id, subset, split, revision) in prepare_capability_sources.SPECS.items():
        row = dict(sources[name])
        population = int(row.get("population_count", -1))
        selected = int(row.get("selected_count", -1))
        _require(
            row.get("repo_id") == repo_id
            and row.get("config") == subset
            and row.get("split") == split
            and row.get("revision") == revision
            and row.get("selection") == "outcome-independent SHA-256 rank, cap 500"
            and population > 0
            and selected == min(500, population),
            f"Capability-source protocol changed for {name}",
        )
        _authenticated(Path(str(row["path"])), str(row["sha256"]))
        if name in {"winogrande", "triviaqa_wiki"}:
            _require(int(row.get("shot_count", -1)) == 5, f"Capability demos changed for {name}")
            _authenticated(
                Path(str(row["demonstrations_path"])),
                str(row["demonstrations_sha256"]),
            )
    return sha256_file(complete_path)


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


def _audit_n1_rows(
    rows: list[Mapping[str, Any]],
    model_key: str,
    *,
    balance_amendment: str | None = None,
) -> None:
    _require(len(rows) == 512, f"N1 pruning count is not 512 for {model_key}")
    _require(
        len({str(row["question_key"]) for row in rows}) == 512,
        f"N1 questions repeat for {model_key}",
    )
    cells = Counter(
        (row["dataset"], row["turn_format"], row["bias_type"], row["template_id"])
        for row in rows
    )
    if balance_amendment == campaign.GEMMA_BALANCED_AMENDMENT_ID:
        _require(model_key == "gemma4_12b", "Gemma balance amendment used by another model")
        bias_templates = Counter(
            (row["bias_type"], campaign._n1_template_index(row["template_id"]))
            for row in rows
        )
        _require(
            len(bias_templates) == 8 and set(bias_templates.values()) == {64},
            "Gemma amended N1 bias-template marginals are not exact",
        )
    else:
        _require(balance_amendment is None, f"Unknown N1 balance amendment: {balance_amendment}")
        _require(
            len(cells) == 32 and set(cells.values()) == {16},
            f"N1 is unbalanced for {model_key}",
        )
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
        _require(
            row.get("qualification_choice_source") == "candidate_renormalized_argmax",
            f"N1 qualification choice source changed: {model_key}",
        )
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
            and row.get("qualification_choice_source")
            == "candidate_renormalized_argmax"
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


def _audit_useful_matched_inputs(root: Path) -> Mapping[str, Any]:
    audits = {}
    for model_key in campaign.MODEL_KEYS:
        family_root = root / "evaluations" / "inputs" / model_key / "useful_assertions"
        tasks = []
        for entry in read_jsonl(family_root / "index.jsonl"):
            shard = int(entry["shard"])
            shard_tasks, manifest_sha256 = evaluations.read_task_manifest(
                family_root / f"shard_{shard:04d}.jsonl"
            )
            _require(
                manifest_sha256 == str(entry["sha256"]),
                f"Useful-assertion input hash changed for {model_key}/shard_{shard:04d}",
            )
            tasks.extend(shard_tasks)
        try:
            audits[model_key] = evaluations.validate_useful_matched_design(
                tasks, model_key
            )
        except evaluations.EvaluationError as error:
            raise AuditError(str(error)) from error
    return audits


def _audit_source_attribution_inputs(root: Path) -> Mapping[str, Any]:
    campaign_receipt_path = (
        root / "evaluations" / "inputs" / "SOURCE_ATTRIBUTION_COMPLETE.json"
    )
    campaign_receipt = read_json(campaign_receipt_path)
    _require(
        campaign_receipt.get("status") == "complete"
        and int(campaign_receipt.get("source_form_count", 0)) == 13,
        "Source-attribution input freeze is incomplete",
    )
    audits = {}
    for model_key in campaign.MODEL_KEYS:
        useful_tasks = []
        useful_root = root / "evaluations" / "inputs" / model_key / "useful_assertions"
        for entry in read_jsonl(useful_root / "index.jsonl"):
            tasks, observed_hash = evaluations.read_task_manifest(
                useful_root / f"shard_{int(entry['shard']):04d}.jsonl"
            )
            _require(
                observed_hash == str(entry["sha256"]),
                f"Useful input changed for source matching: {model_key}",
            )
            useful_tasks.extend(tasks)

        source_tasks = []
        source_root = root / "evaluations" / "inputs" / model_key / "source_attribution"
        for entry in read_jsonl(source_root / "index.jsonl"):
            tasks, observed_hash = evaluations.read_task_manifest(
                source_root / f"shard_{int(entry['shard']):04d}.jsonl"
            )
            _require(
                observed_hash == str(entry["sha256"]),
                f"Source-attribution input changed: {model_key}",
            )
            source_tasks.extend(tasks)
        try:
            audits[model_key] = evaluations.validate_source_attribution_design(
                source_tasks, useful_tasks, model_key
            )
        except evaluations.EvaluationError as error:
            raise AuditError(str(error)) from error
        model_receipt_path = (
            root
            / "evaluations"
            / "inputs"
            / model_key
            / "SOURCE_ATTRIBUTION_COMPLETE.json"
        )
        expected = dict(campaign_receipt.get("models", {})).get(model_key, {})
        _authenticated(model_receipt_path, str(expected.get("receipt_sha256", "")))
    return audits


def _audit_raw_evaluation_records(root: Path) -> int:
    observed = 0
    reasoning_counts = Counter()
    reasoning_template_counts = Counter()
    for model_key in campaign.MODEL_KEYS:
        for state_id in campaign.PRIMARY_STATE_IDS:
            for family in evaluations.EVALUATION_FAMILIES:
                family_root = root / "evaluations" / "results" / model_key / state_id / family
                for directory in canonical_shard_directories(family_root):
                    path = directory / "records.jsonl"
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
                            _audit_raw_probability_payload(
                                row,
                                family=family,
                                location=f"{path}:{line_number}",
                            )
                            if row["prompt_regime"] == "reasoning_backed_pushback":
                                _require(
                                    row["bias_type"] == "incorrect_suggestion"
                                    and row["turn_format"] == "multi_turn"
                                    and row["template_family"] == "generic_justification_pressure",
                                    f"Reasoning-backed pushback labels changed: {path}:{line_number}",
                                )
                                key = (model_key, state_id, str(row["dataset_id"]))
                                reasoning_counts[key] += 1
                                reasoning_template_counts[(*key, str(row["template_id"]))] += 1
                            observed += 1
    expected_datasets = ("commonsense_qa", "arc_challenge", "openbookqa")
    for model_key in campaign.MODEL_KEYS:
        for state_id in campaign.PRIMARY_STATE_IDS:
            for dataset_id in expected_datasets:
                key = (model_key, state_id, dataset_id)
                _require(
                    reasoning_counts[key] == 500,
                    f"Reasoning-backed pushback coverage is incomplete: {key}",
                )
                template_counts = {
                    template_id: count
                    for (*prefix, template_id), count in reasoning_template_counts.items()
                    if tuple(prefix) == key
                }
                _require(
                    len(template_counts) == 4 and set(template_counts.values()) == {125},
                    f"Reasoning-backed pushback templates are unbalanced: {key}/{template_counts}",
                )
    return observed


def _audit_weight_model_analysis(model_key: str, analysis: Mapping[str, Any]) -> None:
    """Verify every preregistered within-model weight-analysis deliverable."""

    _require(analysis.get("exact_nesting") is True, f"Weight nesting failed for {model_key}")
    _require(
        int(analysis.get("structural_null_replicates", 0)) == 10_000,
        f"Structural-null count changed for {model_key}",
    )
    expected_pair_sizes = {
        "seed5_vs_seed17": (1000, 1000),
        "seed5_vs_seed29": (1000, 1000),
        "seed17_vs_seed29": (1000, 1000),
        "user_vs_all_reliable_source": (1000, 1000),
        "user_vs_false_source_only": (1000, 1000),
        "size250_vs_1000": (250, 1000),
        "size500_vs_1000": (500, 1000),
    }
    overlaps = {str(row["pair_id"]): row for row in analysis.get("overlaps", [])}
    _require(
        set(overlaps) == set(expected_pair_sizes),
        f"Weight-analysis pair coverage fails for {model_key}",
    )
    for pair_id, row in overlaps.items():
        left_count, right_count = expected_pair_sizes[pair_id]
        intersection = int(row.get("intersection_count", -1))
        union = int(row.get("union_count", -1))
        _require(
            int(row.get("left_count", -1)) == left_count
            and int(row.get("right_count", -1)) == right_count
            and 0 <= intersection <= min(left_count, right_count)
            and union == left_count + right_count - intersection
            and math.isclose(float(row.get("jaccard", -1.0)), intersection / union)
            and math.isclose(
                float(row.get("left_overlap_fraction", -1.0)),
                intersection / left_count,
            )
            and math.isclose(
                float(row.get("right_overlap_fraction", -1.0)),
                intersection / right_count,
            ),
            f"Weight-overlap statistics are malformed for {model_key}/{pair_id}",
        )
        if pair_id.startswith("size"):
            _require(
                row.get("analysis_role") == "sparsity_nesting_not_independent_stability"
                and "structural_null" not in row
                and intersection == left_count,
                f"Nested size analysis is malformed for {model_key}/{pair_id}",
            )
        else:
            structural_null = dict(row.get("structural_null", {}))
            expected_intersection = float(structural_null.get("expected_intersection", -1.0))
            enrichment = structural_null.get("enrichment")
            _require(
                row.get("analysis_role") == "non_nested_mask_overlap"
                and int(structural_null.get("replicates", 0)) == 10_000
                and int(structural_null.get("observed_intersection", -1)) == intersection
                and expected_intersection >= 0.0
                and (
                    (expected_intersection == 0.0 and enrichment is None)
                    or (
                        expected_intersection > 0.0
                        and math.isclose(float(enrichment), intersection / expected_intersection)
                    )
                ),
                f"Structural null is incomplete for {model_key}/{pair_id}",
            )

    expected_question_pairs = {
        "seed5_vs_seed17",
        "seed5_vs_seed29",
        "seed17_vs_seed29",
    }
    question_overlaps = {
        str(row["pair_id"]): row for row in analysis.get("question_set_overlaps", [])
    }
    _require(
        set(question_overlaps) == expected_question_pairs,
        f"Seed question-set overlap coverage fails for {model_key}",
    )
    for pair_id, row in question_overlaps.items():
        intersection = int(row.get("intersection_count", -1))
        union = int(row.get("union_count", -1))
        _require(
            int(row.get("left_count", -1)) == 512
            and int(row.get("right_count", -1)) == 512
            and 0 <= intersection <= 512
            and union == 1024 - intersection
            and math.isclose(float(row.get("jaccard", -1.0)), intersection / union),
            f"Seed question-set overlap is malformed for {model_key}/{pair_id}",
        )

    expected_composition_sizes = {
        "n1_mechanism": 1000,
        "n1_seed17": 1000,
        "n1_seed29": 1000,
        "n1_prefix_250": 250,
        "n1_prefix_500": 500,
        "n1_prefix_1000": 1000,
        "source_all": 1000,
        "source_false": 1000,
    }
    composition = dict(analysis.get("composition", {}))
    _require(
        set(composition) == set(expected_composition_sizes),
        f"Layer/projection composition coverage fails for {model_key}",
    )
    for mask_id, expected_count in expected_composition_sizes.items():
        rows = list(composition[mask_id])
        _require(
            rows
            and sum(int(row.get("count", 0)) for row in rows) == expected_count
            and all(
                int(row.get("layer", -1)) >= 0
                and str(row.get("projection", "")) in ELIGIBLE_PROJECTIONS
                and int(row.get("count", 0)) > 0
                for row in rows
            ),
            f"Layer/projection composition is malformed for {model_key}/{mask_id}",
        )


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
    _authenticated(
        Path(str(source["suite_source_bindings"])),
        str(source["suite_source_bindings_sha256"]),
    )
    capability_source_complete_sha256 = _audit_capability_sources(root)
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
        manifest_receipt = read_json(manifest_root / "MANIFESTS_COMPLETE.json")
        n1_prune = read_jsonl(manifest_root / "n1_mechanism" / "prune.jsonl")
        n2_prune_path = manifest_root / "n2_selective" / "prune.jsonl"
        n1_preserve = read_jsonl(manifest_root / "n1_general" / "preserve.jsonl")
        n2_preserve = read_jsonl(manifest_root / "n2_selective" / "preserve.jsonl")
        _audit_n1_rows(
            n1_prune,
            model_key,
            balance_amendment=manifest_receipt.get("balance_amendment"),
        )
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
        steering_rows = read_jsonl(
            root / "inputs" / "steering_questions" / f"{model_key}.jsonl"
        )
        _require(
            Counter(
                (str(row["dataset_id"]), str(row["steering_split"]))
                for row in steering_rows
            )
            == {
                ("commonsense_qa", "fit"): 100,
                ("commonsense_qa", "development"): 50,
                ("arc_challenge", "fit"): 100,
                ("arc_challenge", "development"): 50,
            },
            f"Steering cohort is not the full frozen 100/50 split for {model_key}",
        )
        steering_questions = {
            f"{row['dataset_id']}:{row['source_split']}:{row['source_example_id']}"
            for row in steering_rows
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
        score_metadata = {
            score_id: _audit_score_cache(
                root,
                model_key=model_key,
                specification=specification,
                score_id=score_id,
            )
            for score_id in campaign.SCORE_SPECS
        }
        score_roles = {
            score_id: metadata["aggregation"]
            for score_id, metadata in score_metadata.items()
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
                mask_metadata[mask_id]["prune_metadata_sha256"]
                == sha256_file(
                    root
                    / "scores"
                    / model_key
                    / str(mask_metadata[mask_id]["prune_score_id"])
                    / "metadata.json"
                )
                and mask_metadata[mask_id]["preserve_metadata_sha256"]
                == sha256_file(
                    root
                    / "scores"
                    / model_key
                    / str(mask_metadata[mask_id]["preserve_score_id"])
                    / "metadata.json"
                ),
                f"Mask is not bound to its authenticated score caches: {model_key}/{mask_id}",
            )
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
            magnitude_audit = dict(random_metadata.get("magnitude_match_audit", {}))
            _require(
                set(magnitude_audit) == set(random_metadata["counts_by_module"]),
                f"Random magnitude audit lacks modules for {model_key}/{random_id}",
            )
            for name, row in magnitude_audit.items():
                target_bins = [int(value) for value in row.get("target_bin_counts", ())]
                random_bins = [int(value) for value in row.get("random_bin_counts", ())]
                _require(
                    row.get("exact_bin_match") is True
                    and row.get("disjoint") is True
                    and len(target_bins) == 10
                    and target_bins == random_bins
                    and sum(target_bins) == int(random_metadata["counts_by_module"][name])
                    and int(row.get("numel", -1)) == int(random_metadata["counts_by_module"][name]),
                    f"Random magnitude-decile match fails for {model_key}/{random_id}/{name}",
                )
            masks[random_id] = read_json(random_root / "COMPLETE.json")["indices_sha256"]
        for state_id in campaign.PRIMARY_STATE_IDS:
            _require(
                (root / "states" / model_key / f"{state_id}.json").is_file(),
                f"Missing state registry {model_key}/{state_id}",
            )
        exact_supplement_sha256 = None
        exact_source_supplement_sha256 = None
        if model_key == "gemma4_12b":
            supplement_path = (
                root
                / "inputs"
                / "n1_screen_model_supplement_shards"
                / model_key
                / "COMPLETE"
            )
            supplement = read_json(supplement_path)
            _require(
                supplement.get("status") == "complete"
                and supplement.get("relaxes_quota") is False
                and supplement.get("supplement_id")
                == campaign.GEMMA_EXACT_SUPPLEMENT_ID
                and supplement.get("cell") == campaign.GEMMA_EXACT_SUPPLEMENT_CELL,
                "Gemma exact-quota supplement is missing or changed",
            )
            _require(
                manifest_receipt.get("balance_amendment")
                == campaign.GEMMA_BALANCED_AMENDMENT_ID,
                "Gemma did not use the authorized balanced-marginal amendment",
            )
            steering_reservation = dict(
                manifest_receipt.get("steering_reservation") or {}
            )
            reservation_datasets = dict(
                steering_reservation.get("datasets", {})
            )
            _require(
                steering_reservation.get("method")
                == "reserve_paired_questions_outside_feasibility_witness_v1"
                and int(steering_reservation.get("fit_per_dataset", -1)) == 100
                and int(steering_reservation.get("development_per_dataset", -1))
                == 50
                and int(
                    steering_reservation.get("feasibility_witness_seed", -1)
                )
                == 5
                and len(
                    str(
                        steering_reservation.get(
                            "feasibility_witness_question_hash", ""
                        )
                    )
                )
                == 64
                and set(reservation_datasets)
                == {"commonsense_qa", "arc_challenge"}
                and all(
                    int(row.get("reserved_count", -1)) == 150
                    and int(row.get("neutral_correct_count", -1))
                    + int(row.get("neutral_incorrect_or_invalid_count", -1))
                    == 150
                    for row in reservation_datasets.values()
                ),
                "Gemma balanced amendment lacks the authenticated full steering reservation",
            )
            exact_supplement_sha256 = sha256_file(supplement_path)
            source_supplement_path = (
                root
                / "inputs"
                / "source_screen_model_supplement_shards"
                / model_key
                / "COMPLETE"
            )
            source_supplement = read_json(source_supplement_path)
            source_candidate_path = (
                root
                / "inputs"
                / "source_screen_model_supplement_candidates"
                / f"{model_key}.jsonl"
            )
            _require(
                source_supplement.get("status") == "complete"
                and source_supplement.get("relaxes_quota") is False
                and source_supplement.get("relaxes_behavior_qualification") is False
                and source_supplement.get("supplement_id")
                == campaign.GEMMA_SOURCE_SUPPLEMENT_ID
                and source_supplement.get("cell")
                == campaign.GEMMA_SOURCE_SUPPLEMENT_CELL,
                "Gemma exact source-quota supplement is missing or changed",
            )
            _authenticated(
                source_candidate_path,
                str(source_supplement.get("candidate_sha256", "")),
            )
            exact_source_supplement_sha256 = sha256_file(source_supplement_path)
        model_audits[model_key] = {
            "model_id": specification["model_id"],
            "revision": specification["revision"],
            "n1_pool": read_json(manifest_root / "MANIFESTS_COMPLETE.json")["n1_pool"],
            "score_roles": score_roles,
            "mask_hashes": masks,
            "exact_n1_supplement_sha256": exact_supplement_sha256,
            "exact_source_supplement_sha256": exact_source_supplement_sha256,
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
    useful_matched_design = _audit_useful_matched_inputs(root)
    source_attribution_design = _audit_source_attribution_inputs(root)
    _require(
        evaluation_inputs.get("reasoning_backed_prompt_registry")
        == str(REASONING_BACKED_REGISTRY.resolve()),
        "Evaluation manifests point to the wrong reasoning-backed prompt registry",
    )
    _authenticated(
        REASONING_BACKED_REGISTRY,
        str(evaluation_inputs.get("reasoning_backed_prompt_registry_sha256", "")),
    )
    _require(
        evaluation_inputs.get("source_bindings_sha256")
        == source["suite_source_bindings_sha256"]
        and evaluation_inputs.get("external_utility_complete_sha256")
        == capability_source_complete_sha256,
        "Evaluation manifests are not bound to the authenticated source freezes",
    )
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
        "source_attribution_cells.csv",
        "source_form_advantage.csv",
        "source_family_advantage.csv",
        "source_overall_advantage.csv",
        "source_attribution_pruning_effect.csv",
        "source_family_pruning_effect.csv",
        "source_overall_pruning_effect.csv",
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
        "source_family_advantage.png",
        "source_family_advantage.pdf",
        "source_attribution_pruning_effect.png",
        "source_attribution_pruning_effect.pdf",
    }
    figure_names = {Path(str(row["path"])).name for row in report.get("figures", [])}
    _require(figure_names == expected_figures, f"Paper-ready figure coverage is incomplete: {figure_names}")
    for row in report["figures"]:
        _authenticated(Path(str(row["path"])), str(row["sha256"]))
    paper_results = read_json(root / "reports" / "paper_results.json")
    source_family_rows = list(paper_results.get("source_family_advantage", []))
    expected_source_families = {
        "quantified_reliability",
        "human_expertise",
        "vetted_reference",
        "independent_corroboration",
        "native_structured_tool",
    }
    _require(
        {str(row.get("source_family")) for row in source_family_rows}
        == expected_source_families,
        "Source-attribution report lacks one or more source families",
    )
    _require(
        {str(row.get("claim_type")) for row in source_family_rows}
        == {"suggest_c", "suggest_w", "doubt_c", "doubt_w"}
        and {str(row.get("turn_format")) for row in source_family_rows}
        == {"single_turn", "multi_turn"},
        "Source-attribution report lacks a truth-direction or turn-format cell",
    )
    source_overall_rows = list(paper_results.get("source_overall_advantage", []))
    _require(
        {str(row.get("model_key")) for row in source_overall_rows}
        == set(campaign.MODEL_KEYS)
        and {str(row.get("state_id")) for row in source_overall_rows}
        == set(campaign.PRIMARY_STATE_IDS)
        and {str(row.get("claim_type")) for row in source_overall_rows}
        == {"suggest_c", "suggest_w", "doubt_c", "doubt_w"}
        and {str(row.get("turn_format")) for row in source_overall_rows}
        == {"single_turn", "multi_turn"}
        and {str(row.get("metric")) for row in source_overall_rows}
        == set(reporting.USEFUL_METRICS),
        "Pooled matched-source advantage coverage is incomplete",
    )
    source_pruning_rows = list(
        paper_results.get("source_family_pruning_effect", [])
    )
    _require(
        {str(row.get("comparison_attribution")) for row in source_pruning_rows}
        == {"sampled_source", "bare_user"}
        and {str(row.get("state_id")) for row in source_pruning_rows}
        == set(campaign.PRIMARY_STATE_IDS).difference({"unpruned"}),
        "Source-attribution pruning report lacks matched user/source state effects",
    )
    source_overall_pruning_rows = list(
        paper_results.get("source_overall_pruning_effect", [])
    )
    _require(
        {str(row.get("model_key")) for row in source_overall_pruning_rows}
        == set(campaign.MODEL_KEYS)
        and {
            str(row.get("comparison_attribution"))
            for row in source_overall_pruning_rows
        }
        == {"sampled_source", "bare_user"}
        and {str(row.get("state_id")) for row in source_overall_pruning_rows}
        == set(campaign.PRIMARY_STATE_IDS).difference({"unpruned"})
        and {str(row.get("metric")) for row in source_overall_pruning_rows}
        == set(reporting.USEFUL_METRICS),
        "Pooled source/user pruning-effect coverage is incomplete",
    )
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
    for model_key, analysis in weight["models"].items():
        _audit_weight_model_analysis(model_key, analysis)
    receipt = {
        "status": "complete",
        "experiment": campaign.EXPERIMENT,
        "config_sha256": sha256_file(args.config),
        "source_freeze_sha256": sha256_file(root / "inputs" / "SOURCE_FREEZE_COMPLETE.json"),
        "capability_source_complete_sha256": capability_source_complete_sha256,
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
        "useful_matched_design": useful_matched_design,
        "source_attribution_design": source_attribution_design,
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
