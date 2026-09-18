#!/usr/bin/env python3
"""Fail-closed final audit for the complete Bonham publication bundle."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import campaign
from core import DEFAULT_CONFIG, atomic_json, load_config, read_json, read_jsonl, sha256_file


class AuditError(campaign.CampaignError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _authenticated(path: Path, expected: str) -> None:
    _require(path.is_file(), f"Missing artifact: {path}")
    _require(sha256_file(path) == expected, f"Changed artifact: {path}")


def final_audit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    root = Path(args.result_root)
    _require(float(config["selection"]["protection_fraction"]) == 0.00005, "p changed")
    _require(int(config["selection"]["mask_weight_count"]) == 1000, "n changed")
    source = read_json(root / "inputs" / "SOURCE_FREEZE_COMPLETE.json")
    _require(source.get("status") == "complete", "Source freeze is incomplete")
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
        _require(len(n1_prune) == 512, "N1 pruning count is not 512")
        _require(len({row["question_key"] for row in n1_prune}) == 512, "N1 questions repeat")
        n1_cells = Counter(
            (row["dataset"], row["turn_format"], row["bias_type"], row["template_id"])
            for row in n1_prune
        )
        _require(len(n1_cells) == 32 and set(n1_cells.values()) == {16}, "N1 is unbalanced")
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
        source_rows = n2_preserve[512:]
        source_cells = Counter((row["claim_type"], row["turn_format"]) for row in source_rows)
        _require(len(n2_preserve) == 1024, "N2 preservation count is not 1024")
        _require(
            (manifest_root / "n2_selective" / "preserve.jsonl").read_bytes().startswith(
                (manifest_root / "n1_general" / "preserve.jsonl").read_bytes()
            ),
            "N2 does not begin with N1's byte-identical general bank",
        )
        _require(len(source_cells) == 8 and set(source_cells.values()) == {64}, "N2 source cells fail")
        _require(
            all(row.get("claim_attribution") == "reliable_source" for row in source_rows),
            "N2 contains a bare-user preservation row",
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
        masks = {}
        for mask_id in ("n1_mechanism", "n2_selective", "random_n1", "random_n2"):
            mask_root = root / "masks" / model_key / mask_id
            metadata = read_json(mask_root / "metadata.json")
            _require(int(metadata["selected_count"]) == 1000, f"{mask_id} is not 1,000 weights")
            _require(float(metadata["p"]) == 0.00005, f"{mask_id} records the wrong p")
            complete = read_json(mask_root / "COMPLETE.json")
            _authenticated(mask_root / "indices.pt", complete["indices_sha256"])
            _authenticated(mask_root / "metadata.json", complete["metadata_sha256"])
            masks[mask_id] = complete["indices_sha256"]
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
    evalplus_complete = read_json(root / "evalplus" / "results" / "COMPLETE.json")
    _require(evalplus_complete.get("publication_count") == 24, "Code state/model coverage is incomplete")
    report = read_json(root / "reports" / "COMPLETE.json")
    _require(report.get("bootstrap_replicates") == 2000, "Report bootstrap count changed")
    weight = read_json(root / "weight_analysis" / "COMPLETE.json")
    _require(weight.get("cross_architecture_intersections") == 0, "Cross-architecture weights mixed")
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
        "mask_count": 6,
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
