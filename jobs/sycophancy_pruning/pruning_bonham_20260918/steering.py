#!/usr/bin/env python3
"""Frozen prompt-only paired MeanDiff baseline for Bonham."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import campaign
from core import (
    DEFAULT_CONFIG,
    Question,
    atomic_json,
    atomic_jsonl,
    construction_bias,
    designated_wrong,
    load_config,
    read_json,
    read_jsonl,
    render_messages,
    sha256_file,
    sha256_json,
)
from bonham_runtime.evaluation.schemas import StateSpec


ALPHAS = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)


class SteeringError(campaign.CampaignError):
    pass


def _question(row: Mapping[str, Any]) -> Question:
    return Question(
        dataset_id=str(row["dataset_id"]),
        source_example_id=str(row["source_example_id"]),
        source_split=str(row["source_split"]),
        question=str(row["question"]),
        labels=tuple(str(value) for value in row["labels"]),
        answers=tuple(str(value) for value in row["answers"]),
        gold=str(row["gold"]),
    )


def prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    root = Path(args.result_root)
    frozen = read_jsonl(root / "inputs" / "steering_questions" / f"{args.model_key}.jsonl")
    output = []
    for row in frozen:
        question = _question(row)
        wrong = designated_wrong(question)
        sentence = construction_bias(config, question, "incorrect_suggestion", 0, wrong)
        neutral = render_messages(
            question,
            bias_sentence=None,
            turn_format="single_turn",
            assistant_answer=None,
            answer_instruction=str(config["answer_instruction"]),
        )
        harmful = render_messages(
            question,
            bias_sentence=sentence,
            turn_format="single_turn",
            assistant_answer=None,
            answer_instruction=str(config["answer_instruction"]),
        )
        if any(message.get("role") == "assistant" for message in (*neutral, *harmful)):
            raise SteeringError("Prompt-only MeanDiff cannot contain assistant answers")
        output.append(
            {
                "model_key": args.model_key,
                "dataset_id": question.dataset_id,
                "source_example_id": question.source_example_id,
                "partition": str(row["steering_split"]),
                "labels": list(question.labels),
                "gold": question.gold,
                "wrong": wrong,
                "messages_by_condition": {"neutral": list(neutral), "harmful": list(harmful)},
                "messages_sha256_by_condition": {
                    "neutral": sha256_json(neutral),
                    "harmful": sha256_json(harmful),
                },
            }
        )
    counts = {
        dataset_id: {
            partition: sum(
                row["dataset_id"] == dataset_id and row["partition"] == partition
                for row in output
            )
            for partition in ("fit", "development")
        }
        for dataset_id in ("commonsense_qa", "arc_challenge")
    }
    if any(value != {"fit": 100, "development": 50} for value in counts.values()):
        raise SteeringError(f"MeanDiff split is not 100 fit / 50 development: {counts}")
    path = root / "steering" / args.model_key / "inputs" / "paired_prompts.jsonl"
    atomic_jsonl(path, output)
    receipt = {
        "status": "complete",
        "model_key": args.model_key,
        "counts": counts,
        "same_question_pairs": True,
        "assistant_answers_in_fit": 0,
        "direction_definition": "mean(harmful_prompt_final_token-neutral_prompt_final_token)",
        "manifest_sha256": sha256_file(path),
    }
    atomic_json(path.with_suffix(".COMPLETE.json"), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _rows(root: Path, model_key: str, partition: str) -> list[Mapping[str, Any]]:
    path = root / "steering" / model_key / "inputs" / "paired_prompts.jsonl"
    receipt = read_json(path.with_suffix(".COMPLETE.json"))
    if receipt.get("manifest_sha256") != sha256_file(path):
        raise SteeringError("MeanDiff paired-prompt manifest changed")
    rows = [row for row in read_jsonl(path) if row["partition"] == partition]
    expected = 200 if partition == "fit" else 100
    if len(rows) != expected:
        raise SteeringError(f"MeanDiff {partition} expected {expected} rows, found {len(rows)}")
    return rows


def extract(args: argparse.Namespace) -> None:
    from bonham_runtime.interventions.activations import extract_prompt_state

    config = load_config(args.config)
    root = Path(args.result_root)
    specification = campaign.model_spec(config, args.model_key)
    layers = tuple(int(value) for value in specification["steering_layers"])
    rows = _rows(root, args.model_key, "fit")
    model, tokenizer = campaign._load_model(
        campaign.model_snapshot(args.hf_cache, specification)
    )
    neutral_states = []
    harmful_states = []
    prompt_audit = []
    for index, row in enumerate(rows, 1):
        extracted = {}
        for condition in ("neutral", "harmful"):
            messages = tuple(dict(value) for value in row["messages_by_condition"][condition])
            if sha256_json(messages) != row["messages_sha256_by_condition"][condition]:
                raise SteeringError("MeanDiff rendered prompt changed before extraction")
            state = extract_prompt_state(
                model,
                tokenizer,
                messages,
                choices=tuple(row["labels"]),
                residual_layers=layers,
            )
            extracted[condition] = np.stack(
                [np.asarray(state.hidden_by_layer[layer], dtype=np.float32) for layer in layers]
            )
            prompt_audit.append(
                {
                    "dataset_id": row["dataset_id"],
                    "source_example_id": row["source_example_id"],
                    "condition": condition,
                    "messages_sha256": row["messages_sha256_by_condition"][condition],
                    "prompt_token_count": state.prompt_token_count,
                    "final_token_id": state.final_token_id,
                    "final_token_text": state.final_token_text,
                }
            )
        neutral_states.append(extracted["neutral"])
        harmful_states.append(extracted["harmful"])
        if index % 20 == 0:
            print(f"MeanDiff extract {args.model_key}: {index}/{len(rows)}", flush=True)
    neutral_array = np.stack(neutral_states)
    harmful_array = np.stack(harmful_states)
    directions = np.asarray((harmful_array - neutral_array).mean(axis=0), dtype=np.float32)
    if directions.shape[0] != len(layers) or not np.isfinite(directions).all():
        raise SteeringError("MeanDiff direction array is malformed")
    output = root / "steering" / args.model_key / "fit" / "directions.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        layers=np.asarray(layers, dtype=np.int32),
        directions=directions,
        neutral_states=neutral_array,
        harmful_states=harmful_array,
    )
    os.replace(temporary, output)
    receipt = {
        "status": "complete",
        "model_key": args.model_key,
        "model_id": specification["model_id"],
        "model_revision": specification["revision"],
        "layers": list(layers),
        "pair_count": len(rows),
        "direction_definition": "mean(harmful-neutral) at final rendered prompt token",
        "prompt_audit_sha256": sha256_json(prompt_audit),
        "directions_sha256": sha256_file(output),
    }
    atomic_json(output.with_suffix(".COMPLETE.json"), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _margin(scores: Mapping[str, float], gold: str, wrong: str) -> float:
    return float(scores[gold]) - float(scores[wrong])


def develop(args: argparse.Namespace) -> None:
    from bonham_runtime.interventions.activations import score_with_residual_additions

    config = load_config(args.config)
    root = Path(args.result_root)
    specification = campaign.model_spec(config, args.model_key)
    layers_expected = tuple(int(value) for value in specification["steering_layers"])
    fit_path = root / "steering" / args.model_key / "fit" / "directions.npz"
    fit_receipt = read_json(fit_path.with_suffix(".COMPLETE.json"))
    if fit_receipt.get("directions_sha256") != sha256_file(fit_path):
        raise SteeringError("MeanDiff direction archive changed")
    with np.load(fit_path, allow_pickle=False) as archive:
        layers = tuple(int(value) for value in archive["layers"].tolist())
        directions = np.asarray(archive["directions"], dtype=np.float32)
    if layers != layers_expected:
        raise SteeringError("MeanDiff layer grid differs from the frozen model grid")
    rows = _rows(root, args.model_key, "development")
    model, tokenizer = campaign._load_model(
        campaign.model_snapshot(args.hf_cache, specification)
    )
    accumulators = {
        (layer, alpha): {"movement": [], "neutral_margin": [], "neutral_correct": 0}
        for layer in layers
        for alpha in ALPHAS
    }
    for item_index, row in enumerate(rows, 1):
        gold, wrong = str(row["gold"]), str(row["wrong"])
        choices = tuple(row["labels"])
        neutral_messages = tuple(row["messages_by_condition"]["neutral"])
        harmful_messages = tuple(row["messages_by_condition"]["harmful"])
        for layer_index, layer in enumerate(layers):
            additions = np.stack([alpha * directions[layer_index] for alpha in ALPHAS])
            neutral_probabilities, neutral_scores = score_with_residual_additions(
                model,
                tokenizer,
                neutral_messages,
                choices=choices,
                residual_layer=layer,
                addition_vectors=additions,
                max_batch_size=len(ALPHAS),
            )
            _harmful_probabilities, harmful_scores = score_with_residual_additions(
                model,
                tokenizer,
                harmful_messages,
                choices=choices,
                residual_layer=layer,
                addition_vectors=additions,
                max_batch_size=len(ALPHAS),
            )
            for alpha_index, alpha in enumerate(ALPHAS):
                neutral_margin = _margin(neutral_scores[alpha_index], gold, wrong)
                harmful_margin = _margin(harmful_scores[alpha_index], gold, wrong)
                values = accumulators[(layer, alpha)]
                values["neutral_margin"].append(neutral_margin)
                values["movement"].append(neutral_margin - harmful_margin)
                values["neutral_correct"] += (
                    max(
                        neutral_probabilities[alpha_index],
                        key=neutral_probabilities[alpha_index].get,
                    )
                    == gold
                )
        if item_index % 10 == 0:
            print(f"MeanDiff develop {args.model_key}: {item_index}/{len(rows)}", flush=True)
    metrics = []
    for layer in layers:
        baseline = accumulators[(layer, 0.0)]
        baseline_movement = float(np.mean(baseline["movement"]))
        baseline_margin = float(np.mean(baseline["neutral_margin"]))
        for alpha in ALPHAS:
            values = accumulators[(layer, alpha)]
            movement = float(np.mean(values["movement"]))
            margin = float(np.mean(values["neutral_margin"]))
            reduction = baseline_movement - movement
            margin_delta = margin - baseline_margin
            metrics.append(
                {
                    "model_key": args.model_key,
                    "layer": layer,
                    "alpha": alpha,
                    "denominator": len(rows),
                    "targeted_pressure_movement": movement,
                    "targeted_pressure_reduction_vs_alpha0": reduction,
                    "neutral_gold_margin": margin,
                    "neutral_gold_margin_delta_vs_alpha0": margin_delta,
                    "neutral_accuracy": values["neutral_correct"] / len(rows),
                    "selection_score": reduction + margin_delta,
                }
            )
    candidates = [row for row in metrics if float(row["alpha"]) != 0.0]
    candidates.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["targeted_pressure_reduction_vs_alpha0"]),
            -float(row["neutral_accuracy"]),
            int(row["layer"]),
            float(row["alpha"]),
        )
    )
    if len(candidates) != len(layers) * 8:
        raise SteeringError("MeanDiff development grid is incomplete")
    selected = dict(candidates[0])
    output_root = root / "steering" / args.model_key / "frozen"
    metrics_path = output_root / "development_metrics.jsonl"
    atomic_jsonl(metrics_path, metrics)
    direction = directions[layers.index(int(selected["layer"]))]
    direction_path = output_root / "direction.npy"
    output_root.mkdir(parents=True, exist_ok=True)
    if direction_path.exists():
        raise FileExistsError(direction_path)
    with direction_path.open("xb") as handle:
        np.save(handle, direction, allow_pickle=False)
    counts = read_json(root / "model_smoke" / args.model_key / "COMPLETE.json")
    state = StateSpec(
        state_id="prompt_only_meandiff",
        display_name="Prompt-only paired MeanDiff steering",
        intervention_kind="activation_steering",
        model_id=str(specification["model_id"]),
        model_revision=str(specification["revision"]),
        tokenizer_revision=str(specification["revision"]),
        parameters_set_to_zero=0,
        total_model_parameters=int(counts["total_model_parameters"]),
        eligible_pruning_parameters=int(counts["eligible_pruning_parameters"]),
        artifact_sha256={"direction": sha256_file(direction_path)},
        steering_layer=int(selected["layer"]),
        steering_alpha=float(selected["alpha"]),
        metadata={
            "experiment": campaign.EXPERIMENT,
            "direction_path": str(direction_path.resolve()),
            "direction_definition": "mean(harmful-neutral) before any assistant answer",
            "application": "initial_prompt_pass_only",
            "candidate_layers": list(layers),
            "alpha_grid": list(ALPHAS),
            "selection_partition": "development_only",
            "selection_rule": (
                "maximize pressure reduction plus neutral gold-margin preservation "
                "over nonzero layer/alpha candidates"
            ),
            "development_metrics_sha256": sha256_file(metrics_path),
            "selected_metrics": selected,
        },
    )
    state_path = campaign._state_path(root, args.model_key, "prompt_only_meandiff")
    campaign._write_state(state_path, state)
    receipt = {
        "status": "complete",
        "model_key": args.model_key,
        "selected": selected,
        "candidate_count": len(candidates),
        "state_sha256": sha256_file(state_path),
        "direction_sha256": sha256_file(direction_path),
        "development_metrics_sha256": sha256_file(metrics_path),
        "nonzero_alpha_required": True,
    }
    atomic_json(output_root / "COMPLETE.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("extract", extract), ("develop", develop)):
        command = subparsers.add_parser(name)
        command.add_argument("--result-root", type=Path, required=True)
        command.add_argument("--model-key", choices=campaign.MODEL_KEYS, required=True)
        if name in {"extract", "develop"}:
            command.add_argument("--hf-cache", type=Path, required=True)
        command.set_defaults(func=function)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
