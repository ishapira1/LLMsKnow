#!/usr/bin/env python3
"""Freeze and execute Bonham factual, updating, and capability evaluations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import campaign
from core import (
    BIAS_TYPES,
    DEFAULT_CONFIG,
    REASONING_BACKED_REGISTRY,
    TURN_FORMATS,
    Question,
    atomic_json,
    atomic_jsonl,
    balanced_template_assignments,
    canonical_json,
    designated_wrong,
    evaluation_bias,
    load_config,
    load_reasoning_backed_templates,
    option_ref,
    read_json,
    read_jsonl,
    render_messages,
    sha256_file,
    stable_hash,
)
from bonham_runtime.evaluation.artifacts import validate_complete_bundle
from bonham_runtime.evaluation.runner import EvaluationTask, read_task_manifest, run_evaluation_cell
from bonham_runtime.capabilities import build_capability_tasks, utility_evaluation_name


QUESTION_SHARD_SIZE = 25
EVALUATION_SHARD_LIMITS = {
    "generalization": 60,
    "useful_assertions": 120,
    "capabilities": 80,
}
PRIMARY_REGIMES = ("seen", "close_paraphrase", "naturalistic")
CAPABILITY_NAMES = {
    "BoolQ",
    "RTE",
    "HellaSwag",
    "WinoGrande",
    "OpenBookQA",
    "MMLU",
    "MMLU-Pro",
    "TriviaQA-Wiki",
    "SST-2 arbitrary-label ICL",
    "AG News arbitrary-label ICL",
    "HumanEval+",
    "MBPP+",
}


class EvaluationError(campaign.CampaignError):
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


def _question_key(question: Question) -> str:
    return f"{question.dataset_id}:{question.source_split}:{question.source_example_id}"


def _base_metadata(
    *,
    model_key: str,
    question: Question,
    question_axis: str,
    prompt_regime: str,
    bias_type: str,
    turn_format: str,
    template_family: str,
    template_id: str,
    wrong: str,
) -> Mapping[str, Any]:
    return {
        "model_key": model_key,
        "dataset_id": question.dataset_id,
        "question_id": question.source_example_id,
        "question_key": _question_key(question),
        "question_axis": question_axis,
        "prompt_regime": prompt_regime,
        "bias_type": bias_type,
        "turn_format": turn_format,
        "template_family": template_family,
        "template_id": template_id,
        "claim_truth": "false" if bias_type in BIAS_TYPES else "not_applicable",
        "claim_attribution": "bare_user" if bias_type in BIAS_TYPES else "none",
        "asserted_label": wrong if bias_type == "incorrect_suggestion" else None,
        "doubted_label": question.gold if bias_type == "doubt_correct" else None,
        "gold_label": question.gold,
        "neutral_label": None,
        "wrong_label": wrong,
        "retry_on_invalid": False,
    }


def _generalization_tasks_for_model(
    config: Mapping[str, Any], model_key: str, questions: Sequence[Question]
) -> list[EvaluationTask]:
    assignments = {}
    reasoning_assignments = {}
    reasoning_templates = load_reasoning_backed_templates()
    for dataset_id in ("commonsense_qa", "arc_challenge", "openbookqa"):
        ids = [row.source_example_id for row in questions if row.dataset_id == dataset_id]
        reasoning_assignments[dataset_id] = balanced_template_assignments(
            ids,
            4,
            f"{campaign.EXPERIMENT}:generalization:{dataset_id}:reasoning_backed_pushback",
        )
        for bias_type in BIAS_TYPES:
            for regime, count in (("seen", 4), ("close_paraphrase", 8), ("naturalistic", 24)):
                assignments[(dataset_id, bias_type, regime)] = (
                    balanced_template_assignments(
                        ids,
                        count,
                        f"{campaign.EXPERIMENT}:generalization:{dataset_id}:{bias_type}:{regime}",
                    )
                )
    tasks = []
    for question in questions:
        wrong = designated_wrong(question)
        question_axis = (
            "new_dataset" if question.dataset_id == "openbookqa" else "held_out_same_dataset"
        )
        neutral_metadata = _base_metadata(
            model_key=model_key,
            question=question,
            question_axis=question_axis,
            prompt_regime="neutral",
            bias_type="neutral",
            turn_format="single_turn",
            template_family="neutral",
            template_id="neutral",
            wrong=wrong,
        )
        tasks.append(
            EvaluationTask(
                example_id=f"generalization:neutral:{_question_key(question)}",
                evaluator_id="bonham_generalization_v1",
                display_name="Bonham sycophancy generalization",
                dataset_id=question.dataset_id,
                dataset_revision=str(config["datasets"][question.dataset_id]["revision"]),
                split=question.source_split,
                condition_id="generalization.neutral",
                messages=render_messages(
                    question,
                    bias_sentence=None,
                    turn_format="single_turn",
                    assistant_answer=None,
                    answer_instruction=str(config["answer_instruction"]),
                ),
                output_mode="mcq",
                max_new_tokens=8,
                choices=question.labels,
                gold_choice=question.gold,
                metadata=neutral_metadata,
            )
        )
        for bias_type in BIAS_TYPES:
            for turn_format in TURN_FORMATS:
                for regime in PRIMARY_REGIMES:
                    template_index = assignments[
                        (question.dataset_id, bias_type, regime)
                    ][question.source_example_id]
                    sentence, family = evaluation_bias(
                        config,
                        question,
                        bias_type=bias_type,
                        regime=regime,
                        template_index=template_index,
                        wrong=wrong,
                    )
                    template_id = f"{regime}.{bias_type}.{template_index:02d}"
                    metadata = _base_metadata(
                        model_key=model_key,
                        question=question,
                        question_axis=question_axis,
                        prompt_regime=regime,
                        bias_type=bias_type,
                        turn_format=turn_format,
                        template_family=family,
                        template_id=template_id,
                        wrong=wrong,
                    )
                    metadata = {**metadata, "bias_sentence": sentence, "primary_macro_eligible": True}
                    tasks.append(
                        EvaluationTask(
                            example_id=(
                                f"generalization:{regime}:{bias_type}:{turn_format}:"
                                f"{_question_key(question)}"
                            ),
                            evaluator_id="bonham_generalization_v1",
                            display_name="Bonham sycophancy generalization",
                            dataset_id=question.dataset_id,
                            dataset_revision=str(config["datasets"][question.dataset_id]["revision"]),
                            split=question.source_split,
                            condition_id=f"generalization.{regime}.{bias_type}.{turn_format}",
                            messages=render_messages(
                                question,
                                bias_sentence=sentence,
                                turn_format=turn_format,
                                assistant_answer=(
                                    question.gold if turn_format == "multi_turn" else None
                                ),
                                answer_instruction=str(config["answer_instruction"]),
                            ),
                            output_mode="mcq",
                            max_new_tokens=8,
                            choices=question.labels,
                            gold_choice=question.gold,
                            target_choice=(wrong if bias_type == "incorrect_suggestion" else None),
                            metadata=metadata,
                        )
                    )

        reasoning_index = reasoning_assignments[question.dataset_id][question.source_example_id]
        reasoning_template = reasoning_templates[reasoning_index]
        reasoning_sentence = reasoning_template.format(W=option_ref(question, wrong))
        reasoning_metadata = _base_metadata(
            model_key=model_key,
            question=question,
            question_axis=question_axis,
            prompt_regime="reasoning_backed_pushback",
            bias_type="incorrect_suggestion",
            turn_format="multi_turn",
            template_family="generic_justification_pressure",
            template_id=f"reasoning_backed_pushback.{reasoning_index}",
            wrong=wrong,
        )
        reasoning_metadata = {
            **reasoning_metadata,
            "bias_sentence": reasoning_sentence,
            "primary_macro_eligible": False,
            "stress_test": True,
            "reasoning_scope": "generic_justification_pressure",
        }
        tasks.append(
            EvaluationTask(
                example_id=f"stress:reasoning_backed_pushback:{_question_key(question)}",
                evaluator_id="bonham_generalization_v1",
                display_name="Bonham rationale-framed pushback stress test",
                dataset_id=question.dataset_id,
                dataset_revision=str(config["datasets"][question.dataset_id]["revision"]),
                split=question.source_split,
                condition_id="stress.reasoning_backed_pushback.multi_turn",
                messages=render_messages(
                    question,
                    bias_sentence=reasoning_sentence,
                    turn_format="multi_turn",
                    assistant_answer=question.gold,
                    answer_instruction=str(config["answer_instruction"]),
                ),
                output_mode="mcq",
                max_new_tokens=8,
                choices=question.labels,
                gold_choice=question.gold,
                target_choice=wrong,
                metadata=reasoning_metadata,
            )
        )

        stress_index = int(
            stable_hash(campaign.EXPERIMENT, "stress", question.dataset_id, question.source_example_id),
            16,
        ) % 4
        for stress_type, turn_format in (("desire", "single_turn"), ("pushback", "multi_turn")):
            template = str(config["stress_templates"][stress_type][stress_index])
            sentence = template.format(W=option_ref(question, wrong))
            metadata = _base_metadata(
                model_key=model_key,
                question=question,
                question_axis=question_axis,
                prompt_regime="out_of_family_stress",
                bias_type=stress_type,
                turn_format=turn_format,
                template_family=stress_type,
                template_id=f"stress.{stress_type}.{stress_index}",
                wrong=wrong,
            )
            metadata = {
                **metadata,
                "bias_sentence": sentence,
                "primary_macro_eligible": False,
                "stress_test": True,
            }
            tasks.append(
                EvaluationTask(
                    example_id=f"stress:{stress_type}:{_question_key(question)}",
                    evaluator_id="bonham_generalization_v1",
                    display_name="Bonham out-of-family stress test",
                    dataset_id=question.dataset_id,
                    dataset_revision=str(config["datasets"][question.dataset_id]["revision"]),
                    split=question.source_split,
                    condition_id=f"stress.{stress_type}.{turn_format}",
                    messages=render_messages(
                        question,
                        bias_sentence=sentence,
                        turn_format=turn_format,
                        assistant_answer=(question.gold if turn_format == "multi_turn" else None),
                        answer_instruction=str(config["answer_instruction"]),
                    ),
                    output_mode="mcq",
                    max_new_tokens=8,
                    choices=question.labels,
                    gold_choice=question.gold,
                    target_choice=wrong,
                    metadata=metadata,
                )
            )
    return tasks


def _useful_sentence(
    config: Mapping[str, Any],
    *,
    question: Question,
    claim_type: str,
    attribution: str,
    wrong: str,
) -> tuple[str, str | None, str | None, str]:
    if claim_type == "suggest_c":
        asserted, doubted, truth = question.gold, None, "true"
        template_key = f"{attribution}_suggestion"
        proposition = f"the answer is {option_ref(question, question.gold)}"
        option = option_ref(question, question.gold)
    elif claim_type == "suggest_w":
        asserted, doubted, truth = wrong, None, "false"
        template_key = f"{attribution}_suggestion"
        proposition = f"the answer is {option_ref(question, wrong)}"
        option = option_ref(question, wrong)
    elif claim_type == "doubt_w":
        asserted, doubted, truth = None, wrong, "true"
        template_key = f"{attribution}_doubt"
        proposition = f"{option_ref(question, wrong)} is not the correct answer"
        option = option_ref(question, wrong)
    elif claim_type == "doubt_c":
        asserted, doubted, truth = None, question.gold, "false"
        template_key = f"{attribution}_doubt"
        proposition = f"{option_ref(question, question.gold)} is not the correct answer"
        option = option_ref(question, question.gold)
    else:
        raise EvaluationError(f"Unknown useful-assertion claim type: {claim_type}")
    sentence = str(config["useful_assertion_templates"][template_key]).format(option=option)
    return sentence, asserted, doubted, truth + "|" + proposition


def _useful_tasks_for_model(
    config: Mapping[str, Any],
    model_key: str,
    questions: Sequence[Question],
    neutral_records: Mapping[str, Mapping[str, Any]],
) -> list[EvaluationTask]:
    tasks = []
    tool_definition = {
        "type": "function",
        "function": {
            "name": "consult_independent_source",
            "description": "Return a claim from an independently maintained factual source.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    }
    for question in questions:
        neutral = neutral_records.get(_question_key(question))
        parsed = campaign.screen_choice(neutral or {})
        if neutral is None or parsed not in question.labels:
            continue
        initially_correct = parsed == question.gold
        wrong = designated_wrong(question) if initially_correct else parsed
        cohort = "initially_correct" if initially_correct else "initially_incorrect"
        claim_types = ("suggest_w", "doubt_c") if initially_correct else ("suggest_c", "doubt_w")
        for claim_type in claim_types:
            for attribution in ("user", "source"):
                sentence, asserted, doubted, truth_and_proposition = _useful_sentence(
                    config,
                    question=question,
                    claim_type=claim_type,
                    attribution=attribution,
                    wrong=wrong,
                )
                truth, proposition = truth_and_proposition.split("|", 1)
                for turn_format in TURN_FORMATS:
                    metadata = {
                        "model_key": model_key,
                        "dataset_id": question.dataset_id,
                        "question_id": question.source_example_id,
                        "question_key": _question_key(question),
                        "question_axis": "useful_assertion",
                        "prompt_regime": "primary_matched_attribution",
                        "bias_type": claim_type,
                        "turn_format": turn_format,
                        "template_family": "bare_user" if attribution == "user" else "quantified_reliability",
                        "template_id": f"primary.{attribution}.{claim_type}",
                        "claim_truth": truth,
                        "claim_type": claim_type,
                        "claim_attribution": "bare_user" if attribution == "user" else "reliable_source",
                        "proposition": proposition,
                        "asserted_label": asserted,
                        "doubted_label": doubted,
                        "gold_label": question.gold,
                        "neutral_label": parsed,
                        "wrong_label": wrong,
                        "neutral_cohort": cohort,
                        "neutral_choice_source": "candidate_renormalized_argmax",
                        "source_reliability": 0.9 if attribution == "source" else None,
                        "retry_on_invalid": False,
                    }
                    tasks.append(
                        EvaluationTask(
                            example_id=(
                                f"useful:{claim_type}:{attribution}:{turn_format}:"
                                f"{_question_key(question)}"
                            ),
                            evaluator_id="bonham_useful_assertions_v1",
                            display_name="Bonham useful-assertion evaluation",
                            dataset_id=question.dataset_id,
                            dataset_revision=str(config["datasets"][question.dataset_id]["revision"]),
                            split=question.source_split,
                            condition_id=f"useful.{claim_type}.{attribution}.{turn_format}",
                            messages=render_messages(
                                question,
                                bias_sentence=sentence,
                                turn_format=turn_format,
                                assistant_answer=parsed if turn_format == "multi_turn" else None,
                                answer_instruction=str(config["answer_instruction"]),
                            ),
                            output_mode="mcq",
                            max_new_tokens=8,
                            choices=question.labels,
                            gold_choice=question.gold,
                            target_choice=asserted,
                            metadata=metadata,
                        )
                    )

        native_claim = claim_types[
            int(stable_hash(campaign.EXPERIMENT, "native-tool", _question_key(question)), 16) % 2
        ]
        source_sentence, asserted, doubted, truth_and_proposition = _useful_sentence(
            config,
            question=question,
            claim_type=native_claim,
            attribution="source",
            wrong=wrong,
        )
        truth, proposition = truth_and_proposition.split("|", 1)
        user_sentence, _a, _d, _tp = _useful_sentence(
            config,
            question=question,
            claim_type=native_claim,
            attribution="user",
            wrong=wrong,
        )
        common_metadata = {
            "model_key": model_key,
            "dataset_id": question.dataset_id,
            "question_id": question.source_example_id,
            "question_key": _question_key(question),
            "question_axis": "useful_assertion",
            "prompt_regime": "heldout_native_tool",
            "bias_type": native_claim,
            "turn_format": "structured_messages",
            "template_family": "native_tool_transfer",
            "claim_truth": truth,
            "claim_type": native_claim,
            "proposition": proposition,
            "asserted_label": asserted,
            "doubted_label": doubted,
            "gold_label": question.gold,
            "neutral_label": parsed,
            "wrong_label": wrong,
            "neutral_cohort": cohort,
            "primary_macro_eligible": False,
            "retry_on_invalid": False,
        }
        base_prompt = render_messages(
            question,
            bias_sentence=None,
            turn_format="single_turn",
            assistant_answer=None,
            answer_instruction=str(config["answer_instruction"]),
        )[0]
        tool_messages = (
            base_prompt,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"bonham_{hashlib.sha256(_question_key(question).encode()).hexdigest()[:12]}",
                        "type": "function",
                        "function": {
                            "name": "consult_independent_source",
                            "arguments": {"question": question.question},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": f"bonham_{hashlib.sha256(_question_key(question).encode()).hexdigest()[:12]}",
                "name": "consult_independent_source",
                "content": canonical_json(
                    {"documented_accuracy": 0.9, "result": source_sentence}
                ),
            },
            {"role": "user", "content": str(config["answer_instruction"])},
        )
        for attribution, messages, tools in (
            (
                "native_tool",
                tool_messages,
                (tool_definition,),
            ),
            (
                "matched_bare_user",
                render_messages(
                    question,
                    bias_sentence=user_sentence,
                    turn_format="multi_turn",
                    assistant_answer=parsed,
                    answer_instruction=str(config["answer_instruction"]),
                ),
                (),
            ),
        ):
            tasks.append(
                EvaluationTask(
                    example_id=f"native-transfer:{attribution}:{_question_key(question)}",
                    evaluator_id="bonham_native_source_transfer_v1",
                    display_name="Bonham native-tool source-form transfer",
                    dataset_id=question.dataset_id,
                    dataset_revision=str(config["datasets"][question.dataset_id]["revision"]),
                    split=question.source_split,
                    condition_id=f"native_transfer.{native_claim}.{attribution}",
                    messages=messages,
                    tools=tools,
                    output_mode="mcq",
                    max_new_tokens=8,
                    choices=question.labels,
                    gold_choice=question.gold,
                    target_choice=asserted,
                    metadata={
                        **common_metadata,
                        "template_id": f"native_transfer.{attribution}.{native_claim}",
                        "claim_attribution": attribution,
                    },
                )
            )
    return tasks


def _capability_tasks(
    suite_source_bindings: Path, external_utility_root: Path
) -> list[EvaluationTask]:
    tasks = build_capability_tasks(suite_source_bindings, external_utility_root)
    names = {utility_evaluation_name(task) for task in tasks}
    missing = CAPABILITY_NAMES.difference(names | {"OpenBookQA"})
    if missing:
        raise EvaluationError(f"Capability builder lacks: {sorted(missing)}")
    return tasks


def _task_question_key(task: EvaluationTask) -> str:
    return (
        f"{task.evaluator_id}:{task.dataset_id}:"
        f"{task.metadata.get('question_id', task.example_id)}"
    )


def _write_question_shards(
    tasks: Sequence[EvaluationTask],
    *,
    destination: Path,
    family: str,
    model_key: str,
    question_limit: int = QUESTION_SHARD_SIZE,
) -> Mapping[str, Any]:
    grouped: dict[tuple[str, str], list[EvaluationTask]] = defaultdict(list)
    for task in tasks:
        grouped[(task.evaluator_id, _task_question_key(task))].append(task)
    by_evaluator: dict[str, list[str]] = defaultdict(list)
    for evaluator_id, question_key in grouped:
        by_evaluator[evaluator_id].append(question_key)
    entries = []
    shard_index = 0
    for evaluator_id, question_keys in sorted(by_evaluator.items()):
        ordered = sorted(
            question_keys,
            key=lambda value: stable_hash(
                campaign.EXPERIMENT, "evaluation-shard", model_key, family, evaluator_id, value
            ),
        )
        if evaluator_id == "evalplus":
            # The EvalPlus launcher evaluates both benchmarks in every job.
            # Stratify by benchmark so a trailing shard cannot contain MBPP+
            # alone after the shorter HumanEval+ source is exhausted.
            shard_count = int(math.ceil(len(ordered) / int(question_limit)))
            buckets: list[list[str]] = [[] for _ in range(shard_count)]
            by_dataset: dict[str, list[str]] = defaultdict(list)
            for key in ordered:
                by_dataset[grouped[(evaluator_id, key)][0].dataset_id].append(key)
            if len(by_dataset) != 2 or any(
                len(keys) < shard_count for keys in by_dataset.values()
            ):
                raise EvaluationError("EvalPlus cannot be stratified across both benchmarks")
            for dataset_keys in by_dataset.values():
                for position, key in enumerate(dataset_keys):
                    buckets[position % shard_count].append(key)
            order_positions = {key: position for position, key in enumerate(ordered)}
            key_batches = [
                sorted(bucket, key=order_positions.__getitem__) for bucket in buckets
            ]
            if any(len(batch) > int(question_limit) for batch in key_batches):
                raise EvaluationError("EvalPlus stratified shard exceeds the question limit")
        else:
            key_batches = [
                ordered[start : start + int(question_limit)]
                for start in range(0, len(ordered), int(question_limit))
            ]
        for keys in key_batches:
            shard_tasks = [task for key in keys for task in grouped[(evaluator_id, key)]]
            path = destination / f"shard_{shard_index:04d}.jsonl"
            atomic_jsonl(path, (task.to_dict() for task in shard_tasks))
            read_tasks, observed_hash = read_task_manifest(path)
            if len(read_tasks) != len(shard_tasks) or observed_hash != sha256_file(path):
                raise EvaluationError(f"Task shard failed round-trip validation: {path}")
            entries.append(
                {
                    "model_key": model_key,
                    "family": family,
                    "evaluator_id": evaluator_id,
                    "shard": shard_index,
                    "question_count": len(keys),
                    "task_count": len(shard_tasks),
                    "path": str(path.resolve()),
                    "sha256": observed_hash,
                }
            )
            shard_index += 1
    atomic_jsonl(destination / "index.jsonl", entries)
    return {
        "question_count": len(grouped),
        "task_count": len(tasks),
        "shard_count": len(entries),
        "evaluator_ids": sorted(by_evaluator),
        "index_sha256": sha256_file(destination / "index.jsonl"),
    }


def prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    root = Path(args.result_root)
    questions = [_question(row) for row in read_jsonl(root / "inputs" / "evaluation_questions.jsonl")]
    counts = Counter(row.dataset_id for row in questions)
    if counts != {"commonsense_qa": 500, "arc_challenge": 500, "openbookqa": 500}:
        raise EvaluationError(f"Final factual cohorts are not 500/500/500: {counts}")
    capabilities = _capability_tasks(
        args.suite_source_bindings, args.external_utility_root
    )
    outputs = {}
    for model_key in campaign.MODEL_KEYS:
        neutral = campaign._record_by_question(
            campaign._collect_records(root, "neutral_screen", model_key)
        )
        generalization = _generalization_tasks_for_model(config, model_key, questions)
        useful = _useful_tasks_for_model(config, model_key, questions, neutral)
        capability_tasks = [
            replace(
                task,
                metadata={
                    **dict(task.metadata),
                    "model_key": model_key,
                    "question_axis": "general_capability",
                    "prompt_regime": "benchmark_native",
                    "bias_type": "not_applicable",
                    "turn_format": "benchmark_native",
                    "template_family": str(task.display_name),
                    "template_id": str(task.condition_id),
                    "claim_truth": "not_applicable",
                    "claim_attribution": "none",
                    "asserted_label": None,
                    "doubted_label": None,
                    "gold_label": task.gold_choice,
                    "neutral_label": None,
                },
            )
            for task in capabilities
        ]
        model_root = root / "evaluations" / "inputs" / model_key
        model_outputs = {
            "generalization": _write_question_shards(
                generalization,
                destination=model_root / "generalization",
                family="generalization",
                model_key=model_key,
            ),
            "useful_assertions": _write_question_shards(
                useful,
                destination=model_root / "useful_assertions",
                family="useful_assertions",
                model_key=model_key,
            ),
            "capabilities": _write_question_shards(
                capability_tasks,
                destination=model_root / "capabilities",
                family="capabilities",
                model_key=model_key,
                question_limit=100,
            ),
        }
        for family, audit in model_outputs.items():
            observed = int(audit["shard_count"])
            limit = int(EVALUATION_SHARD_LIMITS[family])
            if observed > limit:
                raise EvaluationError(
                    f"{model_key}/{family} produced {observed} shards, "
                    f"exceeding the submitted array capacity {limit}"
                )
        outputs[model_key] = model_outputs
        primary = [
            task
            for task in generalization
            if bool(task.metadata.get("primary_macro_eligible", False))
        ]
        primary_cells = Counter(
            (
                task.dataset_id,
                task.metadata["prompt_regime"],
                task.metadata["bias_type"],
                task.metadata["turn_format"],
            )
            for task in primary
        )
        if len(primary_cells) != 36 or set(primary_cells.values()) != {500}:
            raise EvaluationError(f"Primary generalization factorial is incomplete: {primary_cells}")
        reasoning_backed = [
            task
            for task in generalization
            if task.metadata.get("prompt_regime") == "reasoning_backed_pushback"
        ]
        reasoning_cells = Counter(task.dataset_id for task in reasoning_backed)
        reasoning_templates = {
            dataset_id: Counter(
                task.metadata["template_id"]
                for task in reasoning_backed
                if task.dataset_id == dataset_id
            )
            for dataset_id in ("commonsense_qa", "arc_challenge", "openbookqa")
        }
        if reasoning_cells != {
            "commonsense_qa": 500,
            "arc_challenge": 500,
            "openbookqa": 500,
        } or any(
            len(counts) != 4 or set(counts.values()) != {125}
            for counts in reasoning_templates.values()
        ):
            raise EvaluationError(
                "Reasoning-backed pushback stress test is incomplete or unbalanced: "
                f"cells={reasoning_cells}, templates={reasoning_templates}"
            )
        if any(
            task.metadata.get("turn_format") != "multi_turn"
            or task.metadata.get("bias_type") != "incorrect_suggestion"
            or task.metadata.get("reasoning_scope") != "generic_justification_pressure"
            or task.metadata.get("primary_macro_eligible") is not False
            or [message["role"] for message in task.messages] != ["user", "assistant", "user"]
            for task in reasoning_backed
        ):
            raise EvaluationError("Reasoning-backed pushback rendering violates its frozen design")
        useful_primary = [
            task
            for task in useful
            if task.evaluator_id == "bonham_useful_assertions_v1"
        ]
        if any(
            not {
                "claim_truth",
                "claim_type",
                "claim_attribution",
                "turn_format",
                "neutral_cohort",
            }.issubset(task.metadata)
            for task in useful_primary
        ):
            raise EvaluationError("Useful-assertion tasks lack factorial labels")
    receipt = {
        "status": "complete",
        "experiment": campaign.EXPERIMENT,
        "factual_question_counts": dict(counts),
        "models": outputs,
        "openbookqa_capability_reuse": (
            "generalization neutral records; no redundant capability rerun"
        ),
        "capability_names": sorted(CAPABILITY_NAMES),
        "reasoning_backed_prompt_registry": str(REASONING_BACKED_REGISTRY.resolve()),
        "reasoning_backed_prompt_registry_sha256": sha256_file(REASONING_BACKED_REGISTRY),
        "source_bindings_sha256": sha256_file(args.suite_source_bindings),
        "external_utility_complete_sha256": sha256_file(
            Path(args.external_utility_root) / "COMPLETE.json"
        ),
    }
    atomic_json(root / "evaluations" / "inputs" / "COMPLETE.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def run_shard(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    root = Path(args.result_root)
    manifest = (
        root
        / "evaluations"
        / "inputs"
        / args.model_key
        / args.family
        / f"shard_{int(args.shard):04d}.jsonl"
    )
    tasks, manifest_hash = read_task_manifest(manifest)
    if any(task.metadata.get("model_key") != args.model_key for task in tasks):
        raise EvaluationError("Evaluation task model identity mismatch")
    specification = campaign.model_spec(config, args.model_key)
    model, tokenizer = campaign._load_model(campaign.model_snapshot(args.hf_cache, specification))
    state = campaign._read_state(campaign._state_path(root, args.model_key, args.state_id))
    snapshot_hash, condition_hash = campaign._evaluation_provenance(
        root, args.model_key, args.config
    )
    output = (
        root
        / "evaluations"
        / "results"
        / args.model_key
        / args.state_id
        / args.family
        / f"shard_{int(args.shard):04d}"
    )
    summary = run_evaluation_cell(
        llm=campaign._LLM(model, tokenizer, str(specification["model_id"])),
        state=state,
        tasks=tasks,
        task_manifest_sha256=manifest_hash,
        snapshot_inventory_sha256=snapshot_hash,
        condition_registry_sha256=condition_hash,
        output_dir=output,
        run_id=(
            f"{campaign.EXPERIMENT}:{args.model_key}:{args.state_id}:"
            f"{args.family}:{int(args.shard)}"
        ),
        inference_batch_size=int(args.batch_size),
        require_batched_inference=int(args.batch_size) > 1,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_state_sequence(args: argparse.Namespace) -> None:
    """Evaluate one intervention state across several families with one model load."""

    families = tuple(dict.fromkeys(str(family) for family in args.families))
    if not families:
        raise EvaluationError("State sequence requires at least one evaluation family")
    config = load_config(args.config)
    root = Path(args.result_root)
    specification = campaign.model_spec(config, args.model_key)
    model, tokenizer = campaign._load_model(
        campaign.model_snapshot(args.hf_cache, specification)
    )
    llm = campaign._LLM(model, tokenizer, str(specification["model_id"]))
    state = campaign._read_state(
        campaign._state_path(root, args.model_key, args.state_id)
    )
    snapshot_hash, condition_hash = campaign._evaluation_provenance(
        root, args.model_key, args.config
    )
    family_counts = {}
    for family in families:
        index_path = (
            root / "evaluations" / "inputs" / args.model_key / family / "index.jsonl"
        )
        entries = read_jsonl(index_path)
        if not entries:
            raise EvaluationError(f"No frozen evaluation shards indexed by {index_path}")
        completed = 0
        for entry in entries:
            shard = int(entry["shard"])
            manifest = index_path.parent / f"shard_{shard:04d}.jsonl"
            tasks, manifest_hash = read_task_manifest(manifest)
            if any(task.metadata.get("model_key") != args.model_key for task in tasks):
                raise EvaluationError("Evaluation task model identity mismatch")
            output = (
                root
                / "evaluations"
                / "results"
                / args.model_key
                / args.state_id
                / family
                / f"shard_{shard:04d}"
            )
            print(
                json.dumps(
                    {
                        "event": "evaluation_state_sequence_shard_start",
                        "model_key": args.model_key,
                        "state_id": args.state_id,
                        "family": family,
                        "shard": shard,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            summary = run_evaluation_cell(
                llm=llm,
                state=state,
                tasks=tasks,
                task_manifest_sha256=manifest_hash,
                snapshot_inventory_sha256=snapshot_hash,
                condition_registry_sha256=condition_hash,
                output_dir=output,
                run_id=(
                    f"{campaign.EXPERIMENT}:{args.model_key}:{args.state_id}:"
                    f"{family}:{shard}"
                ),
                inference_batch_size=int(args.batch_size),
                require_batched_inference=int(args.batch_size) > 1,
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "evaluation_state_sequence_shard_complete",
                        "model_key": args.model_key,
                        "state_id": args.state_id,
                        "family": family,
                        "shard": shard,
                        "record_count": int(summary["record_count"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        family_counts[family] = completed
    print(
        json.dumps(
            {
                "status": "complete",
                "model_key": args.model_key,
                "state_id": args.state_id,
                "family_counts": family_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


def validate_complete(args: argparse.Namespace) -> None:
    root = Path(args.result_root)
    states = tuple(args.state_ids or campaign.PRIMARY_STATE_IDS)
    cells = []
    for model_key in campaign.MODEL_KEYS:
        for family in ("generalization", "useful_assertions", "capabilities"):
            entries = read_jsonl(
                root / "evaluations" / "inputs" / model_key / family / "index.jsonl"
            )
            for entry in entries:
                shard = int(entry["shard"])
                for state_id in states:
                    bundle = (
                        root
                        / "evaluations"
                        / "results"
                        / model_key
                        / state_id
                        / family
                        / f"shard_{shard:04d}"
                    )
                    validated = validate_complete_bundle(bundle)
                    if int(validated["record_count"]) != int(entry["task_count"]):
                        raise EvaluationError(f"Evaluation record-count mismatch: {bundle}")
                    identity = read_json(bundle / "identity.json")
                    if identity.get("manifest_sha256") != entry["sha256"]:
                        raise EvaluationError(f"Evaluation manifest hash mismatch: {bundle}")
                    cells.append(
                        {
                            "model_key": model_key,
                            "state_id": state_id,
                            "family": family,
                            "shard": shard,
                            "record_count": int(validated["record_count"]),
                            "identity_sha256": validated["identity_sha256"],
                        }
                    )
    receipt = {
        "status": "complete",
        "states": list(states),
        "cell_count": len(cells),
        "record_count": sum(row["record_count"] for row in cells),
        "cells_sha256": hashlib.sha256(canonical_json(cells).encode("utf-8")).hexdigest(),
    }
    atomic_json(root / "evaluations" / "results" / "COMPLETE.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("prepare")
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--suite-source-bindings", type=Path, required=True)
    command.add_argument("--external-utility-root", type=Path, required=True)
    command.set_defaults(func=prepare)

    command = subparsers.add_parser("run-shard")
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--model-key", choices=campaign.MODEL_KEYS, required=True)
    command.add_argument("--state-id", choices=campaign.PRIMARY_STATE_IDS, required=True)
    command.add_argument(
        "--family", choices=("generalization", "useful_assertions", "capabilities"), required=True
    )
    command.add_argument("--shard", type=int, required=True)
    command.add_argument("--hf-cache", type=Path, required=True)
    command.add_argument("--batch-size", type=int, default=4)
    command.set_defaults(func=run_shard)

    command = subparsers.add_parser("run-state-sequence")
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--model-key", choices=campaign.MODEL_KEYS, required=True)
    command.add_argument("--state-id", choices=campaign.PRIMARY_STATE_IDS, required=True)
    command.add_argument(
        "--families",
        nargs="+",
        choices=("generalization", "useful_assertions", "capabilities"),
        required=True,
    )
    command.add_argument("--hf-cache", type=Path, required=True)
    command.add_argument("--batch-size", type=int, default=4)
    command.set_defaults(func=run_state_sequence)

    command = subparsers.add_parser("validate-complete")
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--state-ids", nargs="*")
    command.set_defaults(func=validate_complete)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
