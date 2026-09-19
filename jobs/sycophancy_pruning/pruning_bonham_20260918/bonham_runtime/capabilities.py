"""Frozen general-capability task builders used only by Bonham.

The builders deliberately consume authenticated source bindings rather than
loading datasets at evaluation time.  This keeps a clean Bonham checkout
self-contained and makes every prompt manifest reproducible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import string
import time
from typing import Any, Callable, Mapping, Sequence

from .evaluation.runner import EvaluationTask
from .evaluation.schemas import sha256_json
from .evaluation.utility_adapters import (
    EVALPLUS,
    MMLU,
    MMLU_PRO,
    SYMBOLIC_IN_CONTEXT_LEARNING,
)


MAX_EXAMPLES = 500
_LETTERS = tuple(string.ascii_uppercase)
_REVISIONS = {
    "mmlu": "4450500f923c49f1fb1dd3d99108a0bd9717b660",
    "mmlu_pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "sst2_symbolic_icl": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    "ag_news_symbolic_icl": "eb185aade064a813bc0b7f42de02595523103ca4f6",
    "evalplus_humaneval": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
    "evalplus_mbpp": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
}


class CapabilityError(RuntimeError):
    """Raised when a frozen capability source cannot satisfy the protocol."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _authenticated_value(path: Path, expected: str, serialization: str | None = None) -> Any:
    path = Path(path).expanduser().resolve()
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            # Hash and parse one byte snapshot.  On a congested shared
            # filesystem, two independent reads can otherwise observe a good
            # hash followed by a transiently truncated parse read.
            payload = path.read_bytes()
            observed = hashlib.sha256(payload).hexdigest()
            if observed != str(expected):
                raise CapabilityError(
                    f"Frozen capability source is absent or changed: {path}"
                )
            text = payload.decode("utf-8")
            if serialization == "jsonl" or path.suffix == ".jsonl":
                return [
                    json.loads(line)
                    for line in text.splitlines()
                    if line.strip()
                ]
            return json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, CapabilityError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(attempt + 1)
    raise CapabilityError(
        f"Could not read authenticated capability source after retries: {path}"
    ) from last_error


def _binding_value(binding_path: Path, key: str) -> Any:
    registry = _read_json(Path(binding_path))
    sources = registry.get("sources")
    if not isinstance(sources, Mapping) or key not in sources:
        raise CapabilityError(f"Suite source bindings lack {key!r}")
    binding = sources[key]
    if not isinstance(binding, Mapping):
        raise CapabilityError(f"Malformed suite source binding for {key!r}")
    value = _authenticated_value(
        Path(str(binding.get("path", ""))),
        str(binding.get("sha256", "")),
        str(binding.get("serialization", "")) or None,
    )
    if isinstance(value, Mapping) and isinstance(value.get("rows"), list):
        return value["rows"]
    return value


def _external_sources(root: Path) -> tuple[Mapping[str, list[Mapping[str, Any]]], Mapping[str, Any]]:
    receipt = _read_json(Path(root) / "COMPLETE.json")
    if receipt.get("status") != "complete" or not isinstance(receipt.get("sources"), Mapping):
        raise CapabilityError("External capability source receipt is incomplete")
    result: dict[str, list[Mapping[str, Any]]] = {}
    for key, metadata in receipt["sources"].items():
        if not isinstance(metadata, Mapping):
            raise CapabilityError(f"Malformed external source receipt for {key!r}")
        value = _authenticated_value(Path(str(metadata["path"])), str(metadata["sha256"]), "jsonl")
        if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
            raise CapabilityError(f"External source {key!r} is not a row list")
        result[str(key)] = [dict(row) for row in value]
        demo_path = metadata.get("demonstrations_path")
        if demo_path is not None:
            demo = _authenticated_value(
                Path(str(demo_path)), str(metadata.get("demonstrations_sha256", "")), "jsonl"
            )
            result[f"{key}_demonstrations"] = [dict(row) for row in demo]
    return result, receipt["sources"]


def _rank(namespace: str, identity: str) -> str:
    return hashlib.sha256(
        f"pruning_bonham_20260918\0{namespace}\0{identity}".encode("utf-8")
    ).hexdigest()


def _hash_limit(
    rows: Sequence[Any], namespace: str, count: int, identity_fn: Callable[[Any], Any]
) -> list[Any]:
    return sorted(
        rows, key=lambda row: (_rank(namespace, str(identity_fn(row))), str(identity_fn(row)))
    )[: min(int(count), len(rows))]


def _stratified_limit(
    rows: Sequence[Mapping[str, Any]],
    *,
    namespace: str,
    count: int,
    stratum_fn: Callable[[Mapping[str, Any]], Any],
    identity_fn: Callable[[Mapping[str, Any]], Any],
) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(stratum_fn(row))].append(row)
    if not rows:
        raise CapabilityError(f"{namespace} source is empty")
    target = min(int(count), len(rows))
    ideals = {key: target * len(values) / len(rows) for key, values in grouped.items()}
    allocations = {key: int(value) for key, value in ideals.items()}
    remaining = target - sum(allocations.values())
    for key in sorted(grouped, key=lambda value: (-(ideals[value] - allocations[value]), value))[:remaining]:
        allocations[key] += 1
    selected: list[Mapping[str, Any]] = []
    for key, values in sorted(grouped.items()):
        selected.extend(
            _hash_limit(values, f"{namespace}:{key}", allocations[key], identity_fn)
        )
    if len(selected) != target:
        raise CapabilityError(f"{namespace} stratified cap did not reach {target}")
    return selected


def _required(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "") or "").strip()
    if not value:
        raise CapabilityError(f"Capability row lacks nonempty {key!r}")
    return value


def _source_id(row: Mapping[str, Any], namespace: str) -> str:
    for key in ("example_id", "id", "task_id", "question_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    raise CapabilityError(f"{namespace} row lacks a stable identity")


def _answer_label(value: Any, labels: Sequence[str]) -> str:
    allowed = tuple(str(label).upper() for label in labels)
    if isinstance(value, int) and 0 <= value < len(allowed):
        return allowed[value]
    text = str(value or "").strip().upper()
    if text.isdigit() and int(text) < len(allowed):
        return allowed[int(text)]
    if text not in allowed:
        raise CapabilityError(f"Answer {value!r} is outside labels {allowed}")
    return text


def _format_mcq(question: str, choices: Sequence[str]) -> str:
    if not 2 <= len(choices) <= len(_LETTERS):
        raise CapabilityError("Multiple-choice row needs 2-26 choices")
    return "\n".join(
        [question]
        + [f"{_LETTERS[index]}. {str(choice).strip()}" for index, choice in enumerate(choices)]
    )


def _simple_mcq(
    *, example_id: str, evaluator_id: str, display_name: str, dataset_id: str,
    revision: str, prompt: str, choices: Sequence[str], gold: str,
) -> EvaluationTask:
    return EvaluationTask(
        example_id=example_id,
        evaluator_id=evaluator_id,
        display_name=display_name,
        dataset_id=dataset_id,
        dataset_revision=revision,
        split="validation",
        condition_id=f"utility.{dataset_id}",
        messages=({"role": "user", "content": prompt},),
        output_mode="mcq",
        max_new_tokens=8,
        choices=tuple(choices),
        gold_choice=gold,
        metadata={"question_id": example_id, "retry_on_invalid": False},
    )


def _unique_external_ids(rows: Sequence[Mapping[str, Any]], dataset: str) -> tuple[str, ...]:
    bases = [str(row.get("stable_row_id", "")).strip() for row in rows]
    if any(not value for value in bases):
        raise CapabilityError(f"{dataset} has an empty stable_row_id")
    counts = Counter(bases)
    identities = []
    for row, base in zip(rows, bases):
        identity = base
        if counts[base] > 1:
            source_index = str(row.get("source_index", "")).strip()
            if not source_index:
                raise CapabilityError(f"{dataset} duplicate {base!r} lacks source_index")
            identity = f"{base}:source_index:{source_index}"
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise CapabilityError(f"{dataset} stable identities remain duplicated")
    return tuple(identities)


def _mmlu_tasks(
    rows: Sequence[Mapping[str, Any]], demonstrations: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[EvaluationTask]:
    tasks = []
    for row in rows:
        identity = _source_id(row, "mmlu")
        subject = _required(row, "subject")
        choices = tuple(str(value).strip() for value in row.get("choices", ()))
        labels = _LETTERS[: len(choices)]
        gold = _answer_label(row.get("answer", row.get("answer_index")), labels)
        shots = list(demonstrations.get(subject, ()))
        if len(shots) < 5:
            raise CapabilityError(f"MMLU subject {subject!r} has fewer than five demonstrations")
        pieces = [f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}."]
        for shot in shots[:5]:
            shot_choices = tuple(str(value).strip() for value in shot.get("choices", ()))
            shot_labels = _LETTERS[: len(shot_choices)]
            shot_gold = _answer_label(shot.get("answer", shot.get("answer_index")), shot_labels)
            pieces.append(f"{_format_mcq(_required(shot, 'question'), shot_choices)}\nAnswer: {shot_gold}")
        pieces.append(f"{_format_mcq(_required(row, 'question'), choices)}\nAnswer:")
        tasks.append(EvaluationTask(
            example_id=f"mmlu:{identity}", evaluator_id="mmlu_full", display_name=MMLU,
            dataset_id="mmlu", dataset_revision=_REVISIONS["mmlu"], split="test",
            condition_id="utility.mmlu", messages=({"role": "user", "content": "\n\n".join(pieces)},),
            output_mode="mcq", max_new_tokens=32, choices=labels, gold_choice=gold,
            metadata={"question_id": identity, "subject": subject, "shot_count": 5},
        ))
    return tasks


def _mmlu_pro_tasks(
    rows: Sequence[Mapping[str, Any]], demonstrations: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[EvaluationTask]:
    tasks = []
    for row in rows:
        identity = _source_id(row, "mmlu_pro")
        category = str(row.get("category", row.get("subject", "")) or "").strip()
        if not category:
            raise CapabilityError("MMLU-Pro row lacks category")
        options = tuple(str(value).strip() for value in row.get("options", row.get("choices", ())))
        labels = _LETTERS[: len(options)]
        gold = _answer_label(row.get("answer", row.get("answer_index")), labels)
        shots = list(demonstrations.get(category, ()))
        if len(shots) < 5:
            raise CapabilityError(f"MMLU-Pro category {category!r} has fewer than five demonstrations")
        pieces = [f"The following are challenging multiple-choice questions about {category}. Think step by step and end with `Answer: X`."]
        for shot in shots[:5]:
            shot_options = tuple(str(value).strip() for value in shot.get("options", shot.get("choices", ())))
            shot_labels = _LETTERS[: len(shot_options)]
            shot_gold = _answer_label(shot.get("answer", shot.get("answer_index")), shot_labels)
            reasoning = str(shot.get("cot_content", shot.get("reasoning", "")) or "").strip()
            if not reasoning:
                raise CapabilityError("MMLU-Pro demonstration lacks official reasoning")
            pieces.append(f"{_format_mcq(_required(shot, 'question'), shot_options)}\n{reasoning}\nAnswer: {shot_gold}")
        pieces.append(f"{_format_mcq(_required(row, 'question'), options)}\nLet's think step by step.")
        tasks.append(EvaluationTask(
            example_id=f"mmlu-pro:{identity}", evaluator_id="mmlu_pro_full", display_name=MMLU_PRO,
            dataset_id="mmlu_pro", dataset_revision=_REVISIONS["mmlu_pro"], split="test",
            condition_id="utility.mmlu_pro", messages=({"role": "user", "content": "\n\n".join(pieces)},),
            output_mode="generation", max_new_tokens=1024, gold_answers=(gold,),
            metadata={"question_id": identity, "category": category, "allowed_labels": list(labels), "gold_label": gold, "shot_count": 5},
        ))
    return tasks


def _symbolic_tasks(rows: Sequence[Mapping[str, Any]]) -> list[EvaluationTask]:
    tasks = []
    for row in rows:
        identity = _source_id(row, "symbolic_icl")
        dataset = str(row.get("dataset", "") or "").strip().lower()
        dataset_id = {"sst2": "sst2_symbolic_icl", "ag_news": "ag_news_symbolic_icl"}.get(dataset)
        if dataset_id is None:
            raise CapabilityError(f"Unsupported symbolic ICL dataset {dataset!r}")
        allowed = tuple(str(value).strip() for value in row.get("allowed_labels", ()))
        expected = str(row.get("expected_label", "") or "").strip()
        messages = tuple(dict(value) for value in row.get("messages", ()))
        if expected not in allowed or not messages:
            raise CapabilityError("Malformed symbolic ICL row")
        tasks.append(EvaluationTask(
            example_id=f"symbolic-icl:{identity}", evaluator_id="symbolic_icl_200",
            display_name=SYMBOLIC_IN_CONTEXT_LEARNING, dataset_id=dataset_id,
            dataset_revision=_REVISIONS[dataset_id], split="validation" if dataset == "sst2" else "test",
            condition_id="utility.symbolic_icl", messages=messages, output_mode="generation",
            max_new_tokens=4, gold_answers=(expected,),
            metadata={"question_id": identity, "dataset": dataset, "allowed_labels": list(allowed), "expected_label": expected},
        ))
    return tasks


def _evalplus_tasks(rows: Sequence[Mapping[str, Any]]) -> list[EvaluationTask]:
    aliases = {
        "humaneval+": ("HumanEval+", "evalplus_humaneval"),
        "humaneval": ("HumanEval+", "evalplus_humaneval"),
        "mbpp+": ("MBPP+", "evalplus_mbpp"),
        "mbpp": ("MBPP+", "evalplus_mbpp"),
    }
    tasks = []
    for row in rows:
        identity = _source_id(row, "evalplus")
        benchmark_key = str(row.get("benchmark", "") or "").strip().lower()
        if benchmark_key not in aliases:
            raise CapabilityError(f"Unknown EvalPlus benchmark {benchmark_key!r}")
        benchmark, dataset_id = aliases[benchmark_key]
        prompt = str(row.get("prompt", row.get("canonical_prompt", "")) or "")
        if not prompt.strip():
            raise CapabilityError("EvalPlus row lacks canonical prompt")
        tasks.append(EvaluationTask(
            example_id=f"evalplus:{benchmark}:{identity}", evaluator_id="evalplus", display_name=EVALPLUS,
            dataset_id=dataset_id, dataset_revision=_REVISIONS[dataset_id], split="test",
            condition_id="utility.evalplus", messages=({"role": "user", "content": prompt},),
            output_mode="generation", max_new_tokens=512,
            metadata={"question_id": identity, "task_id": identity, "benchmark": benchmark},
        ))
    return tasks


def utility_evaluation_name(task: EvaluationTask) -> str:
    if task.evaluator_id == "evalplus":
        return str(task.metadata["benchmark"])
    if task.evaluator_id == "symbolic_icl_200":
        return "SST-2 arbitrary-label ICL" if task.dataset_id == "sst2_symbolic_icl" else "AG News arbitrary-label ICL"
    return str(task.display_name)


def build_capability_tasks(binding: Path, external_root: Path) -> list[EvaluationTask]:
    external, revisions = _external_sources(external_root)
    tasks: list[EvaluationTask] = []
    for row in external["boolq"]:
        tasks.append(_simple_mcq(
            example_id=f"boolq:{row['stable_row_id']}", evaluator_id="bonham_boolq",
            display_name="BoolQ", dataset_id="boolq", revision=str(revisions["boolq"]["revision"]),
            prompt=f"Passage: {row['passage']}\nQuestion: {row['question']}\nAnswer Yes or No.",
            choices=("Yes", "No"), gold="Yes" if bool(row["answer"]) else "No",
        ))
    for row in external["rte"]:
        tasks.append(_simple_mcq(
            example_id=f"rte:{row['stable_row_id']}", evaluator_id="bonham_rte",
            display_name="RTE", dataset_id="rte", revision=str(revisions["rte"]["revision"]),
            prompt=f"Premise: {row['sentence1']}\nHypothesis: {row['sentence2']}\nDoes the premise entail the hypothesis?",
            choices=("Entailment", "Not entailment"),
            gold="Entailment" if int(row["label"]) == 0 else "Not entailment",
        ))
    for row, row_id in zip(external["hellaswag"], _unique_external_ids(external["hellaswag"], "hellaswag")):
        endings = list(row["endings"])
        for index, ending in enumerate(endings):
            tasks.append(EvaluationTask(
                example_id=f"hellaswag:{row_id}:ending:{index}", evaluator_id="bonham_hellaswag_acc_norm",
                display_name="HellaSwag", dataset_id="hellaswag", dataset_revision=str(revisions["hellaswag"]["revision"]),
                split="validation", condition_id="utility.hellaswag",
                messages=({"role": "user", "content": str(row.get("ctx", row.get("ctx_a", "")))},),
                output_mode="response_nll", max_new_tokens=1,
                metadata={"question_id": row_id, "candidate_index": index, "gold_candidate_index": int(row["label"]), "target_text": str(ending)},
            ))
    winogrande_demos = external["winogrande_demonstrations"]
    if len(winogrande_demos) != 5:
        raise CapabilityError("WinoGrande requires exactly five frozen demonstrations")
    demo_text = [str(row["sentence"]).replace("_", str(row[f"option{int(row['answer'])}"])) for row in winogrande_demos]
    for row in external["winogrande"]:
        sentence = str(row["sentence"])
        if sentence.count("_") != 1:
            raise CapabilityError("WinoGrande sentence does not contain one blank")
        prefix, suffix = sentence.split("_", 1)
        for index, option in enumerate((row["option1"], row["option2"])):
            tasks.append(EvaluationTask(
                example_id=f"winogrande:{row['stable_row_id']}:option:{index}", evaluator_id="bonham_winogrande",
                display_name="WinoGrande", dataset_id="winogrande", dataset_revision=str(revisions["winogrande"]["revision"]),
                split="validation", condition_id="utility.winogrande",
                messages=({"role": "user", "content": "Complete each sentence.\n\n" + "\n".join(demo_text) + "\n" + prefix},),
                output_mode="response_nll", max_new_tokens=1,
                metadata={"question_id": str(row["stable_row_id"]), "candidate_index": index, "gold_candidate_index": int(row["answer"]) - 1, "target_text": str(option) + suffix, "shot_count": 5},
            ))
    trivia_demos = external["triviaqa_wiki_demonstrations"]
    if len(trivia_demos) != 5:
        raise CapabilityError("TriviaQA-Wiki requires exactly five frozen demonstrations")
    trivia_prefix = "Answer each question briefly.\n\n" + "\n\n".join(
        f"Question: {row['question']}\nAnswer: {row['answer']['value']}" for row in trivia_demos
    )
    for row in external["triviaqa_wiki"]:
        aliases = list(row.get("answer", {}).get("aliases", ())) or [str(row.get("answer", {}).get("value", ""))]
        tasks.append(EvaluationTask(
            example_id=f"triviaqa:{row['stable_row_id']}", evaluator_id="bonham_triviaqa_wiki",
            display_name="TriviaQA-Wiki", dataset_id="triviaqa_wiki", dataset_revision=str(revisions["triviaqa_wiki"]["revision"]),
            split="validation", condition_id="utility.triviaqa_wiki",
            messages=({"role": "user", "content": trivia_prefix + f"\n\nQuestion: {row['question']}\nAnswer:"},),
            output_mode="generation", max_new_tokens=32,
            gold_answers=tuple(str(value) for value in aliases if str(value).strip()),
            metadata={"question_id": str(row["stable_row_id"]), "accepted_aliases": aliases, "shot_count": 5, "demonstrations_sha256": revisions["triviaqa_wiki"]["demonstrations_sha256"]},
        ))
    mmlu_rows = _stratified_limit(
        _binding_value(binding, "mmlu_test"), namespace="utility:mmlu", count=MAX_EXAMPLES,
        stratum_fn=lambda row: row.get("subject", ""), identity_fn=lambda row: row.get("id", sha256_json(row)),
    )
    mmlu_pro_rows = _stratified_limit(
        _binding_value(binding, "mmlu_pro_test"), namespace="utility:mmlu_pro", count=MAX_EXAMPLES,
        stratum_fn=lambda row: row.get("category", row.get("subject", "")),
        identity_fn=lambda row: row.get("question_id", row.get("id", sha256_json(row))),
    )
    tasks.extend(_mmlu_tasks(mmlu_rows, _binding_value(binding, "mmlu_demonstrations")))
    tasks.extend(_mmlu_pro_tasks(mmlu_pro_rows, _binding_value(binding, "mmlu_pro_demonstrations")))
    tasks.extend(_symbolic_tasks(_binding_value(binding, "symbolic_icl_200")))
    tasks.extend(_evalplus_tasks(_binding_value(binding, "evalplus")))
    identities = [(task.evaluator_id, task.dataset_id, task.example_id, task.condition_id) for task in tasks]
    if not tasks or len(identities) != len(set(identities)):
        raise CapabilityError("Capability task identities are empty or duplicated")
    names = {utility_evaluation_name(task) for task in tasks}
    expected = {"BoolQ", "RTE", "HellaSwag", "WinoGrande", "MMLU", "MMLU-Pro", "TriviaQA-Wiki", "SST-2 arbitrary-label ICL", "AG News arbitrary-label ICL", "HumanEval+", "MBPP+"}
    if names != expected:
        raise CapabilityError(f"Capability task names mismatch: expected {sorted(expected)}, got {sorted(names)}")
    return tasks


__all__ = ["CapabilityError", "build_capability_tasks", "utility_evaluation_name"]
