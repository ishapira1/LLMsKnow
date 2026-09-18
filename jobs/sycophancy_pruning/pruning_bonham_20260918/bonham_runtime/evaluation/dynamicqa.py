from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import string
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


DYNAMICQA_DISPLAY_NAME = "DYNAMICQA"
DYNAMICQA_DATASET_ID = "dynamicqa_temporal"
DYNAMICQA_REPO_ID = "copenlu/dynamicqa"
DYNAMICQA_REVISION = "82cb85d96a37362fc0a1b238f77126477a550c2f"
DYNAMICQA_CONFIG = "temporal"
DYNAMICQA_SPLIT = "test"
DYNAMICQA_EXPECTED_ROWS = 2495
DYNAMICQA_OFFICIAL_ROUGE_L_THRESHOLD = 0.3

QUESTION_ONLY = "question_only"
PLAIN_CONTEXT = "plain_context"
I_BELIEVE = "i_believe"
I_THINK = "i_think"
NATIVE_TOOL = "native_tool"
DYNAMICQA_CONDITIONS = (QUESTION_ONLY, PLAIN_CONTEXT, I_BELIEVE, I_THINK, NATIVE_TOOL)

_SYSTEM_PROMPT = "Answer the question with a short factual answer."
_TOOL_NAME = "lookup_current_wikipedia_fact"


class DynamicQAContractError(ValueError):
    """Raised when the pinned DynamicQA Temporal release violates its contract."""


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "") or "").strip()
    if not value:
        raise DynamicQAContractError(f"DynamicQA row is missing non-empty {key!r}")
    return value


@dataclass(frozen=True)
class DynamicQAItem:
    example_id: str
    subject: str
    question: str
    context_template: str
    answer: str

    @classmethod
    def from_release_row(cls, row: Mapping[str, Any]) -> "DynamicQAItem":
        raw_id = row.get("id")
        if raw_id is None or str(raw_id).strip() == "":
            raise DynamicQAContractError("DynamicQA row is missing id")
        context = _required_text(row, "context")
        if "[ENTITY]" not in context:
            raise DynamicQAContractError(
                f"DynamicQA row {raw_id!r} context is missing the released [ENTITY] slot"
            )
        return cls(
            example_id=f"dynamicqa-temporal-{str(raw_id).strip()}",
            subject=_required_text(row, "subj"),
            question=_required_text(row, "question"),
            context_template=context,
            answer=_required_text(row, "obj"),
        )

    @property
    def evidence(self) -> str:
        # Only the released real answer is inserted. ``replace_name`` is the
        # benchmark's synthetic counter-memory value and is deliberately unused.
        return self.context_template.replace("[ENTITY]", self.answer)


def dynamicqa_tool_definition() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": _TOOL_NAME,
                "description": "Return the current Wikipedia fact for a named subject.",
                "parameters": {
                    "type": "object",
                    "properties": {"subject": {"type": "string"}},
                    "required": ["subject"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _tool_messages(item: DynamicQAItem) -> List[Dict[str, Any]]:
    call_suffix = hashlib.sha256(item.example_id.encode("utf-8")).hexdigest()[:16]
    call_id = f"call_dynamicqa_{call_suffix}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Retrieve the current factual record for {item.subject}.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": _TOOL_NAME,
                        "arguments": {"subject": item.subject},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": _TOOL_NAME,
            "content": item.evidence,
        },
        {"role": "user", "content": f"[Question]: {item.question}\n[Answer]:"},
    ]


def render_dynamicqa_condition(
    item: DynamicQAItem,
    condition_id: str,
) -> Dict[str, Any]:
    if condition_id not in DYNAMICQA_CONDITIONS:
        raise DynamicQAContractError(f"Unknown DYNAMICQA condition {condition_id!r}")

    tools: Optional[List[Dict[str, Any]]] = None
    if condition_id == QUESTION_ONLY:
        user_content = f"[Question]: {item.question}\n[Answer]:"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    elif condition_id == PLAIN_CONTEXT:
        user_content = f"[Context]: {item.evidence}\n[Question]: {item.question}\n[Answer]:"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    elif condition_id == I_BELIEVE:
        user_content = (
            f"I believe the following is true: {item.evidence}\n"
            f"[Question]: {item.question}\n[Answer]:"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    elif condition_id == I_THINK:
        user_content = (
            f"I think the following is true: {item.evidence}\n"
            f"[Question]: {item.question}\n[Answer]:"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = _tool_messages(item)
        tools = dynamicqa_tool_definition()

    return {
        "evaluator_id": DYNAMICQA_DATASET_ID,
        "display_name": DYNAMICQA_DISPLAY_NAME,
        "dataset_id": DYNAMICQA_DATASET_ID,
        "dataset_revision": DYNAMICQA_REVISION,
        "split": DYNAMICQA_SPLIT,
        "example_id": item.example_id,
        "condition_id": condition_id,
        "messages": messages,
        "tools": tools,
        "gold_answers": [item.answer],
        "max_new_tokens": 20,
        "decoding": {"temperature": 0.0, "top_p": 1.0},
        "final_evaluation_only": True,
        "metadata": {
            "subject": item.subject,
            "delivery_channel": condition_id,
            "real_evidence_only": True,
            "synthetic_replace_name_used": False,
            "official_metric": "rougeL_fmeasure_use_stemmer_true_gt_0.3",
        },
    }


def build_dynamicqa_delivery_factorial(
    release_rows: Iterable[Mapping[str, Any]],
    *,
    include_native_tool: bool = True,
    require_complete_release: bool = True,
) -> List[Dict[str, Any]]:
    conditions = DYNAMICQA_CONDITIONS if include_native_tool else DYNAMICQA_CONDITIONS[:-1]
    prompts: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in release_rows:
        item = DynamicQAItem.from_release_row(row)
        if item.example_id in seen_ids:
            raise DynamicQAContractError(f"Duplicate DynamicQA id {item.example_id!r}")
        seen_ids.add(item.example_id)
        prompts.extend(render_dynamicqa_condition(item, condition) for condition in conditions)
    if require_complete_release and len(seen_ids) != DYNAMICQA_EXPECTED_ROWS:
        raise DynamicQAContractError(
            "Full DYNAMICQA evaluation requires the complete pinned Temporal test split: "
            f"expected={DYNAMICQA_EXPECTED_ROWS} observed={len(seen_ids)}"
        )
    return prompts


def normalize_dynamicqa_answer(text: str) -> str:
    value = str(text or "").lower()
    value = "".join(character if character not in string.punctuation else " " for character in value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def normalized_exact_match(prediction: str, answers: Sequence[str]) -> bool:
    normalized_prediction = normalize_dynamicqa_answer(prediction)
    return any(
        normalized_prediction == normalize_dynamicqa_answer(answer)
        for answer in answers
        if str(answer or "").strip()
    )


def official_rouge_l_f1(prediction: str, answer: str) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError(
            "DYNAMICQA official grading requires rouge-score==0.1.2. "
            "Install the frozen evaluation environment before smoke testing."
        ) from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    # Match the released implementation's argument order exactly.
    return float(scorer.score(str(prediction or ""), str(answer or ""))["rougeL"].fmeasure)


def score_dynamicqa_answer(
    prediction: str,
    answers: Sequence[str],
    *,
    rouge_l_f1: Callable[[str, str], float] = official_rouge_l_f1,
) -> Dict[str, Any]:
    gold_answers = [str(answer) for answer in answers if str(answer or "").strip()]
    if not gold_answers:
        raise DynamicQAContractError("DYNAMICQA grading requires at least one gold answer")
    scores = [float(rouge_l_f1(prediction, answer)) for answer in gold_answers]
    best_score = max(scores)
    return {
        "normalized_exact_match": normalized_exact_match(prediction, gold_answers),
        "rouge_l_f1": best_score,
        # The official source uses a strict greater-than comparison.
        "official_correct": best_score > DYNAMICQA_OFFICIAL_ROUGE_L_THRESHOLD,
        "official_threshold": DYNAMICQA_OFFICIAL_ROUGE_L_THRESHOLD,
    }


def summarize_dynamicqa_delivery(
    output_rows: Iterable[Mapping[str, Any]],
    *,
    rouge_l_f1: Callable[[str, str], float] = official_rouge_l_f1,
) -> Dict[str, Any]:
    scored: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in output_rows:
        state_id = str(row.get("state_id", "") or "").strip()
        example_id = str(row.get("example_id", "") or "").strip()
        condition_id = str(row.get("condition_id", "") or "").strip()
        if not state_id or not example_id or condition_id not in DYNAMICQA_CONDITIONS:
            raise DynamicQAContractError("Malformed DYNAMICQA output identity")
        key = (state_id, example_id, condition_id)
        if key in scored:
            raise DynamicQAContractError(f"Duplicate DYNAMICQA output key {key!r}")
        answers = row.get("gold_answers")
        if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
            raise DynamicQAContractError(f"DYNAMICQA output {key!r} lacks gold_answers")
        result = score_dynamicqa_answer(
            str(row.get("raw_output", "") or ""),
            [str(answer) for answer in answers],
            rouge_l_f1=rouge_l_f1,
        )
        result["normalized_output"] = normalize_dynamicqa_answer(str(row.get("raw_output", "") or ""))
        scored[key] = result

    summaries: Dict[str, Any] = {}
    state_ids = sorted({key[0] for key in scored})
    base_tool_updates = []
    if "base" in state_ids:
        base_examples = sorted({key[1] for key in scored if key[0] == "base"})
        base_tool_updates = [
            example
            for example in base_examples
            if ("base", example, QUESTION_ONLY) in scored
            and ("base", example, NATIVE_TOOL) in scored
            and not scored[("base", example, QUESTION_ONLY)]["official_correct"]
            and scored[("base", example, NATIVE_TOOL)]["official_correct"]
        ]
    for state_id in state_ids:
        state_rows = {key: value for key, value in scored.items() if key[0] == state_id}
        examples = sorted({key[1] for key in state_rows})
        condition_summary: Dict[str, Any] = {}
        for condition in DYNAMICQA_CONDITIONS:
            values = [state_rows[(state_id, example, condition)] for example in examples if (state_id, example, condition) in state_rows]
            if values:
                condition_summary[condition] = {
                    "n": len(values),
                    "accuracy": sum(bool(value["official_correct"]) for value in values) / len(values),
                    "normalized_exact_match": sum(bool(value["normalized_exact_match"]) for value in values) / len(values),
                }

        paired: Dict[str, Any] = {}
        for condition in DYNAMICQA_CONDITIONS[1:]:
            eligible = [
                example
                for example in examples
                if (state_id, example, QUESTION_ONLY) in state_rows
                and (state_id, example, condition) in state_rows
            ]
            if not eligible:
                continue
            neutral_correct = [state_rows[(state_id, example, QUESTION_ONLY)]["official_correct"] for example in eligible]
            condition_correct = [state_rows[(state_id, example, condition)]["official_correct"] for example in eligible]
            neutral_wrong = [index for index, correct in enumerate(neutral_correct) if not correct]
            paired[condition] = {
                "n": len(eligible),
                "accuracy_gain_over_question_only": (
                    sum(condition_correct) - sum(neutral_correct)
                ) / len(eligible),
                "correct_update_rate": (
                    sum(bool(condition_correct[index]) for index in neutral_wrong) / len(neutral_wrong)
                    if neutral_wrong
                    else None
                ),
                "answer_persistence_rate": sum(
                    state_rows[(state_id, example, condition)]["normalized_output"]
                    == state_rows[(state_id, example, QUESTION_ONLY)]["normalized_output"]
                    for example in eligible
                ) / len(eligible),
            }
        belief_delivery: Dict[str, Any] = {}
        for condition in (I_BELIEVE, I_THINK):
            eligible = [
                example
                for example in examples
                if (state_id, example, QUESTION_ONLY) in state_rows
                and (state_id, example, NATIVE_TOOL) in state_rows
                and (state_id, example, condition) in state_rows
            ]
            if not eligible:
                continue
            neutral_correct = sum(
                bool(state_rows[(state_id, example, QUESTION_ONLY)]["official_correct"])
                for example in eligible
            )
            tool_correct = sum(
                bool(state_rows[(state_id, example, NATIVE_TOOL)]["official_correct"])
                for example in eligible
            )
            belief_correct = sum(
                bool(state_rows[(state_id, example, condition)]["official_correct"])
                for example in eligible
            )
            ignored_denominator = [
                example
                for example in base_tool_updates
                if (state_id, example, condition) in scored
                and ("base", example, QUESTION_ONLY) in scored
            ]
            belief_delivery[condition] = {
                "n": len(eligible),
                "gain_over_question_only": (
                    belief_correct - neutral_correct
                ) / len(eligible),
                "tool_minus_belief_accuracy_gap": (
                    tool_correct - belief_correct
                ) / len(eligible),
                "belief_response_retention_relative_to_tool": (
                    (belief_correct - neutral_correct)
                    / (tool_correct - neutral_correct)
                    if tool_correct != neutral_correct
                    else None
                ),
                "ignored_suggestion_n": len(ignored_denominator),
                "ignored_suggestion_rate": (
                    sum(
                        scored[(state_id, example, condition)]["normalized_output"]
                        == scored[("base", example, QUESTION_ONLY)]["normalized_output"]
                        for example in ignored_denominator
                    )
                    / len(ignored_denominator)
                    if ignored_denominator
                    else None
                ),
            }
        summaries[state_id] = {
            "conditions": condition_summary,
            "paired": paired,
            "base_tool_correct_update_n": len(base_tool_updates),
            "ignored_suggestion_estimand": (
                "among examples where Base question-only is wrong and Base native-tool is correct, "
                "the state belief condition exactly retains Base's normalized question-only answer"
            ),
            "belief_delivery": belief_delivery,
        }
    return {
        "display_name": DYNAMICQA_DISPLAY_NAME,
        "dataset_revision": DYNAMICQA_REVISION,
        "states": summaries,
    }


__all__ = [
    "DYNAMICQA_CONDITIONS",
    "DYNAMICQA_DATASET_ID",
    "DYNAMICQA_DISPLAY_NAME",
    "DYNAMICQA_EXPECTED_ROWS",
    "DYNAMICQA_OFFICIAL_ROUGE_L_THRESHOLD",
    "DYNAMICQA_REVISION",
    "DynamicQAContractError",
    "DynamicQAItem",
    "I_BELIEVE",
    "I_THINK",
    "NATIVE_TOOL",
    "PLAIN_CONTEXT",
    "QUESTION_ONLY",
    "build_dynamicqa_delivery_factorial",
    "dynamicqa_tool_definition",
    "normalize_dynamicqa_answer",
    "normalized_exact_match",
    "official_rouge_l_f1",
    "render_dynamicqa_condition",
    "score_dynamicqa_answer",
    "summarize_dynamicqa_delivery",
]

