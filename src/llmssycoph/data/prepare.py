from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .prompts import BIAS_TEMPLATE_TO_TYPE, NEUTRAL_TEMPLATE
from .types import QuestionGroup


def as_prompt_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    chunks: List[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content.strip())
    return "\n".join([chunk for chunk in chunks if chunk])


def dataset_name(row: Dict[str, Any]) -> str:
    base = row.get("base", {}) or {}
    value = base.get("dataset")
    if isinstance(value, str):
        return value.strip()
    return ""


def unique_dataset_names(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({name for row in rows if (name := dataset_name(row))})


def _normalized_dataset_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _matches_dataset_name(row: Dict[str, Any], selected_dataset_name: str) -> bool:
    wanted = _normalized_dataset_name(selected_dataset_name)
    if not wanted or wanted == "all":
        return True
    return _normalized_dataset_name(dataset_name(row)) == wanted


def _row_question_text(row: Dict[str, Any]) -> str:
    base = row.get("base", {}) or {}
    return str(base.get("question_text", base.get("question", "")) or "").strip()


def question_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    base = row.get("base", {}) or {}
    dataset = dataset_name(row)
    question = _row_question_text(row)
    correct_answer = str(base.get("correct_answer", "")).strip()
    incorrect_answer = str(base.get("incorrect_answer", "")).strip()
    return dataset, question, correct_answer, incorrect_answer


def template_type(row: Dict[str, Any]) -> Optional[str]:
    metadata = row.get("metadata", {}) or {}
    template = metadata.get("prompt_template")
    if template == NEUTRAL_TEMPLATE:
        return "neutral"
    if template in BIAS_TEMPLATE_TO_TYPE:
        return BIAS_TEMPLATE_TO_TYPE[template]

    prompt_text = as_prompt_text(row.get("prompt", []))
    base = row.get("base", {}) or {}
    correct_answer = str(base.get("correct_answer", ""))
    incorrect_answer = str(base.get("incorrect_answer", ""))
    prompt_text_lower = prompt_text.lower()
    if "i don't think the answer is" in prompt_text_lower and correct_answer and correct_answer.lower() in prompt_text_lower:
        return "doubt_correct"
    if "i think the answer is" in prompt_text_lower and "really not sure" in prompt_text_lower:
        if incorrect_answer and incorrect_answer.lower() in prompt_text_lower:
            return "incorrect_suggestion"
        if correct_answer and correct_answer.lower() in prompt_text_lower:
            return "suggest_correct"
    if prompt_text_lower.strip() == _row_question_text(row).lower():
        return "neutral"
    return None


def deduplicate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduplicated = []
    for row in rows:
        key = (
            question_key(row),
            template_type(row),
            json.dumps(row.get("prompt", []), sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)
    return deduplicated


def build_question_groups(
    rows: Sequence[Dict[str, Any]],
    selected_bias_types: Sequence[str],
    selected_dataset_name: str = "all",
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not _matches_dataset_name(row, selected_dataset_name):
            continue
        row_template_type = template_type(row)
        if row_template_type is None:
            continue
        if row_template_type != "neutral" and row_template_type not in selected_bias_types:
            continue
        grouped[question_key(row)][row_template_type] = row

    groups: List[Dict[str, Any]] = []
    for idx, (key, rows_by_type) in enumerate(grouped.items()):
        if "neutral" not in rows_by_type:
            continue
        if not all(bias_type in rows_by_type for bias_type in selected_bias_types):
            continue
        dataset, question_text, correct_answer, incorrect_answer = key
        group = QuestionGroup(
            question_id=f"q_{idx}",
            dataset=dataset,
            question_text=question_text,
            correct_answer=correct_answer,
            incorrect_answer=incorrect_answer,
            rows_by_type=rows_by_type,
        )
        groups.append(group.to_group_dict())
    return groups


def split_groups_train_val_test(
    groups: Sequence[Dict[str, Any]],
    test_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled_groups = list(groups)
    rng = random.Random(seed)
    rng.shuffle(shuffled_groups)

    n_groups = len(shuffled_groups)
    if n_groups == 0:
        return [], [], []
    if n_groups == 1:
        return shuffled_groups, [], []

    if test_frac <= 0:
        n_test = 0
    else:
        n_test = int(round(n_groups * test_frac))
        n_test = max(1, min(n_groups - 1, n_test))

    test_groups = shuffled_groups[:n_test]
    train_val_groups = shuffled_groups[n_test:]

    remaining_after_test = len(train_val_groups)
    if remaining_after_test <= 1 or val_frac <= 0:
        n_val = 0
    else:
        n_val = int(round(remaining_after_test * val_frac))
        n_val = max(1, min(remaining_after_test - 1, n_val))

    val_groups = train_val_groups[:n_val]
    train_groups = train_val_groups[n_val:]
    return train_groups, val_groups, test_groups


def split_groups(
    groups: Sequence[Dict[str, Any]],
    test_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_groups, _, test_groups = split_groups_train_val_test(
        groups,
        test_frac=test_frac,
        val_frac=0.0,
        seed=seed,
    )
    return train_groups, test_groups


__all__ = [
    "as_prompt_text",
    "build_question_groups",
    "dataset_name",
    "deduplicate_rows",
    "question_key",
    "split_groups",
    "split_groups_train_val_test",
    "template_type",
    "unique_dataset_names",
]
