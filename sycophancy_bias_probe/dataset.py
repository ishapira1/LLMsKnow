from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constants import BIAS_TEMPLATE_TO_TYPE, NEUTRAL_TEMPLATE


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


def question_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    base = row.get("base", {}) or {}
    question = str(base.get("question", "")).strip()
    correct_answer = str(base.get("correct_answer", "")).strip()
    incorrect_answer = str(base.get("incorrect_answer", "")).strip()
    return question, correct_answer, incorrect_answer


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
    if prompt_text_lower.strip() == str(base.get("question", "")).strip().lower():
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
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
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
        question, correct_answer, incorrect_answer = key
        groups.append(
            {
                "question_id": f"q_{idx}",
                "question": question,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "rows_by_type": rows_by_type,
            }
        )
    return groups


def split_groups(
    groups: Sequence[Dict[str, Any]],
    test_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled_groups = list(groups)
    rng = random.Random(seed)
    rng.shuffle(shuffled_groups)

    n_groups = len(shuffled_groups)
    if n_groups == 0:
        return [], []
    if n_groups == 1:
        return shuffled_groups, []

    n_test = int(round(n_groups * test_frac))
    n_test = max(1, min(n_groups - 1, n_test))
    test_groups = shuffled_groups[:n_test]
    train_groups = shuffled_groups[n_test:]
    return train_groups, test_groups
