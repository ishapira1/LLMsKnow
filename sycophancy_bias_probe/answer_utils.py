from __future__ import annotations

import re
from typing import Any, Dict, List


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def extract_gold_answers_from_base(base: Dict[str, Any]) -> List[str]:
    if not isinstance(base, dict):
        return []

    for key in ["answers", "answer", "gold", "label", "target", "gold_answer", "ground_truth"]:
        if key not in base:
            continue
        value = base[key]
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            aliases: List[str] = []
            if "value" in value and isinstance(value["value"], str):
                aliases.append(value["value"])
            if "aliases" in value and isinstance(value["aliases"], list):
                aliases.extend([alias for alias in value["aliases"] if isinstance(alias, str)])
            if "normalized_aliases" in value and isinstance(value["normalized_aliases"], list):
                aliases.extend(
                    [alias for alias in value["normalized_aliases"] if isinstance(alias, str)]
                )
            aliases = [alias for alias in aliases if alias]
            if aliases:
                return list(dict.fromkeys(aliases))
        if isinstance(value, list):
            aliases = []
            for item in value:
                if isinstance(item, str):
                    aliases.append(item)
                elif isinstance(item, dict):
                    if "value" in item and isinstance(item["value"], str):
                        aliases.append(item["value"])
                    if "aliases" in item and isinstance(item["aliases"], list):
                        aliases.extend([alias for alias in item["aliases"] if isinstance(alias, str)])
            aliases = [alias for alias in aliases if alias]
            if aliases:
                return list(dict.fromkeys(aliases))

    if "answer_aliases" in base and isinstance(base["answer_aliases"], list):
        aliases = [alias for alias in base["answer_aliases"] if isinstance(alias, str) and alias]
        if aliases:
            return list(dict.fromkeys(aliases))

    return []


def normalize_answer(text: str) -> str:
    normalized = text.strip().lower()
    normalized = _WS_RE.sub(" ", normalized)
    normalized = _PUNCT_RE.sub("", normalized)
    normalized = _WS_RE.sub(" ", normalized).strip()
    return normalized


def is_correct_short_answer(pred: str, gold_answers: List[str]) -> bool:
    if not pred or not gold_answers:
        return False

    normalized_pred = normalize_answer(pred)
    gold_norm = [normalize_answer(gold) for gold in gold_answers if gold]

    if normalized_pred in set(gold_norm):
        return True

    for gold in gold_norm:
        if not gold:
            continue
        if re.search(rf"(?:^|\s){re.escape(gold)}(?:$|\s)", normalized_pred):
            return True

    return False


def extract_short_answer_from_generation(text: str) -> str:
    short = text.strip()

    match = re.search(r"(?:final answer|answer)\s*:\s*(.+)$", short, flags=re.IGNORECASE)
    if match:
        short = match.group(1).strip()

    short = short.splitlines()[0].strip()
    short = re.split(r"[.?!]\s+", short, maxsplit=1)[0].strip()
    short = short.strip(" \"'“”‘’\t")

    return short


__all__ = [
    "extract_gold_answers_from_base",
    "normalize_answer",
    "is_correct_short_answer",
    "extract_short_answer_from_generation",
]
