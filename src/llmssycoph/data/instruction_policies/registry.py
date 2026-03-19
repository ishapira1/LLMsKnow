from __future__ import annotations

from typing import Dict, Type

from .answer_only_policy import AnswerOnlyPolicy
from .answer_with_reasoning_policy import AnswerWithReasoningPolicy
from .brief_answer_policy import BriefAnswerPolicy
from .instruction_policy import InstructionPolicy


INSTRUCTION_POLICY_ANSWER_ONLY = AnswerOnlyPolicy.name
INSTRUCTION_POLICY_ANSWER_WITH_REASONING = AnswerWithReasoningPolicy.name
DEFAULT_INSTRUCTION_POLICY_NAME = INSTRUCTION_POLICY_ANSWER_ONLY

INSTRUCTION_POLICY_TYPES: tuple[Type[InstructionPolicy], ...] = (
    BriefAnswerPolicy,
    AnswerOnlyPolicy,
    AnswerWithReasoningPolicy,
)
INSTRUCTION_POLICY_REGISTRY: Dict[str, Type[InstructionPolicy]] = {
    policy_type.name: policy_type for policy_type in INSTRUCTION_POLICY_TYPES
}
INSTRUCTION_POLICY_ALIASES: Dict[str, str] = {
    alias: policy_type.name
    for policy_type in INSTRUCTION_POLICY_TYPES
    for alias in policy_type.aliases
}
VISIBLE_INSTRUCTION_POLICY_NAMES = tuple(
    policy_type.name for policy_type in INSTRUCTION_POLICY_TYPES if policy_type.visible_in_cli
)


def canonical_instruction_policy_name(
    value: str | None,
    *,
    default_name: str = DEFAULT_INSTRUCTION_POLICY_NAME,
) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        normalized = default_name
    if normalized in INSTRUCTION_POLICY_REGISTRY:
        return normalized
    aliased = INSTRUCTION_POLICY_ALIASES.get(normalized)
    if aliased:
        return aliased
    valid = sorted(set(INSTRUCTION_POLICY_REGISTRY) | set(INSTRUCTION_POLICY_ALIASES))
    raise ValueError(f"Unknown instruction policy {value!r}. Valid values: {valid}")


def get_instruction_policy(
    value: str | None = None,
    *,
    default_name: str = DEFAULT_INSTRUCTION_POLICY_NAME,
) -> InstructionPolicy:
    policy_name = canonical_instruction_policy_name(value, default_name=default_name)
    return INSTRUCTION_POLICY_REGISTRY[policy_name]()


def legacy_mc_mode_for_instruction_policy(value: str | None) -> str:
    return get_instruction_policy(value).legacy_mc_mode


__all__ = [
    "DEFAULT_INSTRUCTION_POLICY_NAME",
    "INSTRUCTION_POLICY_ALIASES",
    "INSTRUCTION_POLICY_ANSWER_ONLY",
    "INSTRUCTION_POLICY_ANSWER_WITH_REASONING",
    "INSTRUCTION_POLICY_REGISTRY",
    "INSTRUCTION_POLICY_TYPES",
    "VISIBLE_INSTRUCTION_POLICY_NAMES",
    "canonical_instruction_policy_name",
    "get_instruction_policy",
    "legacy_mc_mode_for_instruction_policy",
]
