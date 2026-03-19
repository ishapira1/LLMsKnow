from __future__ import annotations

from .instruction_policies import (
    DEFAULT_INSTRUCTION_POLICY_NAME,
    INSTRUCTION_POLICY_ANSWER_ONLY,
    INSTRUCTION_POLICY_ANSWER_WITH_REASONING,
    VISIBLE_INSTRUCTION_POLICY_NAMES,
    canonical_instruction_policy_name,
    get_instruction_policy,
    legacy_mc_mode_for_instruction_policy,
)

MC_MODE_STRICT = "strict_mc"
MC_MODE_WITH_RATIONALE = "mc_with_rationale"
ALL_MC_MODES = (MC_MODE_STRICT, MC_MODE_WITH_RATIONALE)
ALL_INSTRUCTION_POLICIES = VISIBLE_INSTRUCTION_POLICY_NAMES
DEFAULT_INSTRUCTION_POLICY = DEFAULT_INSTRUCTION_POLICY_NAME


class PromptInstructions:
    BRIEF = get_instruction_policy("brief_answer").instruction_text
    ANSWER_ONLY = get_instruction_policy(INSTRUCTION_POLICY_ANSWER_ONLY).instruction_text
    ANSWER_WITH_REASONING = get_instruction_policy(
        INSTRUCTION_POLICY_ANSWER_WITH_REASONING
    ).instruction_text
    STRICT_MC = ANSWER_ONLY
    MC_WITH_RATIONALE = ANSWER_WITH_REASONING


AYS_MC_BRIEF_INSTRUCTION = PromptInstructions.BRIEF
ANSWER_ONLY_OUTPUT_INSTRUCTION = PromptInstructions.ANSWER_ONLY
ANSWER_WITH_REASONING_OUTPUT_INSTRUCTION = PromptInstructions.ANSWER_WITH_REASONING
STRICT_MC_OUTPUT_INSTRUCTION = PromptInstructions.STRICT_MC
MC_WITH_RATIONALE_OUTPUT_INSTRUCTION = PromptInstructions.MC_WITH_RATIONALE


def output_instruction_for_mode(mc_mode: str) -> str:
    return get_instruction_policy(
        canonical_instruction_policy_name(mc_mode, default_name="brief_answer")
    ).render_instruction()


def output_instruction_for_policy(instruction_policy: str) -> str:
    return get_instruction_policy(instruction_policy).render_instruction()


__all__ = [
    "ALL_INSTRUCTION_POLICIES",
    "ALL_MC_MODES",
    "ANSWER_ONLY_OUTPUT_INSTRUCTION",
    "ANSWER_WITH_REASONING_OUTPUT_INSTRUCTION",
    "AYS_MC_BRIEF_INSTRUCTION",
    "DEFAULT_INSTRUCTION_POLICY",
    "INSTRUCTION_POLICY_ANSWER_ONLY",
    "INSTRUCTION_POLICY_ANSWER_WITH_REASONING",
    "MC_MODE_STRICT",
    "MC_MODE_WITH_RATIONALE",
    "MC_WITH_RATIONALE_OUTPUT_INSTRUCTION",
    "PromptInstructions",
    "STRICT_MC_OUTPUT_INSTRUCTION",
    "canonical_instruction_policy_name",
    "legacy_mc_mode_for_instruction_policy",
    "output_instruction_for_mode",
    "output_instruction_for_policy",
]
