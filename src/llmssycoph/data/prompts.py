from __future__ import annotations

from dataclasses import dataclass

from .instruction_policies import get_instruction_policy
from .prompt import Prompt
from .prompt_instructions import (
    ALL_INSTRUCTION_POLICIES,
    ALL_MC_MODES,
    ANSWER_ONLY_OUTPUT_INSTRUCTION,
    ANSWER_WITH_REASONING_OUTPUT_INSTRUCTION,
    AYS_MC_BRIEF_INSTRUCTION,
    DEFAULT_INSTRUCTION_POLICY,
    INSTRUCTION_POLICY_ANSWER_ONLY,
    INSTRUCTION_POLICY_ANSWER_WITH_REASONING,
    MC_MODE_STRICT,
    MC_MODE_WITH_RATIONALE,
    MC_WITH_RATIONALE_OUTPUT_INSTRUCTION,
    PromptInstructions,
    STRICT_MC_OUTPUT_INSTRUCTION,
    output_instruction_for_mode,
)

PROMPT_SPEC_VERSION = 4
GRADING_SPEC_VERSION = 3
STRICT_OUTPUT_CONTRACT = "answer_line_letter_only"


from .agreement_biases.registry import AGREEMENT_BIAS_TYPES, get_agreement_bias


class BiasInjectionTemplates:
    TEMPLATE_TO_TYPE = {
        bias_class().prompt_template: bias_class.name
        for bias_class in AGREEMENT_BIAS_TYPES
        if bias_class.name != "neutral"
    }
    PROMPT_TEMPLATE_BY_TYPE = {
        bias_class.name: bias_class().prompt_template
        for bias_class in AGREEMENT_BIAS_TYPES
    }
    NEUTRAL = PROMPT_TEMPLATE_BY_TYPE["neutral"]


BIAS_TEMPLATE_TO_TYPE = dict(BiasInjectionTemplates.TEMPLATE_TO_TYPE)
PROMPT_TEMPLATE_BY_TYPE = dict(BiasInjectionTemplates.PROMPT_TEMPLATE_BY_TYPE)
NEUTRAL_TEMPLATE = BiasInjectionTemplates.NEUTRAL
ALL_BIAS_TYPES = tuple(BIAS_TEMPLATE_TO_TYPE.values())


@dataclass(frozen=True)
class PromptBuilder:
    def output_instruction_for_mode(self, mc_mode: str) -> str:
        return output_instruction_for_mode(mc_mode)

    def output_instruction_for_policy(self, instruction_policy: str) -> str:
        return get_instruction_policy(instruction_policy).render_instruction()

    def bias_text(
        self,
        template_type: str,
        correct_answer: str,
        incorrect_answer: str,
    ) -> str:
        from .agreement_biases import get_agreement_bias
        from .question import Question

        question = Question(
            dataset="",
            question_text="",
            correct_answer=correct_answer,
            incorrect_answer=incorrect_answer,
        )
        return get_agreement_bias(template_type).bias_text(question)

    def render_prompt_text(
        self,
        question_text: str,
        template_type: str,
        correct_answer: str,
        incorrect_answer: str,
        mc_mode: str = MC_MODE_STRICT,
        instruction_policy: str | None = None,
    ) -> str:
        question_text = str(question_text or "").strip()
        if not question_text:
            return ""

        from .question import Question

        prompt = Prompt(
            question=Question(
                dataset="",
                question_text=question_text,
                correct_answer=correct_answer,
                incorrect_answer=incorrect_answer,
            ),
            agreement_bias=get_agreement_bias(template_type),
            instruction_policy=get_instruction_policy(instruction_policy or mc_mode),
        )
        return prompt.prompt_text


default_prompt_builder = PromptBuilder()


__all__ = [
    "ALL_BIAS_TYPES",
    "ALL_INSTRUCTION_POLICIES",
    "ALL_MC_MODES",
    "ANSWER_ONLY_OUTPUT_INSTRUCTION",
    "ANSWER_WITH_REASONING_OUTPUT_INSTRUCTION",
    "AYS_MC_BRIEF_INSTRUCTION",
    "BIAS_TEMPLATE_TO_TYPE",
    "BiasInjectionTemplates",
    "DEFAULT_INSTRUCTION_POLICY",
    "GRADING_SPEC_VERSION",
    "INSTRUCTION_POLICY_ANSWER_ONLY",
    "INSTRUCTION_POLICY_ANSWER_WITH_REASONING",
    "MC_MODE_STRICT",
    "MC_MODE_WITH_RATIONALE",
    "MC_WITH_RATIONALE_OUTPUT_INSTRUCTION",
    "NEUTRAL_TEMPLATE",
    "PROMPT_SPEC_VERSION",
    "PROMPT_TEMPLATE_BY_TYPE",
    "PromptBuilder",
    "PromptInstructions",
    "STRICT_MC_OUTPUT_INSTRUCTION",
    "STRICT_OUTPUT_CONTRACT",
    "default_prompt_builder",
    "output_instruction_for_mode",
]
