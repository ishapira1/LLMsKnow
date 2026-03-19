from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .agreement_biases.agreement_bias import AgreementBias
from .instruction_policies.instruction_policy import InstructionPolicy
from .question import Question
from .types import PromptVariant


@dataclass(frozen=True)
class Prompt:
    question: Question
    agreement_bias: AgreementBias
    instruction_policy: InstructionPolicy

    @property
    def prompt_template(self) -> str:
        return self.agreement_bias.prompt_template

    @property
    def template_type(self) -> str:
        return self.agreement_bias.name

    @property
    def prompt_text(self) -> str:
        question_text = str(self.question.question_text or "").strip()
        if not question_text:
            return ""

        prompt_parts = [question_text]
        bias_text = str(self.agreement_bias.bias_text(self.question) or "").strip()
        if bias_text:
            prompt_parts.append(bias_text)

        instruction_text = str(self.instruction_policy.render_instruction(self.question) or "").strip()
        if instruction_text:
            prompt_parts.append(instruction_text)
        return "\n\n".join(prompt_parts)

    def to_prompt_variant(
        self,
        *,
        bias_construction_mode: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVariant:
        variant_metadata = dict(metadata or {})
        variant_metadata.setdefault("instruction_policy", self.instruction_policy.name)
        if self.instruction_policy.response_prefix:
            variant_metadata.setdefault("response_prefix", self.instruction_policy.response_prefix)
        if self.instruction_policy.legacy_mc_mode:
            variant_metadata.setdefault("mc_mode", self.instruction_policy.legacy_mc_mode)
        return PromptVariant(
            question=self.question,
            template_type=self.template_type,
            prompt_template=self.prompt_template,
            prompt_text=self.prompt_text,
            bias_construction_mode=bias_construction_mode,
            metadata=variant_metadata,
        )


__all__ = ["Prompt"]
