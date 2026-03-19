from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..instruction_policies import InstructionPolicy, get_instruction_policy
from ..question import Question
from ..types import PromptVariant


class AgreementBias(ABC):
    name: str = ""

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def bias_text(self, question: Question) -> str:
        raise NotImplementedError

    def _resolve_instruction_policy(
        self,
        instruction_policy: InstructionPolicy | str | None = None,
        *,
        mc_mode: str | None = None,
    ) -> InstructionPolicy:
        if isinstance(instruction_policy, InstructionPolicy):
            return instruction_policy
        if instruction_policy is not None:
            return get_instruction_policy(instruction_policy)
        return get_instruction_policy(mc_mode)

    def render_prompt_text(
        self,
        question: Question,
        instruction_policy: InstructionPolicy | str | None = None,
        *,
        mc_mode: str | None = None,
    ) -> str:
        from ..prompt import Prompt

        resolved_policy = self._resolve_instruction_policy(instruction_policy, mc_mode=mc_mode)
        return Prompt(
            question=question,
            agreement_bias=self,
            instruction_policy=resolved_policy,
        ).prompt_text

    def build_prompt_variant(
        self,
        question: Question,
        *,
        instruction_policy: InstructionPolicy | str | None = None,
        mc_mode: str | None = None,
        bias_construction_mode: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVariant:
        from ..prompt import Prompt

        resolved_policy = self._resolve_instruction_policy(instruction_policy, mc_mode=mc_mode)
        return Prompt(
            question=question,
            agreement_bias=self,
            instruction_policy=resolved_policy,
        ).to_prompt_variant(
            bias_construction_mode=bias_construction_mode,
            metadata=metadata,
        )


__all__ = ["AgreementBias"]
