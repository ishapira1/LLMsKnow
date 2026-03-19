from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..question import Question


class InstructionPolicy(ABC):
    name: str = ""
    aliases: tuple[str, ...] = ()
    legacy_mc_mode: str = ""
    visible_in_cli: bool = True
    response_prefix: str = ""

    @property
    @abstractmethod
    def instruction_text(self) -> str:
        raise NotImplementedError

    def render_instruction(self, question: Optional[Question] = None) -> str:
        del question
        return str(self.instruction_text or "").strip()


__all__ = ["InstructionPolicy"]
