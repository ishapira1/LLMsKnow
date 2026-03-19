from __future__ import annotations

from .instruction_policy import InstructionPolicy


class BriefAnswerPolicy(InstructionPolicy):
    name = "brief_answer"
    aliases = ("brief",)
    legacy_mc_mode = ""
    visible_in_cli = False

    @property
    def instruction_text(self) -> str:
        return "Please answer briefly with the option text or letter first."


__all__ = ["BriefAnswerPolicy"]
