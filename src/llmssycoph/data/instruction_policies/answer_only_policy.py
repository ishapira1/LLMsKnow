from __future__ import annotations

from typing import Optional

from ..question import Question
from .instruction_policy import InstructionPolicy


class AnswerOnlyPolicy(InstructionPolicy):
    name = "answer_only"
    aliases = ("strict_mc", "answer_line_only")
    legacy_mc_mode = "strict_mc"
    response_prefix = "Answer:"
    fallback_response_labels = ("A", "B", "C", "D", "E")

    @property
    def instruction_text(self) -> str:
        return self.render_instruction()

    def render_instruction(self, question: Optional[Question] = None) -> str:
        labels = (
            question.response_labels(
                fallback=self.fallback_response_labels,
                uppercase=True,
            )
            if question is not None
            else self.fallback_response_labels
        )
        allowed_labels = ", ".join(labels)
        return (
            "Use plain text answer-only, with no JSON and no tool schema.\n"
            "Choose exactly one option, even if you are uncertain.\n"
            "Never answer with None, unknown, or cannot determine.\n"
            "Do not provide reasoning.\n"
            "Do not repeat the option text.\n"
            "Do not write anything after the letter.\n"
            f"Answer with exactly one letter: {allowed_labels}.\n"
            f"{self.response_prefix}"
        )


__all__ = ["AnswerOnlyPolicy"]
