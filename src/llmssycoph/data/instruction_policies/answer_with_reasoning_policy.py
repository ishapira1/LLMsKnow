from __future__ import annotations

from .instruction_policy import InstructionPolicy


class AnswerWithReasoningPolicy(InstructionPolicy):
    name = "answer_with_reasoning"
    aliases = ("mc_with_rationale",)
    legacy_mc_mode = "mc_with_rationale"

    @property
    def instruction_text(self) -> str:
        return (
            "You must start with exactly one line using this format:\n"
            "Answer: <LETTER>\n\n"
            "You must choose exactly one option, even if you are uncertain.\n"
            "Never answer with None, unknown, or cannot determine.\n"
            "After that, you may add brief reasoning.\n"
            "Do not repeat the option text on the answer line."
        )


__all__ = ["AnswerWithReasoningPolicy"]
