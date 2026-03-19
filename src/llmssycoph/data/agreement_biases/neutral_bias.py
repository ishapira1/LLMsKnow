from __future__ import annotations

from ..question import Question
from .agreement_bias import AgreementBias


class NeutralBias(AgreementBias):
    name = "neutral"
    template = "{question}"

    @property
    def prompt_template(self) -> str:
        return self.template

    def bias_text(self, question: Question) -> str:
        del question
        return ""


__all__ = ["NeutralBias"]
