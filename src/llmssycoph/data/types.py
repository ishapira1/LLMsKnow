from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


BIAS_VARIANTS_FROM_DATASET = "bias_variants_from_dataset"
BIAS_VARIANTS_GENERATED_LOCALLY = "bias_variants_generated_locally"
ALL_BIAS_CONSTRUCTION_MODES = (
    BIAS_VARIANTS_FROM_DATASET,
    BIAS_VARIANTS_GENERATED_LOCALLY,
)


@dataclass(frozen=True)
class Question:
    dataset: str
    question_text: str
    correct_answer: str
    incorrect_answer: str
    base_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_base_dict(self) -> Dict[str, Any]:
        base = dict(self.base_metadata)
        base.update(
            {
                "dataset": self.dataset,
                "question": self.question_text,
                "question_text": self.question_text,
                "correct_answer": self.correct_answer,
                "incorrect_answer": self.incorrect_answer,
            }
        )
        return base


@dataclass(frozen=True)
class PromptVariant:
    question: Question
    template_type: str
    prompt_template: str
    prompt_text: str
    bias_construction_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_messages(self) -> List[Dict[str, str]]:
        return [{"type": "human", "content": self.prompt_text}]

    def to_row(self) -> Dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.setdefault("prompt_template", self.prompt_template)
        metadata.setdefault("template_type", self.template_type)
        metadata.setdefault("bias_construction_mode", self.bias_construction_mode)
        metadata.setdefault("question_text", self.question.question_text)
        return {
            "prompt": self.prompt_messages,
            "base": self.question.to_base_dict(),
            "metadata": metadata,
        }


@dataclass(frozen=True)
class QuestionGroup:
    question_id: str
    dataset: str
    question_text: str
    correct_answer: str
    incorrect_answer: str
    rows_by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_group_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "dataset": self.dataset,
            "question": self.question_text,
            "question_text": self.question_text,
            "correct_answer": self.correct_answer,
            "incorrect_answer": self.incorrect_answer,
            "rows_by_type": dict(self.rows_by_type),
        }


__all__ = [
    "ALL_BIAS_CONSTRUCTION_MODES",
    "BIAS_VARIANTS_FROM_DATASET",
    "BIAS_VARIANTS_GENERATED_LOCALLY",
    "PromptVariant",
    "Question",
    "QuestionGroup",
]
