from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple


RESPONSE_LABEL_METADATA_KEYS: tuple[str, ...] = (
    "response_labels",
    "option_labels",
    "choice_labels",
    "labels",
    "letters",
)


def _normalize_response_labels(
    value: Any,
    *,
    uppercase: bool = False,
) -> Tuple[str, ...]:
    labels: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if "," in text:
            labels = [part.strip() for part in text.split(",") if part.strip()]
        elif len(text) > 1 and all(ch.isalnum() for ch in text):
            labels = [ch for ch in text]
        else:
            labels = [text]
    elif isinstance(value, dict):
        labels = [str(key).strip() for key in value if str(key).strip()]
    elif isinstance(value, (list, tuple)):
        labels = [str(item).strip() for item in value if str(item).strip()]

    if uppercase:
        labels = [label.upper() for label in labels]
    return tuple(labels)


@dataclass(frozen=True)
class Question:
    dataset: str
    question_text: str
    correct_answer: str
    incorrect_answer: str
    base_metadata: Dict[str, Any] = field(default_factory=dict)

    def response_labels(
        self,
        *,
        fallback: Sequence[str] = (),
        metadata_keys: Sequence[str] = RESPONSE_LABEL_METADATA_KEYS,
        uppercase: bool = False,
    ) -> Tuple[str, ...]:
        for key in metadata_keys:
            labels = _normalize_response_labels(self.base_metadata.get(key), uppercase=uppercase)
            if labels:
                return labels
        return _normalize_response_labels(list(fallback), uppercase=uppercase)

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


__all__ = ["Question", "RESPONSE_LABEL_METADATA_KEYS"]
