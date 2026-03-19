from __future__ import annotations

from typing import Dict, List, Sequence, Type

from .agreement_bias import AgreementBias
from .doubt_correct_bias import DoubtCorrectBias
from .incorrect_suggestion_bias import IncorrectSuggestionBias
from .neutral_bias import NeutralBias
from .suggest_correct_bias import SuggestCorrectBias


AGREEMENT_BIAS_TYPES: tuple[Type[AgreementBias], ...] = (
    NeutralBias,
    IncorrectSuggestionBias,
    DoubtCorrectBias,
    SuggestCorrectBias,
)
AGREEMENT_BIAS_REGISTRY: Dict[str, Type[AgreementBias]] = {
    bias_type.name: bias_type for bias_type in AGREEMENT_BIAS_TYPES
}


def get_agreement_bias(name: str) -> AgreementBias:
    normalized_name = str(name or "").strip()
    bias_type = AGREEMENT_BIAS_REGISTRY.get(normalized_name)
    if bias_type is None:
        raise ValueError(
            f"Unknown agreement bias {normalized_name!r}. Valid: {sorted(AGREEMENT_BIAS_REGISTRY)}"
        )
    return bias_type()


def resolve_agreement_biases(
    names: Sequence[str],
    *,
    include_neutral: bool = False,
) -> List[AgreementBias]:
    ordered_names = [str(name).strip() for name in names if str(name).strip()]
    if include_neutral:
        ordered_names = ["neutral", *ordered_names]
    seen = set()
    resolved: List[AgreementBias] = []
    for name in ordered_names:
        if name in seen:
            continue
        resolved.append(get_agreement_bias(name))
        seen.add(name)
    return resolved


__all__ = [
    "AGREEMENT_BIAS_REGISTRY",
    "AGREEMENT_BIAS_TYPES",
    "get_agreement_bias",
    "resolve_agreement_biases",
]
