from __future__ import annotations

from typing import Dict, List, Sequence, Type

from .agreement_bias import AgreementBias
from .doubt_correct_bias import DoubtCorrectBias
from .doubt_correct_strong_bias import DoubtCorrectStrongBias
from .incorrect_suggestion_bias import IncorrectSuggestionBias
from .incorrect_suggestion_strong_bias import IncorrectSuggestionStrongBias
from .neutral_bias import NeutralBias
from .suggest_correct_bias import SuggestCorrectBias
from .suggest_correct_strong_bias import SuggestCorrectStrongBias
from .suggest_random_bias import SuggestRandomBias
from .suggest_random_strong_bias import SuggestRandomStrongBias
from ..prompt_families import resolve_prompt_families


AGREEMENT_BIAS_TYPES: tuple[Type[AgreementBias], ...] = (
    NeutralBias,
    IncorrectSuggestionBias,
    IncorrectSuggestionStrongBias,
    DoubtCorrectBias,
    DoubtCorrectStrongBias,
    SuggestCorrectBias,
    SuggestCorrectStrongBias,
    SuggestRandomBias,
    SuggestRandomStrongBias,
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
    resolved: List[AgreementBias] = []
    for prompt_family in resolve_prompt_families(names, include_neutral=include_neutral):
        if prompt_family.family_id not in AGREEMENT_BIAS_REGISTRY:
            continue
        resolved.append(get_agreement_bias(prompt_family.family_id))
    return resolved


__all__ = [
    "AGREEMENT_BIAS_REGISTRY",
    "AGREEMENT_BIAS_TYPES",
    "get_agreement_bias",
    "resolve_agreement_biases",
]
