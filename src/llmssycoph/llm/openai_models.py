from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class OpenAIModelSpec:
    model_name: str
    reasoning_effort: Optional[str] = None


OPENAI_MODEL_SPECS: Dict[str, OpenAIModelSpec] = {
    "gpt-5.4": OpenAIModelSpec(model_name="gpt-5.4", reasoning_effort="none"),
    "gpt-5.4-mini": OpenAIModelSpec(model_name="gpt-5.4-mini", reasoning_effort="none"),
    "gpt-5.4-nano": OpenAIModelSpec(model_name="gpt-5.4-nano", reasoning_effort="none"),
    "gpt-5.4-pro": OpenAIModelSpec(model_name="gpt-5.4-pro", reasoning_effort=None),
    "gpt-5.2": OpenAIModelSpec(model_name="gpt-5.2", reasoning_effort="none"),
    "gpt-5.2-pro": OpenAIModelSpec(model_name="gpt-5.2-pro", reasoning_effort="none"),
    "gpt-5.1": OpenAIModelSpec(model_name="gpt-5.1", reasoning_effort=None),
    "gpt-5": OpenAIModelSpec(model_name="gpt-5", reasoning_effort=None),
    "gpt-5-pro": OpenAIModelSpec(model_name="gpt-5-pro", reasoning_effort=None),
    "gpt-5-mini": OpenAIModelSpec(model_name="gpt-5-mini", reasoning_effort=None),
    "gpt-5-nano": OpenAIModelSpec(model_name="gpt-5-nano", reasoning_effort=None),
}


__all__ = ["OPENAI_MODEL_SPECS", "OpenAIModelSpec"]
