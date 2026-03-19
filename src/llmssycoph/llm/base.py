from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GenerationResult:
    response_raw: str
    completion_token_count: Optional[int] = None
    hit_max_new_tokens: bool = False
    stopped_on_eos: bool = False
    finish_reason: str = ""

    @classmethod
    def from_output(cls, output: Any) -> "GenerationResult":
        if isinstance(output, cls):
            return output
        if isinstance(output, dict):
            return cls(
                response_raw=str(output.get("response_raw", "") or ""),
                completion_token_count=output.get("completion_token_count"),
                hit_max_new_tokens=bool(output.get("hit_max_new_tokens", False)),
                stopped_on_eos=bool(output.get("stopped_on_eos", False)),
                finish_reason=str(output.get("finish_reason", "") or ""),
            )
        return cls(response_raw=str(output or ""))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "response_raw": self.response_raw,
            "completion_token_count": self.completion_token_count,
            "hit_max_new_tokens": self.hit_max_new_tokens,
            "stopped_on_eos": self.stopped_on_eos,
            "finish_reason": self.finish_reason,
        }


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        n: int,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 1,
        safe_fallback: bool = True,
        strict_mc_letters: str = "",
    ) -> List[GenerationResult]:
        raise NotImplementedError

    @abstractmethod
    def score_choices(
        self,
        messages: List[Dict[str, Any]],
        choices: List[str],
    ) -> Dict[str, float]:
        raise NotImplementedError

    def get_model_and_tokenizer(self) -> Tuple[Any, Any]:
        raise NotImplementedError(
            f"LLM backend {self.__class__.__name__} does not expose a raw model/tokenizer pair."
        )


__all__ = ["BaseLLM", "GenerationResult"]
