"""Lazy Cohere client construction for optional judge evaluations."""

from __future__ import annotations

import os
from typing import Any

try:
    import cohere
except ModuleNotFoundError:  # The dependency is optional for non-judge runs.
    cohere = None


def cohere_client(purpose: str = "Cohere evaluation") -> Any:
    """Create a client from either supported environment-variable name."""

    api_key = os.getenv("COHERE_API_KEY") or os.getenv("COHERE_KEY")
    if not api_key:
        raise RuntimeError(
            f"{purpose} requires Cohere credentials; set COHERE_API_KEY or COHERE_KEY"
        )
    if cohere is None:
        raise RuntimeError(
            f"{purpose} requires the optional 'cohere' Python package"
        )
    return cohere.ClientV2(api_key=api_key)
