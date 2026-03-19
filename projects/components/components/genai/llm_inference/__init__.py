"""
GenAI — configurable LLM inference with multi-backend support.

Image: genai-gpu
"""

from ._backends import (  # noqa: F401 — re-exported for tests
    _calculate_anthropic_cost,
    _calculate_openai_cost,
    _validate_inputs,
)
from ._task import llm_inference  # noqa: F401

__all__ = ["llm_inference"]
