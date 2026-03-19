"""
Evaluation component — LLM-as-Judge for evaluating model outputs.

Use a strong LLM (GPT-4, Claude, Llama-70B) to evaluate outputs from a weaker model.
Scores for hallucination, relevance, coherence, toxicity, and custom criteria.

Image: genai-gpu
"""

from ._backends import (  # noqa: F401 — re-exported for tests
    _build_evaluation_prompt,
    _call_anthropic_judge,
    _call_openai_judge,
    _call_vllm_judge,
)
from ._evaluation import (  # noqa: F401
    _heuristic_evaluate,
    _load_predictions_for_llm_judge,
    _parse_s3_uri,
)
from ._rubrics import DEFAULT_RUBRICS  # noqa: F401
from ._task import llm_judge  # noqa: F401

__all__ = [
    "DEFAULT_RUBRICS",
    "_build_evaluation_prompt",
    "_call_anthropic_judge",
    "_call_openai_judge",
    "_call_vllm_judge",
    "_heuristic_evaluate",
    "_load_predictions_for_llm_judge",
    "_parse_s3_uri",
    "llm_judge",
]
