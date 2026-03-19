"""
Preference data generator component — AI-powered preference pair creation for DPO/RLHF.

Image: genai-gpu
"""

from ._generators import _get_generator  # noqa: F401 — re-exported for tests
from ._judges import _get_judge  # noqa: F401 — re-exported for tests
from ._task import generate_preference_data  # noqa: F401

__all__ = ["generate_preference_data"]
