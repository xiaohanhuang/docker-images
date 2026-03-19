"""
Evaluation component — benchmark reward models against RewardBench suite.

Image: ml-gpu

This component evaluates reward models on the RewardBench benchmark, which tests
how well a reward model discriminates between good and bad outputs across multiple
categories: chat, safety, reasoning, and code.
"""

from ._task import CATEGORY_MAP, benchmark_reward_model  # noqa: F401

__all__ = ["benchmark_reward_model", "CATEGORY_MAP"]
