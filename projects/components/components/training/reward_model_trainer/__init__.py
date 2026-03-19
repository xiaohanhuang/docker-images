"""
Training component — train a reward model from preference data for RLHF.

Image: ml-gpu
"""

from ._task import RewardModelOutput, train_reward_model  # noqa: F401

__all__ = ["train_reward_model", "RewardModelOutput"]
