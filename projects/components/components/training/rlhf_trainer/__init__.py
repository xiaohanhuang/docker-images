"""
Training component — RL alignment via native PPO / REINFORCE++.

Supports PPO, REINFORCE++, GRPO, and RLOO algorithms using a lightweight
native training loop built on transformers + torch.  When OpenRLHF is
available (production image) it delegates to ``openrlhf.cli.*``; otherwise
it uses the built-in loop which works on any transformers-compatible GPU
container.

**NOTE**: For distributed RLHF training with OpenRLHF on multiple GPUs/nodes,
see ``distributed_rlhf_trainer.py`` which provides a Ray-based multi-service
architecture with DeepSpeed ZeRO-3, vLLM inference, and Redis trajectory
buffering. This component is optimized for single-GPU smoke tests and
prototyping.

Image: ml-gpu
"""

from ._task import _ALGORITHM_MAP, RLHFTrainerOutput, rlhf_trainer  # noqa: F401

__all__ = ["rlhf_trainer", "RLHFTrainerOutput"]
