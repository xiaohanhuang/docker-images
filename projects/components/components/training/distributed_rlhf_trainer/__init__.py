"""
Distributed RLHF training component using OpenRLHF.

This component implements the multi-service distributed RLHF architecture
described in docs/openrlhf-distributed-implementation.md, replacing the
single-GPU training loop with OpenRLHF's Ray-based distributed system.

When OpenRLHF is available (production ``training-llm`` image) it delegates
to ``openrlhf.cli.train_ppo_ray``; otherwise it uses a native PyTorch
training loop that works on any transformers-compatible GPU container (the
standard ``ml-gpu`` image).

Architecture (production mode):
    - Actor: Policy model training + rollout generation (DeepSpeed/FSDP)
    - Critic: Value network for advantage estimation (DeepSpeed/FSDP)
    - Reference: Frozen SFT model for KL penalty (vLLM inference)
    - Reward: Score responses via gRPC (vLLM or custom server)
    - Buffer: Stream rollouts between services (Redis Cluster)

Image: ml-gpu (native fallback) or training-llm (with openrlhf[vllm])

Implementation is split across submodules:
    - ``_helpers``: S3 download, MLflow lifecycle, algorithm map
    - ``_native``: Native PyTorch fallback training loop
    - ``_openrlhf``: Ray/OpenRLHF distributed training backend
    - ``_task``: Flyte task definition
"""

from ._helpers import download_s3, end_mlflow, start_mlflow  # noqa: F401
from ._task import DistributedRLHFTrainerOutput, distributed_rlhf_trainer  # noqa: F401

__all__ = ["distributed_rlhf_trainer", "DistributedRLHFTrainerOutput"]
