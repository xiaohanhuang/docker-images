"""Flyte task definition for rlhf_trainer component."""

from typing import Dict, NamedTuple

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile


class RLHFTrainerOutput(NamedTuple):
    """Output from RLHF training."""

    checkpoint_path: FlyteDirectory
    mlflow_run_id: str
    reward_stats: Dict[str, float]
    kl_divergence: float


# Map user-friendly algorithm names to internal identifiers.
_ALGORITHM_MAP = {
    "ppo": "ppo",
    "reinforce": "reinforce",  # REINFORCE++
    "reinforce_baseline": "reinforce_baseline",
    "grpo": "grpo",  # Group Relative Policy Optimization
    "rloo": "rloo",  # Leave-One-Out
}


@task(
    retries=1,
    requests=Resources(cpu="4", mem="24Gi", gpu="1", ephemeral_storage="80Gi"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1", ephemeral_storage="120Gi"),
    cache=False,
)
def rlhf_trainer(
    sft_model_path: FlyteDirectory,
    reward_model_path: FlyteDirectory,
    dataset_path: FlyteFile,
    algorithm: str = "ppo",
    prompt_column: str = "prompt",
    ppo_epochs: int = 1,
    learning_rate: float = 5e-7,
    batch_size: int = 4,
    mini_batch_size: int = 4,
    kl_penalty: str = "kl",
    init_kl_coef: float = 0.01,
    target_kl: float = 0.1,
    max_new_tokens: int = 128,
    num_gpus: int = 1,
    gradient_checkpointing: bool = True,
    use_deepspeed: bool = False,
    checkpoint_interval: int = -1,
    num_training_steps: int = 1000,
) -> RLHFTrainerOutput:
    """Align an LLM using RL with PPO / REINFORCE++ / GRPO / RLOO.

    Uses a native training loop built on transformers + torch that works
    on any GPU container with standard ML dependencies.  Supports LoRA
    adapter checkpoints for both the SFT and reward models.

    Algorithms:
      * ppo       — Proximal Policy Optimization with clipped ratio
      * reinforce — REINFORCE++ (critic-free, KL-penalized)
      * grpo      — Group Relative Policy Optimization
      * rloo      — Leave-One-Out baseline estimation

    Args:
        sft_model_path: S3 path or HuggingFace Hub ID of the SFT model.
        reward_model_path: S3 path or HuggingFace Hub ID of the reward model.
        dataset_path: S3 path to JSONL prompt dataset.
        algorithm: RL algorithm (see above).
        prompt_column: Column in dataset containing prompt text.
        ppo_epochs: Number of outer epochs over the prompt dataset.
        learning_rate: Actor learning rate.
        batch_size: Global training batch size.
        mini_batch_size: Micro-batch size per GPU (unused in native loop;
            kept for API compatibility).
        kl_penalty: KL penalty type (informational).
        init_kl_coef: KL divergence coefficient.
        target_kl: Target KL for adaptive coefficient.
        max_new_tokens: Maximum generation length during rollouts.
        num_gpus: Number of GPUs (native loop uses 1).
        gradient_checkpointing: Enable gradient checkpointing.
        use_deepspeed: Reserved (unused in native loop).
        checkpoint_interval: Steps between checkpoints (-1 = end only).
        num_training_steps: Maximum training steps.

    Returns:
        RLHFTrainerOutput with checkpoint_path, mlflow_run_id,
        reward_stats, and kl_divergence.
    """
    # Convert Flyte types to S3 path strings
    sft_model_path = getattr(sft_model_path, "remote_source", None) or str(sft_model_path)  # type: ignore[assignment]
    reward_model_path = getattr(reward_model_path, "remote_source", None) or str(reward_model_path)  # type: ignore[assignment]
    dataset_path = getattr(dataset_path, "remote_source", None) or str(dataset_path)  # type: ignore[assignment]

    from ._training import run_rlhf_training

    result = run_rlhf_training(
        sft_model_path=sft_model_path,
        reward_model_path=reward_model_path,
        dataset_path=dataset_path,
        algorithm=algorithm,
        algorithm_map=_ALGORITHM_MAP,
        prompt_column=prompt_column,
        ppo_epochs=ppo_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        kl_penalty=kl_penalty,
        init_kl_coef=init_kl_coef,
        target_kl=target_kl,
        max_new_tokens=max_new_tokens,
        gradient_checkpointing=gradient_checkpointing,
        checkpoint_interval=checkpoint_interval,
        num_training_steps=num_training_steps,
    )

    return RLHFTrainerOutput(
        checkpoint_path=FlyteDirectory(path=result["checkpoint_path"]),
        mlflow_run_id=result["mlflow_run_id"],
        reward_stats=result["reward_stats"],
        kl_divergence=result["kl_divergence"],
    )
