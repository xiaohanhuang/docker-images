"""Flyte task definition for reward_model_trainer component."""

from typing import NamedTuple

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

try:
    from ml_platform_sdk.tasks.efs import build_efs_pod_template

    _efs_pod_template = build_efs_pod_template()
except ImportError:
    _efs_pod_template = None


class RewardModelOutput(NamedTuple):
    """Outputs from reward model training."""

    checkpoint_path: FlyteDirectory
    mlflow_run_id: str
    accuracy: float
    reward_margin: float
    final_loss: float


@task(
    retries=1,
    requests=Resources(cpu="8", mem="32Gi", gpu="1"),
    limits=Resources(cpu="16", mem="64Gi", gpu="1"),
    cache=False,
    pod_template=_efs_pod_template,
)
def train_reward_model(
    base_model: str,
    preference_data_path: FlyteFile,
    prompt_column: str = "prompt",
    chosen_column: str = "chosen",
    rejected_column: str = "rejected",
    modeling_type: str = "bradley_terry",
    epochs: int = 1,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 512,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    num_gpus: int = 1,
    mlflow_tracking_uri: str = "",
    mlflow_experiment_name: str = "reward-model-training",
) -> RewardModelOutput:
    """Train a reward model from preference data.

    Args:
        base_model: HuggingFace model ID (e.g., "meta-llama/Llama-3-8b").
        preference_data_path: S3 path to JSONL preference pairs.
        prompt_column: Column name for prompts in the dataset.
        chosen_column: Column name for chosen (preferred) responses.
        rejected_column: Column name for rejected responses.
        modeling_type: Loss type - "bradley_terry" or "regression".
        epochs: Number of training epochs.
        learning_rate: Peak learning rate for AdamW optimizer.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_length: Maximum sequence length for tokenization.
        use_lora: Whether to use LoRA adapters for efficient training.
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA alpha scaling factor.
        num_gpus: Number of GPUs (informational only; the task currently
            runs on a single GPU). Logged to MLflow for tracking purposes.
        mlflow_tracking_uri: MLflow tracking server URI (optional).
        mlflow_experiment_name: MLflow experiment name.

    Returns:
        RewardModelOutput with checkpoint path, metrics, and MLflow run ID.
    """
    # Convert FlyteFile input to S3 path string
    preference_data_path = (  # type: ignore[assignment]
        getattr(preference_data_path, "remote_source", None) or str(preference_data_path)
    )

    from ._training import run_reward_model_training

    result = run_reward_model_training(
        base_model=base_model,
        preference_data_path=preference_data_path,
        prompt_column=prompt_column,
        chosen_column=chosen_column,
        rejected_column=rejected_column,
        modeling_type=modeling_type,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        num_gpus=num_gpus,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    return RewardModelOutput(
        checkpoint_path=FlyteDirectory(path=result["checkpoint_path"]),
        mlflow_run_id=result["run_id"],
        accuracy=result["accuracy"],
        reward_margin=result["reward_margin"],
        final_loss=result["final_loss"],
    )
