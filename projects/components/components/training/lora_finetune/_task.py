"""Flyte task definition for lora_finetune component."""

from typing import Dict, List, Optional, Tuple

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile


@task(
    retries=1,
    requests=Resources(cpu="8", mem="32Gi", gpu="1"),
    limits=Resources(cpu="16", mem="64Gi", gpu="1"),
    cache=False,
)
def lora_finetune(
    base_model: str,
    train_data_path: FlyteFile,
    val_data_path: Optional[str] = None,
    method: str = "lora",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: Optional[List[str]] = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    mlflow_experiment: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[FlyteDirectory, str, Dict[str, float]]:
    """Fine-tune any HuggingFace causal LM using LoRA or QLoRA (PEFT).

    Supports causal LM models (AutoModelForCausalLM). Logs everything to MLflow.
    Checkpoint directory is returned as a FlyteDirectory so Flyte handles
    S3 upload to its managed data bucket automatically.

    Args:
        base_model: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B").
        train_data_path: S3 URI to tokenized training data (s3://bucket/key, JSONL).
        val_data_path: S3 URI to validation data (s3://bucket/key, JSONL). Optional.
        method: Fine-tuning method - "lora" or "qlora". Default "lora".
        lora_r: LoRA rank. Default 16.
        lora_alpha: LoRA alpha scaling parameter. Default 32.
        lora_target_modules: List of module names to apply LoRA (auto-detect if None).
        epochs: Number of training epochs. Default 3.
        batch_size: Per-device training batch size. Default 4.
        learning_rate: Peak learning rate for optimizer. Default 2e-4.
        gradient_accumulation_steps: Number of gradient accumulation steps. Default 4.
        mlflow_experiment: MLflow experiment name. If None, uses default experiment.
        trust_remote_code: Allow executing custom code from the model repo. Default False.

    Returns:
        Tuple of (checkpoint_dir, mlflow_run_id, final_metrics).
        - checkpoint_dir: FlyteDirectory pointing to saved LoRA adapter + tokenizer
        - mlflow_run_id: MLflow run ID for tracking
        - final_metrics: Dict with training and validation metrics
    """
    # Convert FlyteFile/FlyteDirectory inputs to S3 path strings
    train_data_path = getattr(train_data_path, "remote_source", None) or str(train_data_path)  # type: ignore[assignment]
    if val_data_path is not None:
        val_data_path = getattr(val_data_path, "remote_source", None) or str(val_data_path)

    # Validate method parameter (fast-fail before heavy imports)
    if method not in ("lora", "qlora"):
        raise ValueError(f"Invalid method '{method}'. Must be 'lora' or 'qlora'.")

    # Validate S3 paths (fast-fail before heavy imports)
    if not train_data_path.startswith("s3://"):
        raise ValueError(
            f"train_data_path must be an S3 URI starting with 's3://', got: {train_data_path!r}"
        )
    if val_data_path is not None and not val_data_path.startswith("s3://"):
        raise ValueError(
            f"val_data_path must be an S3 URI starting with 's3://', got: {val_data_path!r}"
        )

    from ._training import run_lora_finetune

    result = run_lora_finetune(
        base_model=base_model,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        method=method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mlflow_experiment=mlflow_experiment,
        trust_remote_code=trust_remote_code,
    )

    return (
        FlyteDirectory(path=result["checkpoint_dir"]),
        result["mlflow_run_id"],
        result["final_metrics"],
    )
