"""
pipeline.py — LLM Supervised Fine-Tuning Pipeline.

Register with: pyflyte register --project ml-platform --domain development pipeline.py
"""

import sys

from flytekit import workflow

sys.path.insert(0, "/app")

from config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_R,
)

# Import components from local task modules within the image
from tasks.data_splitter import data_splitter
from tasks.full_finetune import full_finetune
from tasks.hf_dataset_loader import hf_dataset_loader
from tasks.lora_finetune import lora_finetune
from tasks.model_evaluator import model_evaluator
from tasks.notify import notify_teams
from tasks.registry_publisher import registry_publisher
from tasks.tokenizer import tokenizer


@workflow
def llm_sft_lora_pipeline(
    base_model: str = DEFAULT_BASE_MODEL,
    dataset: str = DEFAULT_DATASET,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    quantization: str = "none",
    model_name: str = "llm-sft",
    teams_webhook: str = "",
) -> str:
    """LoRA/QLoRA supervised fine-tuning pipeline.

    Args:
        base_model: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B").
        dataset: HuggingFace dataset ID or S3 URI.
        epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        batch_size: Per-device training batch size.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        quantization: Quantization mode ("none" or "4bit" for QLoRA).
        model_name: MLflow model name for registration.
        teams_webhook: Optional Teams webhook URL for notifications.

    Returns:
        MLflow model URI.
    """
    # Stage 1: Load dataset
    raw_data = hf_dataset_loader(dataset=dataset)

    # Stage 2: Tokenize
    tokenized = tokenizer(raw_data=raw_data, model_id=base_model)

    # Stage 3: Split data
    train_data, val_data, test_data = data_splitter(tokenized_data=tokenized)

    # Stage 4: Train with LoRA
    checkpoint, _ = lora_finetune(
        base_model=base_model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=epochs,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        quantization=quantization,
        batch_size=batch_size,
    )

    # Stage 5: Evaluate
    eval_metrics = model_evaluator(
        checkpoint_path=checkpoint,
        test_data=test_data,
        base_model=base_model,
    )

    # Stage 6: Register in MLflow
    model_uri = registry_publisher(
        checkpoint_path=checkpoint,
        model_name=model_name,
        eval_metrics=eval_metrics,
        base_model=base_model,
        training_method="lora",
    )

    # Stage 7: Optional notification
    if teams_webhook:
        notify_teams(
            webhook_url=teams_webhook,
            title="LLM LoRA Fine-Tuning Complete",
            message=model_uri,
            success=True,
        )

    return model_uri


@workflow
def llm_sft_full_pipeline(
    base_model: str = DEFAULT_BASE_MODEL,
    dataset: str = DEFAULT_DATASET,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = 2e-5,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = "llm-sft",
    teams_webhook: str = "",
) -> str:
    """Full-parameter supervised fine-tuning pipeline.

    Args:
        base_model: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B").
        dataset: HuggingFace dataset ID or S3 URI.
        epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        batch_size: Per-device training batch size.
        model_name: MLflow model name for registration.
        teams_webhook: Optional Teams webhook URL for notifications.

    Returns:
        MLflow model URI.
    """
    # Stage 1: Load dataset
    raw_data = hf_dataset_loader(dataset=dataset)

    # Stage 2: Tokenize
    tokenized = tokenizer(raw_data=raw_data, model_id=base_model)

    # Stage 3: Split data
    train_data, val_data, test_data = data_splitter(tokenized_data=tokenized)

    # Stage 4: Train full
    checkpoint, _ = full_finetune(
        base_model=base_model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    # Stage 5: Evaluate (no base_model needed — full checkpoint has everything)
    eval_metrics = model_evaluator(
        checkpoint_path=checkpoint,
        test_data=test_data,
        base_model=base_model,
    )

    # Stage 6: Register in MLflow
    model_uri = registry_publisher(
        checkpoint_path=checkpoint,
        model_name=model_name,
        eval_metrics=eval_metrics,
        base_model=base_model,
        training_method="full",
    )

    # Stage 7: Optional notification
    if teams_webhook:
        notify_teams(
            webhook_url=teams_webhook,
            title="LLM Full Fine-Tuning Complete",
            message=model_uri,
            success=True,
        )

    return model_uri
