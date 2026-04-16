"""ML Platform SDK — shared utilities for tasks, workflows, and services."""

from __future__ import annotations

import os

# Suppress noisy flytekit type warnings in notebooks
os.environ["FLYTE_SDK_LOGGING_LEVEL"] = "40"
from flytekit import Resources, dynamic, eager, map_task, task
from flytekit.types.directory import FlyteDirectory  # noqa: F401
from flytekit.types.file import FlyteFile  # noqa: F401

from .catalog import (
    data_splitter,
    deploy_vllm,
    download_dataset,
    full_finetune,
    hf_dataset_loader,
    llm_judge,
    llm_sft_full_pipeline,
    llm_sft_lora_pipeline,
    lora_finetune,
    model_evaluator,
    preprocess_tabular,
    text2sql_pipeline,
    tokenizer,
    train_reward_model,
    vllm_deployer,
)
from .checkpoint import CheckpointManager, HuggingFaceCheckpointManager
from .components import (
    Component,
    Pipeline,
)
from .core import submit, workflow
from .tasks.accelerate import accelerate_task, platform
from .tasks.tensorboard import get_summary_writer

__all__ = [
    # Flytekit re-exports
    "Resources",
    "task",
    "dynamic",
    "eager",
    "map_task",
    "workflow",
    "submit",
    # Component/Pipeline classes
    "Component",
    "Pipeline",
    # Checkpoint managers
    "CheckpointManager",
    "HuggingFaceCheckpointManager",
    # Components
    "data_splitter",
    "deploy_vllm",
    "download_dataset",
    "full_finetune",
    "hf_dataset_loader",
    "llm_judge",
    "llm_sft_full_pipeline",
    "llm_sft_lora_pipeline",
    "lora_finetune",
    "model_evaluator",
    "preprocess_tabular",
    "text2sql_pipeline",
    "tokenizer",
    "train_reward_model",
    "vllm_deployer",
    "accelerate_task",
    "platform",
    "get_summary_writer",
]
