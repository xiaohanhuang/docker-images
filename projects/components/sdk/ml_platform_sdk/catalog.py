"""Concrete component and pipeline instances — the ML Platform component catalog.

Each entry defines a pre-registered Flyte task or launch plan that is callable
inside ``@mp.workflow``, ``@mp.eager``, or via ``mp.submit()``.

To add a new component, call ``_comp()`` (for tasks) or ``_pipe()`` (for
launch plans) with the fully-qualified Flyte entity name, input/output type
maps, and optional defaults.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from .components import Component, Pipeline

# ── Helpers ──────────────────────────────────────────────────────────────────


def _comp(
    name: str,
    inputs: dict,
    outputs: dict,
    defaults: Optional[Dict] = None,
) -> Component:
    """Create a ``Component`` pointing to a pre-registered Flyte task."""
    return Component(name=name, inputs=inputs, outputs=outputs, defaults=defaults)


def _pipe(
    name: str,
    inputs: dict,
    outputs: dict,
    defaults: Optional[Dict] = None,
) -> Pipeline:
    """Create a ``Pipeline`` pointing to a pre-registered Flyte workflow."""
    return Pipeline(name=name, inputs=inputs, outputs=outputs, defaults=defaults)


# ── Data Components ──────────────────────────────────────────────────────────

hf_dataset_loader = _comp(
    "components.data.hf_dataset_loader.task.hf_dataset_loader",
    inputs={
        "dataset_name": str,
        "split": str,
        "subset": Optional[str],
        "num_samples": Optional[int],
    },
    outputs={"dataset": FlyteFile},
    defaults={"split": "train"},
)

download_dataset = _comp(
    "components.data.ingest.task.download_dataset",
    inputs={"s3_uri": str},
    outputs={"dataset": FlyteFile},
)

tokenizer = _comp(
    "components.data.tokenizer.task.tokenizer",
    inputs={
        "raw_data": FlyteFile,
        "model_id": str,
        "prompt_template": str,
        "max_length": int,
    },
    outputs={"tokenized_data": FlyteFile},
    defaults={"prompt_template": "alpaca", "max_length": 2048},
)

data_splitter = _comp(
    "components.data.data_splitter.task.data_splitter",
    inputs={
        "tokenized_data": FlyteFile,
        "train_ratio": float,
        "val_ratio": float,
        "test_ratio": float,
        "seed": int,
    },
    outputs={"train_data": FlyteFile, "val_data": FlyteFile, "test_data": FlyteFile},
    defaults={"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1, "seed": 42},
)

preprocess_tabular = _comp(
    "components.data.preprocess.task.preprocess_tabular",
    inputs={
        "dataset": FlyteFile,
        "target_column": str,
        "test_size": float,
        "val_size": float,
        "seed": int,
    },
    outputs={"train_data": FlyteFile, "val_data": FlyteFile, "test_data": FlyteFile},
    defaults={"test_size": 0.1, "val_size": 0.1, "seed": 42},
)

# ── Training Components ──────────────────────────────────────────────────────

lora_finetune = _comp(
    "components.training.lora_finetune.task.lora_finetune",
    inputs={
        "base_model": str,
        "train_data_path": FlyteFile,
        "val_data_path": Optional[str],
        "method": str,
        "lora_r": int,
        "lora_alpha": int,
        "lora_target_modules": Optional[List[str]],
        "epochs": int,
        "batch_size": int,
        "learning_rate": float,
        "gradient_accumulation_steps": int,
        "mlflow_experiment": Optional[str],
        "trust_remote_code": bool,
    },
    outputs={
        "checkpoint_dir": FlyteDirectory,
        "mlflow_run_id": str,
        "final_metrics": Dict[str, float],
    },
    defaults={
        "method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "gradient_accumulation_steps": 4,
        "trust_remote_code": False,
    },
)

full_finetune = _comp(
    "components.training.full_finetune.task.full_finetune",
    inputs={
        "base_model": str,
        "train_data": FlyteFile,
        "val_data": FlyteFile,
        "num_epochs": int,
        "learning_rate": float,
        "batch_size": int,
        "gradient_accumulation_steps": int,
        "use_efs_checkpoints": bool,
    },
    outputs={"checkpoint_dir": FlyteDirectory, "metrics": Dict[str, float]},
    defaults={
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "use_efs_checkpoints": True,
    },
)

train_reward_model = _comp(
    "components.training.reward_model_trainer.task.train_reward_model",
    inputs={
        "base_model": str,
        "preference_data_path": FlyteFile,
        "prompt_column": str,
        "chosen_column": str,
        "rejected_column": str,
        "modeling_type": str,
        "epochs": int,
        "learning_rate": float,
        "batch_size": int,
        "gradient_accumulation_steps": int,
        "max_length": int,
        "use_lora": bool,
        "lora_rank": int,
        "lora_alpha": int,
        "num_gpus": int,
        "mlflow_tracking_uri": str,
        "mlflow_experiment_name": str,
    },
    outputs={
        "checkpoint_path": FlyteDirectory,
        "mlflow_run_id": str,
        "accuracy": float,
        "reward_margin": float,
        "final_loss": float,
    },
    defaults={
        "prompt_column": "prompt",
        "chosen_column": "chosen",
        "rejected_column": "rejected",
        "modeling_type": "bradley_terry",
        "epochs": 1,
        "learning_rate": 1e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_length": 512,
        "use_lora": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "num_gpus": 1,
        "mlflow_tracking_uri": "",
        "mlflow_experiment_name": "reward-model-training",
    },
)

# ── Evaluation Components ────────────────────────────────────────────────────

model_evaluator = _comp(
    "components.evaluation.model_evaluator.task.model_evaluator",
    inputs={
        "checkpoint_path": FlyteDirectory,
        "test_data": FlyteFile,
        "base_model": Optional[str],
        "metrics": Optional[List[str]],
    },
    outputs={"metrics": Dict[str, float]},
)

llm_judge = _comp(
    "components.evaluation.llm_judge.task.llm_judge",
    inputs={
        "predictions_path": FlyteDirectory,
        "ground_truth_path": Optional[FlyteFile],
        "judge_model": str,
        "scorers": Optional[List[str]],
        "custom_rubric": Optional[str],
        "sample_size": int,
        "thresholds": Optional[Dict[str, float]],
        "mlflow_experiment": Optional[str],
    },
    outputs={"metrics": dict},
    defaults={"judge_model": "gpt-4o", "sample_size": 100},
)

# ── Serving Components ───────────────────────────────────────────────────────

vllm_deployer = _comp(
    "components.serving.vllm_deployer.task.vllm_deployer",
    inputs={
        "model_path": str,
        "service_name": str,
        "gpu_count": int,
        "quantization": Optional[str],
        "max_model_len": Optional[int],
        "min_replicas": int,
        "max_replicas": int,
    },
    outputs={
        "endpoint_url": str,
        "service_name": str,
        "deployment_config": Dict[str, Any],
    },
    defaults={"gpu_count": 1, "min_replicas": 1, "max_replicas": 5},
)

deploy_vllm = _comp(
    "components.serving.vllm_deploy.task.deploy_vllm",
    inputs={
        "model_id": str,
        "port": int,
        "max_model_len": int,
        "gpu_memory_utilization": float,
    },
    outputs={"endpoint_url": str},
    defaults={"port": 8000, "max_model_len": 4096, "gpu_memory_utilization": 0.9},
)

# ── Pipelines ────────────────────────────────────────────────────────────────

llm_sft_lora_pipeline = _pipe(
    "pipeline.llm_sft_lora_pipeline",
    inputs={
        "base_model": str,
        "dataset": str,
        "epochs": int,
        "learning_rate": float,
        "batch_size": int,
        "lora_r": int,
        "lora_alpha": int,
        "quantization": str,
        "model_name": str,
        "teams_webhook": str,
    },
    outputs={"o0": str},
    defaults={
        "quantization": "none",
        "model_name": "llm-sft",
        "teams_webhook": "",
    },
)

llm_sft_full_pipeline = _pipe(
    "pipeline.llm_sft_full_pipeline",
    inputs={
        "base_model": str,
        "dataset": str,
        "epochs": int,
        "learning_rate": float,
        "batch_size": int,
        "model_name": str,
        "teams_webhook": str,
    },
    outputs={"o0": str},
    defaults={
        "model_name": "llm-sft",
        "teams_webhook": "",
    },
)

text2sql_pipeline = _pipe(
    "pipeline.text2sql_pipeline",
    inputs={
        "num_epochs": int,
        "batch_size": int,
        "learning_rate": float,
    },
    outputs={"model_version_uri": str},
)
