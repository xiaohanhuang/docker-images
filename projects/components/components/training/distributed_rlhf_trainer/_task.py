"""Flyte task definition for distributed_rlhf_trainer component."""

import json
import re
import tempfile
from pathlib import Path
from typing import Dict, NamedTuple

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from ._helpers import download_s3, end_mlflow, start_mlflow


def _ensure_prompt_column(dataset_path: str, prompt_column: str) -> str:
    """Ensure the dataset JSONL has the required prompt column.

    If the column already exists, returns the path unchanged.  Otherwise,
    extracts the human turn from Anthropic hh-rlhf style ``chosen`` fields
    (``\\n\\nHuman: ...\\n\\nAssistant: ...``) and writes a new JSONL with
    the prompt column added.
    """
    import os

    with open(dataset_path) as f:
        first_line = f.readline().strip()
    if not first_line:
        return dataset_path

    first_record = json.loads(first_line)
    if prompt_column in first_record:
        print(f"[openrlhf] Dataset already has '{prompt_column}' column")
        return dataset_path

    # Determine source column for prompt extraction
    source_col = None
    for candidate in ("chosen", "text", "input"):
        if candidate in first_record:
            source_col = candidate
            break

    if source_col is None:
        print(
            f"[openrlhf] WARNING: '{prompt_column}' column missing and no known "
            f"source column found. Available: {list(first_record.keys())}"
        )
        return dataset_path

    print(f"[openrlhf] Extracting '{prompt_column}' from '{source_col}' column")

    fd, out_path = tempfile.mkstemp(suffix=".jsonl", prefix="openrlhf-prompted-")
    os.close(fd)
    row_count = 0
    with open(dataset_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if prompt_column not in record:
                text = record.get(source_col, "")
                # Extract prompt from "Human: ... Assistant: ..." format
                match = re.search(r"Human:\s*(.+?)(?:\n\nAssistant:|\Z)", text, re.DOTALL)
                record[prompt_column] = match.group(1).strip() if match else text
            json.dump(record, fout)
            fout.write("\n")
            row_count += 1
        fout.flush()
        os.fsync(fout.fileno())

    print(f"[openrlhf] Added '{prompt_column}' column to {row_count} rows")
    return out_path


class DistributedRLHFTrainerOutput(NamedTuple):
    """Output from distributed RLHF training."""

    checkpoint_path: FlyteDirectory
    mlflow_run_id: str
    reward_stats: Dict[str, float]
    kl_divergence: float
    training_metrics: Dict[str, float]


@task(
    retries=1,
    requests=Resources(cpu="8", mem="32Gi", gpu="1", ephemeral_storage="80Gi"),
    limits=Resources(cpu="16", mem="64Gi", gpu="1", ephemeral_storage="120Gi"),
    cache=False,
)
def distributed_rlhf_trainer(
    sft_model_path: FlyteDirectory,
    reward_model_path: FlyteDirectory,
    dataset_path: FlyteFile,
    # OpenRLHF-specific parameters
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
    colocate_critic_reward: bool = True,
    # Model configuration (empty string = use sft_model_path / reward_model_path)
    pretrain: str = "",
    reward_pretrain: str = "",
    # Training parameters
    algorithm: str = "ppo",
    prompt_column: str = "prompt",
    max_epochs: int = 1,
    rollout_batch_size: int = 64,
    micro_rollout_batch_size: int = 8,
    train_batch_size: int = 128,
    micro_train_batch_size: int = 4,
    max_samples: int = 100000,
    max_len: int = 512,
    prompt_max_len: int = 256,
    generate_max_len: int = 256,
    # PPO parameters
    ppo_epochs: int = 1,
    learning_rate: float = 5e-7,
    critic_learning_rate: float = 0.0,  # 0.0 = auto (5x learning_rate)
    init_kl_coef: float = 0.01,
    kl_target: float = 0.0,  # 0.0 = use default (0.1)
    cliprange: float = 0.2,
    cliprange_value: float = 0.2,
    gamma: float = 1.0,
    lam: float = 0.95,
    # vLLM inference parameters
    vllm_tensor_parallel_size: int = 1,
    vllm_num_engines: int = 1,
    vllm_gpu_memory_utilization: float = 0.5,
    # DeepSpeed configuration
    zero_stage: int = 2,
    gradient_checkpointing: bool = True,
    bf16: bool = True,
    # Optimization
    gradient_accumulation_steps: int = 1,
    adam_offload: bool = False,
    # Reference/Reward service endpoints (empty = not used, for multi-service mode)
    reference_service_url: str = "",
    reward_service_url: str = "",
    redis_url: str = "",
    # Checkpointing
    checkpoint_interval: int = -1,
    save_path: str = "",  # empty = auto-generate temp dir
    async_checkpoint: bool = False,
    # Logging
    logging_steps: int = 1,
    eval_steps: int = -1,
    mlflow_experiment: str = "",  # empty = auto-name
    # Advanced optimizations (Phase 3)
    use_hybrid_engine: bool = False,
    enable_ring_attention: bool = False,
    flash_attention: bool = True,
    enable_efa: bool = False,
) -> DistributedRLHFTrainerOutput:
    """Train LLM using distributed RLHF with OpenRLHF.

    This component orchestrates distributed RLHF training using OpenRLHF's
    Ray-based architecture. It can run in two modes:

    1. **Single-node mode** (colocate_critic_reward=True):
       All roles (actor, critic, reference, reward) run on the same node(s).
       Suitable for small-scale experiments and testing.

    2. **Multi-service mode** (colocate_critic_reward=False):
       Separate services for each role, communicating via gRPC/NCCL.
       Requires reference_service_url, reward_service_url, and redis_url.

    When OpenRLHF is not installed, a native PyTorch training loop is used
    automatically (equivalent to ``rlhf_trainer`` but with the distributed
    interface and output schema).

    Args:
        sft_model_path: S3 path or HuggingFace Hub ID of the SFT model.
        reward_model_path: S3 path or HuggingFace Hub ID of the reward model.
        dataset_path: S3 path to JSONL prompt dataset.
        num_nodes: Number of compute nodes for distributed training.
        num_gpus_per_node: GPUs per node.
        colocate_critic_reward: If True, run all roles on same nodes (single-node mode).
        pretrain: Override model path (HF Hub ID or local path).
        reward_pretrain: Override reward model path.
        algorithm: RL algorithm (ppo, grpo, rloo).
        prompt_column: Column in dataset containing prompt text.
        max_epochs: Number of training epochs.
        rollout_batch_size: Global batch size for rollout generation.
        micro_rollout_batch_size: Per-GPU micro-batch size for rollouts.
        train_batch_size: Global batch size for training.
        micro_train_batch_size: Per-GPU micro-batch size for training.
        max_samples: Maximum samples to use from dataset.
        max_len: Maximum sequence length.
        prompt_max_len: Maximum prompt length.
        generate_max_len: Maximum generation length.
        ppo_epochs: Number of PPO update epochs per rollout.
        learning_rate: Actor learning rate.
        critic_learning_rate: Critic learning rate (defaults to learning_rate).
        init_kl_coef: Initial KL divergence coefficient.
        kl_target: Target KL for adaptive coefficient.
        cliprange: PPO clip range for policy.
        cliprange_value: PPO clip range for value function.
        gamma: Discount factor.
        lam: GAE lambda parameter.
        vllm_tensor_parallel_size: Tensor parallelism for vLLM reference model.
        vllm_num_engines: Number of vLLM engines for generation.
        vllm_gpu_memory_utilization: Fraction of GPU memory for vLLM (0.0-1.0).
            Lower values required in colocated mode where training shares GPU memory.
        zero_stage: DeepSpeed ZeRO stage (0, 1, 2, 3).
        gradient_checkpointing: Enable gradient checkpointing.
        bf16: Use bfloat16 mixed precision.
        gradient_accumulation_steps: Gradient accumulation steps.
        adam_offload: Offload Adam optimizer to CPU.
        reference_service_url: gRPC URL of reference model service (multi-service mode).
        reward_service_url: gRPC URL of reward model service (multi-service mode).
        redis_url: Redis URL for trajectory buffer (multi-service mode).
        checkpoint_interval: Steps between checkpoints (-1 = end only).
        save_path: Override checkpoint save path.
        async_checkpoint: Enable async S3 checkpoint uploads (non-blocking I/O).
        logging_steps: Steps between log outputs.
        eval_steps: Steps between evaluation runs (-1 = disable).
        mlflow_experiment: MLflow experiment name.
        use_hybrid_engine: Enable OpenRLHF Hybrid Engine for models <13B
            (merges training + inference).
        enable_ring_attention: Enable Ring-Attention for 70B+ models across 8+ GPUs.
        flash_attention: Enable Flash Attention 2 for efficient attention computation.
        enable_efa: Enable AWS EFA networking for high-speed cross-node NCCL.

    Returns:
        DistributedRLHFTrainerOutput with checkpoint_path, mlflow_run_id,
        reward_stats, kl_divergence, and training_metrics.
    """
    # ── Normalize sentinel defaults ───────────────────────────────────────
    # Empty strings → None for internal use; 0.0 floats → None where they
    # represent "auto" / "not specified".
    _pretrain = pretrain or None
    _reward_pretrain = reward_pretrain or None
    _critic_lr = critic_learning_rate if critic_learning_rate > 0 else None
    _kl_target = kl_target if kl_target > 0 else None
    _reference_url = reference_service_url or None
    _reward_url = reward_service_url or None
    _redis_url = redis_url or None
    _save_path = save_path or None
    _mlflow_experiment = mlflow_experiment or None

    # ── Runtime validation: prevent toy/undersized execution ─────────────
    import glob
    import os
    import time

    # Diagnostic: environment variables
    nv_vis = os.environ.get("NVIDIA_VISIBLE_DEVICES", "(not set)")
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "(not set)")
    print(f"[gpu-diag] NVIDIA_VISIBLE_DEVICES={nv_vis}")
    print(f"[gpu-diag] CUDA_VISIBLE_DEVICES={cuda_vis}")
    print(f"[gpu-diag] LD_LIBRARY_PATH={ld_path}")

    # Diagnostic: GPU device nodes
    dev_nvidia = glob.glob("/dev/nvidia*")
    print(f"[gpu-diag] /dev/nvidia* devices: {dev_nvidia}")

    # Diagnostic: nvidia-smi
    import subprocess as _sp

    try:
        nvsmi = _sp.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        print(f"[gpu-diag] nvidia-smi -L rc={nvsmi.returncode}: {nvsmi.stdout.strip()}")
        if nvsmi.returncode != 0:
            print(f"[gpu-diag] nvidia-smi stderr: {nvsmi.stderr.strip()}")
    except FileNotFoundError:
        print("[gpu-diag] nvidia-smi not found in PATH — skipping")
        nvsmi = None

    # Diagnostic: CUDA driver via ctypes
    try:
        import ctypes

        libcuda = ctypes.CDLL("libcuda.so.1")
        rc_init = libcuda.cuInit(0)
        count = ctypes.c_int(0)
        rc_count = libcuda.cuDeviceGetCount(ctypes.byref(count))
        print(f"[gpu-diag] cuInit={rc_init}, cuDeviceGetCount={rc_count}, count={count.value}")
    except Exception as e:
        print(f"[gpu-diag] ctypes CUDA probe failed: {e}")

    import torch

    # Retry CUDA detection with backoff (CDI device injection may have a brief delay)
    available_gpus = 0
    for attempt in range(3):
        available_gpus = torch.cuda.device_count()
        print(
            f"[gpu-diag] torch.cuda attempt {attempt + 1}: "
            f"is_available={torch.cuda.is_available()}, device_count={available_gpus}"
        )
        if available_gpus > 0:
            break
        time.sleep(2)

    if available_gpus < 1:
        raise ValueError(
            "distributed_rlhf_trainer requires at least one GPU-enabled pod. "
            f"No CUDA devices were detected. "
            f"NVIDIA_VISIBLE_DEVICES={nv_vis}, CUDA_VISIBLE_DEVICES={cuda_vis}, "
            f"/dev/nvidia*={dev_nvidia}, "
            f"nvidia-smi rc={nvsmi.returncode if nvsmi else 'not found'}"
        )

    # Current implementation executes within a single Flyte task pod.
    # Multi-node orchestration requires an external Ray cluster integration.
    if num_nodes != 1:
        raise ValueError(
            "distributed_rlhf_trainer currently supports num_nodes=1 in this deployment mode. "
            "For true multi-node training, integrate an external Ray cluster orchestration path."
        )

    if num_gpus_per_node > available_gpus:
        raise ValueError(
            f"Requested num_gpus_per_node={num_gpus_per_node}, but only {available_gpus} GPU(s) "
            "are available in this task pod. Increase step resources via recipe infra profile "
            "or reduce num_gpus_per_node."
        )

    if not colocate_critic_reward:
        missing_urls = []
        if not _reference_url:
            missing_urls.append("reference_service_url")
        if not _reward_url:
            missing_urls.append("reward_service_url")
        if not _redis_url:
            missing_urls.append("redis_url")

        if missing_urls:
            raise ValueError(
                "Non-colocated RLHF mode requires non-empty service endpoints. "
                f"Missing: {', '.join(missing_urls)}"
            )

    # ── Resolve S3 paths ──────────────────────────────────────────────────
    sft_model_path_str = getattr(sft_model_path, "remote_source", None) or str(sft_model_path)
    reward_model_path_str = getattr(reward_model_path, "remote_source", None) or str(
        reward_model_path
    )
    dataset_path_str = getattr(dataset_path, "remote_source", None) or str(dataset_path)

    # Download models and dataset from S3
    sft_model_local = download_s3(sft_model_path_str, "sft")
    reward_model_local = download_s3(reward_model_path_str, "reward")

    if dataset_path_str.startswith("s3://"):
        import s3fs

        s3 = s3fs.S3FileSystem()
        local_data = tempfile.mktemp(suffix=".jsonl", prefix="openrlhf-data-")
        print(f"[openrlhf] Downloading dataset from {dataset_path_str}")
        s3.get(dataset_path_str.rstrip("/"), local_data)
        dataset_local = local_data
    else:
        dataset_local = dataset_path_str

    # ── Ensure the prompt column exists in the dataset ────────────────────
    dataset_local = _ensure_prompt_column(dataset_local, prompt_column)

    save_dir = Path(_save_path) if _save_path else Path(tempfile.mkdtemp(prefix="openrlhf-output-"))
    save_dir.mkdir(exist_ok=True, parents=True)

    # ── Detect backend ────────────────────────────────────────────────────
    try:
        import openrlhf  # noqa: F401
        import ray  # noqa: F401

        use_openrlhf = True
        framework = "openrlhf-ray"
    except ImportError:
        use_openrlhf = False
        framework = "native-torch"
        print(
            "[openrlhf] OpenRLHF/Ray not installed — "
            "falling back to native PyTorch training loop"
        )

    # ── Start MLflow ──────────────────────────────────────────────────────
    mlflow_run_id, mlflow_available = start_mlflow(
        algorithm,
        framework,
        {
            "algorithm": algorithm,
            "framework": framework,
            "num_nodes": num_nodes,
            "num_gpus_per_node": num_gpus_per_node,
            "sft_model": sft_model_path_str,
            "reward_model": reward_model_path_str,
            "learning_rate": learning_rate,
            "ppo_epochs": ppo_epochs,
            "rollout_batch_size": rollout_batch_size,
            "train_batch_size": train_batch_size,
            "zero_stage": zero_stage,
            "colocate_critic_reward": colocate_critic_reward,
        },
        _mlflow_experiment,
    )

    try:
        if use_openrlhf:
            from ._openrlhf import run_openrlhf

            reward_stats, final_kl, training_metrics = run_openrlhf(
                sft_model_local,
                reward_model_local,
                dataset_local,
                save_dir,
                pretrain=_pretrain,
                reward_pretrain=_reward_pretrain,
                num_nodes=num_nodes,
                num_gpus_per_node=num_gpus_per_node,
                colocate_critic_reward=colocate_critic_reward,
                algorithm=algorithm,
                prompt_column=prompt_column,
                max_epochs=max_epochs,
                rollout_batch_size=rollout_batch_size,
                micro_rollout_batch_size=micro_rollout_batch_size,
                train_batch_size=train_batch_size,
                micro_train_batch_size=micro_train_batch_size,
                max_samples=max_samples,
                max_len=max_len,
                prompt_max_len=prompt_max_len,
                generate_max_len=generate_max_len,
                ppo_epochs=ppo_epochs,
                learning_rate=learning_rate,
                critic_learning_rate=_critic_lr,
                init_kl_coef=init_kl_coef,
                kl_target=_kl_target,
                cliprange=cliprange,
                cliprange_value=cliprange_value,
                gamma=gamma,
                lam=lam,
                vllm_tensor_parallel_size=vllm_tensor_parallel_size,
                vllm_num_engines=vllm_num_engines,
                vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                zero_stage=zero_stage,
                gradient_checkpointing=gradient_checkpointing,
                bf16=bf16,
                gradient_accumulation_steps=gradient_accumulation_steps,
                adam_offload=adam_offload,
                reference_service_url=_reference_url,
                reward_service_url=_reward_url,
                redis_url=_redis_url,
                checkpoint_interval=checkpoint_interval,
                logging_steps=logging_steps,
                eval_steps=eval_steps,
                use_hybrid_engine=use_hybrid_engine,
                enable_ring_attention=enable_ring_attention,
                flash_attention=flash_attention,
                async_checkpoint=async_checkpoint,
                enable_efa=enable_efa,
                mlflow_available=mlflow_available,
            )
            final_path = str(save_dir)
        else:
            from ._native import run_native

            reward_stats, final_kl, training_metrics, final_path = run_native(
                sft_model_local,
                reward_model_local,
                dataset_local,
                save_dir,
                algorithm=algorithm,
                prompt_column=prompt_column,
                ppo_epochs=ppo_epochs,
                learning_rate=learning_rate,
                init_kl_coef=init_kl_coef,
                kl_target=_kl_target,
                cliprange=cliprange,
                max_samples=max_samples,
                generate_max_len=generate_max_len,
                gradient_checkpointing=gradient_checkpointing,
                train_batch_size=train_batch_size,
                micro_train_batch_size=micro_train_batch_size,
                checkpoint_interval=checkpoint_interval,
                mlflow_available=mlflow_available,
            )
    finally:
        end_mlflow(mlflow_available)

    print(f"[openrlhf] Training complete — {training_metrics.get('total_steps', 0)} steps")
    print(f"[openrlhf] Mean reward: {reward_stats['mean_reward']:.4f}")
    print(f"[openrlhf] Final KL: {final_kl:.4f}")
    print(f"[openrlhf] Checkpoint saved to: {final_path}")

    return DistributedRLHFTrainerOutput(
        checkpoint_path=FlyteDirectory(path=final_path),
        mlflow_run_id=mlflow_run_id,
        reward_stats=reward_stats,
        kl_divergence=final_kl,
        training_metrics=training_metrics,
    )
