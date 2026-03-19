"""OpenRLHF Ray-based distributed RLHF training backend (0.9.5+).

Launches OpenRLHF's PPO training via the CLI entry point with vLLM
for high-throughput generation.  Ray is started locally and training
runs as a subprocess aligned with the documented argparse CLI.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Map user algorithm names → OpenRLHF --advantage_estimator values.
_ADVANTAGE_ESTIMATOR = {
    "ppo": None,  # default (GAE)
    "reinforce": "reinforce",
    "reinforce_baseline": "reinforce_baseline",
    "grpo": "group_norm",
    "rloo": "rloo",
    "dr_grpo": "dr_grpo",
}


def _install_missing_deps() -> None:
    """Install OpenRLHF runtime deps that were skipped by ``--no-deps``."""
    # Install gcc if not present (needed for some extensions)
    cc_check = subprocess.run(["which", "gcc"], capture_output=True)
    if cc_check.returncode != 0:
        print("[openrlhf] Installing gcc via apt...")
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
        r = subprocess.run(
            ["apt-get", "install", "-y", "-qq", "gcc"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(f"[openrlhf] apt install gcc failed: {r.stderr}")
        else:
            print("[openrlhf] gcc installed successfully")

    # Install torchdata if missing
    missing: list[str] = []
    for pkg in ("torchdata",):
        check = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
        )
        if check.returncode != 0:
            missing.append(pkg)
    if missing:
        print(f"[openrlhf] Installing missing deps: {missing}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", *missing],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[openrlhf] pip install failed: {result.stderr}")
        else:
            print(f"[openrlhf] Successfully installed: {missing}")


def _patch_deepspeed_fused_adam() -> None:
    """Replace DeepSpeed's FusedAdam with a torch.optim.AdamW wrapper on disk.

    DeepSpeed's FusedAdam/CPUAdam require JIT-compiling CUDA extensions via
    ninja at runtime.  The training image doesn't ship ninja, and installing
    it at runtime doesn't reliably propagate to Ray actor processes (separate
    PIDs spawned by the Ray runtime).

    This patches the actual files in site-packages so that ALL Python
    processes — including Ray actors — see the patched version without
    needing JIT compilation or ninja.
    """
    import site

    site_pkgs = site.getsitepackages()[0]
    adam_dir = os.path.join(site_pkgs, "deepspeed", "ops", "adam")

    if not os.path.isdir(adam_dir):
        print("[openrlhf] deepspeed/ops/adam not found — skipping FusedAdam patch")
        return

    # Check if already patched (idempotent)
    marker = os.path.join(adam_dir, ".patched_no_jit")
    if os.path.exists(marker):
        print("[openrlhf] DeepSpeed FusedAdam already patched")
        return

    # ── Patch fused_adam.py ────────────────────────────────────────────
    fused_adam_code = '''\
"""Patched FusedAdam: wraps torch.optim.AdamW to avoid JIT compilation."""
import torch.optim


class FusedAdam(torch.optim.AdamW):
    """Drop-in replacement for DeepSpeed FusedAdam.

    Accepts and ignores FusedAdam-specific keyword arguments so that
    existing call sites (OpenRLHF, DeepSpeed internals) work unchanged.
    """

    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, adam_w_mode=True,
                 weight_decay=0.0, amsgrad=False, set_grad_none=True,
                 **kwargs):
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
'''
    fused_path = os.path.join(adam_dir, "fused_adam.py")
    with open(fused_path, "w") as f:
        f.write(fused_adam_code)

    # ── Patch cpu_adam.py (used by --adam_offload) ─────────────────────
    cpu_adam_code = '''\
"""Patched DeepSpeedCPUAdam: wraps torch.optim.AdamW to avoid JIT compilation."""
import torch.optim


class DeepSpeedCPUAdam(torch.optim.AdamW):
    """Drop-in replacement for DeepSpeed CPUAdam."""

    def __init__(self, model_params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 amsgrad=False, adamw_mode=True, fp32_optimizer_states=True,
                 **kwargs):
        super().__init__(
            model_params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
'''
    cpu_path = os.path.join(adam_dir, "cpu_adam.py")
    with open(cpu_path, "w") as f:
        f.write(cpu_adam_code)

    # ── Do NOT replace __init__.py — it exports other classes like
    # ZenFlowSelectiveAdamW_stage3 that we must not remove. The
    # original __init__.py imports FusedAdam/DeepSpeedCPUAdam from the
    # files we just patched, so the swap is automatic. ─────────────────

    # Write marker so we don't re-patch on retry
    with open(marker, "w") as f:
        f.write("patched\n")

    # Verify the patch works
    verify = subprocess.run(
        [
            sys.executable,
            "-c",
            "from deepspeed.ops.adam import FusedAdam; "
            "print(f'FusedAdam base: {FusedAdam.__bases__}')",
        ],
        capture_output=True,
        text=True,
    )
    if verify.returncode == 0:
        print(f"[openrlhf] DeepSpeed FusedAdam patched: {verify.stdout.strip()}")
    else:
        print(f"[openrlhf] FusedAdam patch verification failed: {verify.stderr}")


def _ensure_flash_attn() -> bool:
    """Ensure ``flash_attn`` is importable for OpenRLHF and all Ray workers.

    OpenRLHF unconditionally imports ``flash_attn`` submodules at module
    load time (via ``ring_attn_utils.py``), even when ring attention is
    disabled.  The training-llm image installs OpenRLHF with ``--no-deps``
    to skip flash-attn (requires NVCC to build from source).

    Strategy:
    1. If the real package is already installed → done (return True).
    2. Try ``pip install flash-attn`` (pre-built wheel).
    3. If that fails, write a stub ``flash_attn`` package directly into
       site-packages so all processes (including Ray workers) see it.

    Returns:
        True if real flash_attn is available, False if using stub.
    """
    check = subprocess.run(
        [sys.executable, "-c", "import flash_attn; flash_attn.flash_attn_func"],
        capture_output=True,
    )
    if check.returncode == 0:
        return True

    print("[openrlhf] flash_attn not found, attempting pip install...")
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "flash-attn"],
        capture_output=True,
        text=True,
    )
    if install.returncode == 0:
        print("[openrlhf] flash_attn installed successfully")
        return True

    # Write stub directly into site-packages — no pip needed.
    import site

    site_pkgs = site.getsitepackages()[0]
    pkg_dir = os.path.join(site_pkgs, "flash_attn")
    print(f"[openrlhf] Creating flash_attn stub in {pkg_dir}")
    os.makedirs(pkg_dir, exist_ok=True)

    with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
        f.write(
            "# Stub flash_attn package for OpenRLHF compatibility\n"
            "# Provides importable symbols so transformers detects flash_attn,\n"
            "# then falls back to eager attention at runtime.\n"
            "\n"
            "def flash_attn_func(*args, **kwargs):\n"
            "    raise NotImplementedError('flash_attn stub — use eager attention')\n"
            "\n"
            "def flash_attn_varlen_func(*args, **kwargs):\n"
            "    raise NotImplementedError('flash_attn stub — use eager attention')\n"
            "\n"
            "def flash_attn_with_kvcache(*args, **kwargs):\n"
            "    raise NotImplementedError('flash_attn stub — use eager attention')\n"
        )

    with open(os.path.join(pkg_dir, "bert_padding.py"), "w") as f:
        f.write(
            "# Stub: flash_attn.bert_padding\n"
            "def index_first_axis(*a, **kw): raise NotImplementedError('flash_attn stub')\n"
            "def pad_input(*a, **kw): raise NotImplementedError('flash_attn stub')\n"
            "def unpad_input(*a, **kw): raise NotImplementedError('flash_attn stub')\n"
            "def rearrange(*a, **kw): raise NotImplementedError('flash_attn stub')\n"
        )

    utils_dir = os.path.join(pkg_dir, "utils")
    os.makedirs(utils_dir, exist_ok=True)
    with open(os.path.join(utils_dir, "__init__.py"), "w") as f:
        f.write("# Stub: flash_attn.utils\n")
    with open(os.path.join(utils_dir, "distributed.py"), "w") as f:
        f.write(
            "# Stub: flash_attn.utils.distributed\n"
            "def all_gather(*a, **kw): raise NotImplementedError('flash_attn stub')\n"
        )

    # Create dist-info so importlib.metadata.version("flash_attn") works.
    # Use version 0.0.0 so transformers sees it as too old for
    # flash_attention_2 (requires >=2.1.0) and falls back to eager.
    dist_info = os.path.join(site_pkgs, "flash_attn-0.0.0.dist-info")
    os.makedirs(dist_info, exist_ok=True)
    with open(os.path.join(dist_info, "METADATA"), "w") as f:
        f.write("Metadata-Version: 2.1\nName: flash-attn\nVersion: 0.0.0\n")
    with open(os.path.join(dist_info, "INSTALLER"), "w") as f:
        f.write("stub\n")
    with open(os.path.join(dist_info, "RECORD"), "w") as f:
        f.write("")
    with open(os.path.join(dist_info, "top_level.txt"), "w") as f:
        f.write("flash_attn\n")

    # Verify both import and metadata work
    verify = subprocess.run(
        [
            sys.executable,
            "-c",
            "import flash_attn; import flash_attn.bert_padding; "
            "import flash_attn.utils.distributed; "
            "import importlib.metadata; "
            "v = importlib.metadata.version('flash_attn'); "
            "print(f'OK version={v}')",
        ],
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        print(f"[openrlhf] stub verification failed: {verify.stderr}")
        raise RuntimeError("Failed to create flash_attn stub package")

    print(f"[openrlhf] flash_attn stub created successfully: {verify.stdout.strip()}")
    return False


def run_openrlhf(
    sft_model_local: str,
    reward_model_local: str,
    dataset_local: str,
    save_dir: Path,
    *,
    pretrain: Optional[str],
    reward_pretrain: Optional[str],
    num_nodes: int,
    num_gpus_per_node: int,
    colocate_critic_reward: bool,
    algorithm: str,
    prompt_column: str,
    max_epochs: int,
    rollout_batch_size: int,
    micro_rollout_batch_size: int,
    train_batch_size: int,
    micro_train_batch_size: int,
    max_samples: int,
    max_len: int,
    prompt_max_len: int,
    generate_max_len: int,
    ppo_epochs: int,
    learning_rate: float,
    critic_learning_rate: Optional[float],
    init_kl_coef: float,
    kl_target: Optional[float],
    cliprange: float,
    cliprange_value: float,
    gamma: float,
    lam: float,
    vllm_tensor_parallel_size: int,
    vllm_num_engines: int,
    vllm_gpu_memory_utilization: float,
    zero_stage: int,
    gradient_checkpointing: bool,
    bf16: bool,
    gradient_accumulation_steps: int,
    adam_offload: bool,
    reference_service_url: Optional[str],
    reward_service_url: Optional[str],
    redis_url: Optional[str],
    checkpoint_interval: int,
    logging_steps: int,
    eval_steps: int,
    use_hybrid_engine: bool,
    enable_ring_attention: bool,
    flash_attention: bool,
    async_checkpoint: bool,
    enable_efa: bool,
    mlflow_available: bool,
) -> tuple:
    """Run training via OpenRLHF's Ray + vLLM engine.

    Starts a local Ray head node, then launches ``python -m
    openrlhf.cli.train_ppo_ray`` as a subprocess.  vLLM provides
    high-throughput generation for the rollout phase.

    Returns (reward_stats, final_kl, training_metrics).
    """
    total_gpus = num_nodes * num_gpus_per_node

    # ── Install missing OpenRLHF deps skipped by --no-deps ────────────
    _install_missing_deps()

    # ── Patch DeepSpeed to avoid JIT compilation ──────────────────────
    # Ray actors are separate processes and can't find ninja at runtime.
    # Patch the actual deepspeed package on disk so ALL processes use
    # torch.optim.AdamW instead of JIT-compiled FusedAdam/CPUAdam.
    _patch_deepspeed_fused_adam()

    # ── Ensure flash_attn is available ────────────────────────────────
    has_real_flash_attn = _ensure_flash_attn()
    if not has_real_flash_attn:
        # Stub is installed for import compat only — tell transformers to
        # use eager attention so it never calls the stub functions.
        os.environ["ATTN_BACKEND"] = "eager"
        print("[openrlhf] Using flash_attn stub — forcing eager attention")

    # ── Ensure Ray is running ─────────────────────────────────────────
    ray_started = False
    status = subprocess.run(["ray", "status"], capture_output=True)
    if status.returncode != 0:
        print(f"[openrlhf] Starting Ray head node with {total_gpus} GPUs")
        result = subprocess.run(
            [
                "ray",
                "start",
                "--head",
                "--num-gpus",
                str(total_gpus),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[openrlhf] ray start stdout: {result.stdout}")
            print(f"[openrlhf] ray start stderr: {result.stderr}")
            result.check_returncode()
        ray_started = True
    else:
        print("[openrlhf] Using existing Ray cluster")

    try:
        cmd = _build_command(
            sft_model_local=sft_model_local,
            reward_model_local=reward_model_local,
            dataset_local=dataset_local,
            save_dir=save_dir,
            pretrain=pretrain,
            reward_pretrain=reward_pretrain,
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
            learning_rate=learning_rate,
            critic_learning_rate=critic_learning_rate,
            init_kl_coef=init_kl_coef,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_num_engines=vllm_num_engines,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            zero_stage=zero_stage,
            gradient_checkpointing=gradient_checkpointing,
            bf16=bf16,
            adam_offload=adam_offload,
            checkpoint_interval=checkpoint_interval,
            logging_steps=logging_steps,
            enable_efa=enable_efa,
            has_real_flash_attn=has_real_flash_attn,
        )

        print(f"[openrlhf] Running: {' '.join(cmd)}")

        # ── Execute training, streaming output ────────────────────────
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_metrics: dict = {}
        for line in proc.stdout:
            line = line.rstrip()
            print(line)
            _update_metrics(last_metrics, line)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"OpenRLHF training failed with exit code {proc.returncode}")

        # ── Extract final metrics ─────────────────────────────────────
        reward_stats = {
            "mean_reward": last_metrics.get("reward/mean", 0.0),
            "std_reward": last_metrics.get("reward/std", 0.0),
            "max_reward": last_metrics.get("reward/max", 0.0),
            "min_reward": last_metrics.get("reward/min", 0.0),
        }
        final_kl = last_metrics.get("kl", last_metrics.get("kl_divergence", 0.0))
        training_metrics = {
            "total_steps": float(last_metrics.get("step", 0)),
            "total_episodes": float(last_metrics.get("episodes", 0)),
            "actor_loss": float(last_metrics.get("actor_loss", 0.0)),
            "critic_loss": float(last_metrics.get("critic_loss", 0.0)),
        }

        if mlflow_available:
            try:
                import mlflow

                mlflow.log_metrics(
                    {
                        "final/mean_reward": reward_stats["mean_reward"],
                        "final/kl_divergence": final_kl,
                        "final/total_steps": training_metrics["total_steps"],
                    }
                )
            except Exception:
                pass

    finally:
        if ray_started:
            print("[openrlhf] Stopping Ray")
            subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)

    return reward_stats, final_kl, training_metrics


# ── Private helpers ───────────────────────────────────────────────────


def _build_command(
    *,
    sft_model_local,
    reward_model_local,
    dataset_local,
    save_dir,
    pretrain,
    reward_pretrain,
    num_nodes,
    num_gpus_per_node,
    colocate_critic_reward,
    algorithm,
    prompt_column,
    max_epochs,
    rollout_batch_size,
    micro_rollout_batch_size,
    train_batch_size,
    micro_train_batch_size,
    max_samples,
    max_len,
    prompt_max_len,
    generate_max_len,
    learning_rate,
    critic_learning_rate,
    init_kl_coef,
    vllm_tensor_parallel_size,
    vllm_num_engines,
    vllm_gpu_memory_utilization,
    zero_stage,
    gradient_checkpointing,
    bf16,
    adam_offload,
    checkpoint_interval,
    logging_steps,
    enable_efa,
    has_real_flash_attn=True,
) -> list:
    """Build the ``openrlhf.cli.train_ppo_ray`` CLI argument list."""
    cmd = [
        sys.executable,
        "-m",
        "openrlhf.cli.train_ppo_ray",
        # Model paths
        "--pretrain",
        pretrain or sft_model_local,
        "--reward_pretrain",
        reward_pretrain or reward_model_local,
        # Data
        "--prompt_data",
        dataset_local,
        "--input_key",
        prompt_column,
        # Output
        "--save_path",
        str(save_dir),
        "--save_hf_ckpt",
        # Resource allocation
        "--actor_num_nodes",
        str(num_nodes),
        "--actor_num_gpus_per_node",
        str(num_gpus_per_node),
        "--ref_num_nodes",
        "1",
        "--ref_num_gpus_per_node",
        str(num_gpus_per_node),
        "--reward_num_nodes",
        "1",
        "--reward_num_gpus_per_node",
        str(num_gpus_per_node),
        "--critic_num_nodes",
        "1",
        "--critic_num_gpus_per_node",
        str(num_gpus_per_node),
        # vLLM inference engine
        "--vllm_num_engines",
        str(vllm_num_engines),
        "--vllm_tensor_parallel_size",
        str(vllm_tensor_parallel_size),
        "--vllm_gpu_memory_utilization",
        str(vllm_gpu_memory_utilization),
        "--vllm_sync_backend",
        "nccl",
        "--enforce_eager",
        # Training parameters
        "--max_epochs",
        str(max_epochs),
        "--rollout_batch_size",
        str(rollout_batch_size),
        "--micro_rollout_batch_size",
        str(micro_rollout_batch_size),
        "--train_batch_size",
        str(train_batch_size),
        "--micro_train_batch_size",
        str(micro_train_batch_size),
        "--max_samples",
        str(max_samples),
        "--max_len",
        str(max_len),
        "--prompt_max_len",
        str(prompt_max_len),
        "--generate_max_len",
        str(generate_max_len),
        # Learning rates
        "--actor_learning_rate",
        str(learning_rate),
        "--critic_learning_rate",
        str(critic_learning_rate or learning_rate * 5),
        # KL penalty
        "--init_kl_coef",
        str(init_kl_coef),
        # DeepSpeed
        "--zero_stage",
        str(zero_stage),
        "--param_dtype",
        "bf16" if bf16 else "fp32",
        # Recommended optimizations
        "--normalize_reward",
        # Logging
        "--logging_steps",
        str(logging_steps),
    ]

    # packing_samples requires flash_attention_2; skip when using stub
    if has_real_flash_attn:
        cmd.append("--packing_samples")

    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    # Algorithm selection via advantage estimator
    estimator = _ADVANTAGE_ESTIMATOR.get(algorithm)
    if estimator:
        cmd.extend(["--advantage_estimator", estimator])

    # Colocated mode: all models share GPUs with sleep/wake scheduling
    if colocate_critic_reward:
        cmd.extend(
            [
                "--colocate_all_models",
                "--vllm_enable_sleep",
                "--deepspeed_enable_sleep",
            ]
        )

    if checkpoint_interval > 0:
        cmd.extend(["--save_steps", str(checkpoint_interval)])

    if adam_offload:
        cmd.append("--adam_offload")

    if enable_efa:
        cmd.extend(["--nccl_socket_ifname", "eth0"])

    if not has_real_flash_attn:
        cmd.extend(["--attn_implementation", "eager"])

    return cmd


def _update_metrics(metrics: dict, line: str):
    """Parse OpenRLHF log lines for training metrics."""
    # OpenRLHF logs JSON metrics periodically
    if line.lstrip().startswith("{") and "reward" in line:
        try:
            data = json.loads(line)
            metrics.update(data)
        except json.JSONDecodeError:
            pass
    # Catch key=value format: step=10 reward/mean=0.5
    for match in re.finditer(r"([\w/]+)\s*[=:]\s*([-\d.eE+]+)", line):
        try:
            metrics[match.group(1)] = float(match.group(2))
        except ValueError:
            pass
