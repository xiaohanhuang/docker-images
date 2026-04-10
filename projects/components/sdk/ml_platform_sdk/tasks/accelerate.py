"""
Accelerate Task Decorator — Universal Scaling Wrapper.

Provides ``@accelerate_task`` that wraps a function as a Flyte task with
PyTorch distributed training config (via KFPyTorch operator), Karpenter
GPU node selection/tolerations, and multi-GPU process spawning.

The companion ``platform.setup()`` helper automates model, optimizer, and
dataloader wrapping for DDP, FSDP, or DeepSpeed with strategy heuristics
based on model size and VRAM.

Usage::

    from ml_platform_sdk.tasks.accelerate import accelerate_task, platform

    @accelerate_task(num_nodes=2, gpus_per_node=4, mem="64Gi")
    def train(epochs: int) -> str:
        import torch

        model = torch.nn.Linear(100, 10)
        optimizer = torch.optim.Adam(model.parameters())
        model, optimizer, _ = platform.setup(model, optimizer, strategy="ddp")

        for epoch in range(epochs):
            ...
        return "done"
"""

import functools
import os
import tempfile
from typing import Callable

from flytekit import PodTemplate, Resources, task
from ml_platform_sdk.tasks.gpu import (
    GPU_NODE_LABEL,
    GPU_NODE_LABEL_VALUE,
    GPU_TAINT_EFFECT,
    GPU_TAINT_KEY,
    GPU_TAINT_VALUE,
    GPU_TYPE_SELECTORS,
    INSTANCE_FAMILY_LABEL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_ENV_VAR: str = "ACCELERATE_STRATEGY"


# ---------------------------------------------------------------------------
# Platform helper (lazy — no torch at import time)
# ---------------------------------------------------------------------------


class Platform:
    """Helper for distributed training setup.

    All heavy imports (``torch``, ``deepspeed``, ``mlflow``) happen inside
    method bodies so this module can be imported at Flyte registration time
    without GPU dependencies.
    """

    def setup(
        self,
        model,
        optimizer,
        dataloader=None,
        strategy: str = "auto",
        mixed_precision: str | None = "bf16",
        batch_size: int | None = None,
    ):
        """Wrap *model*, *optimizer*, and optional *dataloader* for distributed training.

        Args:
            model: A ``torch.nn.Module``.
            optimizer: A ``torch.optim.Optimizer``.
            dataloader: Optional ``torch.utils.data.DataLoader``.
            strategy: ``"auto"``, ``"ddp"``, ``"fsdp"``, or ``"deepspeed"``.
            mixed_precision: ``"fp16"``, ``"bf16"``, or ``None``.
            batch_size: Optional per-GPU batch size for better memory estimation
                when ``strategy="auto"``.

        Returns:
            Tuple of (model, optimizer, dataloader).
        """
        import torch
        import torch.distributed as dist

        # Use strategy from decorator env if set
        decorator_strategy = os.environ.get(STRATEGY_ENV_VAR, "auto")
        if decorator_strategy != "auto":
            strategy = decorator_strategy

        # 1. Initialize process group (skip in local/single-process mode)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if world_size > 1 and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        # Always derive rank/world_size from dist when initialized
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            model = model.to(device)

        if strategy == "auto":
            strategy = self._heuristics(model, batch_size=batch_size)

        # 2. Strategy wrapping (skip if single-process / local dev)
        if world_size > 1:
            if strategy == "ddp":
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank] if torch.cuda.is_available() else None,
                )
            elif strategy == "fsdp":
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel,
                )

                model = FullyShardedDataParallel(model)
                # Rebuild optimizer to reference FSDP-wrapped parameters
                optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
            elif strategy == "deepspeed":
                try:
                    import deepspeed
                except ImportError:
                    raise ImportError(
                        "DeepSpeed is required for strategy='deepspeed'. "
                        "Install it with: pip install deepspeed"
                    ) from None

                ds_config = {
                    "train_batch_size": 32,
                    "fp16": {"enabled": mixed_precision == "fp16"},
                    "bf16": {"enabled": mixed_precision == "bf16"},
                    "zero_optimization": {"stage": 3},
                }
                model, optimizer, _, _ = deepspeed.initialize(
                    model=model, optimizer=optimizer, config=ds_config
                )

        # 3. MLflow logging (rank 0 only, best-effort)
        if rank == 0:
            try:
                import mlflow

                mlflow.log_params(
                    {
                        "strategy": strategy,
                        "mixed_precision": mixed_precision or "none",
                        "world_size": world_size,
                    }
                )
            except Exception:
                pass

        # 4. DistributedSampler injection (only in distributed mode)
        if dataloader is not None and world_size > 1:
            if not isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
                if isinstance(dataloader.dataset, torch.utils.data.IterableDataset):
                    raise TypeError(
                        "DistributedSampler does not support IterableDataset. "
                        "Shard the iterable dataset manually per rank/worker."
                    )
                shuffle = isinstance(dataloader.sampler, torch.utils.data.RandomSampler)
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataloader.dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                )
                dataloader = torch.utils.data.DataLoader(
                    dataloader.dataset,
                    batch_size=dataloader.batch_size,
                    num_workers=dataloader.num_workers,
                    pin_memory=dataloader.pin_memory,
                    drop_last=dataloader.drop_last,
                    collate_fn=dataloader.collate_fn,
                    worker_init_fn=dataloader.worker_init_fn,
                    persistent_workers=dataloader.persistent_workers,
                    prefetch_factor=dataloader.prefetch_factor,
                    timeout=dataloader.timeout,
                    generator=dataloader.generator,
                    multiprocessing_context=dataloader.multiprocessing_context,
                    sampler=sampler,
                )

        return model, optimizer, dataloader

    def _heuristics(self, model, *, batch_size: int | None = None) -> str:
        """Choose strategy based on model size and GPU VRAM.

        - DDP when the full training footprint fits in a single GPU (<50% VRAM).
        - FSDP when sharding across GPUs is sufficient.
        - DeepSpeed when the model is too large even after sharding across all
          GPUs, requiring CPU/NVMe offloading.

        Memory estimate includes:
        - Parameters (dtype-aware)
        - Gradients (same size as params)
        - Optimizer states (2x params for Adam momentum + variance)
        - Activations (scaled by batch_size when known, otherwise ~1x params)
        """
        import torch

        num_params = sum(p.numel() for p in model.parameters())
        param_dtype = next(model.parameters()).dtype
        bytes_per_param = 2 if param_dtype in (torch.float16, torch.bfloat16) else 4

        param_mem = num_params * bytes_per_param
        grad_mem = param_mem
        # Adam keeps fp32 copies of momentum + variance regardless of param dtype
        optimizer_mem = num_params * 4 * 2
        # Activation memory: activations are produced and consumed layer by
        # layer, so peak usage is proportional to the largest single layer
        # (not total params).  We approximate per-layer activation size by the
        # parameter count of the largest layer, and scale by batch_size.
        max_layer_params = max(
            (
                sum(p.numel() for p in m.parameters(recurse=False))
                for m in model.modules()
                if sum(1 for _ in m.parameters(recurse=False)) > 0
            ),
            default=num_params,
        )
        activation_mem = max_layer_params * bytes_per_param
        if batch_size is not None and batch_size > 0:
            activation_mem *= batch_size

        total_mem_est = param_mem + grad_mem + optimizer_mem + activation_mem

        if not torch.cuda.is_available():
            return "ddp"

        try:
            gpu_vram = torch.cuda.get_device_properties(0).total_memory
        except Exception:
            gpu_vram = 24 * 1024**3  # assume A10G (24 GB)

        # Reserve ~10% for CUDA context, NCCL buffers, cuDNN workspace,
        # and memory fragmentation overhead.
        usable_vram = gpu_vram * 0.9

        # Model fits on a single GPU → DDP (fastest)
        if total_mem_est < usable_vram * 0.5:
            return "ddp"

        # FSDP/DeepSpeed shard params + grads + optimizer, but NOT activations.
        # Each GPU still holds full activations for its micro-batch.
        shardable_mem = param_mem + grad_mem + optimizer_mem
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        num_gpus = max(1, world_size)
        per_gpu_sharded = shardable_mem / num_gpus + activation_mem

        # If even after sharding the per-GPU footprint exceeds 80% usable VRAM,
        # we need CPU offloading → DeepSpeed ZeRO-Infinity
        if per_gpu_sharded > usable_vram * 0.8:
            return "deepspeed"

        return "fsdp"


platform = Platform()


# ---------------------------------------------------------------------------
# Worker function for multi-GPU spawning
# ---------------------------------------------------------------------------


def _mp_target(
    local_rank,
    fn_bytes,
    strategy,
    rank_offset,
    world_size,
    master_addr,
    master_port,
    result_file,
    args,
    kwargs,
):
    """Entrypoint for ``mp.spawn``. Must be at module scope to be pickleable."""
    import cloudpickle

    fn = cloudpickle.loads(fn_bytes)
    os.environ["LOCAL_RANK"] = str(local_rank)
    _worker_fn(
        fn,
        strategy,
        rank_offset,
        world_size,
        master_addr,
        master_port,
        result_file if local_rank == 0 else None,
        args,
        kwargs,
    )


def _worker_fn(
    fn, strategy, rank_offset, world_size, master_addr, master_port, result_file, args, kwargs
):
    """Run *fn* inside a spawned process with distributed env vars set."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = rank_offset + local_rank

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ[STRATEGY_ENV_VAR] = strategy

    result = fn(*args, **kwargs)

    # Capture rank 0 result atomically via cloudpickle
    if rank == 0 and result_file:
        import cloudpickle

        tmp_path = result_file + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                cloudpickle.dump(result, f)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, result_file)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise RuntimeError(
                "Failed to serialize rank 0 task result with cloudpickle. "
                "Return values from accelerate_task functions must be "
                "cloudpickle-serializable."
            ) from e

    return result


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def accelerate_task(
    strategy: str = "auto",
    num_nodes: int = 1,
    gpus_per_node: int = 1,
    gpu_type: str = "any",
    cpu: str = "4",
    mem: str = "16Gi",
    **task_kwargs,
):
    """Decorator that wraps a function as a Flyte task with distributed PyTorch config.

    Uses the KFPyTorch operator for multi-node training and ``mp.spawn``
    for multi-GPU-per-node.

    Args:
        strategy: ``"auto"``, ``"ddp"``, ``"fsdp"``, or ``"deepspeed"``.
        num_nodes: Number of training nodes.
        gpus_per_node: GPUs per node.
        gpu_type: GPU family to target (``"any"``, ``"a10g"``, ``"a100"``).
        cpu: CPU request per node.
        mem: Memory request per node.
        **task_kwargs: Extra keyword arguments forwarded to ``flytekit.task``.

    Returns:
        A decorated Flyte task with distributed training support.

    Example::

        @accelerate_task(num_nodes=2, gpus_per_node=4, mem="64Gi")
        def train(epochs: int) -> str:
            import torch
            model = torch.nn.Linear(100, 10)
            optimizer = torch.optim.Adam(model.parameters())
            model, optimizer, _ = platform.setup(model, optimizer)
            ...
            return "done"
    """
    try:
        from flytekitplugins.kfpytorch import PyTorch, Worker
    except ImportError as e:
        raise ImportError(
            "accelerate_task() requires the Flyte KFPyTorch plugin. "
            "Install it with: pip install flytekitplugins-kfpytorch"
        ) from e
    from kubernetes.client import V1Container, V1PodSpec, V1Toleration

    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    if gpus_per_node < 1:
        raise ValueError(f"gpus_per_node must be >= 1, got {gpus_per_node}")

    valid_strategies = {"auto", "ddp", "fsdp", "deepspeed"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid strategy {strategy!r}. Must be one of: {', '.join(sorted(valid_strategies))}"
        )

    pytorch_config = PyTorch(
        master=Worker(replicas=1),
        worker=Worker(replicas=max(0, num_nodes - 1)),
        increase_shared_mem=False,
    )

    res = Resources(cpu=cpu, mem=mem, gpu=str(gpus_per_node))

    node_selector = {GPU_NODE_LABEL: GPU_NODE_LABEL_VALUE}
    if gpu_type != "any" and gpu_type in GPU_TYPE_SELECTORS:
        node_selector[INSTANCE_FAMILY_LABEL] = GPU_TYPE_SELECTORS[gpu_type]

    pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary")],
            node_selector=node_selector,
            tolerations=[
                V1Toleration(
                    key=GPU_TAINT_KEY,
                    operator="Equal",
                    value=GPU_TAINT_VALUE,
                    effect=GPU_TAINT_EFFECT,
                )
            ],
        )
    )

    def decorator(fn: Callable) -> Callable:
        reserved_keys = {"task_config", "requests", "limits", "pod_template"}
        conflicts = reserved_keys & set(task_kwargs)
        if conflicts:
            raise ValueError(
                f"accelerate_task manages {conflicts} internally; "
                "do not pass them in **task_kwargs"
            )

        @task(
            task_config=pytorch_config,
            requests=res,
            limits=res,
            pod_template=pod_template,
            shared_memory=True,
            **task_kwargs,
        )
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import torch.multiprocessing as mp

            master_addr = os.environ.get("MASTER_ADDR")
            master_port = os.environ.get("MASTER_PORT")
            node_rank = int(os.environ.get("RANK", "0"))
            world_size_nodes = int(os.environ.get("WORLD_SIZE", "1"))

            if world_size_nodes > 1 and not master_addr:
                raise RuntimeError(
                    "Multi-node distributed execution requires "
                    "MASTER_ADDR to be set when WORLD_SIZE > 1."
                )
            master_addr = master_addr or "localhost"
            master_port = master_port or "29500"

            total_world_size = world_size_nodes * gpus_per_node
            rank_offset = node_rank * gpus_per_node

            if gpus_per_node > 1:
                result_file = None
                if node_rank == 0:
                    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False)
                    result_file = tmp.name
                    tmp.close()

                import cloudpickle

                fn_bytes = cloudpickle.dumps(fn)

                try:
                    mp.spawn(
                        _mp_target,
                        args=(
                            fn_bytes,
                            strategy,
                            rank_offset,
                            total_world_size,
                            master_addr,
                            master_port,
                            result_file,
                            args,
                            kwargs,
                        ),
                        nprocs=gpus_per_node,
                        join=True,
                    )

                    result = None
                    if (
                        node_rank == 0
                        and result_file
                        and os.path.exists(result_file)
                        and os.path.getsize(result_file) > 0
                    ):
                        try:
                            import cloudpickle

                            with open(result_file, "rb") as f:
                                result = cloudpickle.load(f)
                        except Exception as e:
                            raise RuntimeError(
                                "Failed to deserialize result from rank 0 process."
                            ) from e
                    return result
                finally:
                    if result_file and os.path.exists(result_file):
                        os.unlink(result_file)
            else:
                # Single GPU per node
                os.environ["LOCAL_RANK"] = "0"
                return _worker_fn(
                    fn,
                    strategy,
                    rank_offset,
                    total_world_size,
                    master_addr,
                    master_port,
                    None,
                    args,
                    kwargs,
                )

        return wrapper

    return decorator
