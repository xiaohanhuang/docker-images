"""
Fault-Tolerant Task Decorators for the ML Platform SDK.

Provides decorators that automatically enable fault tolerance for training tasks:
- Automatic periodic checkpointing (EFS + S3)
- Flyte retry policies with spot interruption handling
- MLflow checkpoint lineage tracking

Note: SIGTERM handling is *not* built in. Users must call
``CheckpointManager.shutdown()`` explicitly (ideally in a ``try/finally``) to
guarantee that pending S3 uploads complete before the process exits.

Usage:
    from ml_platform_sdk.tasks.fault_tolerant import fault_tolerant_task

    @fault_tolerant_task(
        requests=Resources(cpu="4", mem="14Gi", gpu="1"),
        checkpoint_interval_steps=100,
        checkpoint_interval_seconds=300,
        s3_bucket="my-bucket",
        s3_prefix="checkpoints/my-training",
    )
    def my_training_task(dataset_path: str) -> str:
        # Training logic with automatic checkpointing
        ...
"""

from typing import Callable, Optional

from flytekit import PodTemplate, Resources
from flytekit import task as flyte_task
from kubernetes.client import (
    V1Container,
    V1PodSpec,
    V1Toleration,
)
from ml_platform_sdk.tasks.efs import EFS_MOUNT_PATH, build_efs_pod_template


def fault_tolerant_task(
    requests: Optional[Resources] = None,
    limits: Optional[Resources] = None,
    container_image: Optional[str] = None,
    checkpoint_interval_steps: int = 100,
    checkpoint_interval_seconds: int = 300,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    max_checkpoints_to_keep: int = 3,
    retries: int = 3,
    interruptible: bool = True,
    efs_mount_path: str = EFS_MOUNT_PATH,
    mlflow_tracking: bool = True,
    **kwargs,
) -> Callable:
    """
    Decorator that wraps a function as a fault-tolerant Flyte task.

    The decorated task automatically gets:
    1. EFS volume mount for fast checkpoint storage
    2. S3 backup configuration for durable storage
    3. Flyte retry policy (up to `retries` attempts)
    4. Spot instance support (if `interruptible=True`)
    5. GPU tolerations for GPU workloads
    6. Environment variables for checkpoint configuration

    The task function receives checkpoint configuration via environment variables:
    - CHECKPOINT_BASE_DIR: Base checkpoint directory on EFS
      (combine with FLYTE_INTERNAL_EXECUTION_ID for a per-execution path)
    - CHECKPOINT_DIR: Alias for CHECKPOINT_BASE_DIR (convenience; still a
      *base* path — append the execution ID before writing checkpoints)
    - CHECKPOINT_INTERVAL_STEPS: Save every N steps
    - CHECKPOINT_INTERVAL_SECONDS: Save every M seconds
    - S3_CHECKPOINT_BUCKET: S3 bucket for backup
    - S3_CHECKPOINT_PREFIX: S3 key prefix
    - MAX_CHECKPOINTS_TO_KEEP: Number of checkpoints to retain
    - MLFLOW_CHECKPOINT_TRACKING: Whether to log to MLflow

    Args:
        requests: CPU/memory/GPU resource requests
        limits: CPU/memory/GPU resource limits
        container_image: Container image to use
        checkpoint_interval_steps: Save checkpoint every N steps
        checkpoint_interval_seconds: Save checkpoint every M seconds
        s3_bucket: S3 bucket name for durable backup
        s3_prefix: S3 key prefix for checkpoints
        max_checkpoints_to_keep: Maximum number of checkpoints to retain
        retries: Number of Flyte retries on failure
        interruptible: Whether task can run on spot instances
        efs_mount_path: Mount path for EFS volume
        mlflow_tracking: Whether to log checkpoint metadata to MLflow
        **kwargs: Additional arguments passed to @task decorator

    Returns:
        Decorated fault-tolerant Flyte task

    Example:
        @fault_tolerant_task(
            requests=Resources(cpu="4", mem="14Gi", gpu="1"),
            checkpoint_interval_steps=100,
            s3_bucket="ml-platform-data",
            s3_prefix="checkpoints/my-training",
        )
        def train_model(dataset_path: str, epochs: int) -> str:
            from ml_platform_sdk.checkpoint import CheckpointManager

            # Initialize checkpoint manager from env vars
            exec_id = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "local")
            checkpoint_dir = os.path.join(
                os.environ["CHECKPOINT_BASE_DIR"], exec_id
            )
            ckpt_mgr = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                s3_bucket=os.environ.get("S3_CHECKPOINT_BUCKET"),
                s3_prefix=os.environ.get("S3_CHECKPOINT_PREFIX"),
                save_interval_steps=int(os.environ.get("CHECKPOINT_INTERVAL_STEPS", "100")),
                save_interval_seconds=int(os.environ.get("CHECKPOINT_INTERVAL_SECONDS", "300")),
            )

            # Load latest checkpoint
            checkpoint = ckpt_mgr.load_latest_checkpoint()
            start_step = checkpoint["step"] + 1 if checkpoint else 0

            # Training loop
            for step in range(start_step, num_steps):
                # ... training logic ...

                # Save checkpoint periodically
                if ckpt_mgr.should_save(step):
                    ckpt_mgr.save_checkpoint(
                        step=step,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        metrics={"loss": loss, "accuracy": acc},
                    )

            # Graceful shutdown
            ckpt_mgr.shutdown()

            return "Training complete"
    """
    # Build EFS pod template
    base_template = build_efs_pod_template(mount_path=efs_mount_path)
    # Add GPU toleration if GPU resources are requested or limited
    has_gpu = (requests and requests.gpu) or (limits and limits.gpu)
    if has_gpu:
        base_template.pod_spec.tolerations = [
            V1Toleration(
                key="nvidia.com/gpu",
                operator="Equal",
                value="true",
                effect="NoSchedule",
            )
        ]
    pod_template = base_template

    # Prepare environment variables for checkpoint configuration
    # CHECKPOINT_BASE_DIR is the EFS base; users construct per-execution
    # checkpoint paths at runtime using FLYTE_INTERNAL_EXECUTION_ID
    # (automatically set by Flyte in every task pod).
    checkpoint_base = f"{efs_mount_path}/checkpoints"

    environment = dict(kwargs.get("environment", {}))
    environment.update(
        {
            "CHECKPOINT_BASE_DIR": checkpoint_base,
            "CHECKPOINT_DIR": checkpoint_base,
            "CHECKPOINT_INTERVAL_STEPS": str(checkpoint_interval_steps),
            "CHECKPOINT_INTERVAL_SECONDS": str(checkpoint_interval_seconds),
            "MAX_CHECKPOINTS_TO_KEEP": str(max_checkpoints_to_keep),
            "MLFLOW_CHECKPOINT_TRACKING": str(mlflow_tracking),
        }
    )

    if s3_bucket:
        environment["S3_CHECKPOINT_BUCKET"] = s3_bucket

    if s3_prefix:
        environment["S3_CHECKPOINT_PREFIX"] = s3_prefix

    kwargs["environment"] = environment

    # Build task kwargs
    task_kwargs = dict(
        pod_template=pod_template,
        retries=retries,
        interruptible=interruptible,
        cache=False,  # Disable caching for training tasks
        **kwargs,
    )

    if requests is not None:
        task_kwargs["requests"] = requests

    if limits is not None:
        task_kwargs["limits"] = limits

    if container_image is not None:
        task_kwargs["container_image"] = container_image

    def decorator(fn: Callable) -> Callable:
        return flyte_task(**task_kwargs)(fn)

    return decorator


def fault_tolerant_distributed_task(
    requests: Optional[Resources] = None,
    limits: Optional[Resources] = None,
    container_image: Optional[str] = None,
    checkpoint_interval_steps: int = 100,
    checkpoint_interval_seconds: int = 300,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    max_checkpoints_to_keep: int = 3,
    retries: int = 3,
    interruptible: bool = True,
    num_workers: int = 1,
    mlflow_tracking: bool = True,
    **kwargs,
) -> Callable:
    """
    Decorator for fault-tolerant distributed training tasks.

    Creates a PyTorchJob (via KFPyTorch) for multi-GPU/multi-node training
    with automatic checkpointing. Works with DDP, FSDP, DeepSpeed, or any
    framework that uses ``torch.distributed``.

    Args:
        requests: CPU/memory/GPU resource requests per worker
        limits: CPU/memory/GPU resource limits per worker
        container_image: Container image to use
        checkpoint_interval_steps: Save checkpoint every N steps
        checkpoint_interval_seconds: Save checkpoint every M seconds
        s3_bucket: S3 bucket name for durable backup
        s3_prefix: S3 key prefix for checkpoints
        max_checkpoints_to_keep: Maximum number of checkpoints to retain
        retries: Number of Flyte retries on failure
        interruptible: Whether task can run on spot instances
        num_workers: Number of DDP workers (in addition to master)
        mlflow_tracking: Whether to log checkpoint metadata to MLflow
        **kwargs: Additional arguments passed to @task decorator

    Returns:
        Decorated fault-tolerant PyTorch DDP task

    Example:
        @fault_tolerant_distributed_task(
            requests=Resources(cpu="4", mem="14Gi", gpu="1"),
            num_workers=2,
            checkpoint_interval_steps=50,
            s3_bucket="ml-platform-data",
        )
        def train_ddp(dataset_path: str) -> str:
            import torch.distributed as dist
            from ml_platform_sdk.checkpoint import CheckpointManager

            # DDP setup
            rank = int(os.environ.get("RANK", "0"))
            dist.init_process_group("nccl", init_method="env://")

            # Checkpoint manager (only rank 0 saves)
            if rank == 0:
                exec_id = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "local")
                checkpoint_dir = os.path.join(
                    os.environ["CHECKPOINT_BASE_DIR"], exec_id
                )
                ckpt_mgr = CheckpointManager(
                    checkpoint_dir=checkpoint_dir,
                    s3_bucket=os.environ.get("S3_CHECKPOINT_BUCKET"),
                    s3_prefix=os.environ.get("S3_CHECKPOINT_PREFIX"),
                )

            # Training loop with checkpointing
            ...
    """
    try:
        from flytekitplugins.kfpytorch import PyTorch
    except ImportError:
        raise ImportError(
            "flytekitplugins-kfpytorch is required for fault_tolerant_distributed_task. "
            "Install it with: pip install flytekitplugins-kfpytorch"
        ) from None
    from kubernetes.client import V1PersistentVolumeClaimVolumeSource, V1Volume, V1VolumeMount
    from ml_platform_sdk.tasks.efs import EFS_PVC_NAME, EFS_VOLUME_NAME

    # EFS volume for shared checkpoint storage
    efs_volume = V1Volume(
        name=EFS_VOLUME_NAME,
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=EFS_PVC_NAME),
    )

    pytorch_config = PyTorch(
        num_workers=num_workers,
        increase_shared_mem=False,
    )

    # Pod template for EFS, GPU tolerations, and anti-affinity
    from kubernetes.client import (
        V1Affinity,
        V1LabelSelector,
        V1LabelSelectorRequirement,
        V1PodAffinityTerm,
        V1PodAntiAffinity,
    )

    # Add GPU toleration when GPU resources are requested or limited
    has_gpu = (requests and requests.gpu) or (limits and limits.gpu)
    gpu_tolerations = (
        [
            V1Toleration(
                key="nvidia.com/gpu",
                operator="Equal",
                value="true",
                effect="NoSchedule",
            )
        ]
        if has_gpu
        else []
    )

    pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="primary",
                    volume_mounts=[V1VolumeMount(name=EFS_VOLUME_NAME, mount_path=EFS_MOUNT_PATH)],
                )
            ],
            volumes=[efs_volume],
            tolerations=gpu_tolerations,
            affinity=V1Affinity(
                pod_anti_affinity=V1PodAntiAffinity(
                    required_during_scheduling_ignored_during_execution=[
                        V1PodAffinityTerm(
                            label_selector=V1LabelSelector(
                                match_expressions=[
                                    V1LabelSelectorRequirement(
                                        key="training.kubeflow.org/job-name",
                                        operator="Exists",
                                    )
                                ]
                            ),
                            topology_key="kubernetes.io/hostname",
                        )
                    ]
                )
            ),
        )
    )

    # Prepare environment variables
    checkpoint_base = f"{EFS_MOUNT_PATH}/checkpoints"

    environment = dict(kwargs.get("environment", {}))
    environment.update(
        {
            "CHECKPOINT_BASE_DIR": checkpoint_base,
            "CHECKPOINT_DIR": checkpoint_base,
            "CHECKPOINT_INTERVAL_STEPS": str(checkpoint_interval_steps),
            "CHECKPOINT_INTERVAL_SECONDS": str(checkpoint_interval_seconds),
            "MAX_CHECKPOINTS_TO_KEEP": str(max_checkpoints_to_keep),
            "MLFLOW_CHECKPOINT_TRACKING": str(mlflow_tracking),
        }
    )

    if s3_bucket:
        environment["S3_CHECKPOINT_BUCKET"] = s3_bucket

    if s3_prefix:
        environment["S3_CHECKPOINT_PREFIX"] = s3_prefix

    kwargs["environment"] = environment

    # Build task kwargs
    task_kwargs = dict(
        task_config=pytorch_config,
        pod_template=pod_template,
        retries=retries,
        interruptible=interruptible,
        cache=False,
        shared_memory=True,
        **kwargs,
    )

    if requests is not None:
        task_kwargs["requests"] = requests

    if limits is not None:
        task_kwargs["limits"] = limits

    if container_image is not None:
        task_kwargs["container_image"] = container_image

    def decorator(fn: Callable) -> Callable:
        return flyte_task(**task_kwargs)(fn)

    return decorator


def fault_tolerant_ray_task(
    head_node_resources: Optional[Resources] = None,
    worker_node_resources: Optional[Resources] = None,
    num_workers: int = 2,
    container_image: Optional[str] = None,
    checkpoint_interval_steps: int = 100,
    checkpoint_interval_seconds: int = 300,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    max_checkpoints_to_keep: int = 3,
    retries: int = 3,
    interruptible: bool = True,
    mlflow_tracking: bool = True,
    **kwargs,
) -> Callable:
    """
    Decorator for fault-tolerant Ray training tasks.

    Configures Ray cluster with automatic checkpointing support.
    Ray workers get EFS mounts and GPU resources.

    Args:
        head_node_resources: Resources for Ray head node
        worker_node_resources: Resources for Ray worker nodes (per worker)
        num_workers: Number of Ray workers
        container_image: Container image to use
        checkpoint_interval_steps: Save checkpoint every N steps
        checkpoint_interval_seconds: Save checkpoint every M seconds
        s3_bucket: S3 bucket name for durable backup
        s3_prefix: S3 key prefix for checkpoints
        max_checkpoints_to_keep: Maximum number of checkpoints to retain
        retries: Number of Flyte retries on failure
        interruptible: Whether task can run on spot instances
        mlflow_tracking: Whether to log checkpoint metadata to MLflow
        **kwargs: Additional arguments passed to @task decorator

    Returns:
        Decorated fault-tolerant Ray task

    Example:
        @fault_tolerant_ray_task(
            worker_node_resources=Resources(cpu="4", mem="16Gi", gpu="1"),
            num_workers=2,
            checkpoint_interval_steps=100,
            s3_bucket="ml-platform-data",
        )
        def train_ray(dataset_path: str) -> str:
            from ray.train import ScalingConfig
            from ray.train.torch import TorchTrainer

            def train_func(config):
                from ml_platform_sdk.checkpoint import CheckpointManager
                import ray.train

                rank = ray.train.get_context().get_world_rank()

                # Checkpoint manager (only rank 0)
                if rank == 0:
                    ckpt_mgr = CheckpointManager(...)

                # Training loop
                ...

            trainer = TorchTrainer(
                train_loop_per_worker=train_func,
                scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
            )

            result = trainer.fit()
            return "Training complete"
    """
    from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig
    from ml_platform_sdk.tasks.efs import EFS_MOUNT_PATH, EFS_PVC_NAME, EFS_VOLUME_NAME

    # Default resource configs
    if head_node_resources is None:
        head_node_resources = Resources(cpu="2", mem="8Gi")

    if worker_node_resources is None:
        worker_node_resources = Resources(cpu="4", mem="16Gi", gpu="1")

    # Prepare environment variables early so we can inject into worker pods too
    checkpoint_base = f"{EFS_MOUNT_PATH}/checkpoints"

    checkpoint_env = {
        "CHECKPOINT_BASE_DIR": checkpoint_base,
        "CHECKPOINT_DIR": checkpoint_base,
        "CHECKPOINT_INTERVAL_STEPS": str(checkpoint_interval_steps),
        "CHECKPOINT_INTERVAL_SECONDS": str(checkpoint_interval_seconds),
        "MAX_CHECKPOINTS_TO_KEEP": str(max_checkpoints_to_keep),
        "MLFLOW_CHECKPOINT_TRACKING": str(mlflow_tracking),
    }
    if s3_bucket:
        checkpoint_env["S3_CHECKPOINT_BUCKET"] = s3_bucket
    if s3_prefix:
        checkpoint_env["S3_CHECKPOINT_PREFIX"] = s3_prefix

    # Convert to K8s env var format for the Ray worker pod template
    # User-supplied env vars first, then checkpoint_env overwrites to prevent
    # accidental override of checkpoint configuration
    merged_env = {**kwargs.get("environment", {}), **checkpoint_env}

    from kubernetes.client import (
        V1Container,
        V1EnvVar,
        V1PersistentVolumeClaimVolumeSource,
        V1PodSpec,
        V1ResourceRequirements,
        V1Volume,
        V1VolumeMount,
    )

    worker_env_vars = [V1EnvVar(name=k, value=v) for k, v in merged_env.items()]

    # Build resource dict
    resource_requests = {
        "cpu": str(worker_node_resources.cpu or "4"),
        "memory": str(worker_node_resources.mem or "16Gi"),
    }
    resource_limits = dict(resource_requests)
    if worker_node_resources.gpu:
        resource_requests["nvidia.com/gpu"] = str(worker_node_resources.gpu)
        resource_limits["nvidia.com/gpu"] = str(worker_node_resources.gpu)

    # GPU toleration
    gpu_tolerations = (
        [
            V1Toleration(
                key="nvidia.com/gpu",
                operator="Equal",
                value="true",
                effect="NoSchedule",
            )
        ]
        if worker_node_resources.gpu
        else []
    )

    worker_pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="ray-worker",
                    volume_mounts=[V1VolumeMount(name=EFS_VOLUME_NAME, mount_path=EFS_MOUNT_PATH)],
                    env=worker_env_vars,
                    resources=V1ResourceRequirements(
                        requests=resource_requests,
                        limits=resource_limits,
                    ),
                )
            ],
            volumes=[
                V1Volume(
                    name=EFS_VOLUME_NAME,
                    persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                        claim_name=EFS_PVC_NAME
                    ),
                )
            ],
            tolerations=gpu_tolerations,
        )
    )

    ray_config = RayJobConfig(
        head_node_config=HeadNodeConfig(
            ray_start_params={"dashboard-host": "0.0.0.0", "block": "true"},
            requests=head_node_resources,
        ),
        worker_node_config=[
            WorkerNodeConfig(
                group_name="gpu-group",
                replicas=num_workers,
                min_replicas=num_workers,
                max_replicas=num_workers,
                ray_start_params={},
                pod_template=worker_pod_template,
            )
        ],
    )

    # Prepare Flyte environment variables (for the head node)
    environment = dict(kwargs.get("environment", {}))
    environment.update(checkpoint_env)
    kwargs["environment"] = environment

    # Build task kwargs
    task_kwargs = dict(
        task_config=ray_config,
        retries=retries,
        interruptible=interruptible,
        cache=False,
        requests=head_node_resources,
        limits=head_node_resources,
        **kwargs,
    )

    if container_image is not None:
        task_kwargs["container_image"] = container_image

    def decorator(fn: Callable) -> Callable:
        return flyte_task(**task_kwargs)(fn)

    return decorator
