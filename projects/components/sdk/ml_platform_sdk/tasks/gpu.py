"""
GPU Task Decorator for the ML Platform SDK.

Provides a decorator that wraps a function as a Flyte task with GPU scheduling
(tolerations, nodeSelector, resource requests) and optional PyTorch profiling.

When ``profile=True`` (or the ``ML_PLAT_PROFILE`` environment variable is set to
``"1"`` at runtime), the task automatically wraps execution with
``torch.profiler`` and writes traces to ``/mnt/efs/profiles/<execution_id>/``.

When ``nsight=True`` (or ``ML_PLAT_NSIGHT=1``), an init container injects the
``nsys`` binary into the pod via a shared volume, making it available at
``/opt/nsight/bin/nsys``.  Use ``ml_platform_sdk.profiling.nsight_profile()``
in your training code to collect Nsight Systems traces.

Usage:
    from ml_platform_sdk.tasks.gpu import gpu_task

    @gpu_task(gpu=1, memory="32Gi")
    def train(dataset_path: str) -> str:
        ...  # regular training code
        return "done"

    # With PyTorch profiling:
    @gpu_task(gpu=1, memory="32Gi", profile=True)
    def train_profiled(dataset_path: str) -> str:
        ...
        return "done"

    # With Nsight Systems (nsys injected via init container):
    @gpu_task(gpu=1, memory="32Gi", nsight=True)
    def train_nsight(dataset_path: str) -> str:
        from ml_platform_sdk.profiling import nsight_profile
        with nsight_profile():
            ...  # training loop — nsys traces CUDA kernels, nvtx, etc.
        return "done"

    # Or build the pod template manually:
    from ml_platform_sdk.tasks.gpu import build_gpu_pod_template
    from flytekit import task

    @task(pod_template=build_gpu_pod_template(gpu=1), requests=Resources(gpu="1"))
    def another_task() -> None:
        ...
"""

import functools
import os
from typing import Callable

from flytekit import PodTemplate, Resources, task
from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1EnvVar,
    V1PodSpec,
    V1ResourceRequirements,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPU_TAINT_KEY: str = "nvidia.com/gpu"
GPU_TAINT_VALUE: str = "true"
GPU_TAINT_EFFECT: str = "NoSchedule"
GPU_NODE_LABEL: str = "role"
GPU_NODE_LABEL_VALUE: str = "gpu-worker"
PROFILE_ENV_VAR: str = "ML_PLAT_PROFILE"
NSIGHT_ENV_VAR: str = "ML_PLAT_NSIGHT"
PROFILE_OUTPUT_BASE: str = "/mnt/efs/profiles"
NSIGHT_VOLUME_NAME: str = "nsight-bin"
NSIGHT_MOUNT_PATH: str = "/opt/nsight"
NSIGHT_CUDA_IMAGE: str = "nvidia/cuda:12.9.1-devel-ubuntu22.04"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GPU_TYPE_SELECTORS = {
    "a10g": "g5",
    "a100": "p4d",
}

INSTANCE_FAMILY_LABEL: str = "karpenter.k8s.aws/instance-family"


def build_gpu_pod_template(
    gpu: int = 1,
    gpu_type: str = "any",
    memory: str = "32Gi",
    cpu: str = "4",
    profile: bool = False,
    nsight: bool = False,
) -> PodTemplate:
    """Return a :class:`flytekit.PodTemplate` configured for GPU workloads.

    The template includes:
    - GPU resource requests/limits
    - Toleration for the ``nvidia.com/gpu=true:NoSchedule`` taint
    - nodeSelector for ``role: gpu-worker``
    - Optional GPU type targeting (a10g -> g5, a100 -> p4d instance families)
    - Optional ``ML_PLAT_PROFILE=1`` env var for auto-profiling
    - Optional Nsight Systems injection via init container

    Args:
        gpu: Number of GPUs to request.
        memory: Memory request/limit (e.g., "32Gi").
        cpu: CPU request/limit (e.g., "4").
        gpu_type: GPU type to target ("any", "a10g", "a100").
        profile: If True, inject ``ML_PLAT_PROFILE=1`` into the container env.
        nsight: If True, inject Nsight Systems (nsys) via init container.

    Returns:
        A PodTemplate that Flytekit merges into the task's pod spec.
    """
    node_selector = {GPU_NODE_LABEL: GPU_NODE_LABEL_VALUE}
    if gpu_type != "any" and gpu_type in GPU_TYPE_SELECTORS:
        instance_family = GPU_TYPE_SELECTORS[gpu_type]
        node_selector[INSTANCE_FAMILY_LABEL] = instance_family

    env_vars = []
    if profile:
        env_vars.append(V1EnvVar(name=PROFILE_ENV_VAR, value="1"))
    if nsight:
        env_vars.append(V1EnvVar(name=NSIGHT_ENV_VAR, value="1"))
        # Add nsight bin dir to PATH so nsys is discoverable
        env_vars.append(
            V1EnvVar(name="PATH", value=f"{NSIGHT_MOUNT_PATH}/bin:/usr/local/bin:/usr/bin:/bin")
        )

    resources = {"cpu": cpu, "memory": memory}
    if gpu > 0:
        resources["nvidia.com/gpu"] = str(gpu)

    volume_mounts = []
    volumes = []
    init_containers = []

    if nsight:
        volumes.append(
            V1Volume(
                name=NSIGHT_VOLUME_NAME,
                empty_dir=V1EmptyDirVolumeSource(),
            )
        )
        volume_mounts.append(
            V1VolumeMount(
                name=NSIGHT_VOLUME_NAME,
                mount_path=NSIGHT_MOUNT_PATH,
            )
        )
        init_containers.append(
            V1Container(
                name="nsight-injector",
                image=NSIGHT_CUDA_IMAGE,
                command=["/bin/sh", "-c"],
                args=[
                    "cp -r /opt/nvidia/nsight-systems/*/target/linux-desktop-glibc_*/* "
                    f"{NSIGHT_MOUNT_PATH}/ 2>/dev/null || "
                    "cp -r /opt/nvidia/nsight-systems/* "
                    f"{NSIGHT_MOUNT_PATH}/ 2>/dev/null || "
                    f"echo 'WARN: nsight-systems not found in image'"
                ],
                volume_mounts=[
                    V1VolumeMount(
                        name=NSIGHT_VOLUME_NAME,
                        mount_path=NSIGHT_MOUNT_PATH,
                    )
                ],
            )
        )

    container = V1Container(
        name="primary",
        resources=V1ResourceRequirements(requests=resources, limits=resources),
        env=env_vars or None,
        volume_mounts=volume_mounts or None,
    )

    tolerations = [
        V1Toleration(
            key=GPU_TAINT_KEY,
            operator="Equal",
            value=GPU_TAINT_VALUE,
            effect=GPU_TAINT_EFFECT,
        )
    ]

    return PodTemplate(
        pod_spec=V1PodSpec(
            containers=[container],
            init_containers=init_containers or None,
            node_selector=node_selector,
            tolerations=tolerations,
            volumes=volumes or None,
        )
    )


def _maybe_profile(fn: Callable) -> Callable:
    """Wrap *fn* so it auto-profiles when ``ML_PLAT_PROFILE=1`` at runtime."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.getenv(PROFILE_ENV_VAR) != "1":
            return fn(*args, **kwargs)

        import torch
        from ml_platform_sdk.profiling import _upload_to_s3

        execution_id = os.getenv("FLYTE_INTERNAL_EXECUTION_ID", "unknown")
        output_dir = os.path.join(PROFILE_OUTPUT_BASE, execution_id)
        os.makedirs(output_dir, exist_ok=True)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ):
            result = fn(*args, **kwargs)

        print(f"Profiling complete. Traces saved to {output_dir}")
        _upload_to_s3(output_dir)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def gpu_task(
    gpu: int = 1,
    gpu_type: str = "any",
    memory: str = "32Gi",
    cpu: str = "4",
    profile: bool = False,
    nsight: bool = False,
    retries: int = 0,
    cache: bool = False,
    cache_version: str = "1.0",
    **kwargs,
) -> Callable:
    """Decorator that wraps a function as a Flyte task with GPU scheduling.

    Injects a PodTemplate with GPU tolerations, nodeSelector, and resource
    requests.  When *profile* is ``True`` (or ``ML_PLAT_PROFILE=1`` is set at
    runtime), the training function is automatically wrapped with
    ``torch.profiler`` and traces are written to
    ``/mnt/efs/profiles/<execution_id>/``.

    When *nsight* is ``True`` (or ``ML_PLAT_NSIGHT=1``), an init container
    injects the ``nsys`` binary from a CUDA devel image.  Use
    ``ml_platform_sdk.profiling.nsight_profile()`` in your training code
    to capture Nsight Systems traces.

    Args:
        gpu: Number of GPUs to request.
        gpu_type: GPU family to target ("any", "a10g", "a100").
        memory: Memory request/limit (e.g., "32Gi").
        cpu: CPU request/limit (e.g., "4").
        profile: If True, enable PyTorch profiling via env var injection.
        nsight: If True, inject Nsight Systems (nsys) via init container.
        retries: Number of Flyte retries on failure.
        cache: Whether to enable Flyte output caching.
        cache_version: Cache version string.
        **kwargs: Additional keyword arguments forwarded to :func:`flytekit.task`.

    Returns:
        A decorated Flyte task function with GPU scheduling injected.

    Example::

        @gpu_task(gpu=1, memory="32Gi")
        def train(dataset: str) -> str:
            import torch
            model = torch.nn.Linear(100, 10).cuda()
            return "done"

        @gpu_task(gpu=4, gpu_type="a100", memory="64Gi", profile=True)
        def train_profiled(dataset: str) -> str:
            ...

        @gpu_task(gpu=1, nsight=True)
        def train_nsight(dataset: str) -> str:
            from ml_platform_sdk.profiling import nsight_profile
            with nsight_profile():
                ...  # nsys traces this section
            return "done"
    """
    pod_template = build_gpu_pod_template(
        gpu=gpu,
        gpu_type=gpu_type,
        memory=memory,
        cpu=cpu,
        profile=profile,
        nsight=nsight,
    )

    gpu_resources = {}
    if gpu > 0:
        gpu_resources["gpu"] = str(gpu)

    def decorator(fn: Callable) -> Callable:
        # Wrap with auto-profiler (activates only when env var is set)
        wrapped = _maybe_profile(fn)

        task_kwargs = dict(
            pod_template=pod_template,
            retries=retries,
            cache=cache,
            cache_version=cache_version,
            requests=Resources(cpu=cpu, mem=memory, **gpu_resources),
            limits=Resources(cpu=cpu, mem=memory, **gpu_resources),
            **kwargs,
        )

        return task(**task_kwargs)(wrapped)

    return decorator
