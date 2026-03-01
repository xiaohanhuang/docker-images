"""
EFS Task Decorator for the ML Platform SDK.

Provides utilities to auto-inject an EFS (Elastic File System) PersistentVolumeClaim
volume mount into Flyte task pod specs, making shared storage available at /mnt/efs
inside every task container.

Usage:
    from ml_platform_sdk.tasks.efs import efs_task

    @efs_task(requests=Resources(cpu="2", mem="4Gi"))
    def my_task(input: str) -> str:
        # /mnt/efs is available inside this container
        with open("/mnt/efs/data.txt") as f:
            return f.read()

    # Or build the pod template manually and pass it to a standard @task:
    from ml_platform_sdk.tasks.efs import build_efs_pod_template
    from flytekit import task

    @task(pod_template=build_efs_pod_template(), requests=Resources(cpu="1", mem="2Gi"))
    def another_task() -> None:
        ...
"""

from typing import Callable, Optional

from flytekit import PodTemplate, Resources, task
from kubernetes.client import (
    V1Container,
    V1PersistentVolumeClaimVolumeSource,
    V1PodSpec,
    V1Volume,
    V1VolumeMount,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EFS_MOUNT_PATH: str = "/mnt/efs"
EFS_PVC_NAME: str = "efs-claim"
EFS_VOLUME_NAME: str = "efs-storage"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def build_efs_pod_template(
    mount_path: str = EFS_MOUNT_PATH,
    pvc_name: str = EFS_PVC_NAME,
    read_only: bool = False,
) -> PodTemplate:
    """Return a :class:`flytekit.PodTemplate` that mounts the EFS PVC.

    Args:
        mount_path: The path inside the container where EFS will be mounted.
        pvc_name: Name of the Kubernetes PersistentVolumeClaim backed by EFS.
        read_only: Whether to mount the volume as read-only.

    Returns:
        A PodTemplate that Flytekit merges into the task's pod spec.
    """
    return PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="primary",
                    volume_mounts=[
                        V1VolumeMount(
                            name=EFS_VOLUME_NAME,
                            mount_path=mount_path,
                            read_only=read_only,
                        )
                    ],
                )
            ],
            volumes=[
                V1Volume(
                    name=EFS_VOLUME_NAME,
                    persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                        claim_name=pvc_name,
                        read_only=read_only,
                    ),
                )
            ],
        )
    )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def efs_task(
    requests: Optional[Resources] = None,
    limits: Optional[Resources] = None,
    retries: int = 0,
    cache: bool = False,
    cache_version: str = "1.0",
    mount_path: str = EFS_MOUNT_PATH,
    pvc_name: str = EFS_PVC_NAME,
    read_only: bool = False,
    **kwargs,
) -> Callable:
    """Decorator that wraps a function as a Flyte task with an EFS volume mount.

    The decorated function will have /mnt/efs (or *mount_path*) available inside
    its container, backed by the ``efs-claim`` PVC (or *pvc_name*).

    Args:
        requests: CPU/memory resource requests.
        limits: CPU/memory resource limits.
        retries: Number of Flyte retries on failure.
        cache: Whether to enable Flyte output caching.
        cache_version: Cache version string.
        mount_path: Container path where EFS is mounted (default: /mnt/efs).
        pvc_name: Name of the EFS-backed PVC (default: efs-claim).
        read_only: Mount the volume read-only.
        **kwargs: Additional keyword arguments forwarded to :func:`flytekit.task`.

    Returns:
        A decorated Flyte task function with the EFS volume injected.
    """
    pod_template = build_efs_pod_template(
        mount_path=mount_path,
        pvc_name=pvc_name,
        read_only=read_only,
    )

    def decorator(fn: Callable) -> Callable:
        task_kwargs = dict(
            pod_template=pod_template,
            retries=retries,
            cache=cache,
            cache_version=cache_version,
            **kwargs,
        )
        if requests is not None:
            task_kwargs["requests"] = requests
        if limits is not None:
            task_kwargs["limits"] = limits

        # Apply @task directly to fn so Flytekit sees the original type annotations.
        return task(**task_kwargs)(fn)

    return decorator
