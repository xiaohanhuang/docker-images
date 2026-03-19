"""Flyte task definition for full_finetune component."""

import logging
from typing import Dict, Tuple

from flytekit import PodTemplate, Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

logger = logging.getLogger(__name__)

# ── GPU pod template with EFS mount and GPU toleration ─────────────────
try:
    from kubernetes.client import (
        V1Container,
        V1PersistentVolumeClaimVolumeSource,
        V1PodSpec,
        V1Toleration,
        V1Volume,
        V1VolumeMount,
    )

    _efs_volume = V1Volume(
        name="efs-storage",
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name="efs-claim"),
    )
    _efs_mount = V1VolumeMount(name="efs-storage", mount_path="/mnt/efs")

    _gpu_efs_pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(name="primary", volume_mounts=[_efs_mount]),
            ],
            volumes=[_efs_volume],
            tolerations=[
                V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Equal",
                    value="true",
                    effect="NoSchedule",
                )
            ],
            node_selector={"role": "gpu-worker"},
        )
    )
except ImportError:
    _gpu_efs_pod_template = None


@task(
    retries=1,
    requests=Resources(cpu="8", mem="32Gi", gpu="1"),
    limits=Resources(cpu="16", mem="64Gi", gpu="1"),
    cache=False,
    pod_template=_gpu_efs_pod_template,
)
def full_finetune(
    base_model: str,
    train_data: FlyteFile,
    val_data: FlyteFile,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    use_efs_checkpoints: bool = True,
) -> Tuple[FlyteDirectory, Dict[str, float]]:
    """Full-parameter fine-tuning.

    Args:
        base_model: HuggingFace model ID.
        train_data: Training dataset (Arrow format, tar.gz).
        val_data: Validation dataset (Arrow format, tar.gz).
        num_epochs: Training epochs.
        learning_rate: Learning rate.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        use_efs_checkpoints: Save to EFS.

    Returns:
        Tuple of (checkpoint_path, metrics).
    """
    train_data.download()
    val_data.download()

    from ._training import run_full_finetune

    result = run_full_finetune(
        base_model=base_model,
        train_data_path=train_data.path,
        val_data_path=val_data.path,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_efs_checkpoints=use_efs_checkpoints,
    )

    return FlyteDirectory(path=result["checkpoint_path"]), result["metrics"]
