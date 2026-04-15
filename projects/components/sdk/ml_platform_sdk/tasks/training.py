import json
import logging
from typing import Any, Dict

from flytekit import PodTemplate, Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

logger = logging.getLogger(__name__)


def ray_gpu_config(
    num_workers: int = 2,
    gpu_per_worker: int = 1,
    worker_cpu: str = "3",
    worker_mem: str = "12Gi",
    head_cpu: str = "2",
    head_mem: str = "8Gi",
    separate_nodes: bool = True,
) -> RayJobConfig:
    """Build a RayJobConfig for GPU training with sensible defaults.

    Handles GPU tolerations, anti-affinity (one worker per node), and
    resource requests — so pipeline code stays simple.

    Args:
        num_workers: Number of Ray GPU workers.
        gpu_per_worker: GPUs per worker (default 1).
        worker_cpu: CPU request per worker.
        worker_mem: Memory request per worker.
        head_cpu: CPU request for the Ray head node.
        head_mem: Memory request for the Ray head node.
        separate_nodes: Force workers onto different nodes (anti-affinity).

    Returns:
        A RayJobConfig ready to use as ``task_config``.
    """
    from kubernetes.client import (
        V1Affinity,
        V1Container,
        V1LabelSelector,
        V1LabelSelectorRequirement,
        V1PodAffinityTerm,
        V1PodAntiAffinity,
        V1PodSpec,
        V1ResourceRequirements,
        V1Toleration,
    )

    resources = {
        "cpu": worker_cpu,
        "memory": worker_mem,
        "nvidia.com/gpu": str(gpu_per_worker),
    }

    tolerations = [
        V1Toleration(
            key="nvidia.com/gpu",
            operator="Equal",
            value="true",
            effect="NoSchedule",
        )
    ]

    affinity = None
    if separate_nodes:
        affinity = V1Affinity(
            pod_anti_affinity=V1PodAntiAffinity(
                required_during_scheduling_ignored_during_execution=[
                    V1PodAffinityTerm(
                        label_selector=V1LabelSelector(
                            match_expressions=[
                                V1LabelSelectorRequirement(
                                    key="ray.io/group",
                                    operator="In",
                                    values=["gpu-workers"],
                                )
                            ]
                        ),
                        topology_key="kubernetes.io/hostname",
                    )
                ]
            )
        )

    worker_pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="ray-worker",
                    resources=V1ResourceRequirements(
                        requests=resources,
                        limits={**resources, "cpu": str(int(worker_cpu) + 1)},
                    ),
                )
            ],
            tolerations=tolerations,
            affinity=affinity,
        )
    )

    return RayJobConfig(
        head_node_config=HeadNodeConfig(
            ray_start_params={"dashboard-host": "0.0.0.0"},
            requests=Resources(cpu=head_cpu, mem=head_mem),
        ),
        worker_node_config=[
            WorkerNodeConfig(
                group_name="gpu-workers",
                replicas=num_workers,
                min_replicas=num_workers,
                max_replicas=num_workers,
                pod_template=worker_pod_template,
            )
        ],
    )


def pytorch_ddp_config(
    num_workers: int = 2,
    gpu_per_worker: int = 1,
    worker_cpu: str = "4",
    worker_mem: str = "14Gi",
    separate_nodes: bool = True,
    efs_claim: str | None = "efs-claim",
    efs_mount_path: str = "/mnt/efs",
):
    """Build a PyTorch DDP task config (KFPyTorch) with sensible defaults.

    Hides all kubernetes.client boilerplate — GPU tolerations, pod
    anti-affinity (one worker per node), and optional EFS volume mounts
    are handled automatically.

    Args:
        num_workers: Number of DDP workers (excluding master). Total
            processes = num_workers + 1 (master). Use 1 for 2-GPU DDP.
        gpu_per_worker: GPUs per worker (default 1).
        worker_cpu: CPU request per worker pod.
        worker_mem: Memory request per worker pod (should leave ~2Gi
            headroom below instance memory for OS/runtime).
        separate_nodes: Force each worker onto a different physical node
            via pod anti-affinity. Recommended for multi-node DDP.
        efs_claim: Name of the EFS PersistentVolumeClaim to mount, or
            None to skip volume mounting.
        efs_mount_path: Mount path inside the container for the EFS PVC.

    Returns:
        A ``PyTorch`` task config ready to use as ``task_config``, plus
        a ``PodTemplate`` to pass as ``pod_template`` to ``@task``.
        Returns a ``(task_config, pod_template)`` tuple.

    Example::

        from ml_platform_sdk.tasks.training import pytorch_ddp_config

        pytorch_config, pod_template = pytorch_ddp_config(num_workers=1)

        @task(
            task_config=pytorch_config,
            requests=Resources(cpu="4", mem="14Gi", gpu="1"),
            limits=Resources(cpu="4", mem="14Gi", gpu="1"),
            container_image=TRAINING_IMAGE,
            pod_template=pod_template,
        )
        def ddp_train() -> str:
            ...
    """
    from flytekitplugins.kfpytorch import PyTorch, Worker
    from kubernetes.client import (
        V1Affinity,
        V1Container,
        V1LabelSelector,
        V1LabelSelectorRequirement,
        V1PersistentVolumeClaimVolumeSource,
        V1PodAffinityTerm,
        V1PodAntiAffinity,
        V1PodSpec,
        V1Toleration,
        V1Volume,
        V1VolumeMount,
    )

    gpu_toleration = V1Toleration(
        key="nvidia.com/gpu",
        operator="Equal",
        value="true",
        effect="NoSchedule",
    )

    # Anti-affinity: spread workers across separate physical nodes
    affinity = None
    if separate_nodes:
        affinity = V1Affinity(
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
        )

    # Optional EFS volume mount (for checkpointing)
    volumes = []
    volume_mounts = []
    if efs_claim:
        volumes.append(
            V1Volume(
                name="efs-storage",
                persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=efs_claim),
            )
        )
        volume_mounts.append(V1VolumeMount(name="efs-storage", mount_path=efs_mount_path))

    # Pod spec shared by master and worker — tolerations + EFS
    worker_pod_spec = V1PodSpec(
        containers=[V1Container(name="primary", volume_mounts=volume_mounts or None)],
        tolerations=[gpu_toleration],
        affinity=affinity,
        volumes=volumes or None,
    )
    worker_pod_template = PodTemplate(pod_spec=worker_pod_spec)

    # Separate PodTemplate for the @task decorator (anti-affinity only,
    # no EFS mounts — the task's own container handles that via the worker spec)
    task_pod_spec = V1PodSpec(
        containers=[V1Container(name="primary")],
        tolerations=[gpu_toleration],
        affinity=affinity,
    )
    task_pod_template = PodTemplate(pod_spec=task_pod_spec)

    pytorch_config = PyTorch(
        master=Worker(
            replicas=1,
            pod_template=worker_pod_template,
        ),
        worker=Worker(
            replicas=num_workers,
            pod_template=worker_pod_template,
        ),
        increase_shared_mem=False,
    )

    return pytorch_config, task_pod_template


# Define the Ray Cluster Config
# This configures the ephemeral Ray cluster that Flyte spins up on K8s
ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(
        ray_start_params={"dashboard-host": "0.0.0.0", "block": "true"},
        requests=Resources(cpu="2", mem="8Gi"),
    ),
    worker_node_config=[
        WorkerNodeConfig(
            group_name="gpu-group",
            replicas=2,  # Number of workers
            min_replicas=1,
            max_replicas=4,
            ray_start_params={},
            # requests/limits are mutually exclusive with pod_template in some Flytekit versions,
            # but usually they can coexist if correctly structured.
            # The error "Cannot specify both pod_template and requests/limits" suggests we must
            # choose one.
            # We will use pod_template as it allows nodeSelector/tolerations.
            pod_template={
                "spec": {
                    "containers": [
                        {
                            "name": "ray-worker",
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": "16Gi",
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "32Gi",
                                    "nvidia.com/gpu": "1",
                                },
                            },
                        }
                    ],
                    "nodeSelector": {"karpenter.k8s.aws/instance-type": "g5.xlarge"},
                    "tolerations": [
                        {
                            "key": "nvidia.com/gpu",
                            "operator": "Equal",
                            "value": "true",
                            "effect": "NoSchedule",
                        }
                    ],
                }
            },
        )
    ],
)


@task(
    task_config=ray_config,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
)
def train_ray_task(dataset_path: str, training_config: str = "{}") -> FlyteDirectory:
    """
    A Ray task that runs distributed training.

    Args:
        dataset_path: Path to training dataset.
        training_config: Training configuration as a JSON string.
    """
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    # Accept dict for backwards compatibility (previous API accepted dict directly).
    if isinstance(training_config, dict):
        config: Dict[str, Any] = training_config
    else:
        config = json.loads(training_config)

    # This function runs on the Ray Head node (driver)
    logger.info("Starting training with config: %s", config)

    # Define the training loop that runs on workers
    def train_func(config):
        import logging

        # In a real scenario, imports should be here to ensure they are on workers
        import ray.train

        logger = logging.getLogger(__name__)

        # Simulate training loop
        logger.info("Training started...")
        # ... logic to load data from config["dataset_path"] ...
        # ... logic to init model ...

        # Report metrics
        ray.train.report({"loss": 0.01, "accuracy": 0.99})
        logger.info("Training finished.")

    # Configure scaling
    scaling_config = ScalingConfig(
        num_workers=2,  # Should match or be less than replicas defined in ray_config
        use_gpu=True,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"dataset_path": dataset_path, **config},
        scaling_config=scaling_config,
    )

    result = trainer.fit()
    logger.info("Training result: %s", result)

    # Return model checkpoint path (simulated)
    return FlyteDirectory(path="/tmp/checkpoint")
