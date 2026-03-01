from typing import Any, Dict

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

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
def train_ray_task(
    dataset_path: str, training_config: Dict[str, Any]
) -> FlyteDirectory:
    """
    A Ray task that runs distributed training.
    """
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    # This function runs on the Ray Head node (driver)
    print(f"Starting training with config: {training_config}")

    # Define the training loop that runs on workers
    def train_func(config):
        # In a real scenario, imports should be here to ensure they are on workers
        import ray.train

        # Simulate training loop
        print("Training started...")
        # ... logic to load data from config["dataset_path"] ...
        # ... logic to init model ...

        # Report metrics
        ray.train.report({"loss": 0.01, "accuracy": 0.99})
        print("Training finished.")

    # Configure scaling
    scaling_config = ScalingConfig(
        num_workers=2,  # Should match or be less than replicas defined in ray_config
        use_gpu=True,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"dataset_path": dataset_path, **training_config},
        scaling_config=scaling_config,
    )

    result = trainer.fit()
    print(f"Training result: {result}")

    # Return model checkpoint path (simulated)
    return FlyteDirectory(path="/tmp/checkpoint")
