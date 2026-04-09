"""Kubernetes manifest generation for vLLM deployments."""

from typing import Any, Dict, Optional


def _generate_deployment_manifest(
    service_name: str,
    model_path: str,
    gpu_count: int,
    quantization: Optional[str],
    max_model_len: Optional[int],
    min_replicas: int,
    vllm_image: str,
    namespace: str,
) -> Dict[str, Any]:
    """Generate Kubernetes Deployment manifest for vLLM.

    Args:
        service_name: Name for the deployment and service.
        model_path: HuggingFace model ID or S3 path.
        gpu_count: Number of GPUs per pod.
        quantization: Quantization method.
        max_model_len: Maximum context length.
        min_replicas: Initial replica count.
        vllm_image: Docker image for vLLM.
        namespace: Target Kubernetes namespace.

    Returns:
        Deployment manifest as a dictionary.
    """
    # Base command and args for vLLM server
    command = ["python", "-m", "vllm.entrypoints.openai.api_server"]
    args = [
        "--model",
        model_path,
        "--port",
        "8000",
        "--host",
        "0.0.0.0",
    ]
    # Only specify tensor parallelism when using multiple GPUs
    if gpu_count > 1:
        args.extend(["--tensor-parallel-size", str(gpu_count)])

    # Add optional arguments
    if quantization:
        args.extend(["--quantization", quantization])
    if max_model_len:
        args.extend(["--max-model-len", str(max_model_len)])

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": service_name, "namespace": namespace},
        "spec": {
            "replicas": min_replicas,
            "selector": {"matchLabels": {"app": service_name}},
            "template": {
                "metadata": {"labels": {"app": service_name}},
                "spec": {
                    "tolerations": [
                        {
                            "key": "nvidia.com/gpu",
                            "operator": "Equal",
                            "value": "true",
                            "effect": "NoSchedule",
                        }
                    ],
                    "nodeSelector": {"role": "gpu-worker"},
                    "containers": [
                        {
                            "name": "vllm",
                            "image": vllm_image,
                            "command": command,
                            "args": args,
                            "ports": [{"containerPort": 8000, "name": "http"}],
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": "16Gi",
                                    "nvidia.com/gpu": str(gpu_count),
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "32Gi",
                                    "nvidia.com/gpu": str(gpu_count),
                                },
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 120,
                                "periodSeconds": 30,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3,
                            },
                        }
                    ],
                },
            },
        },
    }


def _generate_service_manifest(service_name: str, namespace: str) -> Dict[str, Any]:
    """Generate Kubernetes Service manifest.

    Args:
        service_name: Name for the service.
        namespace: Target Kubernetes namespace.

    Returns:
        Service manifest as a dictionary.
    """
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": service_name, "namespace": namespace},
        "spec": {
            "type": "ClusterIP",
            "selector": {"app": service_name},
            "ports": [{"port": 8000, "targetPort": 8000, "protocol": "TCP", "name": "http"}],
        },
    }


def _generate_scaled_object_manifest(
    service_name: str,
    namespace: str,
    min_replicas: int,
    max_replicas: int,
    prometheus_address: str,
) -> Dict[str, Any]:
    """Generate KEDA ScaledObject manifest for autoscaling.

    Args:
        service_name: Name of the target deployment.
        namespace: Target Kubernetes namespace.
        min_replicas: Minimum replica count.
        max_replicas: Maximum replica count.
        prometheus_address: Prometheus server address.

    Returns:
        ScaledObject manifest as a dictionary.
    """
    return {
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {"name": f"{service_name}-scaler", "namespace": namespace},
        "spec": {
            "scaleTargetRef": {"name": service_name},
            "minReplicaCount": min_replicas,
            "maxReplicaCount": max_replicas,
            "triggers": [
                {
                    "type": "prometheus",
                    "metadata": {
                        "serverAddress": prometheus_address,
                        "query": (
                            f'sum(vllm:num_requests_waiting{{job="vllm",app="{service_name}"}})'  # noqa: E501
                        ),
                        "threshold": "1",
                    },
                }
            ],
        },
    }
