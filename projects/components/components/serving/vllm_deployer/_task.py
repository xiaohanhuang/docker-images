"""Flyte task definition for vllm_deployer component."""

import os
import time
from typing import Any, Dict, NamedTuple, Optional

from flytekit import Resources, task

from ._manifests import (
    _generate_deployment_manifest,
    _generate_scaled_object_manifest,
    _generate_service_manifest,
)
from ._validation import _validate_inputs


class VLLMDeploymentOutput(NamedTuple):
    """Output from vLLM deployment containing endpoint info and configuration."""

    endpoint_url: str
    service_name: str
    deployment_config: Dict[str, Any]


@task(
    retries=1,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
    cache=False,
)
def vllm_deployer(
    model_path: str,
    service_name: str,
    gpu_count: int = 1,
    quantization: Optional[str] = None,
    max_model_len: Optional[int] = None,
    min_replicas: int = 1,
    max_replicas: int = 5,
) -> VLLMDeploymentOutput:
    """Deploy a HuggingFace model as a vLLM inference endpoint on Kubernetes.

    This task creates a production-grade inference service with:
    - Automatic GPU configuration and tensor parallelism
    - Health checks and resource limits
    - Optional KEDA-based autoscaling
    - OpenAI-compatible API endpoint

    The deployed service exposes:
    - Chat completions API: http://{service_name}.{namespace}.svc.cluster.local:8000/v1
    - Health endpoint: http://{service_name}.{namespace}.svc.cluster.local:8000/health

    Args:
        model_path: HuggingFace model ID (e.g., ``mistralai/Mistral-7B-v0.1``)
            or S3 path (e.g., ``s3://bucket/checkpoints/model/``).
        service_name: Kubernetes Service name (DNS-safe: lowercase alphanumeric
            + hyphens, max 63 chars).
        gpu_count: Number of GPUs per pod. Automatically configures tensor
            parallelism when > 1.
        quantization: Quantization method for reduced memory usage. Options:
            ``"awq"`` (Activation-aware Weight Quantization),
            ``"gptq"`` (GPTQ 4-bit),
            ``"fp8"`` (FP8 quantization),
            or ``None`` (no quantization).
        max_model_len: Maximum context length in tokens. If ``None``, uses
            the model's native maximum.
        min_replicas: Minimum number of inference pods. Set to 0 to enable
            scale-to-zero with KEDA.
        max_replicas: Maximum number of inference pods. When autoscaling is
            enabled (min < max), KEDA scales based on request queue depth.

    Returns:
        VLLMDeploymentOutput containing:
            - endpoint_url: Full URL to the inference API endpoint
            - service_name: Kubernetes Service name
            - deployment_config: Dictionary with deployment details

    Raises:
        ValueError: If input validation fails (invalid service_name, gpu_count, etc.)
        RuntimeError: If Kubernetes deployment fails or times out

    Example:
        >>> # Deploy Mistral-7B with AWQ quantization
        >>> result = vllm_deployer(
        ...     model_path="mistralai/Mistral-7B-Instruct-v0.2",
        ...     service_name="mistral-7b",
        ...     gpu_count=1,
        ...     quantization="awq",
        ...     min_replicas=1,
        ...     max_replicas=1,
        ... )
        >>> print(result.endpoint_url)
        http://mistral-7b.default.svc.cluster.local:8000/v1

        >>> # Deploy Llama-70B with 4-way tensor parallelism and autoscaling
        >>> result = vllm_deployer(
        ...     model_path="meta-llama/Llama-2-70b-chat-hf",
        ...     service_name="llama-70b",
        ...     gpu_count=4,
        ...     quantization="gptq",
        ...     min_replicas=0,
        ...     max_replicas=10,
        ... )
    """
    # Lazy imports to avoid loading kubernetes at registration time
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    # Validate inputs
    _validate_inputs(model_path, service_name, gpu_count, quantization, min_replicas, max_replicas)

    # Read configuration from environment
    namespace = os.getenv("KUBERNETES_NAMESPACE", "default")
    vllm_image = os.getenv(
        "VLLM_IMAGE",
        "123456.dkr.ecr.us-west-2.amazonaws.com/genai-gpu:latest",
    )
    prometheus_address = os.getenv(
        "PROMETHEUS_ADDRESS",
        "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
    )

    # Load Kubernetes configuration (in-cluster)
    try:
        config.load_incluster_config()
    except config.ConfigException:
        # Fall back to kubeconfig for local testing
        config.load_kube_config()

    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    custom_objects_api = client.CustomObjectsApi()

    # Generate manifests
    deployment_manifest = _generate_deployment_manifest(
        service_name=service_name,
        model_path=model_path,
        gpu_count=gpu_count,
        quantization=quantization,
        max_model_len=max_model_len,
        min_replicas=min_replicas,
        vllm_image=vllm_image,
        namespace=namespace,
    )

    service_manifest = _generate_service_manifest(service_name=service_name, namespace=namespace)

    autoscaling_enabled = min_replicas < max_replicas
    if autoscaling_enabled:
        scaled_object_manifest = _generate_scaled_object_manifest(
            service_name=service_name,
            namespace=namespace,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            prometheus_address=prometheus_address,
        )

    # Apply Deployment
    try:
        apps_v1.read_namespaced_deployment(name=service_name, namespace=namespace)
        # Deployment exists, update it (patch avoids needing resourceVersion)
        apps_v1.patch_namespaced_deployment(
            name=service_name,
            namespace=namespace,
            body=deployment_manifest,
        )
        print(f"Updated Deployment: {service_name}")
    except ApiException as e:
        if e.status == 404:
            # Deployment doesn't exist, create it
            apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment_manifest,
            )
            print(f"Created Deployment: {service_name}")
        else:
            raise RuntimeError(f"Failed to apply Deployment: {e}") from e

    # Apply Service
    try:
        core_v1.read_namespaced_service(name=service_name, namespace=namespace)
        # Service exists, update it (patch avoids needing resourceVersion)
        core_v1.patch_namespaced_service(
            name=service_name,
            namespace=namespace,
            body=service_manifest,
        )
        print(f"Updated Service: {service_name}")
    except ApiException as e:
        if e.status == 404:
            # Service doesn't exist, create it
            core_v1.create_namespaced_service(
                namespace=namespace,
                body=service_manifest,
            )
            print(f"Created Service: {service_name}")
        else:
            raise RuntimeError(f"Failed to apply Service: {e}") from e

    # Apply KEDA ScaledObject if autoscaling is enabled
    if autoscaling_enabled:
        try:
            custom_objects_api.get_namespaced_custom_object(
                group="keda.sh",
                version="v1alpha1",
                namespace=namespace,
                plural="scaledobjects",
                name=f"{service_name}-scaler",
            )
            # ScaledObject exists, update it (patch avoids needing resourceVersion)
            custom_objects_api.patch_namespaced_custom_object(
                group="keda.sh",
                version="v1alpha1",
                namespace=namespace,
                plural="scaledobjects",
                name=f"{service_name}-scaler",
                body=scaled_object_manifest,
            )
            print(f"Updated ScaledObject: {service_name}-scaler")
        except ApiException as e:
            if e.status == 404:
                # ScaledObject doesn't exist, create it
                custom_objects_api.create_namespaced_custom_object(
                    group="keda.sh",
                    version="v1alpha1",
                    namespace=namespace,
                    plural="scaledobjects",
                    body=scaled_object_manifest,
                )
                print(f"Created ScaledObject: {service_name}-scaler")
            else:
                raise RuntimeError(f"Failed to apply ScaledObject: {e}") from e

    # Wait for Deployment to become ready (timeout: 300s)
    # For scale-to-zero deployments (min_replicas=0), KEDA manages replica count;
    # the deployment is immediately considered ready with 0 initial pods.
    if min_replicas == 0:
        print(f"Deployment {service_name} created with scale-to-zero (KEDA-managed)")
    else:
        timeout = 300
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=service_name, namespace=namespace
                )
                if (
                    deployment.status.ready_replicas is not None
                    and deployment.status.ready_replicas >= min_replicas
                ):
                    print(f"Deployment {service_name} is ready ({min_replicas} replicas)")
                    break
            except ApiException as e:
                raise RuntimeError(f"Failed to check Deployment status: {e}") from e
            time.sleep(10)
        else:
            # Timeout reached, gather pod status for debugging
            try:
                pods = core_v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"app={service_name}",
                )
                pod_statuses = [f"{pod.metadata.name}: {pod.status.phase}" for pod in pods.items]
                raise RuntimeError(
                    f"Deployment {service_name} did not become ready within {timeout}s. "
                    f"Pod statuses: {', '.join(pod_statuses)}"
                )
            except ApiException:
                raise RuntimeError(
                    f"Deployment {service_name} did not become ready within {timeout}s"
                )

    # Build endpoint URL
    endpoint_url = f"http://{service_name}.{namespace}.svc.cluster.local:8000/v1"

    # Build deployment configuration
    deployment_config = {
        "namespace": namespace,
        "deployment_name": service_name,
        "replicas": min_replicas,
        "gpu_count": gpu_count,
        "quantization": quantization,
        "max_model_len": max_model_len,
        "tensor_parallel_size": gpu_count,
        "autoscaling_enabled": autoscaling_enabled,
        "min_replicas": min_replicas,
        "max_replicas": max_replicas,
        "vllm_image": vllm_image,
    }

    return VLLMDeploymentOutput(
        endpoint_url=endpoint_url,
        service_name=service_name,
        deployment_config=deployment_config,
    )
