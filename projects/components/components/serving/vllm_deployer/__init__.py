"""
Serving component — deploy a HuggingFace model as a vLLM inference endpoint.

This component deploys models as production-grade Kubernetes services with
automatic GPU configuration, tensor parallelism, and KEDA-based autoscaling.

Image: genai-gpu
"""

from ._manifests import (  # noqa: F401 — re-exported for tests
    _generate_deployment_manifest,
    _generate_scaled_object_manifest,
    _generate_service_manifest,
)
from ._task import VLLMDeploymentOutput, vllm_deployer  # noqa: F401
from ._validation import _validate_inputs, _validate_service_name  # noqa: F401

__all__ = ["vllm_deployer", "VLLMDeploymentOutput"]
