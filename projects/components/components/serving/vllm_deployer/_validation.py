"""Input validation for vLLM deployments."""

import re
from typing import Optional


def _validate_service_name(name: str) -> None:
    """Validate that service_name is a valid DNS label.

    Args:
        name: Service name to validate.

    Raises:
        ValueError: If service name is invalid.
    """
    if not name:
        raise ValueError("service_name cannot be empty")

    if len(name) > 63:
        raise ValueError(f"service_name too long: {len(name)} > 63 characters")

    # DNS label regex: lowercase alphanumeric + hyphens, must start/end with alphanumeric
    dns_regex = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
    if not re.match(dns_regex, name):
        raise ValueError(
            f"Invalid service_name '{name}': must match DNS label format "
            f"(lowercase alphanumeric + hyphens, start/end with alphanumeric)"
        )


def _validate_inputs(
    model_path: str,
    service_name: str,
    gpu_count: int,
    quantization: Optional[str],
    min_replicas: int,
    max_replicas: int,
) -> None:
    """Validate all input parameters.

    Args:
        model_path: HuggingFace model ID or S3 path.
        service_name: Kubernetes service name.
        gpu_count: Number of GPUs per pod.
        quantization: Quantization method.
        min_replicas: Minimum replicas.
        max_replicas: Maximum replicas.

    Raises:
        ValueError: If any parameter is invalid.
    """
    _validate_service_name(service_name)

    if not model_path or not model_path.strip():
        raise ValueError("model_path cannot be empty")

    if gpu_count < 1:
        raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")

    if gpu_count > 8:
        raise ValueError(f"gpu_count={gpu_count} exceeds practical limit of 8 GPUs per pod")

    valid_quantizations = ["awq", "gptq", "fp8", None]
    if quantization not in valid_quantizations:
        raise ValueError(f"quantization must be one of {valid_quantizations}, got '{quantization}'")

    if min_replicas < 0:
        raise ValueError(f"min_replicas must be >= 0, got {min_replicas}")

    if max_replicas < min_replicas:
        raise ValueError(f"max_replicas ({max_replicas}) must be >= min_replicas ({min_replicas})")
