"""
Remote GPU Execution Decorator for ML Platform SDK.

Enables zero-config remote GPU execution of arbitrary Python functions on the cluster.
Functions decorated with @remote are serialized via cloudpickle and executed on GPU pods
without requiring Docker builds, ECR pushes, or Flyte registration.

Usage:
    from ml_platform_sdk.remote import remote

    @remote(gpu=1, memory="32Gi")
    def train_model(epochs: int):
        import torch
        model = torch.nn.Linear(100, 10).cuda()
        # training loop
        return model.state_dict()

    result = train_model(epochs=10)  # Runs on GPU in cluster
"""

import base64
import functools
import os
from typing import Any, Callable, Dict, Optional

try:
    import cloudpickle
except ImportError:
    raise ImportError(
        "cloudpickle is required for remote execution. " "Install it with: pip install cloudpickle"
    )

try:
    import requests
except ImportError:
    raise ImportError(
        "requests is required for remote execution. " "Install it with: pip install requests"
    )


def remote(
    gpu: int = 0,
    gpu_type: str = "any",
    memory: str = "16Gi",
    cpu: str = "4",
    image: Optional[str] = None,
    packages: Optional[list] = None,
    timeout: int = 3600,
    retries: int = 0,
    env: Optional[Dict[str, str]] = None,
    spot: bool = False,
    ttl: Optional[int] = None,
    execution_url: Optional[str] = None,
) -> Callable:
    """
    Decorator for zero-config remote GPU execution.

    Execution always runs in the 'default' namespace where the execution-service's RBAC
    permissions are configured. Multi-namespace support is planned for Phase 2.

    Args:
        gpu: Number of GPUs to request (0 for CPU-only)
        gpu_type: GPU type to target ("any", "a10g", "a100")
        memory: Memory request (e.g., "16Gi", "32Gi")
        cpu: CPU request (e.g., "4", "8")
        image: Docker image to use (defaults to base-gpu:latest)
        packages: List of pip packages to install before execution
            (e.g., ["transformers", "accelerate"])
        timeout: Max execution time in seconds
        retries: Number of retries on failure
        env: Environment variables to inject
        spot: Use spot instances (future feature)
        ttl: TTL for warm container in seconds
            (default: execution-service's POD_POOL_TTL_SECONDS)
            Set to 0 to use one-shot mode (no warm container reuse)
        execution_url: Execution service URL. Defaults to:
            1. execution_url parameter (highest priority)
            2. ML_PLAT_EXECUTION_URL environment variable
            3. http://execution-service.default.svc.cluster.local:8080 (when inside cluster)
            4. http://localhost:8765 (fallback for local development via port-forward)

    Returns:
        Decorated function that executes remotely on cluster GPU

    Example:
        >>> from ml_platform_sdk.remote import remote
        >>>
        >>> @remote(gpu=1, memory="32Gi")
        >>> def train_model(epochs: int):
        >>>     import torch
        >>>     model = torch.nn.Linear(100, 10).cuda()
        >>>     # training loop
        >>>     return model.state_dict()
        >>>
        >>> result = train_model(epochs=10)  # Runs on GPU in cluster
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return _execute_remote(
                fn=fn,
                args=args,
                kwargs=kwargs,
                gpu=gpu,
                gpu_type=gpu_type,
                memory=memory,
                cpu=cpu,
                image=image,
                packages=packages or [],
                timeout=timeout,
                retries=retries,
                env=env or {},
                spot=spot,
                ttl=ttl,
                execution_url=execution_url,
            )

        return wrapper

    return decorator


def _execute_remote(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    **config,
) -> Any:
    """Execute function remotely on cluster."""

    print(f"[remote] Executing {fn.__name__} on cluster...")
    print(f"[remote] GPU: {config['gpu']}, Memory: {config['memory']}, CPU: {config['cpu']}")

    # 1. Serialize function + arguments + closure
    try:
        payload_dict = {
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "config": config,
        }
        payload_bytes = cloudpickle.dumps(payload_dict)
        print(f"[remote] Serialized payload: {len(payload_bytes)} bytes")
    except Exception as e:
        raise RuntimeError(f"Failed to serialize function: {e}") from e

    # 2. Determine execution service URL
    # Priority order:
    # 1. Explicit execution_url parameter
    # 2. ML_PLAT_EXECUTION_URL environment variable
    # 3. In-cluster DNS (http://execution-service.default.svc.cluster.local:8080)
    # 4. Localhost port-forward for local development
    execution_url = config.get("execution_url") or os.getenv("ML_PLAT_EXECUTION_URL")

    if not execution_url:
        # Try in-cluster DNS first (works when running inside Kubernetes)
        execution_url = "http://execution-service.default.svc.cluster.local:8080"
        # Note: If this fails with connection error, the user should either:
        # 1. Set ML_PLAT_EXECUTION_URL=http://localhost:8765 for local port-forward
        # 2. Run from inside the cluster (e.g., from a JupyterHub pod)

    print(f"[remote] Connecting to execution service at {execution_url}")

    # 3. Send to execution service with separate connect and read timeouts
    try:
        # Use separate timeouts: short connect timeout, long read timeout for streaming
        connect_timeout = 10  # 10 seconds to establish connection
        read_timeout = config.get("timeout", 3600)  # Full execution timeout for reading

        headers = {"Content-Type": "application/json"}
        api_token = os.getenv("EXECUTION_SERVICE_API_TOKEN")
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        request_json = {
            "fn_name": fn.__name__,
            "config": config,
            "payload": base64.b64encode(payload_bytes).decode("utf-8"),
        }

        response = requests.post(
            f"{execution_url}/execute",
            json=request_json,
            headers=headers,
            stream=True,
            timeout=(connect_timeout, read_timeout),
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        print(f"\n[remote] ❌ Failed to connect to execution service at {execution_url}")
        print("[remote] Make sure the execution service is running and accessible.")
        print(
            "[remote] To deploy: cd projects/components/services/"
            "execution-service && make deploy"
        )
        print("[remote] To port-forward: kubectl port-forward svc/execution-service 8765:8080")
        raise RuntimeError(f"Failed to connect to execution service: {e}") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to send request to execution service: {e}") from e

    # 4. Stream logs and collect result
    result_data = b""
    in_result_section = False

    try:
        for chunk in response.iter_content(chunk_size=None):
            if not chunk:
                continue

            # Decode chunk
            try:
                text = chunk.decode("utf-8")
                # Strip trailing spaces added server-side to flush uvicorn's buffer.
                # Strip trailing spaces added server-side to flush uvicorn's buffer.
                # Remove large blocks of spaces from the middle of the chunk if it was segmented.
                text = text.rstrip(" ")
                text = text.replace(" " * 100, "")
            except UnicodeDecodeError:
                # Binary result data
                result_data += chunk
                continue

            # Check for result markers
            if "__RESULT_START__" in text:
                in_result_section = True
                # Print everything before the marker
                split_at_start = text.split("__RESULT_START__", 1)
                before_marker = split_at_start[0]
                if before_marker.strip():
                    print(before_marker, end="", flush=True)
                # Capture data after the marker in the same chunk
                after_marker = split_at_start[1] if len(split_at_start) > 1 else ""
                if "__RESULT_END__" in after_marker:
                    # Both START and END are in this chunk (small result)
                    result_part = after_marker.split("__RESULT_END__", 1)[0]
                    result_data += result_part.encode("utf-8")
                    break
                elif after_marker:
                    result_data += after_marker.encode("utf-8")
                continue

            if "__RESULT_END__" in text:
                # Extract result between markers
                parts = text.split("__RESULT_END__", 1)
                if in_result_section:
                    result_part = parts[0]
                    result_data += result_part.encode("utf-8")
                # Print anything after the end marker
                if len(parts) > 1 and parts[1].strip():
                    print(parts[1], end="", flush=True)
                break

            if in_result_section:
                result_data += chunk
            else:
                # Regular log output
                print(text, end="", flush=True)

    except Exception as e:
        raise RuntimeError(f"Failed to process response: {e}") from e

    # 5. Deserialize result
    if not result_data:
        raise RuntimeError("No result received from remote execution")

    try:
        # Result is base64 encoded in the output
        result_b64 = result_data.decode("utf-8").strip()
        result_bytes = base64.b64decode(result_b64)
        result_dict = cloudpickle.loads(result_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize result: {e}") from e

    # 6. Handle errors or return result
    if result_dict.get("error"):
        error_msg = result_dict["error"]
        traceback_str = result_dict.get("traceback", "")
        print("\n[remote] ❌ Remote execution failed:")
        print(traceback_str)
        raise RuntimeError(f"Remote execution failed: {error_msg}")

    print("\n[remote] ✅ Execution completed successfully")
    return result_dict["return_value"]
