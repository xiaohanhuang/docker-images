"""
Serving component — deploy a model as a vLLM inference server.

Image: genai-gpu
"""

from flytekit import Resources, task


@task(
    retries=1,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1"),
    cache=False,
)
def deploy_vllm(
    model_id: str,
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
) -> str:
    """Deploy a model as a vLLM OpenAI-compatible inference server.

    .. note::

       This task starts vLLM as a subprocess inside the task pod.  The returned
       ``localhost`` endpoint is only reachable from within the same pod.  For
       cluster-wide access, wrap this in a Kubernetes Deployment/Service or use
       the platform's serving infrastructure.

    Args:
        model_id: HuggingFace model ID or local path (e.g. ``mistralai/Mistral-7B-v0.1``).
        port: Port to expose the server on.
        max_model_len: Maximum sequence length (tokens).
        gpu_memory_utilization: Fraction of GPU memory reserved for the KV cache.

    Returns:
        URL of the deployed inference endpoint (e.g. ``http://localhost:8000/v1``).
    """
    import subprocess
    import time

    import requests as http_requests

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    proc = subprocess.Popen(cmd)

    # Wait for the server to become healthy (up to 120 s)
    health_url = f"http://localhost:{port}/health"
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server process exited unexpectedly (rc={proc.returncode})")
        try:
            resp = http_requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        proc.terminate()
        raise RuntimeError(f"vLLM server did not become healthy within 120 s on port {port}")

    return f"http://localhost:{port}/v1"
