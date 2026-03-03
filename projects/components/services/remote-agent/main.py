"""
ML Platform Remote Execution Agent Service.

FastAPI service that receives serialized Python functions from @remote decorator,
creates Kubernetes Jobs to execute them on GPU nodes, streams logs, and returns results.
"""

import asyncio
import base64
import os
import time
import uuid
from typing import Optional

import cloudpickle
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Load Kubernetes config
try:
    config.load_incluster_config()
    print("[agent] Running in-cluster, loaded service account credentials")
except config.ConfigException:
    try:
        config.load_kube_config()
        print("[agent] Running locally, loaded kubeconfig")
    except config.ConfigException as e:
        print(f"[agent] ❌ Could not load Kubernetes config: {e}")
        raise SystemExit("[agent] Failed to load Kubernetes config, exiting") from e

# Initialize Kubernetes clients
batch_v1 = client.BatchV1Api()
core_v1 = client.CoreV1Api()

# Create FastAPI app
app = FastAPI(
    title="ML Platform Remote Execution Agent",
    description="Executes serialized Python functions on Kubernetes GPU pods",
    version="0.1.0",
)

# Execution runner script that runs inside the GPU pod
EXECUTION_RUNNER_SCRIPT = """
import subprocess
import sys

# Ensure cloudpickle is available in the execution environment
try:
    import cloudpickle
except ImportError:
    print("[REMOTE] Installing cloudpickle...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "--root-user-action=ignore", "cloudpickle"]
    )
    import cloudpickle

import os
import base64
import traceback

def main():
    # 1. Load serialized payload from environment
    payload_b64 = os.environ["PAYLOAD_B64"]
    payload_bytes = base64.b64decode(payload_b64)
    payload = cloudpickle.loads(payload_bytes)

    fn = payload["fn"]
    args = payload["args"]
    kwargs = payload["kwargs"]
    config = payload["config"]

    # 2. Install user-specified packages
    packages = config.get("packages", [])
    if packages:
        print(f"[REMOTE] Installing packages: {', '.join(packages)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             "--root-user-action=ignore"] + list(packages)
        )
        print(f"[REMOTE] Packages installed successfully")

    print(f"[REMOTE] Executing {fn.__name__}(...)")
    print(f"[REMOTE] Args: {len(args)} positional, {len(kwargs)} keyword")
    print(f"[REMOTE] Config: GPU={config['gpu']}, Memory={config['memory']}, CPU={config['cpu']}")
    print("")

    try:
        # 2. Execute function
        result = fn(*args, **kwargs)

        # 3. Serialize result
        print("")
        print("[REMOTE] Execution completed successfully")
        result_dict = {"return_value": result, "error": None}

    except Exception as e:
        # Capture exception
        print("")
        print(f"[REMOTE] ❌ Execution failed: {e}")
        traceback.print_exc()
        result_dict = {"return_value": None, "error": str(e), "traceback": traceback.format_exc()}

    # 4. Print serialized result with markers (agent will capture from logs)
    print("__RESULT_START__")
    print(base64.b64encode(cloudpickle.dumps(result_dict)).decode())
    print("__RESULT_END__")

if __name__ == "__main__":
    main()
"""


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "ml-platform-remote-agent",
        "version": "0.1.0",
        "status": "healthy",
    }


@app.get("/health")
async def health():
    """Kubernetes health check endpoint."""
    return {"status": "ok"}


@app.post("/execute")
async def execute_remote(request: Request):
    """
    Execute a serialized function on a GPU pod.

    ⚠️  SECURITY WARNING (Phase 1 MVP):
    This endpoint accepts arbitrary cloudpickle payloads and executes them with cluster
    privileges. This is a remote code execution (RCE) vulnerability by design for Phase 1.

    Production deployment MUST implement:
    1. Authentication (API tokens, mTLS, or OAuth)
    2. Authorization (per-user RBAC for job creation)
    3. NetworkPolicy to restrict access to trusted clients only
    4. Consider: move deserialization into execution pods (not agent process)

    For now, this service should ONLY be exposed via port-forward for local dev,
    or restricted to specific source IPs/namespaces via NetworkPolicy.

    Request body: cloudpickle-serialized dict with:
    - fn: function to execute
    - args: positional arguments
    - kwargs: keyword arguments
    - config: execution config (gpu, memory, cpu, etc.)

    Returns: Streaming response with logs, followed by pickled result
    """

    # 1. Deserialize payload
    payload_bytes = await request.body()

    try:
        payload = cloudpickle.loads(payload_bytes)
    except Exception as e:
        return Response(
            content=f"Failed to deserialize payload: {e}",
            status_code=400,
        )

    fn = payload.get("fn")
    config = payload.get("config", {})

    if not fn:
        return Response(content="Missing 'fn' in payload", status_code=400)

    # 2. Create unique execution ID
    execution_id = f"remote-exec-{uuid.uuid4().hex[:8]}"
    namespace = config.get("namespace", "default")

    print(f"[agent] New execution request: {execution_id}")
    print(f"[agent] Function: {fn.__name__}")
    print(f"[agent] Namespace: {namespace}")

    # 3. Create Kubernetes Job
    try:
        _create_execution_job(
            execution_id=execution_id,
            payload_bytes=payload_bytes,
            config=config,
            namespace=namespace,
        )
        print(f"[agent] Created Job: {execution_id}")
    except Exception as e:
        print(f"[agent] Failed to create Job: {e}")
        return Response(
            content=f"Failed to create execution job: {e}",
            status_code=500,
        )

    # 4. Stream logs and return result
    async def log_streamer():
        """Stream logs from execution pod and return result."""

        try:
            # Wait for pod to be created
            pod_name = await _wait_for_pod(execution_id, namespace, timeout=120)
            if not pod_name:
                yield "[agent] Timeout waiting for pod to be created\n".encode()
                return

            yield f"[agent] Pod {pod_name} created, waiting for logs...\n".encode()

            # Stream logs from pod
            async for log_line in _stream_pod_logs(pod_name, namespace):
                yield log_line.encode()

            # Wait for job completion
            timeout_val = config.get("timeout", 3600)
            await _wait_for_job_completion(execution_id, namespace, timeout=timeout_val)

            yield "\n[agent] Execution completed\n".encode()

        except Exception as e:
            yield f"\n[agent] Error during execution: {e}\n".encode()

        finally:
            # Cleanup job (with grace period for log retrieval)
            try:
                await asyncio.sleep(5)  # Give time for final logs
                batch_v1.delete_namespaced_job(
                    name=execution_id,
                    namespace=namespace,
                    propagation_policy="Background",
                )
                print(f"[agent] Cleaned up Job: {execution_id}")
            except Exception as e:
                print(f"[agent] Failed to cleanup Job: {e}")

    return StreamingResponse(
        log_streamer(),
        media_type="text/plain",
    )


def _create_execution_job(
    execution_id: str,
    payload_bytes: bytes,
    config: dict,
    namespace: str,
) -> client.V1Job:
    """Create Kubernetes Job for remote execution."""

    # Determine node selector based on GPU type
    node_selector = {}
    if config.get("gpu", 0) > 0:
        node_selector["role"] = "gpu-worker"
        gpu_type = config.get("gpu_type", "any")
        if gpu_type == "a10g":
            node_selector["karpenter.k8s.aws/instance-gpu-name"] = "a10g"
        elif gpu_type == "a100":
            node_selector["karpenter.k8s.aws/instance-gpu-name"] = "a100"

    # GPU tolerations
    tolerations = []
    if config.get("gpu", 0) > 0:
        tolerations.append(
            client.V1Toleration(
                key="nvidia.com/gpu",
                operator="Exists",
                effect="NoSchedule",
            )
        )

    # Determine image - use lighter CPU image when no GPU requested
    if config.get("gpu", 0) > 0:
        default_image = "ml-platform/base-gpu:latest"
    else:
        default_image = "python:3.12-slim"
    image = config.get("image") or default_image
    ecr_registry = os.getenv(
        "ECR_REGISTRY", "805673386114.dkr.ecr.us-west-2.amazonaws.com"
    )
    # Only prefix with ECR for ml-platform images, not public images like python:3.12-slim
    if image.startswith("ml-platform/"):
        image = f"{ecr_registry}/{image}"

    # Build environment variables
    env_vars = [
        client.V1EnvVar(
            name="PAYLOAD_B64",
            value=base64.b64encode(payload_bytes).decode(),
        ),
    ]

    # Add user-specified env vars
    user_env = config.get("env", {})
    for key, value in user_env.items():
        env_vars.append(client.V1EnvVar(name=key, value=value))

    # Resource requests and limits
    gpu_count = config.get("gpu", 0)
    resources_dict = {
        "cpu": config.get("cpu", "4"),
        "memory": config.get("memory", "16Gi"),
    }
    if gpu_count > 0:
        resources_dict["nvidia.com/gpu"] = str(gpu_count)

    # Create Job manifest
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=execution_id,
            namespace=namespace,
            labels={
                "app": "remote-execution",
                "execution-id": execution_id,
            },
        ),
        spec=client.V1JobSpec(
            ttl_seconds_after_finished=300,  # Auto-cleanup after 5 minutes
            backoff_limit=config.get("retries", 0),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        "app": "remote-execution",
                        "execution-id": execution_id,
                    },
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    node_selector=node_selector if node_selector else None,
                    tolerations=tolerations if tolerations else None,
                    containers=[
                        client.V1Container(
                            name="executor",
                            image=image,
                            command=["python", "-c", EXECUTION_RUNNER_SCRIPT],
                            env=env_vars,
                            resources=client.V1ResourceRequirements(
                                requests=resources_dict,
                                limits=resources_dict,
                            ),
                        )
                    ],
                ),
            ),
        ),
    )

    # Create job in cluster
    return batch_v1.create_namespaced_job(namespace=namespace, body=job)


async def _wait_for_pod(
    execution_id: str, namespace: str, timeout: int = 120
) -> Optional[str]:
    """Wait for pod to be created and return its name."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            pods = core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"execution-id={execution_id}",
            )
            if pods.items:
                pod_name = pods.items[0].metadata.name
                print(f"[agent] Found pod: {pod_name}")
                return pod_name
        except Exception as e:
            print(f"[agent] Error listing pods: {e}")

        await asyncio.sleep(2)

    return None


async def _stream_pod_logs(pod_name: str, namespace: str):
    """Stream logs from a pod with retries and continuous following."""
    log_deadline = time.time() + 300  # Wait up to 5 min for logs to start

    # Wait for pod to be running or finished
    while time.time() < log_deadline:
        try:
            pod = core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            if pod.status.phase in ["Running", "Succeeded", "Failed"]:
                break
        except Exception as e:
            print(f"[agent] Error reading pod status: {e}")

        await asyncio.sleep(2)

    # Try to stream logs with retries
    while time.time() < log_deadline:
        try:
            # First check if pod is in a terminal state
            pod = core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)

            # If pod finished, just get all logs once
            if pod.status.phase in ["Succeeded", "Failed"]:
                logs = core_v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    follow=False,
                )
                for line in logs.split("\n"):
                    if line:
                        yield line + "\n"
                        await asyncio.sleep(0)
                return

            # Pod is running, stream with follow=True using watch
            # Use watch to stream logs in real-time
            from kubernetes import watch

            w = watch.Watch()
            try:
                for log_line in w.stream(
                    core_v1.read_namespaced_pod_log,
                    name=pod_name,
                    namespace=namespace,
                    follow=True,
                    _preload_content=False,
                    timestamps=False,
                ):
                    if log_line:
                        yield (
                            log_line + "\n" if not log_line.endswith("\n") else log_line
                        )
                        await asyncio.sleep(0)
                w.stop()
                return
            except Exception as stream_err:
                w.stop()
                # If streaming fails, fall back to polling
                print(f"[agent] Streaming failed, will retry: {stream_err}")
                raise stream_err

        except ApiException as e:
            if e.status == 400:
                # Pod not ready yet, wait and retry
                await asyncio.sleep(2)
                yield "[agent] Waiting for pod to start...\n"
                continue
            else:
                yield f"[agent] Error streaming logs: {e}\n"
                return
        except Exception:
            # Retry on other errors (transient network issues, etc.)
            await asyncio.sleep(2)
            yield "[agent] Log stream interrupted, retrying...\n"
            continue

    # Timeout waiting for logs
    yield "[agent] Timed out waiting for pod logs.\n"


async def _wait_for_job_completion(execution_id: str, namespace: str, timeout: int):
    """Wait for job to complete (succeed or fail)."""
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            job = batch_v1.read_namespaced_job_status(
                name=execution_id, namespace=namespace
            )

            # Check if job completed
            if job.status.succeeded:
                print(f"[agent] Job {execution_id} succeeded")
                return
            if job.status.failed:
                print(f"[agent] Job {execution_id} failed")
                return

        except Exception as e:
            print(f"[agent] Error reading job status: {e}")

        await asyncio.sleep(5)

    print(f"[agent] Job {execution_id} timed out after {timeout}s")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"[agent] Starting ML Platform Remote Execution Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
