"""
ML Platform Remote Execution Agent Service.

FastAPI service that receives serialized Python functions from @remote decorator,
creates Kubernetes Jobs to execute them on GPU nodes, streams logs, and returns results.

Phase 2: Pod Pool support — reuses warm containers to reduce cold-start latency.
"""

import asyncio
import base64
import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import cloudpickle
import requests
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

# Pod Pool Configuration
POD_POOL_TTL_SECONDS = int(os.getenv("POD_POOL_TTL_SECONDS", "600"))  # 10 minutes default
POD_POOL_ENABLED = os.getenv("POD_POOL_ENABLED", "true").lower() == "true"


@dataclass
class PodState:
    """State of a pod in the pool."""

    pod_name: str
    status: str  # "idle" or "busy"
    config_hash: str
    config: Dict
    last_used: str
    created: str
    pod_ip: Optional[str] = None
    namespace: str = "default"


# Global pod pool state (in-memory)
# In production, this could be Redis for HA across multiple agent replicas
pod_pool: Dict[str, PodState] = {}
pool_lock = asyncio.Lock()


def compute_config_hash(config: dict) -> str:
    """
    Compute a hash of the execution config for pod matching.

    Pods with the same config hash can be reused.
    Hash is based on: image, gpu count, gpu_type, and sorted packages.
    """
    image = config.get("image", "")
    gpu = config.get("gpu", 0)
    gpu_type = config.get("gpu_type", "any")
    packages = sorted(config.get("packages", []))

    # Create deterministic hash string
    hash_input = f"{image}:{gpu}:{gpu_type}:{','.join(packages)}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

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


@app.get("/pool")
async def get_pool_status():
    """Get current pod pool status."""
    async with pool_lock:
        pool_status = []
        for pod_name, state in pod_pool.items():
            pool_status.append(
                {
                    "pod_name": pod_name,
                    "status": state.status,
                    "config_hash": state.config_hash,
                    "config": state.config,
                    "last_used": state.last_used,
                    "created": state.created,
                    "pod_ip": state.pod_ip,
                }
            )
        return {
            "pool_enabled": POD_POOL_ENABLED,
            "ttl_seconds": POD_POOL_TTL_SECONDS,
            "total_pods": len(pod_pool),
            "idle_pods": sum(1 for s in pod_pool.values() if s.status == "idle"),
            "busy_pods": sum(1 for s in pod_pool.values() if s.status == "busy"),
            "pods": pool_status,
        }


@app.get("/pool/stats")
async def get_pool_stats():
    """Get pool statistics."""
    async with pool_lock:
        return {
            "pool_enabled": POD_POOL_ENABLED,
            "ttl_seconds": POD_POOL_TTL_SECONDS,
            "total_pods": len(pod_pool),
            "idle_pods": sum(1 for s in pod_pool.values() if s.status == "idle"),
            "busy_pods": sum(1 for s in pod_pool.values() if s.status == "busy"),
            "config_hashes": list(set(s.config_hash for s in pod_pool.values())),
        }


@app.post("/execute")
async def execute_remote(request: Request):
    """
    Execute a serialized function on a GPU pod.

    Phase 2: Pod Pool support — reuses warm containers when config matches.

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

    # 2. Compute config hash
    config_hash = compute_config_hash(config)
    namespace = config.get("namespace", "default")

    print("[agent] New execution request")
    print(f"[agent] Function: {fn.__name__}")
    print(f"[agent] Config hash: {config_hash}")
    print(f"[agent] Namespace: {namespace}")
    print(f"[agent] Pool enabled: {POD_POOL_ENABLED}")

    # 3. Check per-call ttl: ttl=0 forces legacy one-shot mode
    per_call_ttl = config.get("ttl")
    use_pool = POD_POOL_ENABLED and per_call_ttl != 0

    # 4. Try to find an idle pod with matching config (if pool enabled)
    pod_name = None
    pod_ip = None

    if use_pool:
        async with pool_lock:
            v1 = client.CoreV1Api()
            for name, state in list(pod_pool.items()):
                if state.status != "idle" or state.config_hash != config_hash:
                    continue

                # Validate that the pod still exists and is Running with a valid IP
                try:
                    pod = v1.read_namespaced_pod(name=name, namespace=state.namespace)
                except ApiException as exc:
                    if exc.status == 404:
                        print(f"[agent] Pooled pod {name} no longer exists; removing")
                        pod_pool.pop(name, None)
                        continue
                    print(f"[agent] Error reading pod {name}: {exc}")
                    continue

                pod_phase = getattr(pod.status, "phase", None)
                current_ip = getattr(pod.status, "pod_ip", None)
                if pod_phase != "Running" or not current_ip:
                    print(
                        f"[agent] Pooled pod {name} not ready "
                        f"(phase={pod_phase}, ip={current_ip}); skipping"
                    )
                    continue

                # Found a matching, healthy idle pod
                print(f"[agent] ♻️  Reusing warm pod: {name} with IP: {current_ip}")
                state.pod_ip = current_ip
                state.status = "busy"
                state.last_used = datetime.utcnow().isoformat()
                pod_name = name
                pod_ip = current_ip
                break

    # 5. If no matching pod found, create a new one
    new_pod_needed = pod_name is None
    execution_id = None  # only set when creating a new job (legacy mode)
    if new_pod_needed:
        execution_id = f"remote-exec-{uuid.uuid4().hex[:8]}"
        print(f"[agent] Creating new pod: {execution_id}")

        if not use_pool:
            try:
                # Legacy mode: create one-shot Job
                _create_execution_job(
                    execution_id=execution_id,
                    payload_bytes=payload_bytes,
                    config=config,
                    namespace=namespace,
                )
                print(f"[agent] Created Job: {execution_id}")
            except Exception as e:
                print(f"[agent] Failed to create job: {e}")
                return Response(
                    content=f"Failed to create execution job: {e}",
                    status_code=500,
                )

    # 6. Execute function
    if use_pool:
        # Pool mode: use StreamingResponse so the ALB/client connection stays alive
        # while we wait for the pod to start (can take 60-120s on cold start).
        _pod_name = pod_name
        _pod_ip = pod_ip
        _new_pod_needed = new_pod_needed
        _execution_id = execution_id if new_pod_needed else None

        async def pool_streamer():
            nonlocal _pod_name, _pod_ip

            try:
                if _new_pod_needed:
                    # Create the executor pod
                    try:
                        _pod_name = await _create_executor_pod(
                            execution_id=_execution_id,
                            config=config,
                            namespace=namespace,
                            config_hash=config_hash,
                        )
                    except Exception as e:
                        yield f"[agent] Failed to create pool pod: {e}\n".encode()
                        return

                    yield f"[agent] Pool pod {_pod_name} created, waiting for ready...\n".encode()

                    # Wait for pod ready, streaming heartbeats every 5s
                    deadline = time.time() + 120
                    while time.time() < deadline:
                        try:
                            pod = core_v1.read_namespaced_pod(
                                name=_pod_name, namespace=namespace
                            )
                            phase = getattr(pod.status, "phase", None)
                            ip = getattr(pod.status, "pod_ip", None)
                            if phase == "Running" and ip:
                                _pod_ip = ip
                                break
                        except Exception:
                            pass
                        yield f"[agent] Waiting for pod {_pod_name} (phase={phase})...\n".encode()
                        await asyncio.sleep(5)
                    else:
                        yield "[agent] Timeout waiting for pool pod to be ready\n".encode()
                        return

                    yield f"[agent] Pool pod ready with IP: {_pod_ip}\n".encode()

                    # Register in pool
                    async with pool_lock:
                        pod_pool[_pod_name] = PodState(
                            pod_name=_pod_name,
                            status="busy",
                            config_hash=config_hash,
                            config=config,
                            last_used=datetime.utcnow().isoformat(),
                            created=datetime.utcnow().isoformat(),
                            pod_ip=_pod_ip,
                            namespace=namespace,
                        )
                else:
                    yield f"[agent] ♻️  Reusing warm pod: {_pod_name}\n".encode()

                # Execute on pod
                yield "[agent] Sending payload to executor...\n".encode()
                try:
                    result = await _execute_on_pod(
                        pod_ip=_pod_ip,
                        payload_bytes=payload_bytes,
                        timeout=config.get("timeout", 3600),
                    )
                    # Mark pod idle
                    async with pool_lock:
                        if _pod_name in pod_pool:
                            pod_pool[_pod_name].status = "idle"
                            pod_pool[_pod_name].last_used = datetime.utcnow().isoformat()
                    yield result.encode()
                except Exception as e:
                    async with pool_lock:
                        if _pod_name in pod_pool:
                            pod_pool[_pod_name].status = "idle"
                    yield f"[agent] Execution failed: {e}\n".encode()

            except Exception as e:
                yield f"[agent] Pool streamer error: {e}\n".encode()

        return StreamingResponse(pool_streamer(), media_type="text/plain")

    else:
        # Legacy mode: stream logs from Job
        async def log_streamer():
            """Stream logs from execution pod and return result."""

            try:
                # Wait for pod to be created
                job_pod_name = await _wait_for_pod(execution_id, namespace, timeout=120)
                if not job_pod_name:
                    yield "[agent] Timeout waiting for pod to be created\n".encode()
                    return

                yield f"[agent] Pod {job_pod_name} created, waiting for logs...\n".encode()

                # Stream logs from pod
                async for log_line in _stream_pod_logs(job_pod_name, namespace):
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


async def _create_executor_pod(
    execution_id: str,
    config: dict,
    namespace: str,
    config_hash: str,
) -> str:
    """
    Create a long-lived executor pod for the pool.

    Returns: pod name
    """
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

    # Determine image — custom images must include the executor HTTP server
    if config.get("gpu", 0) > 0:
        default_image = "ml-platform/executor-pool:latest"
    else:
        default_image = "ml-platform/executor-pool:latest"  # Same for now
    user_image = config.get("image")
    if user_image:
        print(
            f"[agent] ⚠️  Custom image '{user_image}' in pool mode — "
            "it must run the executor HTTP server on :8080 (/health, /execute)"
        )
    image = user_image or default_image
    ecr_registry = os.getenv("ECR_REGISTRY", "805673386114.dkr.ecr.us-west-2.amazonaws.com")
    if image.startswith("ml-platform/"):
        image = f"{ecr_registry}/{image}"

    # Resource requests and limits
    gpu_count = config.get("gpu", 0)
    resources_dict = {
        "cpu": config.get("cpu", "4"),
        "memory": config.get("memory", "16Gi"),
    }
    if gpu_count > 0:
        resources_dict["nvidia.com/gpu"] = str(gpu_count)

    # Create Pod manifest
    pod = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(
            name=execution_id,
            namespace=namespace,
            labels={
                "app": "remote-execution-pool",
                "config-hash": config_hash,
                "managed-by": "remote-agent",
            },
        ),
        spec=client.V1PodSpec(
            restart_policy="OnFailure",
            node_selector=node_selector if node_selector else None,
            tolerations=tolerations if tolerations else None,
            containers=[
                client.V1Container(
                    name="executor",
                    image=image,
                    ports=[client.V1ContainerPort(container_port=8080, name="http")],
                    env=(
                        [client.V1EnvVar(name="PORT", value="8080")]
                        + [
                            client.V1EnvVar(name=str(k), value=str(v))
                            for k, v in (config.get("env") or {}).items()
                        ]
                    ),
                    resources=client.V1ResourceRequirements(
                        requests=resources_dict,
                        limits=resources_dict,
                    ),
                    readiness_probe=client.V1Probe(
                        http_get=client.V1HTTPGetAction(
                            path="/health",
                            port=8080,
                        ),
                        initial_delay_seconds=5,
                        period_seconds=5,
                    ),
                )
            ],
        ),
    )

    # Create pod in cluster
    core_v1.create_namespaced_pod(namespace=namespace, body=pod)
    return execution_id


async def _wait_for_pod_ready(
    pod_name: str, namespace: str, timeout: int = 120
) -> Optional[str]:
    """
    Wait for pod to be ready and return its IP.

    Returns: pod IP or None if timeout
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            pod = core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)

            # Check if ready
            if pod.status.phase == "Running" and pod.status.pod_ip:
                # Check readiness conditions
                if pod.status.conditions:
                    ready = False
                    for condition in pod.status.conditions:
                        if condition.type == "Ready" and condition.status == "True":
                            ready = True
                            break
                    if ready:
                        print(f"[agent] Pod {pod_name} is ready with IP {pod.status.pod_ip}")
                        return pod.status.pod_ip

        except Exception as e:
            print(f"[agent] Error checking pod readiness: {e}")

        await asyncio.sleep(2)

    print(f"[agent] Timeout waiting for pod {pod_name} to be ready")
    return None


async def _execute_on_pod(
    pod_ip: str,
    payload_bytes: bytes,
    timeout: int = 3600,
) -> str:
    """
    Execute payload on a running executor pod via HTTP.

    Returns: result string (with markers for client parsing)
    """
    url = f"http://{pod_ip}:8080/execute"

    try:
        # Run blocking HTTP call in a thread to avoid blocking the event loop
        response = await asyncio.to_thread(
            requests.post,
            url,
            data=payload_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=(10, timeout),
        )
        response.raise_for_status()

        result = response.json()

        if result.get("success"):
            # Format result with markers for client parsing
            result_b64 = result.get("result", "")
            output = "[agent] Execution completed on warm pod\n"
            output += f"__RESULT_START__\n{result_b64}\n__RESULT_END__\n"
            return output
        else:
            # Execution failed
            result_b64 = result.get("result", "")
            output = "[agent] Execution failed on warm pod\n"
            output += f"__RESULT_START__\n{result_b64}\n__RESULT_END__\n"
            return output

    except Exception as e:
        raise RuntimeError(f"Failed to execute on pod {pod_ip}: {e}")


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
        default_image = "ml-platform/base-cpu:latest"
    image = config.get("image") or default_image
    ecr_registry = os.getenv(
        "ECR_REGISTRY", "805673386114.dkr.ecr.us-west-2.amazonaws.com"
    )
    # Prefix all ml-platform images with ECR registry
    if image.startswith("ml-platform/"):
        image = f"{ecr_registry}/{image}"

    # Build environment variables
    # ⚠️  PAYLOAD SIZE LIMITATION (Phase 1):
    # The payload is passed as a base64-encoded environment variable. Kubernetes has a
    # practical limit of ~1MB for the total size of all env vars in a pod spec. Large
    # closures or arguments will cause "request entity too large" errors.
    # Phase 2 will store payloads in S3 and pass a reference URI instead.
    payload_b64 = base64.b64encode(payload_bytes).decode()
    if len(payload_b64) > 900_000:  # ~900KB safety margin
        raise ValueError(
            f"Serialized payload too large ({len(payload_b64)} bytes). "
            "Kubernetes env vars have a ~1MB limit. "
            "Pass large data via S3 URIs instead of function arguments."
        )
    env_vars = [
        client.V1EnvVar(
            name="PAYLOAD_B64",
            value=payload_b64,
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


async def _ttl_eviction_task():
    """
    Background task to evict idle pods that exceed TTL.

    Runs every 60 seconds, checking all idle pods and deleting those
    where (now - last_used) > POD_POOL_TTL_SECONDS.
    """
    print(f"[agent] Starting TTL eviction task (TTL={POD_POOL_TTL_SECONDS}s)")

    while True:
        try:
            await asyncio.sleep(60)  # Check every minute

            if not POD_POOL_ENABLED:
                continue

            now = datetime.utcnow()
            pods_to_evict = []

            async with pool_lock:
                for pod_name, state in pod_pool.items():
                    if state.status == "idle":
                        last_used = datetime.fromisoformat(state.last_used)
                        idle_seconds = (now - last_used).total_seconds()

                        if idle_seconds > POD_POOL_TTL_SECONDS:
                            print(
                                f"[agent] Pod {pod_name} idle for {idle_seconds:.0f}s "
                                f"(TTL={POD_POOL_TTL_SECONDS}s), evicting"
                            )
                            pods_to_evict.append(pod_name)

            # Delete pods outside the lock
            for pod_name in pods_to_evict:
                try:
                    namespace = pod_pool[pod_name].namespace
                    core_v1.delete_namespaced_pod(
                        name=pod_name,
                        namespace=namespace,
                        grace_period_seconds=30,
                    )
                    print(f"[agent] Evicted pod {pod_name}")

                    # Remove from pool
                    async with pool_lock:
                        if pod_name in pod_pool:
                            del pod_pool[pod_name]

                except Exception as e:
                    print(f"[agent] Failed to evict pod {pod_name}: {e}")

        except Exception as e:
            print(f"[agent] Error in TTL eviction task: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    if POD_POOL_ENABLED:
        print(f"[agent] Pod Pool enabled with TTL={POD_POOL_TTL_SECONDS}s")
        asyncio.create_task(_ttl_eviction_task())
    else:
        print("[agent] Pod Pool disabled, using legacy one-shot Jobs")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"[agent] Starting ML Platform Remote Execution Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
