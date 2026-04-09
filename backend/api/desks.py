"""Desk management API endpoints.

Desks are interactive, hermetically sealed GPU workspaces
that scientists use for development, debugging, and experimentation.

Uses the kubernetes Python client for reliable local/in-cluster connectivity.
"""

import asyncio
import datetime
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.aws_config import ECR_REGISTRY
from backend.k8s import get_core_v1
from backend.k8s import is_available as k8s_available
from backend.pricing import GPU_TYPE_PRICING, estimate_desk_burn_rate

try:
    from kubernetes import client
except ImportError:
    client = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
router = APIRouter()

# Namespace where desks are launched
DESK_NAMESPACE = os.getenv("DESK_NAMESPACE", "default")

# Number of time-sliced replicas per physical GPU.
# Must match the NVIDIA device-plugin ConfigMap (replicas=4).
GPU_TIME_SLICE_REPLICAS = 4


class DeskSpec(BaseModel):
    """Specification for launching a new desk."""

    name: str
    image: str = "ml-platform/notebook-marimo-vscode:1.0.0"
    gpu_type: str = "CPU"
    gpu_count: int = 0
    cpu_count: int = 4
    memory: str = "12Gi"
    storage: str = "100Gi"


class DeskInfo(BaseModel):
    """Desk status and metadata."""

    id: str
    name: str
    status: str
    gpu: str
    cpu_count: int = 1
    memory: str = "16Gi"
    uptime: str
    burn_rate: str
    image: str
    user: str = "unknown"
    created_at: str | None = None


def _format_uptime(created_at: datetime.datetime | None) -> str:
    """Format uptime from creation timestamp."""
    if not created_at:
        return "—"
    now = datetime.datetime.now(datetime.timezone.utc)
    delta = now - created_at
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes:02d}m"


@router.get("/pricing")
async def get_pricing() -> dict[str, Any]:
    """Return AWS on-demand pricing for desk instance types."""
    options = []
    for gpu_type, info in GPU_TYPE_PRICING.items():
        options.append(
            {
                "gpu_type": gpu_type,
                "family": info["family"],
                "gpus_per_instance": info["gpus_per_instance"],
                "rate_per_gpu": info["rate_per_gpu"],
                "rate_per_instance": info["rate_instance"],
            }
        )
    return {"pricing": options}


@router.get("")
async def list_desks(user: str | None = None) -> dict[str, Any]:
    """
    List all active desks (interactive pods) with cost metadata.

    Desks are identified by the label `ml-platform/type=desk`.
    """
    if not k8s_available():
        raise HTTPException(status_code=503, detail="kubernetes package not installed")

    try:
        v1 = get_core_v1()

        label_selector = "ml-platform/type=desk"
        if user:
            label_selector += f",ml-platform/user={user}"

        pods = v1.list_namespaced_pod(
            namespace=DESK_NAMESPACE,
            label_selector=label_selector,
            _request_timeout=10,
        )

        desks = []
        for pod in pods.items:
            metadata = pod.metadata
            spec = pod.spec
            labels = metadata.labels or {}
            containers = spec.containers or []

            # Determine GPU type and count
            gpu_type = labels.get("ml-platform/gpu-type", "CPU")
            gpu_count = 0
            cpu_count = 1
            memory_str = "16Gi"
            for container in containers:
                limits = (container.resources.limits or {}) if container.resources else {}
                gpu_count += int(limits.get("nvidia.com/gpu", 0))
                cpu_limit = limits.get("cpu", "1")
                # cpu_limit can be "3" or "500m" etc.
                if isinstance(cpu_limit, str) and cpu_limit.endswith("m"):
                    cpu_count = max(cpu_count, int(cpu_limit[:-1]) // 1000)
                elif cpu_limit:
                    cpu_count = max(cpu_count, int(cpu_limit))

                mem_limit = limits.get("memory", "16Gi")
                memory_str = str(mem_limit)

            # Parse creation timestamp
            created_at = metadata.creation_timestamp

            # Convert time-sliced GPU replicas back to physical GPU counts
            physical_gpus = gpu_count / GPU_TIME_SLICE_REPLICAS if gpu_count > 0 else 0
            # format as an integer if it's a whole number, otherwise keep decimal
            if physical_gpus.is_integer():
                physical_str = f"{int(physical_gpus)}"
            else:
                physical_str = f"{physical_gpus}"

            desks.append(
                DeskInfo(
                    id=metadata.name,
                    name=labels.get("ml-platform/desk-name", metadata.name),
                    status=pod.status.phase or "Unknown",
                    gpu=f"{gpu_type} x{physical_str}" if gpu_count > 0 else "CPU",
                    cpu_count=cpu_count,
                    memory=memory_str,
                    uptime=_format_uptime(created_at),
                    burn_rate=estimate_desk_burn_rate(gpu_type, physical_gpus),
                    image=containers[0].image if containers else "unknown",
                    user=labels.get("ml-platform/user", "unknown"),
                    created_at=created_at.isoformat() if created_at else None,
                )
            )

        return {"desks": desks, "count": len(desks)}

    except Exception as e:
        logger.error(f"Failed to list desks: {e}")
        return {"desks": [], "count": 0}


def _build_pod_manifest(spec: DeskSpec, pod_name: str, user: str) -> "client.V1Pod":
    # Resolve full image path
    image = spec.image
    if "/" not in image or image.startswith("ml-platform/"):
        image = f"{ECR_REGISTRY.rstrip('/')}/{image}"

    # Build container resources
    resources: dict = {
        "requests": {"cpu": str(spec.cpu_count), "memory": spec.memory},
        "limits": {"cpu": str(spec.cpu_count), "memory": spec.memory},
    }
    if spec.gpu_count > 0:
        resources["requests"]["nvidia.com/gpu"] = str(spec.gpu_count)
        resources["limits"]["nvidia.com/gpu"] = str(spec.gpu_count)

    # Node selector — for GPU types we need the specific GPU node,
    # for CPU we let Karpenter auto-provision the best-fit instance
    node_selector: dict = {}
    if spec.gpu_type.upper() != "CPU":
        pricing_info = GPU_TYPE_PRICING.get(spec.gpu_type.upper(), {})
        if pricing_info.get("family"):
            node_selector["karpenter.k8s.aws/instance-family"] = pricing_info["family"]

    # Pod manifest
    pod_manifest = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(
            name=pod_name,
            namespace=DESK_NAMESPACE,
            labels={
                "ml-platform/type": "desk",
                "ml-platform/desk-name": spec.name,
                "ml-platform/user": user,
                "ml-platform/gpu-type": spec.gpu_type,
            },
        ),
        spec=client.V1PodSpec(
            restart_policy="Never",
            node_selector=node_selector if node_selector else None,
            containers=[
                client.V1Container(
                    name="main",
                    image=image,
                    # Start all IDE tools alongside the default entrypoint
                    command=["sh", "-c"],
                    args=[
                        "code-server --port 9000 --auth none --bind-addr 0.0.0.0:9000 "
                        "--disable-getting-started-override /home/jovyan "
                        "> /tmp/code-server.log 2>&1 & "
                        "mkdir -p /home/jovyan/.config/marimo && "
                        "echo '[display]' > /home/jovyan/.config/marimo/marimo.toml && "
                        "echo 'theme = \"dark\"' >> /home/jovyan/.config/marimo/marimo.toml && "
                        "echo '[tool.marimo.display]' > /home/jovyan/pyproject.toml && "
                        "echo 'theme = \"dark\"' >> /home/jovyan/pyproject.toml && "
                        "marimo edit --host 0.0.0.0 --port 2718 --headless --no-token "
                        f"--base-url /desk-marimo/{pod_name} "
                        "--allow-origins '*' --no-skew-protection "
                        "> /tmp/marimo.log 2>&1 & "
                        "exec start-notebook.py "
                        "--IdentityProvider.token='' "
                        "--ServerApp.disable_check_xsrf=True "
                        "--ServerApp.allow_origin='*' "
                        "--ServerApp.tornado_settings="
                        "\"{'headers':{'Content-Security-Policy':"
                        "'frame-ancestors *'}}\" "
                        f"--ServerApp.base_url="
                        f"/desk-jupyter/{pod_name}/"
                    ],
                    resources=client.V1ResourceRequirements(**resources),
                    ports=[
                        client.V1ContainerPort(container_port=22, name="ssh"),
                        client.V1ContainerPort(container_port=8888, name="jupyter"),
                        client.V1ContainerPort(container_port=9000, name="vscode"),
                        client.V1ContainerPort(container_port=2718, name="marimo"),
                    ],
                    volume_mounts=[
                        client.V1VolumeMount(
                            name="efs-storage",
                            mount_path="/shared",
                        ),
                    ],
                    env=[
                        client.V1EnvVar(name="ML_PLAT_USER", value=user),
                        client.V1EnvVar(name="ML_PLAT_DESK", value=pod_name),
                        client.V1EnvVar(name="JUPYTER_TOKEN", value=""),
                    ],
                ),
            ],
            volumes=[
                client.V1Volume(
                    name="efs-storage",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name="efs-claim",
                    ),
                ),
            ],
        ),
    )
    return pod_manifest


@router.post("")
async def launch_desk(spec: DeskSpec, request: Request) -> dict[str, str]:
    """
    Launch a new desk (interactive GPU workspace).

    Creates a Kubernetes pod with the specified resources, GPU requests,
    and EFS volume mounts. The pod is labeled for discovery by list_desks.
    """
    if not k8s_available():
        raise HTTPException(status_code=503, detail="kubernetes package not installed")

    import os

    user_state = getattr(request.state, "user", "")
    if isinstance(user_state, dict):
        user = user_state.get("sub", os.getenv("USER", "xiaohan"))
    else:
        user = user_state if user_state else os.getenv("USER", "xiaohan")

    pod_name = spec.name if spec.name.startswith("desk-") else f"desk-{spec.name}"

    pod_manifest = _build_pod_manifest(spec, pod_name, user)

    try:
        v1 = get_core_v1()
        v1.create_namespaced_pod(
            namespace=DESK_NAMESPACE,
            body=pod_manifest,
            _request_timeout=15,
        )
        logger.info(f"Desk created: {pod_name} ({spec.gpu_type} x{spec.gpu_count})")
        return {
            "status": "created",
            "desk_id": pod_name,
            "message": f"Desk '{spec.name}' launched with {spec.gpu_type} x{spec.gpu_count}",
        }
    except Exception as e:
        logger.error(f"Failed to create desk {pod_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create desk: {e}")


@router.delete("/{desk_id}")
async def stop_desk(desk_id: str) -> dict[str, str]:
    """Stop and remove a desk."""
    if not k8s_available():
        raise HTTPException(status_code=503, detail="kubernetes package not installed")

    try:
        v1 = get_core_v1()
        v1.delete_namespaced_pod(name=desk_id, namespace=DESK_NAMESPACE)
        return {"status": "stopped", "desk_id": desk_id, "message": "EFS volume preserved"}
    except Exception as e:
        if hasattr(e, "status") and e.status == 404:
            return {"status": "stopped", "desk_id": desk_id, "message": "EFS volume preserved"}
        logger.error(f"Failed to stop desk {desk_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop desk: {e}")


@router.websocket("/{desk_id}/logs")
async def desk_logs_ws(websocket: WebSocket, desk_id: str):
    """WebSocket endpoint to stream pod logs."""
    await websocket.accept()
    if not k8s_available():
        await websocket.close(code=1011, reason="kubernetes not available")
        return

    import os

    user = os.getenv("USER", "xiaohan")
    # For websockets, state is in scope if custom ASGI middleware is used
    if "state" in websocket.scope and hasattr(websocket.scope["state"], "user"):
        w_user = websocket.scope["state"].user
        if isinstance(w_user, dict):
            user = w_user.get("sub", user)
        elif isinstance(w_user, str) and w_user:
            user = w_user

    try:
        v1 = get_core_v1()
        pod = v1.read_namespaced_pod(name=desk_id, namespace=DESK_NAMESPACE)
        labels = pod.metadata.labels or {}
        if labels.get("ml-platform/user") != user and labels.get("ml-platform/user"):
            await websocket.send_text(f"Error: Not authorized to view logs for desk '{desk_id}'.")
            await websocket.close(code=1008, reason="Not authorized")
            return
    except Exception as e:
        await websocket.send_text(f"Error checking pod: {e}")
        await websocket.close(code=1011, reason="Pod check failed")
        return

    try:
        v1 = get_core_v1()
        # Use the K8s Python client with follow + _preload_content=False for streaming
        log_stream = v1.read_namespaced_pod_log(
            name=desk_id,
            namespace=DESK_NAMESPACE,
            follow=True,
            _preload_content=False,
        )
        for chunk in log_stream:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            await websocket.send_text(chunk)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Logs WS error for {desk_id}: {e}")


@router.post("/{desk_id}/start-vscode")
async def start_vscode(desk_id: str):
    """Apply VS Code theming and verify code-server is running.

    Code-server starts automatically with the pod entrypoint (port 9000).
    This endpoint applies platform-consistent settings and verifies readiness.
    The frontend accesses code-server through the K8s API proxy
    (kubectl-proxy sidecar on the dashboard pod).
    """
    if not k8s_available():
        raise HTTPException(status_code=503, detail="kubernetes package not installed")

    from kubernetes.stream import stream as k8s_stream

    v1 = get_core_v1()

    # First check the pod is actually Running
    try:
        pod = v1.read_namespaced_pod(name=desk_id, namespace=DESK_NAMESPACE)
        if pod.status.phase != "Running":
            raise HTTPException(
                status_code=409,
                detail=f"Desk pod is {pod.status.phase}, not Running yet",
            )
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=f"Pod not found: {e.reason}")

    def _exec_in_pod(cmd: str) -> str:
        """Execute a shell command inside the desk pod and return stdout."""
        resp = k8s_stream(
            v1.connect_get_namespaced_pod_exec,
            desk_id,
            DESK_NAMESPACE,
            command=["sh", "-c", cmd],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )
        return resp

    try:
        import json as _json

        # Platform-consistent VS Code settings
        settings = {
            "workbench.colorTheme": "Default Dark Modern",
            "workbench.startupEditor": "none",
            "workbench.tips.enabled": False,
            "workbench.welcomePage.walkthroughs.openOnInstall": False,
            "security.workspace.trust.enabled": False,
            "chat.commandCenter.enabled": False,
            "workbench.secondarySideBar.visible": False,
            "workbench.activityBar.location": "default",
            "window.titleBarStyle": "native",
            "window.menuBarVisibility": "compact",
            "window.commandCenter": False,
            "workbench.layoutControl.enabled": False,
            "editor.fontFamily": "JetBrains Mono, Fira Code, monospace",
            "editor.fontSize": 14,
            "editor.minimap.enabled": False,
            "editor.renderWhitespace": "none",
            "editor.smoothScrolling": True,
            "terminal.integrated.fontFamily": "JetBrains Mono, monospace",
            "terminal.integrated.fontSize": 14,
            "workbench.colorCustomizations": {
                "editor.background": "#0d1117",
                "sideBar.background": "#0a0e1a",
                "sideBarSectionHeader.background": "#0a0e1a",
                "activityBar.background": "#080c16",
                "titleBar.activeBackground": "#080c16",
                "titleBar.inactiveBackground": "#080c16",
                "tab.activeBackground": "#0d1117",
                "tab.inactiveBackground": "#080c16",
                "panel.background": "#0a0e1a",
                "terminal.background": "#0a0e1a",
                "statusBar.background": "#080c16",
                "editorGroupHeader.tabsBackground": "#080c16",
            },
        }
        settings_json = _json.dumps(settings)

        # Write settings into the pod
        settings_cmd = (
            "mkdir -p /home/jovyan/.local/share/code-server/User && "
            f"echo '{settings_json}' > /home/jovyan/.local/share/code-server/User/settings.json"
        )
        _exec_in_pod(settings_cmd)

        # Poll until code-server is listening on port 9000
        # (started by pod entrypoint, may take a few seconds to boot)
        for _ in range(15):
            health_output = _exec_in_pod(
                "curl -s -o /dev/null -w '%{http_code}'"
                " http://localhost:9000/ 2>/dev/null || echo 000"
            )
            if health_output.strip() in ("200", "302"):
                return {"status": "started", "desk_id": desk_id}
            await asyncio.sleep(1)

        raise HTTPException(
            status_code=503,
            detail="Code-server did not become ready within 15 seconds",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start code-server in {desk_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RunCodeRequest(BaseModel):
    code: str


# Python wrapper that captures stdout, stderr, and matplotlib plots
_EXEC_WRAPPER = r"""
import sys, json, io, traceback as _tb
_out = io.StringIO()
_err_msg = None
_images = []
_old_stdout = sys.stdout
try:
    sys.stdout = _out
    exec(compile(CODE_PLACEHOLDER, "<cell>", "exec"))
    sys.stdout = _old_stdout
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64 as _b64
        for _fn in plt.get_fignums():
            _buf = io.BytesIO()
            plt.figure(_fn).savefig(
                _buf, format="png",
                bbox_inches="tight", dpi=120,
                facecolor="#0f1423",
            )
            _buf.seek(0)
            _images.append(_b64.b64encode(_buf.read()).decode())
        plt.close("all")
    except ImportError:
        pass
except Exception:
    sys.stdout = _old_stdout
    _err_msg = _tb.format_exc()

# Write result as JSON to a marker file so we can parse it
_result = json.dumps({
    "stdout": _out.getvalue(),
    "error": _err_msg,
    "images": _images,
})
sys.stdout.write("__RICH_OUTPUT_JSON__" + _result)
"""


@router.post("/{desk_id}/run")
async def run_desk_code(desk_id: str, req: RunCodeRequest):
    """Execute code inside the desk pod with rich output capture.

    Returns stdout text, error traceback, and base64-encoded
    matplotlib plot images.
    """
    if not k8s_available():
        raise HTTPException(status_code=503, detail="kubernetes package not installed")

    from kubernetes.stream import stream as k8s_stream

    v1 = get_core_v1()

    # Build the wrapper with the user's code injected
    import base64

    code_b64 = base64.b64encode(req.code.encode()).decode()
    # Inject user code safely via base64 decode
    wrapper = _EXEC_WRAPPER.replace(
        "CODE_PLACEHOLDER",
        f'__import__("base64").b64decode("{code_b64}")' f".decode()",
    )

    try:
        # Write wrapper to file and execute
        import json

        escaped = wrapper.replace("'", "'\\''")
        cmd = f"echo '{escaped}' > /tmp/_nb_exec.py " f"&& python -u /tmp/_nb_exec.py"

        resp = k8s_stream(
            v1.connect_get_namespaced_pod_exec,
            desk_id,
            DESK_NAMESPACE,
            command=["sh", "-c", cmd],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        stdout_data = ""
        stderr_data = ""
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                stdout_data += resp.read_stdout()
            if resp.peek_stderr():
                stderr_data += resp.read_stderr()

        # Try to parse rich output
        marker = "__RICH_OUTPUT_JSON__"
        if marker in stdout_data:
            json_part = stdout_data.split(marker, 1)[1]
            try:
                result = json.loads(json_part)
                return {
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("error") or stderr_data,
                    "images": result.get("images", []),
                    "returncode": 1 if result.get("error") else 0,
                }
            except json.JSONDecodeError:
                pass

        return {
            "stdout": stdout_data,
            "stderr": stderr_data,
            "images": [],
            "returncode": 1 if stderr_data else 0,
        }
    except Exception as e:
        return {
            "stderr": f"Failed to execute: {e}",
            "stdout": "",
            "images": [],
            "returncode": 1,
        }
