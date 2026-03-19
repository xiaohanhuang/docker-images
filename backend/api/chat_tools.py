"""Tools available to the Bedrock agent for querying platform data sources."""

import json
import logging
import os
import urllib.parse
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
)
MLFLOW_URL = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow.monitoring.svc.cluster.local:80",
)
RAY_URL = os.getenv(
    "RAY_DASHBOARD_URL",
    "http://ray-cluster-head-svc.ray.svc.cluster.local:8265",
)


# ── Pod Exec Query Helper ────────────────────────────────────────
# When running outside the cluster (local dev), cluster-internal DNS
# (*.svc.cluster.local) is unreachable. These helpers automatically
# fall back to exec-ing into service pods via the K8s API, running
# wget/python to query the service on localhost from inside the pod.

# Maps service name → pod discovery config.  For StatefulSets the
# deterministic pod name is used directly; for Deployments we look
# up a running pod by label selector.
_POD_EXEC_CONFIG: dict[str, dict[str, Any]] = {
    "kube-prometheus-stack-prometheus": {
        "namespace": "monitoring",
        "pod_name": "prometheus-kube-prometheus-stack-prometheus-0",
        "container": "prometheus",
        "port": 9090,
        "tool": "wget",  # Alpine-based image — wget available
    },
    "mlflow": {
        "namespace": "monitoring",
        "label_selector": "app.kubernetes.io/name=mlflow",
        "container": "mlflow",
        "port": 5000,  # container port (service port 80 → target 5000)
        "tool": "python3",  # Python image — no wget/curl
    },
    "ray-cluster-head-svc": {
        "namespace": "ray",
        "label_selector": "ray.io/node-type=head",
        "port": 8265,
        "tool": "python3",
    },
}


def _parse_svc_url(url: str) -> tuple[str, str, int]:
    """Parse a K8s cluster-internal URL → (namespace, service_name, port)."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or 80
    parts = host.split(".")
    return parts[1] if len(parts) > 1 else "default", parts[0], port


def _find_exec_config(base_url: str) -> dict[str, Any] | None:
    """Match a service URL to its pod exec config."""
    _, svc_name, _ = _parse_svc_url(base_url)
    for key, cfg in _POD_EXEC_CONFIG.items():
        if svc_name == key or svc_name.startswith(key):
            return cfg
    return None


def _build_exec_command(
    tool: str,
    port: int,
    path: str,
    method: str = "GET",
    params: dict | None = None,
    json_body: dict | None = None,
) -> list[str]:
    """Build the shell command to run inside the pod."""
    url = f"http://localhost:{port}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    if tool == "wget":
        if method == "POST" and json_body:
            return [
                "wget",
                "-qO-",
                "--timeout=8",
                "--header=Content-Type: application/json",
                f"--post-data={json.dumps(json_body)}",
                url,
            ]
        return ["wget", "-qO-", "--timeout=8", url]

    # python3 — works in any Python-based container
    if method == "POST" and json_body:
        body_json = json.dumps(json_body)
        py_script = (
            "import urllib.request,json;"
            f"req=urllib.request.Request('{url}',"
            f"data={body_json!r}.encode(),"
            "headers={{'Content-Type':'application/json'}});"
            "print(urllib.request.urlopen(req,timeout=8).read().decode())"
        )
    else:
        py_script = (
            "import urllib.request;"
            f"print(urllib.request.urlopen('{url}',timeout=8).read().decode())"
        )
    return ["python3", "-c", py_script]


async def _pod_exec_query(
    base_url: str,
    path: str,
    method: str = "GET",
    params: dict | None = None,
    json_body: dict | None = None,
    timeout: float = 15.0,
) -> Any:
    """Query a service by exec-ing into its pod via kubectl."""
    import asyncio
    import subprocess

    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    cfg = _find_exec_config(base_url)
    if cfg is None:
        raise RuntimeError(f"No pod exec config for {base_url}")

    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()

    v1 = k8s_client.CoreV1Api()
    namespace = cfg["namespace"]
    container = cfg.get("container")
    port = cfg["port"]
    tool = cfg.get("tool", "wget")

    # Resolve pod name — deterministic for StatefulSets, label lookup otherwise
    pod_name = cfg.get("pod_name")
    if not pod_name:
        pods = v1.list_namespaced_pod(
            namespace, label_selector=cfg["label_selector"], _request_timeout=5
        )
        running = [p for p in pods.items if p.status.phase == "Running"]
        if not running:
            raise RuntimeError(f"No running pods for {cfg['label_selector']} in {namespace}")
        pod_name = running[0].metadata.name
        if not container:
            container = running[0].spec.containers[0].name

    cmd = _build_exec_command(tool, port, path, method, params, json_body)

    kubectl_cmd = ["kubectl", "exec", "-n", namespace, pod_name]
    if container:
        kubectl_cmd += ["-c", container]
    kubectl_cmd += ["--"] + cmd

    def _run() -> str:
        result = subprocess.run(kubectl_cmd, capture_output=True, text=True, timeout=int(timeout))
        if result.returncode != 0:
            raise RuntimeError(f"kubectl exec failed: {result.stderr.strip()}")
        return result.stdout

    loop = asyncio.get_running_loop()
    response = await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=timeout)
    return json.loads(response)


async def _svc_get(
    base_url: str, path: str, params: dict | None = None, timeout: float = 15.0
) -> Any:
    """GET from a K8s service — direct HTTP first, then pod exec fallback."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{base_url}{path}", params=params)
            resp.raise_for_status()
            return resp.json()
    except (httpx.ConnectError, httpx.ConnectTimeout, OSError):
        if ".svc.cluster.local" not in base_url:
            raise
        logger.debug(f"Direct HTTP failed for {base_url}, falling back to pod exec")

    return await _pod_exec_query(base_url, path, params=params, timeout=timeout)


async def _svc_post(
    base_url: str,
    path: str,
    json_body: dict | None = None,
    params: dict | None = None,
    timeout: float = 15.0,
) -> Any:
    """POST to a K8s service — direct HTTP first, then pod exec fallback."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(f"{base_url}{path}", json=json_body, params=params)
            resp.raise_for_status()
            return resp.json()
    except (httpx.ConnectError, httpx.ConnectTimeout, OSError):
        if ".svc.cluster.local" not in base_url:
            raise
        logger.debug(f"Direct HTTP failed for {base_url}, falling back to pod exec")

    return await _pod_exec_query(
        base_url, path, method="POST", json_body=json_body, params=params, timeout=timeout
    )


# ── Prometheus ───────────────────────────────────────────────────


async def query_prometheus(promql: str, duration: str = "1h", step: str = "60s") -> dict[str, Any]:
    """Execute a PromQL query_range and return time-series results.

    Args:
        promql: The PromQL expression.
        duration: How far back to query (e.g. '1h', '6h', '24h').
        step: Resolution step (e.g. '15s', '60s', '300s').

    Returns:
        Prometheus API response with 'status' and 'data' keys.
    """
    import time as _time

    end = _time.time()
    duration_map = {
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "6h": 21600,
        "12h": 43200,
        "24h": 86400,
        "7d": 604800,
    }
    seconds = duration_map.get(duration, 3600)
    start = end - seconds

    try:
        return await _svc_get(
            PROMETHEUS_URL,
            "/api/v1/query_range",
            params={"query": promql, "start": start, "end": end, "step": step},
        )
    except Exception as e:
        logger.warning(f"Prometheus query failed: {e}")
        return {"status": "error", "error": str(e), "data": {"result": []}}


async def query_prometheus_instant(promql: str) -> dict[str, Any]:
    """Execute an instant PromQL query and return current values.

    Args:
        promql: The PromQL expression.

    Returns:
        Prometheus API response.
    """
    try:
        return await _svc_get(
            PROMETHEUS_URL,
            "/api/v1/query",
            params={"query": promql},
        )
    except Exception as e:
        logger.warning(f"Prometheus instant query failed: {e}")
        return {"status": "error", "error": str(e), "data": {"result": []}}


# ── MLflow ───────────────────────────────────────────────────────


async def query_mlflow_experiments() -> list[dict[str, Any]]:
    """List all MLflow experiments.

    Returns:
        List of experiment dicts with experiment_id, name, lifecycle_stage.
    """
    try:
        result = await _svc_get(
            MLFLOW_URL,
            "/api/2.0/mlflow/experiments/search",
            params={"max_results": 100},
        )
        return result.get("experiments", [])
    except Exception as e:
        logger.warning(f"MLflow experiments query failed: {e}")
        return []


async def query_mlflow_runs(experiment_id: str, max_results: int = 50) -> list[dict[str, Any]]:
    """List runs for a given MLflow experiment.

    Args:
        experiment_id: The MLflow experiment ID.
        max_results: Maximum number of runs to return.

    Returns:
        List of run dicts.
    """
    try:
        result = await _svc_post(
            MLFLOW_URL,
            "/api/2.0/mlflow/runs/search",
            json_body={"experiment_ids": [experiment_id], "max_results": max_results},
        )
        return result.get("runs", [])
    except Exception as e:
        logger.warning(f"MLflow runs query failed: {e}")
        return []


# ── Kubernetes ───────────────────────────────────────────────────


async def query_kubernetes_pods(namespace: str | None = None) -> list[dict[str, Any]]:
    """List pods, optionally filtered by namespace.

    Args:
        namespace: Optional namespace filter. If None, lists all namespaces.

    Returns:
        List of pod summary dicts.
    """
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        if namespace:
            pods = v1.list_namespaced_pod(namespace, _request_timeout=10)
        else:
            pods = v1.list_pod_for_all_namespaces(_request_timeout=10)

        results = []
        for p in pods.items:
            results.append(
                {
                    "name": p.metadata.name,
                    "namespace": p.metadata.namespace,
                    "status": p.status.phase,
                    "node": p.spec.node_name or "unscheduled",
                    "gpu": (
                        p.spec.containers[0].resources.limits.get("nvidia.com/gpu", "0")
                        if p.spec.containers and p.spec.containers[0].resources.limits
                        else "0"
                    ),
                }
            )
        return results
    except Exception as e:
        logger.warning(f"Kubernetes pods query failed: {e}")
        return []


async def query_kubernetes_nodes() -> list[dict[str, Any]]:
    """List cluster nodes with capacity information.

    Returns:
        List of node summary dicts.
    """
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        nodes = v1.list_node(_request_timeout=10)

        results = []
        for n in nodes.items:
            conditions = {c.type: c.status for c in n.status.conditions}
            results.append(
                {
                    "name": n.metadata.name,
                    "status": "Ready" if conditions.get("Ready") == "True" else "NotReady",
                    "instance_type": n.metadata.labels.get(
                        "node.kubernetes.io/instance-type", "unknown"
                    ),
                    "cpu_capacity": n.status.capacity.get("cpu", "0"),
                    "memory_capacity": n.status.capacity.get("memory", "0"),
                    "gpu_capacity": n.status.capacity.get("nvidia.com/gpu", "0"),
                }
            )
        return results
    except Exception as e:
        logger.warning(f"Kubernetes nodes query failed: {e}")
        return []


# ── Cost ─────────────────────────────────────────────────────────


async def query_cost(days: int = 7) -> dict[str, Any]:
    """Get cost report for the specified number of days.

    Args:
        days: Number of days to include in the report.

    Returns:
        Cost report dict with total_cost, period_start, period_end, jobs.
    """
    try:
        from backend.api.cost import get_cost_report

        report = await get_cost_report(days=days)
        return report.model_dump()
    except Exception as e:
        logger.warning(f"Cost query failed: {e}")
        return {"total_cost": 0, "period_start": "", "period_end": "", "jobs": []}


# ── Job ID → Pod Label Resolver ──────────────────────────────────


async def lookup_job_pods(job_id: str) -> dict[str, Any]:
    """Resolve a Flyte execution ID or Ray job ID to pod label selectors.

    Args:
        job_id: A Flyte execution ID (e.g. 'fXXXX') or Ray job ID.

    Returns:
        Dict with 'job_type', 'label_selector', and 'pods' list.
    """
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()

        # Try Flyte first — Flyte pods are labeled with execution-id
        flyte_selector = f"execution-id={job_id}"
        flyte_pods = v1.list_pod_for_all_namespaces(
            label_selector=flyte_selector, _request_timeout=10
        )
        if flyte_pods.items:
            return {
                "job_type": "flyte",
                "label_selector": flyte_selector,
                "pods": [
                    {
                        "name": p.metadata.name,
                        "namespace": p.metadata.namespace,
                        "status": p.status.phase,
                        "node": p.spec.node_name or "unscheduled",
                    }
                    for p in flyte_pods.items
                ],
            }

        # Try Ray — Ray job pods are labeled with ray.io/job-id
        ray_selector = f"ray.io/job-id={job_id}"
        ray_pods = v1.list_pod_for_all_namespaces(label_selector=ray_selector, _request_timeout=10)
        if ray_pods.items:
            return {
                "job_type": "ray",
                "label_selector": ray_selector,
                "pods": [
                    {
                        "name": p.metadata.name,
                        "namespace": p.metadata.namespace,
                        "status": p.status.phase,
                        "node": p.spec.node_name or "unscheduled",
                    }
                    for p in ray_pods.items
                ],
            }

        return {"job_type": "unknown", "label_selector": "", "pods": []}
    except Exception as e:
        logger.warning(f"Job pod lookup failed for {job_id}: {e}")
        return {"job_type": "error", "label_selector": "", "pods": [], "error": str(e)}


# ── Ray Cluster ──────────────────────────────────────────────────


async def query_ray_cluster() -> dict[str, Any]:
    """Get Ray cluster status including nodes and resource usage.

    Returns:
        Dict with cluster_status, nodes, alive_nodes count.
    """
    try:
        return await _svc_get(RAY_URL, "/api/cluster_status")
    except Exception as e:
        logger.warning(f"Ray cluster query failed: {e}")
        return {}


async def query_ray_jobs() -> list[dict[str, Any]]:
    """List Ray jobs.

    Returns:
        List of Ray job dicts.
    """
    try:
        return await _svc_get(RAY_URL, "/api/jobs/")
    except Exception as e:
        logger.warning(f"Ray jobs query failed: {e}")
        return []


# ── Tool Registry ────────────────────────────────────────────────
# Bedrock tool definitions for the converse API. Each entry maps
# the tool name to the function and the JSON schema for its inputs.

TOOL_FUNCTIONS = {
    "query_prometheus": query_prometheus,
    "query_prometheus_instant": query_prometheus_instant,
    "query_mlflow_experiments": query_mlflow_experiments,
    "query_mlflow_runs": query_mlflow_runs,
    "query_kubernetes_pods": query_kubernetes_pods,
    "query_kubernetes_nodes": query_kubernetes_nodes,
    "query_cost": query_cost,
    "lookup_job_pods": lookup_job_pods,
    "query_ray_cluster": query_ray_cluster,
    "query_ray_jobs": query_ray_jobs,
}

BEDROCK_TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "query_prometheus",
            "description": (
                "Execute a PromQL query_range against Prometheus and return time-series data. "
                "Use for GPU utilization (DCGM_FI_DEV_GPU_UTIL), CPU, memory, network metrics. "
                "Common queries: DCGM_FI_DEV_GPU_UTIL (GPU %), "
                "rate(node_cpu_seconds_total{mode!='idle'}[5m]) (CPU), "
                "node_memory_MemAvailable_bytes (memory)."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "promql": {
                            "type": "string",
                            "description": "The PromQL expression to execute",
                        },
                        "duration": {
                            "type": "string",
                            "description": "How far back to query: 5m, 15m, 1h, 6h, 24h, 7d",
                        },
                        "step": {
                            "type": "string",
                            "description": "Resolution step: 15s, 60s, 300s",
                        },
                    },
                    "required": ["promql"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_prometheus_instant",
            "description": (
                "Execute an instant PromQL query for current metric values. "
                "Use when the user asks 'what is the current X?' rather than a time range."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "promql": {
                            "type": "string",
                            "description": "The PromQL expression",
                        },
                    },
                    "required": ["promql"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_mlflow_experiments",
            "description": "List all MLflow experiments with their IDs and names.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_mlflow_runs",
            "description": (
                "List runs for a specific MLflow experiment. "
                "Returns run metrics, parameters, and status."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "The MLflow experiment ID",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max runs to return (default 50)",
                        },
                    },
                    "required": ["experiment_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_kubernetes_pods",
            "description": (
                "List Kubernetes pods. Optionally filter by namespace. "
                "Returns pod name, namespace, status, node, and GPU count."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace to filter pods",
                        },
                    },
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_kubernetes_nodes",
            "description": ("List cluster nodes with CPU, memory, GPU capacity and status."),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_cost",
            "description": (
                "Get a cost report for the last N days. "
                "Returns total_cost, per-job breakdown with instance types and durations."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to include (default 7)",
                        },
                    },
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "lookup_job_pods",
            "description": (
                "Resolve a Flyte execution ID or Ray job ID to its Kubernetes pods. "
                "Returns the job type (flyte/ray), label selector, and pod list."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "Flyte execution ID (e.g. fXXXX) or Ray job ID",
                        },
                    },
                    "required": ["job_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_ray_cluster",
            "description": "Get Ray cluster status including node count and resources.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_ray_jobs",
            "description": "List all Ray jobs with their status, submission time, and runtime.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                }
            },
        }
    },
]
