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

    from backend.k8s import ensure_config

    cfg = _find_exec_config(base_url)
    if cfg is None:
        raise RuntimeError(f"No pod exec config for {base_url}")

    ensure_config()
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


def _url_to_svc_name(base_url: str) -> str | None:
    """Map a cluster-internal URL to its svc_proxy service name."""
    url_lower = base_url.lower()
    if "prometheus" in url_lower:
        return "prometheus"
    if "mlflow" in url_lower:
        return "mlflow"
    if "kubecost" in url_lower:
        return "kubecost"
    if "grafana" in url_lower:
        return "grafana"
    if "flyte" in url_lower:
        return "flyte_http"
    return None


async def _svc_get(
    base_url: str, path: str, params: dict | None = None, timeout: float = 15.0
) -> Any:
    """GET from a K8s service — delegates to svc_proxy for ingress-based access."""
    from backend.api.svc_proxy import svc_get as proxy_get

    svc_name = _url_to_svc_name(base_url)
    if svc_name:
        return await proxy_get(svc_name, path.lstrip("/"), params=params, timeout=timeout)
    # Unknown service — try direct
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"{base_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


async def _svc_post(
    base_url: str,
    path: str,
    json_body: dict | None = None,
    params: dict | None = None,
    timeout: float = 15.0,
) -> Any:
    """POST to a K8s service — delegates to svc_proxy for ingress-based access."""
    from backend.api.svc_proxy import svc_post as proxy_post

    svc_name = _url_to_svc_name(base_url)
    if svc_name:
        # svc_proxy.svc_post takes path and body; handle params by appending to path
        full_path = path.lstrip("/")
        if params:
            full_path += "?" + urllib.parse.urlencode(params)
        return await proxy_post(svc_name, full_path, body=json_body, timeout=timeout)
    # Unknown service — try direct
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{base_url}{path}", json=json_body, params=params)
        resp.raise_for_status()
        return resp.json()


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
        from backend.k8s import get_core_v1

        v1 = get_core_v1()
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
        from backend.k8s import get_core_v1

        v1 = get_core_v1()
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
        from backend.k8s import get_core_v1

        v1 = get_core_v1()

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


# ── Kubecost ─────────────────────────────────────────────────────

KUBECOST_URL = os.getenv(
    "KUBECOST_URL",
    "http://kubecost-cost-analyzer.kubecost.svc.cluster.local:9090",
)


async def query_kubecost(window: str = "1d", aggregate: str = "namespace") -> dict[str, Any]:
    """Query Kubecost for Kubernetes workload cost allocation.

    Args:
        window: Time window (e.g., '1d', '7d', 'today', 'lastweek').
        aggregate: How to aggregate costs ('namespace', 'pod', 'label', 'controller').

    Returns:
        Dict with cost breakdown by the requested aggregate.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{KUBECOST_URL}/model/allocation",
                params={"window": window, "aggregate": aggregate},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            # Flatten Kubecost's nested format
            results = []
            for window_data in data:
                for name, alloc in window_data.items():
                    results.append(
                        {
                            "name": name,
                            "cpu_cost": round(alloc.get("cpuCost", 0), 2),
                            "gpu_cost": round(alloc.get("gpuCost", 0), 2),
                            "ram_cost": round(alloc.get("ramCost", 0), 2),
                            "total_cost": round(alloc.get("totalCost", 0), 2),
                        }
                    )
            return {"allocations": results, "window": window, "aggregate": aggregate}
    except Exception as e:
        logger.warning(f"Kubecost query failed: {e}")
        return {"error": str(e), "allocations": []}


# ── AWS Cost Explorer ────────────────────────────────────────────


async def query_aws_cost_explorer(days: int = 7) -> dict[str, Any]:
    """Query AWS Cost Explorer for precise cloud billing metrics.

    Args:
        days: Number of days of billing data to fetch.

    Returns:
        Dict with daily cost breakdown.
    """
    import asyncio
    import datetime as dt

    def _fetch():
        import boto3

        client = boto3.client("ce", region_name="us-west-2")
        end = dt.datetime.now().strftime("%Y-%m-%d")
        start = (dt.datetime.now() - dt.timedelta(days=days)).strftime("%Y-%m-%d")

        response = client.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="DAILY",
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
        )
        results = []
        for entry in response.get("ResultsByTime", []):
            date = entry["TimePeriod"]["Start"]
            for group in entry.get("Groups", []):
                service = group["Keys"][0]
                cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
                if cost > 0.01:
                    results.append({"date": date, "service": service, "cost_usd": round(cost, 2)})
            if not entry.get("Groups"):
                total = float(entry.get("Total", {}).get("UnblendedCost", {}).get("Amount", 0))
                results.append({"date": date, "service": "Total", "cost_usd": round(total, 2)})
        return results

    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _fetch)
        total = sum(r["cost_usd"] for r in results)
        return {"costs": results, "total_usd": round(total, 2), "days": days}
    except Exception as e:
        logger.warning(f"AWS Cost Explorer query failed: {e}")
        return {"error": str(e), "costs": []}


async def get_task_source_code(task_name: str) -> dict[str, Any]:
    """Fetch the source code of a registered Flyte task/component.

    Maps a fully-qualified Flyte task name like
    'components.training.lora_finetune.task.lora_finetune' to its source file
    and returns the code for analysis.

    Args:
        task_name: Fully-qualified Flyte task name (e.g. from job node executions).

    Returns:
        Dict with 'source_path' and 'source_code', or 'error'.
    """
    import pathlib

    # Flyte task names follow: components.<category>.<component>.task.<function>
    # Source lives at: projects/components/components/<category>/<component>/task.py
    parts = task_name.split(".")
    base = pathlib.Path(__file__).resolve().parent.parent.parent

    # Try direct mapping: components.<cat>.<comp>.task.<fn>
    if len(parts) >= 4 and parts[0] == "components":
        category = parts[1]
        component = parts[2]
        comp_dir = base / "projects" / "components" / "components"
        task_file = comp_dir / category / component / "task.py"
        if task_file.exists():
            code = task_file.read_text()
            # Truncate if very large
            if len(code) > 8000:
                code = code[:8000] + "\n... [truncated]"
            return {"source_path": str(task_file.relative_to(base)), "source_code": code}

    # Fallback: search for matching task.py by component name
    components_dir = base / "projects" / "components" / "components"
    if components_dir.exists():
        # Extract the likely component name from task_name
        search_name = parts[2] if len(parts) >= 3 else parts[-1]
        for task_file in components_dir.rglob("task.py"):
            if search_name in str(task_file):
                code = task_file.read_text()
                if len(code) > 8000:
                    code = code[:8000] + "\n... [truncated]"
                return {"source_path": str(task_file.relative_to(base)), "source_code": code}

    return {"error": f"Source not found for task: {task_name}"}


async def get_job_tasks(job_id: str) -> dict[str, Any]:
    """Get per-task breakdown of a Flyte execution.

    Returns each task node's name, status, duration, and task type so the
    agent can analyze the job task by task.

    Args:
        job_id: The Flyte execution ID.

    Returns:
        Dict with 'tasks' list, each containing node_id, task_name, status, duration.
    """
    try:
        from backend.api.svc_proxy import get_flyte_remote

        remote = get_flyte_remote()
        execution = remote.fetch_execution(name=job_id)
        remote.sync_execution(execution, sync_nodes=True)

        tasks = []
        for node_id, node_exec in execution.node_executions.items():
            if node_id.startswith("start-node") or node_id.startswith("end-node"):
                continue
            task_info = {
                "node_id": node_id,
                "task_name": None,
                "status": str(node_exec.closure.phase),
                "started_at": None,
                "duration": None,
            }
            if hasattr(node_exec, "task_executions") and node_exec.task_executions:
                te = node_exec.task_executions[0]
                task_info["task_name"] = te.id.task_id.name
                if te.closure.started_at:
                    task_info["started_at"] = te.closure.started_at.isoformat()
                if te.closure.duration:
                    task_info["duration"] = str(te.closure.duration)
                elif te.closure.started_at and te.closure.updated_at:
                    task_info["duration"] = str(te.closure.updated_at - te.closure.started_at)
            tasks.append(task_info)

        return {"job_id": job_id, "tasks": tasks}
    except Exception as e:
        logger.warning(f"Failed to fetch tasks for {job_id}: {e}")
        return {"error": str(e), "tasks": []}


async def get_job_metrics(job_id: str) -> dict[str, Any]:
    """Fetch GPU/hardware metrics (DCGM) and MLflow data for a job.

    Args:
        job_id: The job/execution ID.

    Returns:
        Dict with 'gpu' metrics dict, 'mlflow_metrics', and 'mlflow_params'.
    """
    try:
        # 1. Resolve pods to get DCGM GPU metrics
        pods = await lookup_job_pods(job_id)
        pod_list = pods.get("pods", [])
        gpu_metrics: dict[str, Any] = {}

        if pod_list:
            import re

            pod_names = "|".join(re.escape(p["name"]) for p in pod_list)
            pod_filter = f'pod=~"{pod_names}"'

            # Query key DCGM metrics in parallel
            dcgm_queries = {
                "gpu_utilization": f"avg(DCGM_FI_DEV_GPU_UTIL{{{pod_filter}}})",
                "memory_utilization": f"avg(DCGM_FI_DEV_MEM_COPY_UTIL{{{pod_filter}}})",
                "tensor_core_utilization": (
                    f"avg(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{{{pod_filter}}}) * 100"
                ),
                "sm_occupancy": f"avg(DCGM_FI_PROF_SM_OCCUPANCY{{{pod_filter}}}) * 100",
                "sm_active": f"avg(DCGM_FI_PROF_SM_ACTIVE{{{pod_filter}}}) * 100",
                "fb_used_mb": (f"avg(DCGM_FI_DEV_FB_USED{{{pod_filter}}})"),
                "fb_free_mb": (f"avg(DCGM_FI_DEV_FB_FREE{{{pod_filter}}})"),
                "gpu_temp_c": f"avg(DCGM_FI_DEV_GPU_TEMP{{{pod_filter}}})",
                "power_usage_w": f"avg(DCGM_FI_DEV_POWER_USAGE{{{pod_filter}}})",
                "pcie_tx_mbps": (
                    f"rate(DCGM_FI_PROF_PCIE_TX_BYTES{{{pod_filter}}}[5m])" " / 1048576"
                ),
                "pcie_rx_mbps": (
                    f"rate(DCGM_FI_PROF_PCIE_RX_BYTES{{{pod_filter}}}[5m])" " / 1048576"
                ),
            }

            for metric_name, promql in dcgm_queries.items():
                try:
                    prom_res = await query_prometheus_instant(promql)
                    results = prom_res.get("data", {}).get("result", [])
                    if results:
                        val = float(results[0].get("value", [0, "0"])[1])
                        gpu_metrics[metric_name] = round(val, 1)
                except Exception:
                    pass

        # 2. Fetch MLflow metrics — search by tag first, fallback to sequential
        all_metrics: dict[str, Any] = {}
        all_params: dict[str, Any] = {}
        exps: list[dict[str, Any]] = []

        try:
            exps = await query_mlflow_experiments()
            experiment_ids = [exp["experiment_id"] for exp in exps if "experiment_id" in exp]

            # Try targeted search by flyte_execution_id tag first
            async with httpx.AsyncClient(timeout=10.0) as client:
                search_payload: dict[str, Any] = {
                    "filter": f"tags.flyte_execution_id = '{job_id}'",
                    "max_results": 1,
                }
                if experiment_ids:
                    search_payload["experiment_ids"] = experiment_ids
                resp = await client.post(
                    f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
                    json=search_payload,
                )
                if resp.status_code == 200:
                    runs = resp.json().get("runs", [])
                    if runs:
                        run = runs[0]
                        all_metrics = {
                            m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])
                        }
                        all_params = {
                            p["key"]: p["value"] for p in run.get("data", {}).get("params", [])
                        }
        except Exception:
            pass

        # Fallback: search sequentially if tag search found nothing
        if not all_metrics:
            if not exps:
                exps = await query_mlflow_experiments()
            for exp in exps:
                runs = await query_mlflow_runs(exp["experiment_id"])
                for run in runs:
                    tags = {t["key"]: t["value"] for t in run.get("data", {}).get("tags", [])}
                    run_info = run.get("info", {})
                    if (
                        job_id in run_info.get("run_name", "")
                        or tags.get("flyte_execution_id") == job_id
                    ):
                        all_metrics = {
                            m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])
                        }
                        all_params = {
                            p["key"]: p["value"] for p in run.get("data", {}).get("params", [])
                        }
                        break
                if all_metrics:
                    break

        return {
            "gpu": gpu_metrics,
            "mlflow_metrics": all_metrics,
            "mlflow_params": all_params,
        }
    except Exception as e:
        logger.warning(f"Failed to fetch metrics for {job_id}: {e}")
        return {"error": str(e)}


async def get_job_logs(job_id: str) -> dict[str, Any]:
    """Fetch logs for a specific Flyte or Ray job.

    Args:
        job_id: The job/execution ID.

    Returns:
        Dict with 'logs' string (last 10,000 chars).
    """
    max_log_chars = 10_000

    # Try Ray logs first via the Ray Dashboard API
    try:
        ray_jobs = await query_ray_jobs()
        for rj in ray_jobs:
            if rj.get("job_id") == job_id or rj.get("submission_id") == job_id:
                ray_job_id = rj.get("job_id", job_id)
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{RAY_URL}/api/jobs/{ray_job_id}/logs")
                    if resp.status_code == 200:
                        log_text = resp.json().get("logs", resp.text)
                        # Truncate from the end (keep last N chars)
                        if len(log_text) > max_log_chars:
                            log_text = "...[truncated]\n" + log_text[-max_log_chars:]
                        return {"logs": log_text}
    except Exception:
        pass

    # Try Flyte logs
    try:
        from backend.api.svc_proxy import get_flyte_remote

        remote = get_flyte_remote()
        execution = remote.fetch_execution(name=job_id)
        remote.sync_execution(execution, sync_nodes=True)

        from backend.k8s import ensure_config

        ensure_config()

        from kubernetes import client as k8s_client

        v1 = k8s_client.CoreV1Api()
        _proj = os.getenv("FLYTE_PROJECT", "ml-platform")
        _dom = os.getenv("FLYTE_DOMAIN", "development")
        target_namespace = os.getenv("FLYTE_NAMESPACE", f"{_proj}-{_dom}")

        chunks: list[str] = []
        total_len = 0
        for node_id, node_exec in execution.node_executions.items():
            if hasattr(node_exec, "task_executions"):
                for task_exec in node_exec.task_executions:
                    chunks.append(f"=== Node: {node_id} ===\n")
                    try:
                        pods = v1.list_namespaced_pod(
                            namespace=target_namespace,
                            label_selector=(
                                f"flyte.org/execution={job_id}" f",flyte.org/node-id={node_id}"
                            ),
                        )
                        for pod in pods.items:
                            log = v1.read_namespaced_pod_log(
                                name=pod.metadata.name,
                                namespace=target_namespace,
                                tail_lines=500,
                            )
                            chunks.append(log + "\n")
                            total_len += len(log) + 1
                    except Exception:
                        chunks.append("[log fetch failed]\n")
                    if total_len > max_log_chars * 2:
                        break
            if total_len > max_log_chars * 2:
                break

        log_text = "".join(chunks)
        if len(log_text) > max_log_chars:
            log_text = "...[truncated]\n" + log_text[-max_log_chars:]
        return {"logs": log_text}
    except Exception as e:
        logger.warning(f"Failed to fetch logs for {job_id}: {e}")
        return {"error": str(e), "logs": ""}


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
    "query_kubecost": query_kubecost,
    "query_aws_cost_explorer": query_aws_cost_explorer,
    "get_job_logs": get_job_logs,
    "get_job_metrics": get_job_metrics,
    "get_job_tasks": get_job_tasks,
    "get_task_source_code": get_task_source_code,
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
            "name": "get_job_metrics",
            "description": (
                "Fetch GPU utilization and MLflow metrics/parameters for a specific job. "
                "Use this for performance analysis and optimization advice."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The Flyte execution ID or Ray job ID",
                        },
                    },
                    "required": ["job_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_job_logs",
            "description": (
                "Fetch the last 10,000 characters of logs for a job. "
                "Use this to diagnose failures, OOMs, or specific errors."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The Flyte execution ID or Ray job ID",
                        },
                    },
                    "required": ["job_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_job_tasks",
            "description": (
                "Get a per-task breakdown of a Flyte job execution. "
                "Returns each task node's name, status, and duration. "
                "Use this first when analyzing a job to understand its structure."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The Flyte execution ID",
                        },
                    },
                    "required": ["job_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_task_source_code",
            "description": (
                "Fetch the Python source code of a registered Flyte task/component. "
                "Use this to review the actual training code for optimization "
                "opportunities (batch size, data loading, checkpointing, etc)."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "task_name": {
                            "type": "string",
                            "description": (
                                "Fully-qualified Flyte task name "
                                "(e.g. components.training.lora_finetune.task.lora_finetune)"
                            ),
                        },
                    },
                    "required": ["task_name"],
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
    {
        "toolSpec": {
            "name": "query_kubecost",
            "description": (
                "Query Kubecost for Kubernetes workload cost allocation"
                " by namespace, pod, or label. "
                "Returns CPU, GPU, RAM, and total costs."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "window": {
                            "type": "string",
                            "description": "Time window: '1d', '7d', 'today', 'lastweek'",
                        },
                        "aggregate": {
                            "type": "string",
                            "description": (
                                "Aggregate by: 'namespace', 'pod'," " 'label', 'controller'"
                            ),
                        },
                    },
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_aws_cost_explorer",
            "description": (
                "Query AWS Cost Explorer for precise cloud billing metrics (EC2, S3, EKS, etc). "
                "Returns daily cost breakdown by AWS service."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days of billing data to fetch (default 7)",
                        },
                    },
                }
            },
        }
    },
]
