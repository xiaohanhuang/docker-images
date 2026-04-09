"""Dashboard aggregation API endpoints.

Uses svc_proxy for Prometheus metrics via ingress. No dev_data fallbacks.
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.svc_proxy import svc_get

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/overview")
async def get_dashboard_overview() -> dict[str, Any]:
    """Get unified dashboard overview with active pods, costs, and recent jobs.

    Returns:
        Dict containing overview metrics
    """
    desk_count = 0
    total_gpu_count = 0
    active_pod_count = 0
    pods_list: list[dict] = []

    try:
        from backend.k8s import get_core_v1

        v1 = get_core_v1()

        # Count desk pods specifically (ml-platform/type=desk)
        desk_pods = await asyncio.to_thread(
            v1.list_namespaced_pod,
            namespace="default",
            label_selector="ml-platform/type=desk",
            _request_timeout=10,
        )
        desk_count = len([p for p in desk_pods.items if p.status.phase in ("Running", "Pending")])

        # Count all running pods and GPU usage
        all_pods = await asyncio.to_thread(
            v1.list_namespaced_pod,
            namespace="default",
            field_selector="status.phase=Running",
            _request_timeout=10,
        )

        active_pod_count = len(all_pods.items)
        for pod in all_pods.items:
            containers = pod.spec.containers or []
            gpu_limits = containers[0].resources.limits or {} if containers else {}
            gpu_val = gpu_limits.get("nvidia.com/gpu", "0")
            pod_gpus = int(gpu_val) if str(gpu_val).isdigit() else 0
            total_gpu_count += pod_gpus

            pods_list.append(
                {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "node": pod.spec.node_name,
                    "gpu": str(pod_gpus),
                    "image": containers[0].image if containers else "",
                    "created_at": (
                        pod.metadata.creation_timestamp.isoformat()
                        if pod.metadata.creation_timestamp
                        else None
                    ),
                }
            )

        pods_list = pods_list[:20]

    except Exception as e:
        logger.warning(f"Dashboard overview: pods unavailable ({e})")

    # Get recent jobs (uses ingress-backed Flyte)
    try:
        from backend.api.jobs import list_jobs

        jobs_data = await list_jobs(limit=10)
        running_jobs = len(
            [j for j in jobs_data.get("jobs", []) if j.get("status") in ("RUNNING", "Running")]
        )
        recent_jobs = jobs_data.get("jobs", [])
    except Exception as e:
        logger.warning(f"Dashboard overview: jobs unavailable ({e})")
        running_jobs = 0
        recent_jobs = []

    # Get cost report (uses ingress-backed Flyte)
    try:
        from backend.api.cost import get_cost_report

        cost_data = await get_cost_report(days=7)
        total_cost = cost_data.total_cost
    except Exception as e:
        logger.warning(f"Dashboard overview: cost unavailable ({e})")
        # Fall back to baseline infrastructure cost estimate
        try:
            from backend.api.cost import _get_cluster_baseline_cost

            total_cost = round(_get_cluster_baseline_cost(7), 2)
        except Exception:
            total_cost = 0.0

    return {
        "active_desks": desk_count,
        "active_pods": active_pod_count,
        "active_gpus": total_gpu_count,
        "running_jobs": running_jobs,
        "recent_jobs": recent_jobs,
        "total_cost": total_cost,
        "pods": pods_list,
    }


@router.get("/metrics/{metric}")
async def get_metrics(metric: str, timeframe: str = "1h") -> dict[str, Any]:
    """Get metrics from Prometheus via ingress.

    Args:
        metric: Metric name (e.g., 'gpu', 'cpu', 'memory')
        timeframe: Timeframe for query (reserved for future query_range)

    Returns:
        Dict containing metric data
    """
    metric_queries = {
        "gpu": "DCGM_FI_DEV_GPU_UTIL",
        "cpu": "rate(node_cpu_seconds_total[5m])",
        "memory": "node_memory_MemAvailable_bytes",
    }

    query = metric_queries.get(metric, metric)

    try:
        data = await svc_get("prometheus", "api/v1/query", params={"query": query})
        return data
    except Exception as e:
        logger.error(f"Prometheus query failed for metric {metric}: {e}")
        raise HTTPException(status_code=503, detail=f"Prometheus unavailable: {e}")
