"""Dashboard aggregation API endpoints."""

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/overview")
async def get_dashboard_overview() -> dict[str, Any]:
    """
    Get unified dashboard overview with active pods, costs, and recent jobs.

    Returns:
        Dict containing overview metrics
    """
    try:
        # Get active pods from pods API
        from backend.api.pods import list_pods

        pods_data = await list_pods(all_namespaces=True)
        active_pods = [p for p in pods_data["pods"] if p["status"] == "Running"]
        # Fix: Filter GPU pods by numeric comparison (gpu > 0)
        gpu_pods = [
            p
            for p in active_pods
            if (isinstance(p.get("gpu"), (int, float)) and p.get("gpu", 0) > 0)
            or (isinstance(p.get("gpu"), str) and p.get("gpu") not in ("0", "", None))
        ]

        # Get recent jobs
        from backend.api.jobs import list_jobs

        jobs_data = await list_jobs(limit=10)

        # Get cost report
        from backend.api.cost import get_cost_report

        cost_data = await get_cost_report(days=7)

        return {
            "active_pods": len(active_pods),
            "gpu_pods": len(gpu_pods),
            "recent_jobs": jobs_data.get("jobs", []),
            "total_cost": cost_data.total_cost,
            "pods": active_pods[:10],
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard overview: {e}")
        logger.exception("Internal error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/metrics/{metric}")
async def get_metrics(metric: str, timeframe: str = "1h") -> dict[str, Any]:
    """
    Get metrics from Prometheus.

    Args:
        metric: Metric name (e.g., 'gpu', 'cpu', 'memory')
        timeframe: Timeframe for query (e.g., '1h', '6h', '24h') - currently unused,
            reserved for future query_range implementation

    Returns:
        Dict containing metric data
    """
    prometheus_url = os.getenv(
        "PROMETHEUS_URL",
        "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
    )

    metric_queries = {
        "gpu": "DCGM_FI_DEV_GPU_UTIL",
        "cpu": "rate(node_cpu_seconds_total[5m])",
        "memory": "node_memory_MemAvailable_bytes",
    }

    query = metric_queries.get(metric, metric)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{prometheus_url}/api/v1/query", params={"query": query})
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Prometheus returned error for metric {metric}: {e}")
        raise HTTPException(status_code=503, detail=f"Prometheus unavailable: {str(e)}")
    except httpx.TimeoutException as e:
        logger.error(f"Prometheus timeout for metric {metric}: {e}")
        raise HTTPException(status_code=504, detail="Prometheus timeout")
    except Exception as e:
        logger.error(f"Failed to query Prometheus for metric {metric}: {e}")
        logger.exception("Internal error")
        raise HTTPException(status_code=500, detail="Internal server error")
