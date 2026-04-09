"""Ray cluster integration API endpoints."""

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

RAY_DASHBOARD_URL = os.getenv(
    "RAY_DASHBOARD_URL", "http://ray-cluster-head-svc.ray.svc.cluster.local:8265"
)


@router.get("/cluster")
async def get_cluster_status() -> dict[str, Any]:
    """
    Get Ray cluster status.

    Returns:
        Dict containing cluster status information
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RAY_DASHBOARD_URL}/api/cluster_status")
            response.raise_for_status()
            return response.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.warning(f"Ray cluster not available: {e}")
        return {
            "active_nodes": 0,
            "total_cpus": 0,
            "total_gpus": 0,
            "available_cpus": 0,
            "available_gpus": 0,
        }
    except Exception as e:
        logger.error(f"Failed to get Ray cluster status: {e}")
        return {
            "active_nodes": 0,
            "total_cpus": 0,
            "total_gpus": 0,
            "available_cpus": 0,
            "available_gpus": 0,
        }


@router.get("/jobs")
async def get_jobs() -> list[dict[str, Any]]:
    """
    Get list of Ray jobs.

    Returns:
        List of Ray jobs
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RAY_DASHBOARD_URL}/api/jobs")
            response.raise_for_status()
            return response.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.warning(f"Ray jobs not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to get Ray jobs: {e}")
        logger.exception("Internal error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/nodes")
async def get_nodes() -> dict[str, Any]:
    """
    Get list of Ray nodes.

    Returns:
        Dict containing node information
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RAY_DASHBOARD_URL}/api/nodes")
            response.raise_for_status()
            return response.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.warning(f"Ray nodes not available: {e}")
        return {"nodes": []}
    except Exception as e:
        logger.error(f"Failed to get Ray nodes: {e}")
        logger.exception("Internal error")
        raise HTTPException(status_code=500, detail="Internal server error")
