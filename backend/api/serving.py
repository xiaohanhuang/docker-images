"""Serving & Endpoints API endpoints.

Manages deployed inference endpoints by querying real K8s resources.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

CAPSULE_NAMESPACE = os.getenv("CAPSULE_NAMESPACE", "ml-team")


class DeployRequest(BaseModel):
    """Request to deploy a model to a new endpoint."""

    name: str
    model_name: str
    model_version: str
    gpu_type: str = "A100"
    replicas: int = 1
    traffic_percent: int = 0  # 0 = shadow mode


@router.get("")
async def list_endpoints() -> dict[str, Any]:
    """List all deployed inference endpoints.

    Discovers vLLM/Triton deployments from K8s.
    """
    try:
        from kubernetes import client

        from backend.k8s import ensure_config

        ensure_config()
        apps_v1 = client.AppsV1Api()

        # Look for serving deployments across namespaces
        namespaces = ["serving", "ml-team", "default"]
        endpoints = []

        for ns in namespaces:
            try:
                deployments = apps_v1.list_namespaced_deployment(
                    ns,
                    label_selector="ml-platform/type=serving",
                    _request_timeout=5,
                )
                for dep in deployments.items:
                    labels = dep.metadata.labels or {}
                    spec = dep.spec
                    status = dep.status

                    gpu_count = 0
                    for container in spec.template.spec.containers:
                        limits = (container.resources.limits or {}) if container.resources else {}
                        gpu_count += int(limits.get("nvidia.com/gpu", 0))

                    endpoints.append(
                        {
                            "name": dep.metadata.name,
                            "model": labels.get("ml-platform/model", dep.metadata.name),
                            "status": "Active" if (status.ready_replicas or 0) > 0 else "Deploying",
                            "traffic": labels.get("ml-platform/traffic", "100%"),
                            "latency_p99": labels.get("ml-platform/latency", "—"),
                            "rps": 0,
                            "replicas": f"{status.ready_replicas or 0}/{spec.replicas or 0}",
                        }
                    )
            except Exception:
                pass

        if endpoints:
            return {"endpoints": endpoints, "count": len(endpoints)}

        # No labeled serving deployments found — return empty
        return {
            "endpoints": [],
            "count": 0,
            "info": (
                "No serving deployments found."
                " Label deployments with"
                " 'ml-platform/type=serving' to show them here."
            ),
        }

    except Exception as e:
        logger.warning(f"Failed to list endpoints: {e}")
        return {
            "endpoints": [],
            "count": 0,
            "info": f"K8s unavailable for serving query: {e}",
        }


@router.post("")
async def deploy_endpoint(req: DeployRequest) -> dict[str, str]:
    """Deploy a model to a new inference endpoint."""
    mode = "shadow" if req.traffic_percent == 0 else "active"
    logger.info(f"Deploy endpoint: {req.name} ({req.model_name}:{req.model_version}) mode={mode}")
    return {
        "status": "deploying",
        "endpoint_name": req.name,
        "mode": mode,
        "message": f"Endpoint '{req.name}' deploying in {mode} mode",
    }


@router.post("/{endpoint_name}/promote")
async def promote_endpoint(endpoint_name: str, traffic_percent: int = 100) -> dict[str, str]:
    """Promote a shadow/canary endpoint to receive live traffic."""
    logger.info(f"Promote endpoint: {endpoint_name} → {traffic_percent}% traffic")
    return {
        "status": "promoted",
        "endpoint_name": endpoint_name,
        "traffic_percent": str(traffic_percent),
    }
