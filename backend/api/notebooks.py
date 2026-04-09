"""Notebook management API endpoints.

Provides:
- Notebook server CRUD (list, launch, stop)
- Reverse proxy to JupyterHub that strips CSP/X-Frame-Options headers
  so the dashboard can embed JupyterHub services in iframes (§5.2)
"""

import logging
import os

from fastapi import APIRouter, HTTPException, Request

try:
    from kubernetes import client
except ImportError:
    client = None  # type: ignore[assignment]
from pydantic import BaseModel

from backend.audit import log_audit_event
from backend.k8s import get_core_v1

logger = logging.getLogger(__name__)
router = APIRouter()

# JupyterHub URL — used for the reverse proxy
_JUPYTERHUB_URL = os.getenv(
    "JUPYTERHUB_INGRESS_URL",
    "http://k8s-jupyter-jupyterh-827e6a6320-482154231.us-west-2.elb.amazonaws.com",
)


class NotebookLaunchRequest(BaseModel):
    """Request model for launching a notebook."""

    namespace: str = "jupyter"
    port: int = 8080


@router.post("")
async def launch_notebook(req: NotebookLaunchRequest, request: Request):
    """Get JupyterHub connection information."""
    try:
        # Audit log
        await log_audit_event(
            user=request.state.user,
            action="launch",
            resource_type="notebook",
            resource_name="jupyterhub",
            details={"namespace": req.namespace},
        )

        return {
            "status": "success",
            "message": "JupyterHub is available",
            "namespace": req.namespace,
            "service": "proxy-public",
            "port": 80,
            "local_port": req.port,
        }

    except Exception as e:
        logger.error(f"Failed to get notebook info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_notebooks(namespace: str = "jupyter"):
    """List running notebook servers."""
    try:
        v1 = get_core_v1()
        pods = v1.list_namespaced_pod(
            namespace=namespace,
            label_selector="component=singleuser-server",
        )

        notebooks = []
        for pod in pods.items:
            notebooks.append(
                {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "created_at": (
                        pod.metadata.creation_timestamp.isoformat()
                        if pod.metadata.creation_timestamp
                        else None
                    ),
                }
            )

        return {"notebooks": notebooks}

    except Exception as e:
        logger.error(f"Failed to list notebooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{username}")
async def stop_notebook(
    username: str,
    namespace: str = "jupyter",
    request: Request = None,
):
    """Stop a user's notebook server."""
    try:
        v1 = get_core_v1()
        pod_name = f"jupyter-{username}"
        v1.delete_namespaced_pod(name=pod_name, namespace=namespace)

        # Audit log
        if request:
            await log_audit_event(
                user=request.state.user,
                action="stop",
                resource_type="notebook",
                resource_name=pod_name,
                details={"namespace": namespace, "username": username},
            )

        return {"status": "success", "message": f"Notebook for {username} stopped"}

    except client.exceptions.ApiException as e:
        if e.status == 404:
            raise HTTPException(status_code=404, detail=f"Notebook for user {username} not found")
        logger.error(f"Failed to stop notebook: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to stop notebook: {e}")
        raise HTTPException(status_code=500, detail=str(e))
