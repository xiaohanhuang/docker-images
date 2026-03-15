"""Notebook management API endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Request
from kubernetes import client, config
from pydantic import BaseModel

from backend.audit import log_audit_event

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize Kubernetes client
try:
    config.load_incluster_config()
except config.ConfigException:
    try:
        config.load_kube_config()
    except config.ConfigException:
        logger.warning("Could not load Kubernetes config")


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
        v1 = client.CoreV1Api()
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
        v1 = client.CoreV1Api()
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
