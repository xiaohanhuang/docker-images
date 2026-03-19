"""Kubernetes cluster API endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from kubernetes import client, config

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_v1() -> client.CoreV1Api:
    """Return a CoreV1Api client, loading k8s config on first use."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        try:
            config.load_kube_config()
        except config.ConfigException:
            logger.warning("Failed to load Kubernetes config")
    return client.CoreV1Api()


@router.get("/nodes")
async def list_nodes() -> list[dict[str, Any]]:
    """
    List all Kubernetes nodes.

    Returns:
        List of node information
    """
    try:
        v1 = _get_v1()
        nodes = v1.list_node(_request_timeout=10)
        node_list = []

        for node in nodes.items:
            conditions = {c.type: c.status for c in node.status.conditions}
            node_info = {
                "name": node.metadata.name,
                "status": "Ready" if conditions.get("Ready") == "True" else "NotReady",
                "instance_type": node.metadata.labels.get(
                    "node.kubernetes.io/instance-type", "unknown"
                ),
                "zone": node.metadata.labels.get("topology.kubernetes.io/zone", "unknown"),
                "capacity": {
                    "cpu": node.status.capacity.get("cpu", "0"),
                    "memory": node.status.capacity.get("memory", "0"),
                    "gpu": node.status.capacity.get("nvidia.com/gpu", "0"),
                },
                "allocatable": {
                    "cpu": node.status.allocatable.get("cpu", "0"),
                    "memory": node.status.allocatable.get("memory", "0"),
                    "gpu": node.status.allocatable.get("nvidia.com/gpu", "0"),
                },
            }
            node_list.append(node_info)

        return node_list
    except Exception as e:
        logger.warning(f"Failed to list nodes: {e}")
        return []


@router.get("/events")
async def list_events(namespace: str | None = Query(None)) -> list[dict[str, Any]]:
    """
    List Kubernetes events.

    Args:
        namespace: Optional namespace filter

    Returns:
        List of events
    """
    try:
        v1 = _get_v1()
        if namespace:
            events = v1.list_namespaced_event(namespace, _request_timeout=10)
        else:
            events = v1.list_event_for_all_namespaces(_request_timeout=10)

        event_list = []
        for event in events.items:
            event_info = {
                "namespace": event.metadata.namespace,
                "name": event.metadata.name,
                "type": event.type,
                "reason": event.reason,
                "message": event.message,
                "timestamp": (
                    event.last_timestamp.isoformat()
                    if event.last_timestamp
                    else event.metadata.creation_timestamp.isoformat()
                ),
                "involved_object": {
                    "kind": event.involved_object.kind,
                    "name": event.involved_object.name,
                },
            }
            event_list.append(event_info)

        # Sort by timestamp, most recent first
        event_list.sort(key=lambda x: x["timestamp"], reverse=True)

        return event_list[:100]  # Return last 100 events
    except Exception as e:
        logger.warning(f"Failed to list events: {e}")
        return []


@router.get("/pods/{pod_name}/logs")
async def get_pod_logs(pod_name: str, namespace: str = Query("default")) -> dict[str, str]:
    """
    Get logs for a specific pod.

    Args:
        pod_name: Name of the pod
        namespace: Namespace of the pod

    Returns:
        Dict containing pod logs
    """
    try:
        v1 = _get_v1()
        logs = v1.read_namespaced_pod_log(
            name=pod_name, namespace=namespace, tail_lines=1000, timestamps=True
        )

        return {"pod_name": pod_name, "namespace": namespace, "logs": logs}
    except Exception as e:
        logger.error(f"Failed to get logs for pod {pod_name}: {e}")
        logger.exception("Internal error")
        raise HTTPException(status_code=500, detail="Internal server error")
