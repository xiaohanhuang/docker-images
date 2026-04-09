"""Pod management API endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Request

try:
    from kubernetes import client
except ImportError:
    client = None  # type: ignore[assignment]
from pydantic import BaseModel, Field

from backend.audit import log_audit_event
from backend.k8s import get_core_v1

logger = logging.getLogger(__name__)
router = APIRouter()


class PodLaunchRequest(BaseModel):
    """Request model for launching a pod."""

    name: str
    image: str
    gpu_type: str = "any"
    gpu_count: int = 1
    namespace: str = "default"
    env_vars: dict[str, str] = Field(default_factory=dict)
    cpu: str = "4"
    memory: str = "16Gi"
    shared: bool = False
    pvc: str = "efs-claim"
    mount_path: str = "/shared"


class PodSSHRequest(BaseModel):
    """Request model for SSH connection info."""

    namespace: str = "default"


@router.post("")
async def launch_pod(req: PodLaunchRequest, request: Request):
    """Launch an interactive GPU pod."""
    try:
        v1 = get_core_v1()

        # Build pod spec
        pod_name = req.name

        # EFS volume and mount (defined before container so volume_mounts is available)
        volumes = [
            client.V1Volume(
                name="efs-storage",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=req.pvc,
                ),
            ),
        ]
        volume_mounts = [
            client.V1VolumeMount(
                name="efs-storage",
                mount_path=req.mount_path,
            ),
        ]

        # Start sshd so kubectl port-forward + SSH works for the user.
        # Generate host keys, allow root login, enable pubkey auth, then keep alive.
        startup_cmd = (
            "ssh-keygen -A && "
            "mkdir -p /run/sshd /root/.ssh && "
            "chmod 700 /root/.ssh && "
            "sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && "
            "sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && "  # noqa: E501
            "sed -i 's/#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config && "
            "echo 'root:root' | chpasswd && "
            "/usr/sbin/sshd && "
            "exec sleep infinity"
        )
        container = client.V1Container(
            name="main",
            image=req.image,
            command=["/bin/bash", "-c", startup_cmd],
            resources=client.V1ResourceRequirements(
                requests={
                    "cpu": req.cpu,
                    "memory": req.memory,
                    **({"nvidia.com/gpu": str(req.gpu_count)} if req.gpu_count > 0 else {}),
                },
                limits={
                    "cpu": req.cpu,
                    "memory": req.memory,
                    **({"nvidia.com/gpu": str(req.gpu_count)} if req.gpu_count > 0 else {}),
                },
            ),
            env=[client.V1EnvVar(name=k, value=v) for k, v in req.env_vars.items()],
            volume_mounts=volume_mounts,
        )

        # Add GPU toleration and node selector if needed
        tolerations = []
        node_selector = {}
        if req.shared:
            node_selector["nvidia.com/device-plugin.config"] = "time-slicing"
            if req.gpu_type != "any":
                node_selector["karpenter.k8s.aws/instance-gpu-name"] = req.gpu_type
            tolerations.append(
                client.V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Equal",
                    value="true",
                    effect="NoSchedule",
                )
            )
        elif req.gpu_count > 0:
            tolerations.append(
                client.V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Equal",
                    value="true",
                    effect="NoSchedule",
                )
            )
            node_selector["role"] = "gpu-worker"
            if req.gpu_type != "any":
                node_selector["karpenter.k8s.aws/instance-gpu-name"] = req.gpu_type

        pod_spec = client.V1PodSpec(
            containers=[container],
            tolerations=tolerations if tolerations else None,
            node_selector=node_selector if node_selector else None,
            restart_policy="Never",
            volumes=volumes,
        )

        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=pod_name,
                labels={
                    "app": "interactive-pod",
                    "managed-by": "ml-platform",
                    "user": request.state.user,
                },
            ),
            spec=pod_spec,
        )

        # Create the pod
        v1.create_namespaced_pod(namespace=req.namespace, body=pod)

        # Audit log
        await log_audit_event(
            user=request.state.user,
            action="create",
            resource_type="pod",
            resource_name=pod_name,
            details={"namespace": req.namespace, "image": req.image},
        )

        return {
            "status": "success",
            "pod_name": pod_name,
            "namespace": req.namespace,
        }

    except Exception as e:
        logger.error(f"Failed to launch pod: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_pods(
    namespace: str = "default",
    all_namespaces: bool = False,
):
    """List all interactive pods."""
    try:
        v1 = get_core_v1()

        if all_namespaces:
            pods = v1.list_pod_for_all_namespaces(
                label_selector="app=interactive-pod", _request_timeout=10
            )
        else:
            pods = v1.list_namespaced_pod(
                namespace, label_selector="app=interactive-pod", _request_timeout=10
            )

        result = []
        for pod in pods.items:
            containers = pod.spec.containers or []
            gpu_limits = containers[0].resources.limits or {} if containers else {}
            gpu_count = gpu_limits.get("nvidia.com/gpu", "0")
            user = (pod.metadata.labels or {}).get("user", "unknown")

            # Determine if all containers are actually ready (not just phase=Running)
            container_statuses = pod.status.container_statuses or []
            all_ready = len(container_statuses) > 0 and all(cs.ready for cs in container_statuses)

            result.append(
                {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "ready": all_ready,
                    "node": pod.spec.node_name,
                    "gpu": str(gpu_count),
                    "user": user,
                    "created_at": (
                        pod.metadata.creation_timestamp.isoformat()
                        if pod.metadata.creation_timestamp
                        else None
                    ),
                }
            )

        return {"pods": result}

    except Exception as e:
        logger.warning(f"Failed to list pods: {e}")
        return {"pods": []}


@router.delete("/{pod_name}")
async def delete_pod(
    pod_name: str,
    namespace: str = "default",
    request: Request = None,
):
    """Delete a pod."""
    try:
        v1 = get_core_v1()
        v1.delete_namespaced_pod(name=pod_name, namespace=namespace)

        # Audit log
        if request:
            await log_audit_event(
                user=request.state.user,
                action="delete",
                resource_type="pod",
                resource_name=pod_name,
                details={"namespace": namespace},
            )

        return {"status": "success", "message": f"Pod {pod_name} deleted"}

    except client.exceptions.ApiException as e:
        if e.status == 404:
            raise HTTPException(status_code=404, detail=f"Pod {pod_name} not found")
        logger.error(f"Failed to delete pod: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pod_name}/ssh")
async def get_ssh_info(
    pod_name: str,
    req: PodSSHRequest,
    request: Request,
):
    """Get SSH connection information for a pod."""
    try:
        v1 = get_core_v1()

        # Verify pod exists
        pod = v1.read_namespaced_pod(name=pod_name, namespace=req.namespace)

        # Audit log
        await log_audit_event(
            user=request.state.user,
            action="ssh_info",
            resource_type="pod",
            resource_name=pod_name,
            details={"namespace": req.namespace},
        )

        return {
            "status": "success",
            "pod_name": pod_name,
            "namespace": req.namespace,
            "node": pod.spec.node_name,
            "ip": pod.status.pod_ip,
        }

    except client.exceptions.ApiException as e:
        if e.status == 404:
            raise HTTPException(status_code=404, detail=f"Pod {pod_name} not found")
        logger.error(f"Failed to get SSH info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
