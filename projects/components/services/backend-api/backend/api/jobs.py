"""Job management API endpoints."""

import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from flytekit.models.core.execution import WorkflowExecutionPhase
from pydantic import BaseModel

from backend.main import log_audit_event

logger = logging.getLogger(__name__)
router = APIRouter()


class JobSubmitRequest(BaseModel):
    """Request model for submitting a job."""

    workflow_name: str
    version: str = "v1"
    inputs: dict


class JobListQuery(BaseModel):
    """Query parameters for listing jobs."""

    limit: int = 50
    project: str | None = None
    domain: str | None = None


def _get_flyte_remote():
    """Get Flyte remote client."""
    from flytekit.configuration import Config
    from flytekit.remote import FlyteRemote

    endpoint = os.getenv("FLYTE_ENDPOINT", "flyteadmin.ml-platform.internal:80")
    project = os.getenv("FLYTE_PROJECT", "flytesnacks")
    domain = os.getenv("FLYTE_DOMAIN", "development")

    config = Config.auto(endpoint=endpoint)
    return FlyteRemote(config=config, default_project=project, default_domain=domain)


@router.post("")
async def submit_job(req: JobSubmitRequest, request: Request):
    """Submit a training job to Flyte."""
    try:
        remote = _get_flyte_remote()

        # Fetch the workflow
        flyte_workflow = remote.fetch_workflow(name=req.workflow_name, version=req.version)

        # Execute it
        execution = remote.execute(
            flyte_workflow,
            inputs=req.inputs,
            wait=False,
        )

        # Audit log
        await log_audit_event(
            user=request.state.user,
            action="submit",
            resource_type="job",
            resource_name=execution.id.name,
            details={
                "workflow": req.workflow_name,
                "version": req.version,
                "inputs": req.inputs,
            },
        )

        console_url = os.getenv(
            "FLYTE_CONSOLE_URL",
            "http://k8s-flyte-flytecon-a425d1f87c-1407955100.us-west-2.elb.amazonaws.com",
        )
        url = (
            f"{console_url}/console/projects/{execution.id.project}"
            f"/domains/{execution.id.domain}/executions/{execution.id.name}"
        )

        return {
            "status": "success",
            "job_id": execution.id.name,
            "project": execution.id.project,
            "domain": execution.id.domain,
            "console_url": url,
        }

    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_jobs(
    limit: int = 50,
    project: str | None = None,
    domain: str | None = None,
):
    """List recent jobs."""
    try:
        remote = _get_flyte_remote()

        # Use remote project/domain if not specified
        proj = project or remote.default_project
        dom = domain or remote.default_domain

        # List recent executions
        executions, _ = remote.client.list_executions_paginated(
            project=proj,
            domain=dom,
            limit=limit,
        )

        result = []
        for execution in executions:
            result.append(
                {
                    "job_id": execution.id.name,
                    "workflow": execution.spec.launch_plan.name,
                    "status": WorkflowExecutionPhase.enum_to_string(execution.closure.phase),
                    "created_at": (
                        execution.closure.created_at.isoformat()
                        if execution.closure.created_at
                        else None
                    ),
                    "duration": (
                        str(execution.closure.duration) if execution.closure.duration else None
                    ),
                }
            )

        return {"jobs": result}

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Get detailed status of a specific job."""
    try:
        remote = _get_flyte_remote()
        execution = remote.fetch_execution(name=job_id)
        remote.sync_execution(execution, sync_nodes=True)

        # Try to determine instance type from K8s
        instance_type = "unknown"
        try:
            from kubernetes import client as k8s_client

            v1 = k8s_client.CoreV1Api()
            _proj = os.getenv("FLYTE_PROJECT", "flytesnacks")
            _dom = os.getenv("FLYTE_DOMAIN", "development")
            target_namespace = os.getenv("FLYTE_NAMESPACE", f"{_proj}-{_dom}")
            pods = v1.list_namespaced_pod(
                namespace=target_namespace,
                label_selector=f"flyte-execution-id={job_id}",
            )
            if not pods.items:
                pods = v1.list_namespaced_pod(
                    namespace=target_namespace,
                    label_selector=f"flyte.org/execution={job_id}",
                )
            if pods.items and pods.items[0].spec.node_name:
                node_info = v1.read_node(name=pods.items[0].spec.node_name)
                instance_type = (
                    node_info.metadata.labels.get("node.kubernetes.io/instance-type")
                    or node_info.metadata.labels.get("karpenter.k8s.aws/instance-type")
                    or "unknown"
                )
        except Exception:
            pass

        # Calculate duration
        started_at = execution.closure.started_at
        phase = execution.closure.phase
        if not started_at:
            duration = None
        elif phase in [
            WorkflowExecutionPhase.SUCCEEDED,
            WorkflowExecutionPhase.FAILED,
            WorkflowExecutionPhase.ABORTED,
        ]:
            if execution.closure.duration:
                duration = execution.closure.duration
            else:
                duration = execution.closure.updated_at - started_at
        else:
            import datetime

            now = datetime.datetime.now(datetime.timezone.utc)
            duration = now - started_at

        return {
            "job_id": execution.id.name,
            "workflow": execution.spec.launch_plan.name,
            "status": WorkflowExecutionPhase.enum_to_string(execution.closure.phase),
            "instance_type": instance_type,
            "created_at": (
                execution.closure.created_at.isoformat() if execution.closure.created_at else None
            ),
            "started_at": (
                execution.closure.started_at.isoformat() if execution.closure.started_at else None
            ),
            "duration": str(duration) if duration else None,
            "error": execution.closure.error.message if execution.closure.error else None,
        }

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e}")


@router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    request: Request,
):
    """Get logs for a job (streaming)."""
    try:
        remote = _get_flyte_remote()
        execution = remote.fetch_execution(name=job_id)
        remote.sync_execution(execution, sync_nodes=True)

        # Audit log
        await log_audit_event(
            user=request.state.user,
            action="view_logs",
            resource_type="job",
            resource_name=job_id,
            details={},
        )

        async def log_generator():
            """Generate log lines."""
            # Get logs from all task executions
            for node_id, node_exec in execution.node_executions.items():
                if hasattr(node_exec, "task_executions"):
                    for task_exec in node_exec.task_executions:
                        yield f"=== Node: {node_id} ===\n".encode()
                        # In a real implementation, fetch actual logs from K8s or log storage
                        yield f"Task: {task_exec.id.task_id.name}\n".encode()
                        yield f"Status: {task_exec.closure.phase}\n".encode()
                        yield "\n".encode()

        return StreamingResponse(log_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Failed to get job logs: {e}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
