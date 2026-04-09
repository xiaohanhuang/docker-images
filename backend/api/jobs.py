"""Job management API endpoints."""

import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

try:
    from flytekit.models.core.execution import WorkflowExecutionPhase
except ImportError:
    WorkflowExecutionPhase = None  # type: ignore[assignment,misc]
from pydantic import BaseModel

from backend.audit import log_audit_event

logger = logging.getLogger(__name__)
router = APIRouter()


class JobSubmitRequest(BaseModel):
    """Request model for submitting a job."""

    workflow_name: str
    version: str = "v1"
    inputs: dict
    env_overrides: dict[str, str] | None = None


class JobListQuery(BaseModel):
    """Query parameters for listing jobs."""

    limit: int = 50
    project: str | None = None
    domain: str | None = None


@router.post("")
async def submit_job(req: JobSubmitRequest, request: Request):
    """Submit a training job to Flyte."""
    try:
        from backend.api.svc_proxy import get_flyte_remote

        remote = get_flyte_remote()

        # Fetch the workflow
        flyte_workflow = remote.fetch_workflow(name=req.workflow_name, version=req.version)

        # Execute it
        execute_kwargs: dict = {
            "inputs": req.inputs,
            "wait": False,
        }
        if req.env_overrides:
            execute_kwargs["envs"] = req.env_overrides
        execution = remote.execute(
            flyte_workflow,
            **execute_kwargs,
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

        console_url = os.getenv("FLYTE_CONSOLE_URL", "")
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
    """List recent jobs.

    Uses Flyte SDK (gRPC via NLB) first, falls back to Flyte HTTP API via ingress.
    """
    proj = project or os.getenv("FLYTE_PROJECT", "ml-platform")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    # Strategy 1: Flyte SDK via gRPC NLB
    try:
        from backend.api.svc_proxy import get_flyte_remote

        remote = get_flyte_remote()
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

        if result:
            return {"jobs": result}
        # gRPC returned empty — fall through to HTTP
        logger.warning("Flyte gRPC returned 0 executions, trying HTTP ingress...")

    except Exception as e:
        logger.warning(f"Flyte gRPC failed ({e}), trying HTTP ingress...")

    # Strategy 2: Flyte HTTP API via ingress
    try:
        from backend.api.svc_proxy import svc_get

        data = await svc_get(
            "flyte_http",
            f"api/v1/executions/{proj}/{dom}",
            params={"limit": str(limit)},
            timeout=10.0,
        )

        result = []
        for exec_item in data.get("executions", []):
            eid = exec_item.get("id", {})
            closure = exec_item.get("closure", {})
            spec = exec_item.get("spec", {})

            # Parse duration — Flyte returns either a dict {"seconds": N}
            # or a string like "3577.155014510s"
            dur = closure.get("duration")
            dur_str = None
            if dur:
                if isinstance(dur, dict):
                    secs = int(dur.get("seconds", 0))
                elif isinstance(dur, str) and dur.endswith("s"):
                    secs = int(float(dur.rstrip("s")))
                else:
                    secs = 0
                if secs > 0:
                    hrs, rem = divmod(secs, 3600)
                    mins, _ = divmod(rem, 60)
                    dur_str = f"{hrs}h {mins:02d}m" if hrs else f"{mins}m"

            # Parse phase — Flyte returns either an int or a string
            phase = closure.get("phase", "UNDEFINED")
            # Flyte HTTP returns numeric phase or string
            phase_map = {
                0: "UNDEFINED",
                1: "QUEUED",
                2: "RUNNING",
                3: "SUCCEEDING",
                4: "SUCCEEDED",
                5: "FAILING",
                6: "FAILED",
                7: "ABORTED",
                8: "TIMED_OUT",
            }
            if isinstance(phase, int):
                phase = phase_map.get(phase, str(phase))

            result.append(
                {
                    "job_id": eid.get("name", ""),
                    "workflow": spec.get("launch_plan", {}).get("name", "unknown"),
                    "status": phase,
                    "created_at": closure.get("created_at") or closure.get("createdAt"),
                    "duration": dur_str,
                }
            )

        return {"jobs": result}

    except Exception as e2:
        logger.error(f"Both Flyte gRPC and HTTP failed: {e2}")
        raise HTTPException(status_code=503, detail=f"Flyte unavailable: {e2}")


def _get_instance_type_from_k8s(job_id: str) -> str:
    """Synchronously fetch instance type from Kubernetes pods/nodes for a job."""
    try:
        from backend.k8s import get_core_v1

        v1 = get_core_v1()
        _proj = os.getenv("FLYTE_PROJECT", "ml-platform")
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
            return (
                node_info.metadata.labels.get("node.kubernetes.io/instance-type")
                or node_info.metadata.labels.get("karpenter.k8s.aws/instance-type")
                or "unknown"
            )
    except Exception:
        pass
    return "unknown"


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Get detailed status of a specific job."""
    try:
        from backend.api.svc_proxy import get_flyte_remote

        remote = get_flyte_remote()
        execution = remote.fetch_execution(name=job_id)
        remote.sync_execution(execution, sync_nodes=True)

        # Try to determine instance type from K8s in a non-blocking thread
        instance_type = await run_in_threadpool(_get_instance_type_from_k8s, job_id)

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
        from backend.api.svc_proxy import get_flyte_remote

        remote = get_flyte_remote()
        execution = remote.fetch_execution(name=job_id)
        remote.sync_execution(execution, sync_nodes=True)

        # Audit log
        user = getattr(request.state, "user", "anonymous")
        await log_audit_event(
            user=user,
            action="view_logs",
            resource_type="job",
            resource_name=job_id,
            details={},
        )

        async def log_generator():
            """Generate log lines from K8s pod logs."""
            from backend.k8s import get_core_v1

            v1 = get_core_v1()
            _proj = os.getenv("FLYTE_PROJECT", "ml-platform")
            _dom = os.getenv("FLYTE_DOMAIN", "development")
            target_namespace = os.getenv("FLYTE_NAMESPACE", f"{_proj}-{_dom}")

            for node_id, node_exec in execution.node_executions.items():
                if hasattr(node_exec, "task_executions"):
                    for task_exec in node_exec.task_executions:
                        yield f"=== Node: {node_id} ===\n".encode()
                        yield f"Task: {task_exec.id.task_id.name}\n".encode()
                        yield f"Status: {task_exec.closure.phase}\n".encode()
                        # Fetch real pod logs from K8s
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
                                    tail_lines=1000,
                                )
                                yield f"{log}\n".encode()
                        except Exception as log_err:
                            yield f"(logs unavailable: {log_err})\n".encode()
                        yield "\n".encode()

        return StreamingResponse(log_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Failed to get job logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs for job {job_id}: {e}")
