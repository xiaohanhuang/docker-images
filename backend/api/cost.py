"""Cost reporting API endpoints."""

import datetime
import logging
import os

from fastapi import APIRouter
from pydantic import BaseModel

from backend.pricing import get_cost_estimate

logger = logging.getLogger(__name__)
router = APIRouter()


class CostReport(BaseModel):
    """Cost report model."""

    total_cost: float
    period_start: str
    period_end: str
    jobs: list[dict]


def _get_cluster_baseline_cost(days: int) -> float:
    """Calculate the baseline cost of running the EKS cluster (nodes)."""
    try:
        import datetime

        from backend.k8s import get_core_v1

        v1 = get_core_v1()
        nodes = v1.list_node().items
        baseline = 0.0
        # EKS control plane is ~$0.10/hour
        baseline += 0.10 * 24 * days
        for node in nodes:
            labels = node.metadata.labels or {}
            instance_type = (
                labels.get("node.kubernetes.io/instance-type")
                or labels.get("beta.kubernetes.io/instance-type")
                or "m5.large"
            )
            cost = get_cost_estimate(instance_type, datetime.timedelta(days=days))
            baseline += cost
        return baseline
    except Exception as e:
        logger.warning(f"Failed to calculate baseline cluster cost: {e}")
        # Default fallback estimate if K8s is unreachable (~$250/week for a small cluster)
        return 250.0 * (days / 7.0)


def _fetch_grpc_cost_report(days: int, project: str | None, domain: str | None) -> CostReport:
    """Fetch cost report using Flyte gRPC client."""
    from backend.api.svc_proxy import get_flyte_remote

    remote = get_flyte_remote()

    # Use remote project/domain if not specified
    proj = project or remote.default_project
    dom = domain or remote.default_domain

    # Calculate time range
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=days)

    # List executions in the time range
    executions, _ = remote.client.list_executions_paginated(
        project=proj,
        domain=dom,
        limit=100,
    )

    total_cost = _get_cluster_baseline_cost(days)
    job_costs = []

    for execution in executions:
        # Filter by time range
        created = execution.closure.created_at
        if created:
            if hasattr(created, "ToDatetime"):
                # Pass timezone so the result is tz-aware (matches start_time/end_time)
                created = created.ToDatetime(datetime.timezone.utc)
            elif isinstance(created, datetime.datetime) and created.tzinfo is None:
                # Normalize naive datetimes to UTC to match start_time/end_time
                created = created.replace(tzinfo=datetime.timezone.utc)
            if created < start_time:
                continue

        # Calculate duration
        dur = execution.closure.duration
        if dur is None:
            duration = datetime.timedelta(0)
        elif hasattr(dur, "ToTimedelta"):
            duration = dur.ToTimedelta()
        elif isinstance(dur, datetime.timedelta):
            duration = dur
        else:
            duration = datetime.timedelta(seconds=float(dur.seconds))

        # Estimate instance type (simplified - in production, query K8s)
        instance_type = "g5.xlarge"  # Default assumption

        # Calculate cost
        cost = get_cost_estimate(instance_type, duration)
        total_cost += cost

        job_costs.append(
            {
                "job_id": execution.id.name,
                "workflow": execution.spec.launch_plan.name,
                "instance_type": instance_type,
                "duration_hours": duration.total_seconds() / 3600.0,
                "cost_usd": round(cost, 2),
                "created_at": created.isoformat() if created else None,
            }
        )

    return CostReport(
        total_cost=round(total_cost, 2),
        period_start=start_time.isoformat(),
        period_end=end_time.isoformat(),
        jobs=job_costs,
    )


async def _fetch_http_cost_report(days: int, project: str | None, domain: str | None) -> CostReport:
    """Fetch cost report using Flyte HTTP fallback via ingress."""
    from backend.api.svc_proxy import svc_get

    proj = project or os.getenv("FLYTE_PROJECT", "ml-platform")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=days)

    data = await svc_get(
        "flyte_http",
        f"api/v1/executions/{proj}/{dom}",
        params={"limit": "100"},
    )

    total_cost = _get_cluster_baseline_cost(days)
    job_costs = []
    for exec_item in data.get("executions", []):
        closure = exec_item.get("closure", {})
        spec = exec_item.get("spec", {})
        eid = exec_item.get("id", {})

        # Filter by time range — mirror the gRPC path behavior
        created_at_str = closure.get("created_at") or closure.get("createdAt")
        created_at = None
        if created_at_str:
            try:
                # Python 3.11+ fromisoformat handles the trailing 'Z' natively
                created_at = datetime.datetime.fromisoformat(created_at_str)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                if created_at < start_time:
                    continue
            except (ValueError, AttributeError):
                pass

        dur = closure.get("duration", {})
        secs = int(dur.get("seconds", 0)) if isinstance(dur, dict) else 0
        duration = datetime.timedelta(seconds=secs)

        instance_type = "g5.xlarge"
        cost = get_cost_estimate(instance_type, duration)
        total_cost += cost

        job_costs.append(
            {
                "job_id": eid.get("name", ""),
                "workflow": spec.get("launch_plan", {}).get("name", "unknown"),
                "instance_type": instance_type,
                "duration_hours": duration.total_seconds() / 3600.0,
                "cost_usd": round(cost, 2),
                "created_at": created_at_str,
            }
        )

    return CostReport(
        total_cost=round(total_cost, 2),
        period_start=start_time.isoformat(),
        period_end=end_time.isoformat(),
        jobs=job_costs,
    )


@router.get("/report")
async def get_cost_report(
    days: int = 7,
    project: str | None = None,
    domain: str | None = None,
) -> CostReport:
    """Generate a cost report for recent jobs."""
    try:
        return _fetch_grpc_cost_report(days, project, domain)
    except Exception as e:
        logger.warning(f"Flyte gRPC failed for cost report ({e}), trying HTTP ingress...")

    # Fallback: Flyte HTTP API via ingress
    try:
        return await _fetch_http_cost_report(days, project, domain)
    except Exception as e2:
        logger.error(f"Both Flyte gRPC and HTTP failed for cost report: {e2}")
        # Return at least the baseline cluster infrastructure cost
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(days=days)
        return CostReport(
            total_cost=round(_get_cluster_baseline_cost(days), 2),
            period_start=start_time.isoformat(),
            period_end=end_time.isoformat(),
            jobs=[],
        )
