"""Cost reporting API endpoints."""

import datetime
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class CostReport(BaseModel):
    """Cost report model."""

    total_cost: float
    period_start: str
    period_end: str
    jobs: list[dict]


class FinOpsManager:
    """Cost management utilities."""

    # Cost per hour for common instance types (On-Demand pricing)
    COST_PER_HOUR = {
        "g5.xlarge": 1.006,
        "g5.2xlarge": 1.212,
        "g5.4xlarge": 1.624,
        "g5.8xlarge": 2.448,
        "g5.12xlarge": 5.672,
        "g5.16xlarge": 4.096,
        "g5.24xlarge": 8.144,
        "g5.48xlarge": 16.288,
        "m5.large": 0.096,
        "m5.xlarge": 0.192,
        "m5.2xlarge": 0.384,
        "m5.4xlarge": 0.768,
        "m5.8xlarge": 1.536,
        "p3.2xlarge": 3.06,
        "p3.8xlarge": 12.24,
    }

    @classmethod
    def get_cost_estimate(cls, instance_type: str, duration: datetime.timedelta) -> float:
        """Calculate cost estimate for an instance type and duration."""
        hourly_rate = cls.COST_PER_HOUR.get(instance_type, 0.0)
        total_hours = duration.total_seconds() / 3600.0
        return total_hours * hourly_rate


def _get_flyte_remote():
    """Get Flyte remote client."""
    from flytekit.configuration import Config
    from flytekit.remote import FlyteRemote

    endpoint = os.getenv("FLYTE_ENDPOINT", "flyteadmin.ml-platform.internal:80")
    project = os.getenv("FLYTE_PROJECT", "flytesnacks")
    domain = os.getenv("FLYTE_DOMAIN", "development")

    config = Config.auto(endpoint=endpoint)
    return FlyteRemote(config=config, default_project=project, default_domain=domain)


@router.get("/report")
async def get_cost_report(
    days: int = 7,
    project: str | None = None,
    domain: str | None = None,
) -> CostReport:
    """Generate a cost report for recent jobs."""
    try:
        remote = _get_flyte_remote()

        # Use remote project/domain if not specified
        proj = project or remote.default_project
        dom = domain or remote.default_domain

        # Calculate time range
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(days=days)

        # List executions in the time range
        executions, _ = remote.client.list_executions_paginated(
            project=proj,
            domain=dom,
            limit=100,
        )

        total_cost = 0.0
        job_costs = []

        for execution in executions:
            # Filter by time range
            if execution.closure.created_at:
                created_at = execution.closure.created_at.ToDatetime()
                if created_at < start_time:
                    continue

            # Calculate duration
            duration = execution.closure.duration or datetime.timedelta(0)

            # Estimate instance type (simplified - in production, query K8s)
            instance_type = "g5.xlarge"  # Default assumption

            # Calculate cost
            cost = FinOpsManager.get_cost_estimate(instance_type, duration)
            total_cost += cost

            job_costs.append(
                {
                    "job_id": execution.id.name,
                    "workflow": execution.spec.launch_plan.name,
                    "instance_type": instance_type,
                    "duration_hours": duration.total_seconds() / 3600.0,
                    "cost_usd": round(cost, 2),
                    "created_at": (
                        execution.closure.created_at.isoformat()
                        if execution.closure.created_at
                        else None
                    ),
                }
            )

        return CostReport(
            total_cost=round(total_cost, 2),
            period_start=start_time.isoformat(),
            period_end=end_time.isoformat(),
            jobs=job_costs,
        )

    except Exception as e:
        logger.error(f"Failed to generate cost report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
