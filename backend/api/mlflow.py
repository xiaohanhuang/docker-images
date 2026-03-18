"""MLflow integration API endpoints."""

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.monitoring.svc.cluster.local:5000")


@router.get("/experiments")
async def list_experiments() -> dict[str, Any]:
    """
    List all MLflow experiments.

    Returns:
        Dict containing list of experiments
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # MLflow experiments/search is a POST endpoint, but also supports GET with no filters
            response = await client.post(
                f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
                json={},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to list MLflow experiments: {e}")
        return {"experiments": []}


@router.get("/experiments/{experiment_id}/runs")
async def list_runs(experiment_id: str) -> dict[str, Any]:
    """
    List runs for a specific experiment.

    Args:
        experiment_id: MLflow experiment ID

    Returns:
        Dict containing list of runs
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
                json={"experiment_ids": [experiment_id]},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to list runs for experiment {experiment_id}: {e}")
        logger.exception("Internal error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """
    List all registered models in MLflow.

    Returns:
        Dict containing list of registered models
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # MLflow registered-models/search is a POST endpoint
            response = await client.post(
                f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search",
                json={},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to list MLflow models: {e}")
        return {"registered_models": []}


@router.get("/models/{model_name}")
async def get_model(model_name: str) -> dict[str, Any]:
    """
    Get details of a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dict containing model details
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/get",
                params={"name": model_name},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get model {model_name}: {e}")
        logger.exception("Internal error")
        raise HTTPException(status_code=500, detail="Internal server error")
