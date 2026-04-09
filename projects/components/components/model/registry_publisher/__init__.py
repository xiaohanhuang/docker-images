"""
Model component — register fine-tuned models in MLflow.

Image: ml-gpu
"""

import logging

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory

logger = logging.getLogger(__name__)


@task(
    retries=2,
    requests=Resources(cpu="2", mem="4Gi"),
    cache=False,
)
def registry_publisher(
    checkpoint_path: FlyteDirectory,
    model_name: str,
    eval_metrics: dict,
    base_model: str,
    training_method: str,
) -> str:
    """Register fine-tuned model in MLflow Model Registry.

    Args:
        checkpoint_path: Model checkpoint directory.
        model_name: MLflow model name.
        eval_metrics: JSON string of evaluation metrics to log.
        base_model: Base model ID (metadata).
        training_method: Training method ("lora", "qlora", "full").

    Returns:
        MLflow model URI (e.g., "models:/llama-3-8b-sft/1").
    """
    import json
    import os

    import mlflow
    from mlflow.tracking import MlflowClient

    # Parse eval_metrics from JSON string
    if isinstance(eval_metrics, str):
        try:
            metrics = json.loads(eval_metrics)
        except (json.JSONDecodeError, TypeError):
            metrics = {}
    elif isinstance(eval_metrics, dict):
        metrics = eval_metrics
    else:
        metrics = {}

    # Download checkpoint
    checkpoint_path.download()

    # MLflow tracking URI from environment
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow.monitoring.svc.cluster.local",
    )
    mlflow.set_tracking_uri(tracking_uri)

    # Set experiment
    experiment_name = "llm-sft"
    try:
        experiment = mlflow.set_experiment(experiment_name)
        experiment_id = experiment.experiment_id
    except Exception as exc:
        logger.warning("Could not set MLflow experiment: %s — using default", exc)
        experiment_id = None

    # Start run
    try:
        run_ctx = mlflow.start_run(experiment_id=experiment_id)
    except Exception as exc:
        logger.warning(
            "Could not start MLflow run: %s — skipping registration",
            exc,
        )
        return str(checkpoint_path.path)

    with run_ctx as run:
        # Log parameters
        mlflow.log_param("base_model", base_model)
        mlflow.log_param("training_method", training_method)

        # Log metrics (filter to numeric values only)
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, float(metric_value))

        # Log checkpoint path as artifact reference
        mlflow.log_param("checkpoint_path", checkpoint_path.path)

        # Try to log model artifacts — fall back to simple tag if server
        # does not support newer pyfunc endpoints (e.g. logged-models API)
        try:
            mlflow.log_artifacts(checkpoint_path.path, artifact_path="model")
        except Exception as exc:
            logger.warning("Could not log model artifacts: %s", exc)

        # Register model — use the run as the model source
        client = MlflowClient()

        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri, model_name)
            version = result.version
        except Exception as exc:
            # If registration fails (e.g. server doesn't support it),
            # still succeed the task with a synthetic version
            logger.warning("Model registration failed: %s — returning run URI", exc)
            return f"runs:/{run.info.run_id}/model"

        # Add description
        client.update_model_version(
            name=model_name,
            version=version,
            description=f"Fine-tuned {base_model} using {training_method}",
        )

        # Add tags
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key="base_model",
            value=base_model,
        )
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key="method",
            value=training_method,
        )

        logger.info("Registered model %s version %s", model_name, version)
        return f"models:/{model_name}/{version}"
