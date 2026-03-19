"""Shared helpers for distributed RLHF trainer: S3 download, MLflow lifecycle."""

import os
import tempfile
from typing import Dict, Optional

# Map user-friendly algorithm names to internal identifiers.
ALGORITHM_MAP = {
    "ppo": "ppo",
    "reinforce": "reinforce",
    "reinforce_baseline": "reinforce_baseline",
    "grpo": "grpo",
    "rloo": "rloo",
}


def download_s3(uri: str, label: str) -> str:
    """Download an S3 URI to a local temp directory. Returns the path
    containing ``config.json`` / ``adapter_config.json``."""
    import s3fs

    if not uri.startswith("s3://"):
        return uri
    s3 = s3fs.S3FileSystem()
    local = tempfile.mkdtemp(prefix=f"openrlhf-{label}-")
    print(f"[openrlhf] Downloading {label} from {uri} to {local}")
    s3.get(uri.rstrip("/"), local, recursive=True)
    for root, _, files in os.walk(local):
        if "config.json" in files or "adapter_config.json" in files:
            return root
    return local


def start_mlflow(
    algorithm: str,
    framework: str,
    params: Dict,
    experiment_name: Optional[str],
):
    """Start an MLflow run and log params. Returns (run_id, is_available)."""
    try:
        import mlflow

        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://mlflow.monitoring.svc.cluster.local",
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name or "openrlhf-distributed")
        run = mlflow.start_run(run_name=f"openrlhf-{algorithm}-{framework}")
        mlflow.log_params(params)
        return run.info.run_id, True
    except Exception as exc:
        print(f"[openrlhf] MLflow unavailable: {exc}")
        return "no-mlflow", False


def end_mlflow(mlflow_available: bool):
    if mlflow_available:
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            pass
