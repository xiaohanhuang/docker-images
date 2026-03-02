"""
register_model.py — Task 5: Register the evaluated model in the MLflow
Model Registry and send a Microsoft Teams notification card.
"""

import os
import sys

from flytekit import Resources, task

sys.path.insert(0, "/app")

from config import (
    CPU_IMAGE,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "")


@task(
    requests=Resources(cpu="1", mem="2Gi"),
    container_image=CPU_IMAGE,
    environment={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
)
def register_model(eval_metrics: dict) -> str:
    """
    1. Register the model from the MLflow run into the Model Registry.
    2. Transition to 'Staging'.
    3. Send Microsoft Teams Adaptive Card notification.
    Returns the registered model version URI.
    """

    import mlflow
    from mlflow.tracking import MlflowClient

    run_id = eval_metrics["run_id"]
    exact_acc = eval_metrics["exact_match_accuracy"]
    exec_acc = eval_metrics["execution_accuracy"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # ── Register model ────────────────────────────────────────────────
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MLFLOW_MODEL_NAME,
    )
    version = registered.version
    print(f"✅ Registered model '{MLFLOW_MODEL_NAME}' version {version}")

    # ── Transition to Staging ─────────────────────────────────────────
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print("✅ Model transitioned to Staging")

    # ── Tag the run ───────────────────────────────────────────────────
    client.set_registered_model_tag(MLFLOW_MODEL_NAME, "latest_version", str(version))

    model_version_uri = f"models:/{MLFLOW_MODEL_NAME}/{version}"

    # ── Teams notification ────────────────────────────────────────────
    if TEAMS_WEBHOOK_URL:
        _send_teams_card(
            webhook_url=TEAMS_WEBHOOK_URL,
            exact_acc=exact_acc,
            exec_acc=exec_acc,
            version=version,
            run_id=run_id,
        )
    else:
        print("ℹ️  TEAMS_WEBHOOK_URL not set, skipping notification")

    return model_version_uri


def _send_teams_card(
    webhook_url: str, exact_acc: float, exec_acc: float, version: int, run_id: str
):
    """Send a rich Microsoft Teams Adaptive Card via webhook."""

    import requests

    status_color = "Good" if exact_acc >= 0.5 else "Warning"
    exact_pct = f"{exact_acc * 100:.1f}%"
    exec_pct = f"{exec_acc * 100:.1f}%"

    card = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.3",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": "🤖 ML Platform — Training Complete",
                            "weight": "Bolder",
                            "size": "Large",
                            "color": status_color,
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {"title": "Workflow", "value": "text2sql-pipeline"},
                                {"title": "Status", "value": "✅ SUCCEEDED"},
                                {
                                    "title": "Model",
                                    "value": f"{MLFLOW_MODEL_NAME} v{version}  \u2192  Staging",
                                },
                                {"title": "Run ID", "value": run_id[:8] + "..."},
                            ],
                        },
                        {
                            "type": "TextBlock",
                            "text": "📊 Evaluation Results",
                            "weight": "Bolder",
                            "spacing": "Medium",
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {"title": "Exact Match Accuracy", "value": exact_pct},
                                {"title": "Execution Accuracy", "value": exec_pct},
                            ],
                        },
                    ],
                    "actions": [
                        {
                            "type": "Action.OpenUrl",
                            "title": "🔗 View in MLflow",
                            "url": f"{MLFLOW_TRACKING_URI}/#/experiments/{MLFLOW_EXPERIMENT}",
                        },
                    ],
                },
            }
        ],
    }

    try:
        resp = requests.post(webhook_url, json=card, timeout=10)
        if resp.status_code == 200:
            print("✅ Teams notification sent")
        else:
            print(f"⚠️  Teams webhook returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"⚠️  Teams notification failed: {e}")
