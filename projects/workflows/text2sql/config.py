"""
config.py — Centralized configuration for the Text2SQL ML workflow.
All S3 paths, model names, and cluster settings live here.
"""

import os

# ── AWS ───────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET = os.getenv("S3_BUCKET", "ml-platform-data-ml-platform-eks-805673386114")
ECR_REGISTRY = os.getenv("ECR_REGISTRY", "805673386114.dkr.ecr.us-west-2.amazonaws.com")

# ── S3 paths (all relative to S3_BUCKET) ─────────────────────────────
S3_RAW_DATA = f"s3://{S3_BUCKET}/text2sql/raw"
S3_PROCESSED = f"s3://{S3_BUCKET}/text2sql/processed"
S3_CHECKPOINTS = f"s3://{S3_BUCKET}/text2sql/checkpoints"
S3_EXPORTS = f"s3://{S3_BUCKET}/text2sql/exports"

# ── Model ─────────────────────────────────────────────────────────────
BASE_MODEL = os.getenv("BASE_MODEL", "Salesforce/codet5p-220m")
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128

# ── HuggingFace Dataset ───────────────────────────────────────────────
HF_DATASET = "b-mc2/sql-create-context"
HF_DATASET_SPLIT = "train"  # only split available; we do our own train/val split

# ── Training defaults ─────────────────────────────────────────────────
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 5e-4
DEFAULT_WARMUP_STEPS = 200
DEFAULT_EVAL_STEPS = 200

# ── MLflow ────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow.monitoring.svc.cluster.local:5000",
)
MLFLOW_EXPERIMENT = "text2sql"
MLFLOW_MODEL_NAME = "text2sql"

# ── Flyte project/domain ──────────────────────────────────────────────
FLYTE_PROJECT = os.getenv("FLYTE_PROJECT", "ml-platform")
FLYTE_DOMAIN = os.getenv("FLYTE_DOMAIN", "development")

# ── ECR image URIs (built by Makefile) ───────────────────────────────
CPU_IMAGE = f"{ECR_REGISTRY}/ml-platform/workflow-cpu:latest"
GPU_IMAGE = f"{ECR_REGISTRY}/ml-platform/workflow-gpu:latest"
