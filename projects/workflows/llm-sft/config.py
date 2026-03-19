"""
config.py — Centralized configuration for LLM-SFT workflow.
All S3 paths, model names, and cluster settings live here.
"""

import os

# ── AWS ───────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET = os.getenv("S3_BUCKET", "ml-platform-data-ml-platform-eks-805673386114")
ECR_REGISTRY = os.getenv("ECR_REGISTRY", "805673386114.dkr.ecr.us-west-2.amazonaws.com")

# ── S3 paths (all relative to S3_BUCKET) ─────────────────────────────
S3_RAW_DATA = f"s3://{S3_BUCKET}/llm-sft/raw"
S3_TOKENIZED = f"s3://{S3_BUCKET}/llm-sft/tokenized"
S3_CHECKPOINTS = f"s3://{S3_BUCKET}/llm-sft/checkpoints"

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_DATASET = "tatsu-lab/alpaca"
DEFAULT_METHOD = "lora"
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32

# ── MLflow ────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow.monitoring.svc.cluster.local",
)
MLFLOW_EXPERIMENT = "llm-sft"

# ── Flyte project/domain ──────────────────────────────────────────────
FLYTE_PROJECT = os.getenv("FLYTE_PROJECT", "ml-platform")
FLYTE_DOMAIN = os.getenv("FLYTE_DOMAIN", "development")
