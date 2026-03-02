"""
pipeline.py — The Flyte @workflow that chains all Text2SQL tasks together.
Register with: pyflyte register --project ml-platform --domain development pipeline.py
"""

import sys

from flytekit import workflow

sys.path.insert(0, "/app")

from config import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR
from tasks.evaluate import evaluate
from tasks.ingest import ingest_data
from tasks.preprocess import preprocess
from tasks.register_model import register_model
from tasks.train import train


@workflow
def text2sql_pipeline(
    num_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
) -> str:
    """
    End-to-end Text-to-SQL fine-tuning pipeline:

    ingest_data  →  preprocess  →  train (GPU+Ray)
                                      ↓
                              evaluate  →  register_model
                                               ↓
                                       Teams notification
    """
    raw_path = ingest_data()
    processed_path = preprocess(raw_s3_path=raw_path)
    checkpoint_path = train(
        processed_s3_path=processed_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    metrics = evaluate(checkpoint_s3_path=checkpoint_path)
    model_uri = register_model(eval_metrics=metrics)
    return model_uri


if __name__ == "__main__":
    # Local smoke-test: run with pyflyte run pipeline.py text2sql_pipeline
    print(text2sql_pipeline())
