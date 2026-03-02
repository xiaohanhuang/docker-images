"""
evaluate.py — Task 4: Evaluate the fine-tuned model on the held-out test set.

Computes:
  1. Exact Match Accuracy  — predictions must match gold SQL exactly
  2. Execution Accuracy    — both queries must return the same rows in SQLite
  3. Logs an examples table to MLflow (question / predicted / gold)
"""

import sys

from flytekit import Resources, task

sys.path.insert(0, "/app")
from config import (
    CPU_IMAGE,
    MAX_TARGET_LEN,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    S3_BUCKET,
)


@task(
    requests=Resources(cpu="4", mem="8Gi"),
    container_image=CPU_IMAGE,
    environment={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
)
def evaluate(checkpoint_s3_path: str) -> dict:
    """
    Load checkpoint from S3, generate SQL predictions on test set,
    compute exact match and execution accuracy, log all to MLflow.
    Returns a metrics dict.
    """
    import io
    import os
    import re
    import sqlite3
    import tempfile

    import boto3
    import mlflow
    import pandas as pd
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    s3 = boto3.client("s3")
    bucket = S3_BUCKET

    # ── Extract mlflow run_id from checkpoint path ────────────────────
    # path format: s3://<bucket>/text2sql/checkpoints/<run_id>
    run_id = checkpoint_s3_path.rstrip("/").split("/")[-1]

    # ── Download checkpoint ───────────────────────────────────────────
    ckpt_dir = tempfile.mkdtemp()
    prefix = f"text2sql/checkpoints/{run_id}/"
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key.split("/")[-1]
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            with open(os.path.join(ckpt_dir, fname), "wb") as f:
                f.write(body)
    print(f"✅ Downloaded checkpoint to {ckpt_dir}")

    # ── Load model ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir)

    sql_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_TARGET_LEN,
        device="cpu",
    )

    # ── Load raw test data ────────────────────────────────────────────
    obj = s3.get_object(Bucket=bucket, Key="text2sql/processed/test_raw.parquet")
    test_df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    # Limit to 500 examples for speed
    test_df = test_df.head(500).reset_index(drop=True)
    print(f"📊 Evaluating on {len(test_df)} examples...")

    # ── Generate predictions ──────────────────────────────────────────
    prompts = [
        f"translate English to SQL: {row['question']} </s> Tables: {row['context']}"
        for _, row in test_df.iterrows()
    ]

    outputs = sql_pipe(prompts, batch_size=16)
    predictions = [o[0]["generated_text"].strip() for o in outputs]
    golds = test_df["answer"].tolist()

    # ── Exact match ───────────────────────────────────────────────────
    def normalize(s):
        return re.sub(r"\s+", " ", s.lower().strip())

    exact_matches = [normalize(p) == normalize(g) for p, g in zip(predictions, golds)]
    exact_match_acc = sum(exact_matches) / len(exact_matches)

    # ── Execution accuracy ────────────────────────────────────────────
    def execute_sql(sql: str, context: str) -> list:
        """Create in-memory SQLite DB from CREATE TABLE context and run SQL."""
        try:
            conn = sqlite3.connect(":memory:")
            # Execute CREATE TABLE statements from context
            for stmt in context.split(";"):
                stmt = stmt.strip()
                if stmt.upper().startswith("CREATE"):
                    conn.execute(stmt + ";")
            conn.commit()
            cursor = conn.execute(sql)
            return cursor.fetchall()
        except Exception:
            return None

    exec_matches = 0
    for pred, gold, (_, row) in zip(predictions, golds, test_df.iterrows()):
        pred_rows = execute_sql(pred, row["context"])
        gold_rows = execute_sql(gold, row["context"])
        if pred_rows is not None and gold_rows is not None and pred_rows == gold_rows:
            exec_matches += 1
    exec_acc = exec_matches / len(test_df)

    print(f"✅ Exact Match Accuracy:  {exact_match_acc:.4f}")
    print(f"✅ Execution Accuracy:    {exec_acc:.4f}")

    # ── Log to MLflow ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "test_exact_match": exact_match_acc,
                "test_execution_accuracy": exec_acc,
            }
        )

        # Log examples table (20 rows)
        examples = pd.DataFrame(
            {
                "question": test_df["question"].head(20),
                "predicted": predictions[:20],
                "gold": golds[:20],
                "exact_match": exact_matches[:20],
            }
        )
        mlflow.log_table(data=examples, artifact_file="eval_examples.json")

        # Log confusion: exact match vs exec accuracy
        mlflow.log_text(
            f"Exact Match: {exact_match_acc:.4f}\nExecution Accuracy: {exec_acc:.4f}",
            "eval_summary.txt",
        )

    return {
        "exact_match_accuracy": exact_match_acc,
        "execution_accuracy": exec_acc,
        "num_examples": len(test_df),
        "run_id": run_id,
    }
