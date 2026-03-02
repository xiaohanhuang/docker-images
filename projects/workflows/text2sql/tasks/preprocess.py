"""
preprocess.py — Task 2: Tokenize and split the raw dataset into
train / validation / test Arrow shards stored in S3.

Input prompt format:
  "translate English to SQL: {question} </s> Tables: {context}"
"""

import sys

from flytekit import Resources, task

sys.path.insert(0, "/app")
from config import (
    BASE_MODEL,
    CPU_IMAGE,
    MAX_INPUT_LEN,
    MAX_TARGET_LEN,
    S3_BUCKET,
    S3_PROCESSED,
)


@task(
    requests=Resources(cpu="4", mem="8Gi"),
    container_image=CPU_IMAGE,
    cache=True,
    cache_version="preprocess-v1.0",
    environment={
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        "HF_DATASETS_CACHE": "/tmp/hf_cache",
    },
)
def preprocess(raw_s3_path: str) -> str:
    """
    1. Read Parquet shards from S3
    2. Format prompts
    3. Tokenize with CodeT5+ tokenizer
    4. Split 80/10/10 train/val/test
    5. Save Arrow datasets to S3
    Returns the S3 base path of the processed dataset.
    """
    import io
    import os
    import shutil
    import tempfile

    import boto3
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoTokenizer

    # ── Load raw shards ───────────────────────────────────────────────
    s3 = boto3.client("s3")
    bucket = S3_BUCKET
    prefix = "text2sql/raw"

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    keys = [
        obj["Key"]
        for page in pages
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]

    print(f"📂 Loading {len(keys)} Parquet shards...")
    dfs = []
    for key in sorted(keys):
        obj = s3.get_object(Bucket=bucket, Key=key)
        dfs.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))
    df = pd.concat(dfs, ignore_index=True)
    print(f"✅ Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    # ── Format prompts ────────────────────────────────────────────────
    def make_prompt(row):
        return (
            f"translate English to SQL: {row['question']} </s> Tables: {row['context']}"
        )

    df["input_text"] = df.apply(make_prompt, axis=1)
    df["target_text"] = df["answer"]

    # ── Tokenise ──────────────────────────────────────────────────────
    print(f"🔤 Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    hf_dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = hf_dataset.map(
        tokenize,
        batched=True,
        batch_size=512,
        remove_columns=["input_text", "target_text"],
        desc="Tokenizing",
    )

    # ── Split ─────────────────────────────────────────────────────────
    splits = tokenized.train_test_split(test_size=0.2, seed=42)
    val_test = splits["test"].train_test_split(test_size=0.5, seed=42)

    train_ds = splits["train"]
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    print(f"Split — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # Also keep raw text for evaluation (exact match / SQL execution)
    test_df = df.iloc[list(range(len(df) - len(val_ds) - len(test_ds), len(df)))].tail(
        len(test_ds)
    )
    raw_test_key = "text2sql/processed/test_raw.parquet"
    s3.put_object(Bucket=bucket, Key=raw_test_key, Body=test_df.to_parquet(index=False))

    # ── Upload Arrow shards to S3 ─────────────────────────────────────
    tmpdir = tempfile.mkdtemp()
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        local_path = os.path.join(tmpdir, name)
        ds.save_to_disk(local_path)
        for fname in os.listdir(local_path):
            fpath = os.path.join(local_path, fname)
            if os.path.isfile(fpath):
                with open(fpath, "rb") as f:
                    s3.put_object(
                        Bucket=bucket,
                        Key=f"text2sql/processed/{name}/{fname}",
                        Body=f.read(),
                    )
        print(f"  Uploaded {name} split to s3://{bucket}/text2sql/processed/{name}/")

    shutil.rmtree(tmpdir)
    print(f"✅ Preprocessing complete → {S3_PROCESSED}")
    return S3_PROCESSED
