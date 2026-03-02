"""
ingest.py — Task 1: Download the sql-create-context dataset from HuggingFace
and cache it as Parquet in S3.

Caches by HuggingFace dataset version hash so re-runs skip the download.
"""

import sys

from flytekit import Resources, task

sys.path.insert(0, "/app")
from config import CPU_IMAGE, HF_DATASET, S3_BUCKET, S3_RAW_DATA


@task(
    requests=Resources(cpu="2", mem="4Gi"),
    container_image=CPU_IMAGE,
    cache=True,
    cache_version="sql-create-context-v1.0",
    environment={
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        "HF_DATASETS_CACHE": "/tmp/hf_cache",
    },
)
def ingest_data() -> str:
    """
    Download b-mc2/sql-create-context from HuggingFace Hub.
    Saves as Parquet shards to S3. Returns the S3 path.
    """

    import boto3
    from datasets import load_dataset

    print(f"⬇️  Downloading dataset: {HF_DATASET}")
    dataset = load_dataset(HF_DATASET, split="train", trust_remote_code=True)

    print(f"✅ Loaded {len(dataset)} examples. Schema: {dataset.column_names}")

    # Convert to pandas and write shards to S3
    df = dataset.to_pandas()
    s3 = boto3.client("s3")

    shard_size = 10_000
    s3_path = S3_RAW_DATA
    bucket = S3_BUCKET
    prefix = "text2sql/raw"

    shard_count = 0
    for start in range(0, len(df), shard_size):
        shard_df = df.iloc[start : start + shard_size]
        parquet_bytes = shard_df.to_parquet(index=False)
        key = f"{prefix}/shard_{shard_count:04d}.parquet"
        s3.put_object(Bucket=bucket, Key=key, Body=parquet_bytes)
        print(
            f"  Uploaded shard {shard_count}: {len(shard_df)} rows → s3://{bucket}/{key}"
        )
        shard_count += 1

    print(f"✅ Ingestion complete: {shard_count} shards at {s3_path}")
    return s3_path
