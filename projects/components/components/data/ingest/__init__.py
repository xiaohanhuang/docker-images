"""
Data ingestion component — downloads a dataset from S3.

Image: data-cpu
"""

from flytekit import Resources, task
from flytekit.types.file import FlyteFile


@task(
    retries=3,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
    cache=True,
    cache_version="1.0",
)
def download_dataset(s3_uri: str) -> FlyteFile:
    """Download a dataset from S3 and return it as a :class:`flytekit.types.file.FlyteFile`.

    Args:
        s3_uri: S3 URI of the dataset (e.g. ``s3://bucket/path/data.parquet``).

    Returns:
        The downloaded file managed by Flyte's blob store.
    """
    import os

    import boto3

    # Validate S3 URI format
    s3_uri_stripped = s3_uri.strip()
    if not s3_uri_stripped.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI '{s3_uri}': URI must start with 's3://'.")
    without_scheme = s3_uri_stripped.removeprefix("s3://")
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(
            f"Invalid S3 URI '{s3_uri}': expected format 's3://<bucket>/<key>'. "
            "Both bucket and key must be non-empty."
        )

    # Preserve original filename/extension so downstream tasks can infer type
    filename = os.path.basename(key)
    if not filename:
        filename = "dataset"
    local_path = os.path.join("/tmp", filename)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    return FlyteFile(path=local_path)
