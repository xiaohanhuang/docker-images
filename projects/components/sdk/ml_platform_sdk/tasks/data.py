from flytekit import Resources, task
from flytekit.types.file import FlyteFile


@task(
    retries=3,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
    cache=True,
    cache_version="1.0",
)
def download_dataset(s3_path: str) -> FlyteFile:
    """
    Downloads a dataset from S3 to a local path and returns it as a FlyteFile.
    In a real scenario, this might just return the S3 path for Ray to stream,
    but here we demonstrate a basic download task.
    """
    print(f"Downloading dataset from {s3_path}...")
    # In a real implementation, use boto3 to download.
    # For now, we simulate a file.

    # local_path = "/tmp/dataset.parquet"
    # s3_client.download_file(bucket, key, local_path)

    # Returning the input path as a FlyteFile so Flyte manages the blob passing
    return FlyteFile(path=s3_path)
