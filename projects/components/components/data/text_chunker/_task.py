"""Flyte task definition for text_chunker component."""

from typing import Tuple

from flytekit import Resources, task

from ._strategies import _chunk_text, _load_document


@task(
    retries=2,
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="4", mem="16Gi"),
    cache=True,
    cache_version="1.0",
)
def chunk_documents(
    s3_input_path: str,
    s3_output_path: str,
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Tuple[str, int, float]:
    """Split documents into overlapping chunks for RAG indexing.

    Supports character-based, token-based, sentence-based, and recursive splitting strategies.

    Args:
        s3_input_path: S3 URI to documents (txt, pdf, md, html).
            Can be a single file (e.g., ``s3://bucket/doc.txt``) or a prefix
            for multiple files (e.g., ``s3://bucket/docs/``).
        s3_output_path: S3 URI where chunked output will be saved as JSONL.
        strategy: Chunking strategy. One of:
            - ``"character"``: Fixed character-length chunks with overlap
            - ``"token"``: Token-based chunking using tiktoken
            - ``"sentence"``: Sentence-boundary-aware splitting
            - ``"recursive"``: Hierarchical splitting (paragraphs → sentences → characters)
        chunk_size: Target chunk size in characters (for character/recursive strategies)
            or tokens (for token strategy).
        chunk_overlap: Number of characters or tokens to overlap between chunks.

    Returns:
        Tuple of (s3_output_path, num_chunks, avg_chunk_size):
            - s3_output_path: S3 path to the chunked output JSONL file
            - num_chunks: Total number of chunks created
            - avg_chunk_size: Average chunk size in characters
    """
    import json
    import os

    import boto3

    # Validate inputs
    if strategy not in ["character", "token", "sentence", "recursive"]:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one of: "
            "character, token, sentence, recursive"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than " f"chunk_size ({chunk_size})"
        )

    # Parse S3 URIs
    def parse_s3_uri(uri):
        uri = uri.strip()
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI '{uri}': must start with 's3://'")
        without_scheme = uri.removeprefix("s3://")
        bucket, _, key = without_scheme.partition("/")
        if not bucket:
            raise ValueError(f"Invalid S3 URI '{uri}': bucket is empty")
        return bucket, key

    input_bucket, input_key = parse_s3_uri(s3_input_path)
    output_bucket, output_key = parse_s3_uri(s3_output_path)
    if not output_key:
        raise ValueError(f"Invalid S3 URI '{s3_output_path}': output key must not be empty")
    if output_key.endswith("/"):
        raise ValueError(f"Invalid S3 URI '{s3_output_path}': output key must not end with '/'")

    s3 = boto3.client("s3")

    # Download input file(s)
    input_files = []
    os.makedirs("/tmp/inputs", exist_ok=True)

    # Check if input is a single file or a prefix
    try:
        s3.head_object(Bucket=input_bucket, Key=input_key)
        # It's a single file
        local_path = os.path.join("/tmp/inputs", os.path.basename(input_key) or "doc")
        s3.download_file(input_bucket, input_key, local_path)
        input_files.append((local_path, input_key))
    except s3.exceptions.ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code not in ("404", "NoSuchKey", "NotFound"):
            raise
        # It's a prefix — paginate to handle >1000 objects
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=input_bucket, Prefix=input_key)
        found_any = False
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                found_any = True
                import hashlib

                ext = os.path.splitext(key)[1]
                filename = hashlib.md5(key.encode()).hexdigest()[:16] + ext
                local_path = os.path.join("/tmp/inputs", filename)
                s3.download_file(input_bucket, key, local_path)
                input_files.append((local_path, key))
        if not found_any:
            raise ValueError(
                f"No files found at S3 path '{s3_input_path}'. "
                "Ensure the path exists and contains files."
            )

    if not input_files:
        raise ValueError(f"No files found at S3 path '{s3_input_path}'")

    # Stream-process: load → chunk → write directly; avoids holding all chunks in RAM
    output_path = "/tmp/chunks.jsonl"
    num_chunks = 0
    total_chars = 0
    with open(output_path, "w") as out_f:
        for local_path, s3_key in input_files:
            text = _load_document(local_path)
            source_uri = f"s3://{input_bucket}/{s3_key}"
            chunks = _chunk_text(
                text,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for i, chunk_str in enumerate(chunks):
                chunk = {
                    "chunk_id": num_chunks,
                    "text": chunk_str,
                    "metadata": {
                        "source_file": source_uri,
                        "chunk_index": i,
                        "strategy": strategy,
                    },
                }
                out_f.write(json.dumps(chunk) + "\n")
                num_chunks += 1
                total_chars += len(chunk_str)

    avg_chunk_size = total_chars / num_chunks if num_chunks > 0 else 0.0

    # Upload to S3
    s3.upload_file(output_path, output_bucket, output_key)
    final_s3_path = f"s3://{output_bucket}/{output_key}"

    return final_s3_path, num_chunks, avg_chunk_size
