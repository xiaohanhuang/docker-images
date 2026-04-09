"""Flyte task definition for vector_store_indexer component."""

import re
from typing import Any, Dict, NamedTuple, Optional

from flytekit import Resources, task


def _safe_name(name: str) -> str:
    """Strip path-unsafe characters from a name for use in local file paths."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


class IndexResult(NamedTuple):
    """Result of vector indexing operation."""

    collection_name: str
    num_vectors: int
    index_config: Dict[str, Any]


@task(
    retries=2,
    requests=Resources(cpu="4", mem="16Gi"),
    limits=Resources(cpu="8", mem="32Gi"),
    cache=False,
)
def index_embeddings(
    embeddings_path: str,
    metadata_path: str,
    collection_name: str,
    backend: str = "pgvector",
    connection_string: Optional[str] = None,
    index_type: str = "ivfflat",
) -> IndexResult:
    """Index embeddings into a vector database.

    Args:
        embeddings_path: S3 path to embeddings (NumPy .npy file).
        metadata_path: S3 path to metadata (JSONL file with text, source, etc.).
        collection_name: Name of the collection/table to create or append to.
        backend: Vector DB backend. Options: "pgvector", "faiss", "chromadb".
        connection_string: Database connection string (required for pgvector).
            Format: "postgresql://user:password@host:port/dbname"
        index_type: Index type for pgvector. Options: "ivfflat", "hnsw".
            Ignored for FAISS and ChromaDB.

    Returns:
        IndexResult containing collection_name, num_vectors, and index_config.

    Raises:
        ValueError: If backend is unsupported or required params are missing.
    """
    import json

    import boto3
    import numpy as np

    from ._backends import _download_from_s3, _index_chromadb, _index_faiss, _index_pgvector

    # Validate backend
    valid_backends = ["pgvector", "faiss", "chromadb"]
    if backend not in valid_backends:
        raise ValueError(f"Unsupported backend '{backend}'. Valid options: {valid_backends}")

    # Download embeddings from S3
    s3 = boto3.client("s3")

    embeddings_local = _download_from_s3(s3, embeddings_path)
    metadata_local = _download_from_s3(s3, metadata_path)

    # Load embeddings and metadata
    embeddings = np.load(embeddings_local)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 2D embeddings array, got shape {embeddings.shape}")
    num_vectors = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]

    metadata_list = []
    with open(metadata_local) as f:
        for line in f:
            metadata_list.append(json.loads(line.strip()))

    if len(metadata_list) != num_vectors:
        raise ValueError(
            f"Mismatch: {num_vectors} embeddings but {len(metadata_list)} metadata entries"
        )

    print(f"Loaded {num_vectors} vectors of dimension {embedding_dim}")
    print(f"Indexing into {backend} backend...")

    if backend == "pgvector":
        if not connection_string:
            raise ValueError("connection_string is required for pgvector backend")
        index_config = _index_pgvector(
            embeddings,
            metadata_list,
            collection_name,
            embedding_dim,
            num_vectors,
            index_type,
            connection_string,
        )
    elif backend == "faiss":
        index_config = _index_faiss(
            embeddings,
            metadata_list,
            collection_name,
            embedding_dim,
            num_vectors,
            embeddings_path,
            s3,
        )
    elif backend == "chromadb":
        index_config = _index_chromadb(
            embeddings,
            metadata_list,
            collection_name,
            num_vectors,
            embeddings_path,
            s3,
        )
    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return IndexResult(
        collection_name=collection_name,
        num_vectors=num_vectors,
        index_config=index_config,
    )
