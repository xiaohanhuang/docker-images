"""Vector store backend implementations: pgvector, FAISS, ChromaDB."""

from typing import Any, Dict, List


def _download_from_s3(s3_client: Any, s3_path: str) -> str:
    """Download a file from S3 and return the local path."""
    import os
    import tempfile

    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path '{s3_path}': must start with 's3://'")

    without_scheme = s3_path.removeprefix("s3://")
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 path '{s3_path}': expected format 's3://<bucket>/<key>'")

    filename = os.path.basename(key)
    if not filename:
        raise ValueError(f"Invalid S3 path '{s3_path}': key must not end with '/'")
    local_path = os.path.join(tempfile.gettempdir(), filename)
    s3_client.download_file(bucket, key, local_path)
    return local_path


def _index_pgvector(
    embeddings: Any,
    metadata_list: List[Dict[str, Any]],
    collection_name: str,
    embedding_dim: int,
    num_vectors: int,
    index_type: str,
    connection_string: str,
) -> Dict[str, Any]:
    """Index vectors into pgvector (PostgreSQL)."""
    import json

    import psycopg2
    from pgvector.psycopg2 import register_vector
    from psycopg2 import sql
    from psycopg2.extras import execute_values

    conn = psycopg2.connect(connection_string)
    try:
        register_vector(conn)
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()

        cur.execute(sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        id SERIAL PRIMARY KEY,
                        embedding vector({}),
                        metadata JSONB
                    )
                """).format(sql.Identifier(collection_name), sql.Literal(embedding_dim)))
        conn.commit()

        def _row_iter():
            for embedding, meta in zip(embeddings, metadata_list):
                yield (embedding.tolist(), json.dumps(meta))

        execute_values(
            cur,
            sql.SQL("INSERT INTO {} (embedding, metadata) VALUES %s").format(
                sql.Identifier(collection_name)
            ),
            _row_iter(),
            template="(%s::vector, %s)",
            page_size=500,
        )
        conn.commit()

        index_name = f"{collection_name}_{index_type}_idx"
        if index_type == "ivfflat":
            lists = max(1, min(num_vectors // 10, 100))
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} "
                    "ON {} USING ivfflat (embedding vector_cosine_ops) "
                    "WITH (lists = {})"
                ).format(
                    sql.Identifier(index_name),
                    sql.Identifier(collection_name),
                    sql.Literal(lists),
                )
            )
            index_config = {"type": "ivfflat", "lists": lists}
        elif index_type == "hnsw":
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} "
                    "ON {} USING hnsw (embedding vector_cosine_ops)"
                ).format(
                    sql.Identifier(index_name),
                    sql.Identifier(collection_name),
                )
            )
            index_config: Dict[str, Any] = {"type": "hnsw"}
        else:
            raise ValueError(f"Unsupported index_type '{index_type}' for pgvector")

        conn.commit()
        cur.close()
    finally:
        conn.close()

    print(f"✅ Indexed {num_vectors} vectors into pgvector table '{collection_name}'")
    return index_config


def _index_faiss(
    embeddings: Any,
    metadata_list: List[Dict[str, Any]],
    collection_name: str,
    embedding_dim: int,
    num_vectors: int,
    embeddings_path: str,
    s3_client: Any,
) -> Dict[str, Any]:
    """Index vectors into FAISS and upload to S3."""
    import json
    import os
    import tempfile

    import faiss
    import numpy as np

    from . import _safe_name

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.astype(np.float32))

    safe_col = _safe_name(collection_name)

    index_local_path = os.path.join(tempfile.gettempdir(), f"{safe_col}.index")
    metadata_local_path = os.path.join(tempfile.gettempdir(), f"{safe_col}_metadata.jsonl")

    faiss.write_index(index, index_local_path)

    with open(metadata_local_path, "w") as f:
        for meta in metadata_list:
            f.write(json.dumps(meta) + "\n")

    embeddings_bucket = embeddings_path.split("/")[2]
    index_s3_key = f"vector_indexes/{collection_name}.index"
    metadata_s3_key = f"vector_indexes/{collection_name}_metadata.jsonl"

    s3_client.upload_file(index_local_path, embeddings_bucket, index_s3_key)
    s3_client.upload_file(metadata_local_path, embeddings_bucket, metadata_s3_key)

    index_config = {
        "type": "faiss_flat_l2",
        "index_path": f"s3://{embeddings_bucket}/{index_s3_key}",
        "metadata_path": f"s3://{embeddings_bucket}/{metadata_s3_key}",
    }

    print(
        f"✅ Indexed {num_vectors} vectors into FAISS " f"(s3://{embeddings_bucket}/{index_s3_key})"
    )
    return index_config


def _index_chromadb(
    embeddings: Any,
    metadata_list: List[Dict[str, Any]],
    collection_name: str,
    num_vectors: int,
    embeddings_path: str,
    s3_client: Any,
) -> Dict[str, Any]:
    """Index vectors into ChromaDB and upload archive to S3."""
    import os
    import shutil
    import tempfile
    import uuid

    import chromadb
    from chromadb.config import Settings

    from . import _safe_name

    db_path = tempfile.mkdtemp(prefix="chromadb_")
    try:
        client = chromadb.Client(
            Settings(
                persist_directory=db_path,
                anonymized_telemetry=False,
            )
        )

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [str(uuid.uuid4()) for _ in range(num_vectors)]
        documents = [meta.get("text", "") for meta in metadata_list]
        metadatas = metadata_list

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

        safe_col = _safe_name(collection_name)
        archive_path = os.path.join(tempfile.gettempdir(), f"{safe_col}_chromadb")
        shutil.make_archive(archive_path, "zip", db_path)
    finally:
        shutil.rmtree(db_path, ignore_errors=True)

    embeddings_bucket = embeddings_path.split("/")[2]
    archive_s3_key = f"vector_indexes/{collection_name}_chromadb.zip"
    s3_client.upload_file(f"{archive_path}.zip", embeddings_bucket, archive_s3_key)

    index_config = {
        "type": "chromadb",
        "archive_path": f"s3://{embeddings_bucket}/{archive_s3_key}",
    }

    print(
        f"✅ Indexed {num_vectors} vectors into ChromaDB "
        f"(s3://{embeddings_bucket}/{archive_s3_key})"
    )
    return index_config
