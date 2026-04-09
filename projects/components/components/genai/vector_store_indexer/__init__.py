"""
GenAI — Index embeddings into a vector database.

Supports pgvector (PostgreSQL), FAISS (S3), and ChromaDB.
Appends vectors and metadata into the target collection (append-only).

Image: genai-gpu
"""

from ._task import IndexResult, _safe_name, index_embeddings  # noqa: F401

__all__ = ["index_embeddings", "IndexResult"]
