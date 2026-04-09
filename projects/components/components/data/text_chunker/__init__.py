"""
Document chunking component — split documents into overlapping chunks for RAG indexing.

Image: data-cpu
"""

from ._strategies import (  # noqa: F401 — re-exported for tests
    _chunk_by_characters,
    _chunk_by_sentences,
    _chunk_by_tokens,
    _chunk_recursive,
    _chunk_text,
    _load_document,
)
from ._task import chunk_documents  # noqa: F401

__all__ = ["chunk_documents"]
