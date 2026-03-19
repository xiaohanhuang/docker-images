"""
Example workflow demonstrating the text_chunker component for RAG indexing.

This workflow:
1. Downloads sample documents from S3
2. Chunks them using the text_chunker component
3. Generates embeddings (optional, shown as placeholder)
"""

import sys
from pathlib import Path

from flytekit import workflow

# Import from the components directory (repo-relative: works locally and in containers)
_COMPONENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_COMPONENTS_DIR))

from data.text_chunker import chunk_documents  # noqa: E402


@workflow
def rag_document_processing_workflow(
    s3_input_path: str = "s3://my-bucket/documents/sample.txt",
    s3_output_path: str = "s3://my-bucket/processed/chunks.jsonl",
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> tuple[str, int, float]:
    """RAG document processing workflow.

    This workflow demonstrates how to use the text_chunker component to prepare
    documents for RAG (Retrieval-Augmented Generation) indexing.

    Args:
        s3_input_path: S3 path to input documents (single file or directory prefix)
        s3_output_path: S3 path where chunked output will be saved
        strategy: Chunking strategy (character, token, sentence, recursive)
        chunk_size: Target chunk size in characters or tokens
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        Tuple of (s3_output_path, num_chunks, avg_chunk_size)
    """
    # Chunk the documents
    s3_path, num_chunks, avg_chunk_size = chunk_documents(
        s3_input_path=s3_input_path,
        s3_output_path=s3_output_path,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return s3_path, num_chunks, avg_chunk_size


# Example of a more complex RAG pipeline workflow (pseudo-code for demonstration)
@workflow
def full_rag_pipeline_workflow(
    s3_input_path: str = "s3://my-bucket/documents/",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> str:
    """Full RAG pipeline: chunk documents, generate embeddings, index for search.

    Args:
        s3_input_path: S3 path to input documents
        embedding_model: Model to use for generating embeddings

    Returns:
        Status message indicating completion
    """
    # Step 1: Chunk documents
    chunks_path, num_chunks, avg_size = chunk_documents(
        s3_input_path=s3_input_path,
        s3_output_path="s3://my-bucket/processed/chunks.jsonl",
        strategy="recursive",
        chunk_size=512,
        chunk_overlap=50,
    )

    # Step 2: Generate embeddings (would use genai/embeddings component)
    # embeddings_path = generate_embeddings(
    #     texts=chunks_path,
    #     model_name=embedding_model,
    # )

    # Step 3: Index in vector database (future component)
    # index_path = index_embeddings(
    #     embeddings=embeddings_path,
    #     texts=chunks_path,
    # )

    return f"Processed {num_chunks} chunks with average size {avg_size:.1f} chars"
