# Document Chunking Component (`text_chunker`)

## Overview

The `text_chunker` component splits documents into overlapping chunks optimized for Retrieval-Augmented Generation (RAG) indexing. It supports multiple chunking strategies and document formats, making it ideal for preparing text data for vector databases and semantic search.

## Component Details

- **Category**: Data Processing / GenAI
- **Image**: `data-cpu`
- **Location**: `projects/components/components/data/text_chunker.py`
- **Task Name**: `chunk_documents`

## Features

### Chunking Strategies

1. **Character-based (`character`)**:
   - Fixed character-length chunks with configurable overlap
   - Fast and predictable chunk sizes
   - Best for: Simple use cases, consistent chunk sizes

2. **Token-based (`token`)**:
   - Uses tiktoken (GPT tokenizer) for accurate token counting
   - Ensures chunks fit within model token limits
   - Best for: LLM applications with strict token limits

3. **Sentence-based (`sentence`)**:
   - Respects sentence boundaries using NLTK
   - Groups sentences to approximate target size
   - Best for: Maintaining semantic coherence, question-answering

4. **Recursive (`recursive`)** (default):
   - Hierarchical splitting: paragraphs → sentences → characters
   - Preserves document structure when possible
   - Best for: Long documents, maintaining context

### Supported Document Formats

| Format | Extensions | Processing |
|--------|-----------|------------|
| Plain Text | `.txt` | Raw text extraction |
| Markdown | `.md` | Raw text extraction |
| HTML | `.html`, `.htm` | Strips scripts/styles, extracts text |
| PDF | `.pdf` | Multi-page text extraction |

## API Reference

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `s3_input_path` | `str` | Required | S3 URI to documents (single file or directory prefix) |
| `s3_output_path` | `str` | Required | S3 URI for output JSONL file |
| `strategy` | `str` | `"recursive"` | Chunking strategy: `character`, `token`, `sentence`, or `recursive` |
| `chunk_size` | `int` | `512` | Target chunk size (characters or tokens) |
| `chunk_overlap` | `int` | `50` | Overlap between consecutive chunks |

### Output

Returns a tuple: `(s3_path: str, num_chunks: int, avg_chunk_size: float)`

- `s3_path`: S3 URI of the output JSONL file
- `num_chunks`: Total number of chunks created
- `avg_chunk_size`: Average chunk size in characters

### Output Format (JSONL)

Each line in the output file is a JSON object:

```json
{
  "chunk_id": 0,
  "text": "This is the chunk text...",
  "metadata": {
    "source_file": "s3://bucket/path/document.txt",
    "chunk_index": 0,
    "strategy": "recursive"
  }
}
```

## Usage Examples

### Example 1: Basic Usage (Single File)

```python
from flytekit import workflow
from data.text_chunker import chunk_documents

@workflow
def process_single_document():
    s3_path, num_chunks, avg_size = chunk_documents(
        s3_input_path="s3://my-bucket/documents/paper.pdf",
        s3_output_path="s3://my-bucket/chunks/paper_chunks.jsonl",
        strategy="recursive",
        chunk_size=512,
        chunk_overlap=50,
    )
    return s3_path, num_chunks, avg_size
```

### Example 2: Process Multiple Documents

```python
@workflow
def process_document_directory():
    # Using a prefix will process all files under that path
    s3_path, num_chunks, avg_size = chunk_documents(
        s3_input_path="s3://my-bucket/documents/",  # All files in directory
        s3_output_path="s3://my-bucket/chunks/all_chunks.jsonl",
        strategy="sentence",
        chunk_size=256,
        chunk_overlap=30,
    )
    return s3_path, num_chunks, avg_size
```

### Example 3: Token-based Chunking for LLMs

```python
@workflow
def chunk_for_llm_context():
    # Use token-based chunking to ensure chunks fit in model context
    s3_path, num_chunks, avg_size = chunk_documents(
        s3_input_path="s3://my-bucket/docs/manual.txt",
        s3_output_path="s3://my-bucket/chunks/manual_tokens.jsonl",
        strategy="token",
        chunk_size=1024,  # 1024 tokens per chunk
        chunk_overlap=100,  # 100 token overlap
    )
    return s3_path, num_chunks, avg_size
```

### Example 4: RAG Pipeline Integration

```python
from flytekit import workflow
from data.text_chunker import chunk_documents
from genai.embeddings import generate_embeddings

@workflow
def rag_indexing_pipeline(
    docs_path: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5"
):
    # Step 1: Chunk documents
    chunks_path, num_chunks, avg_size = chunk_documents(
        s3_input_path=docs_path,
        s3_output_path="s3://my-bucket/rag/chunks.jsonl",
        strategy="recursive",
        chunk_size=512,
        chunk_overlap=50,
    )

    # Step 2: Generate embeddings
    embeddings_path = generate_embeddings(
        texts=chunks_path,
        model_name=embedding_model,
        batch_size=64,
    )

    return embeddings_path, num_chunks
```

## Strategy Selection Guide

### Character Strategy
- **Use when**: You need fast, predictable chunks
- **Pros**: Simple, fast, consistent sizes
- **Cons**: May split mid-sentence or mid-word
- **Recommended settings**: `chunk_size=500`, `chunk_overlap=50`

### Token Strategy
- **Use when**: Working with LLMs with token limits
- **Pros**: Accurate token counting, fits model limits
- **Cons**: Slightly higher overhead than character-based chunking due to tokenization
- **Recommended settings**: `chunk_size=1024`, `chunk_overlap=100`

### Sentence Strategy
- **Use when**: Semantic coherence is important
- **Pros**: Maintains sentence boundaries, better for Q&A
- **Cons**: Variable chunk sizes; requires NLTK sentence tokenizer data to be pre-baked into the image (handled in `data-cpu` Dockerfile)
- **Recommended settings**: `chunk_size=300`, `chunk_overlap=50`

### Recursive Strategy (Default)
- **Use when**: Processing diverse document types
- **Pros**: Preserves structure, handles various content
- **Cons**: More complex, slightly slower
- **Recommended settings**: `chunk_size=512`, `chunk_overlap=50`

## Resource Configuration

The component is configured with:
- **Requests**: 2 CPU, 8Gi memory
- **Limits**: 4 CPU, 16Gi memory
- **Retries**: 2 (automatic retry on failure)
- **Caching**: Enabled (cache_version="1.0")

## Registration

Register the component with Flyte:

```bash
# From the repository root
ml-plat component register \
    projects/components/components/data/text_chunker.py \
    --image ml-platform/data-cpu:latest
```

Or register all data components:

```bash
ml-plat component register \
    projects/components/components/data/ \
    --image ml-platform/data-cpu:latest
```

## Testing

Run the test suite:

```bash
pytest tests/components/test_text_chunker.py -v
```

The test suite includes:
- Unit tests for each chunking strategy
- Document loading tests for all formats
- Integration tests with S3 mocking
- Edge case validation

## Best Practices

1. **Choose the right strategy**: Use recursive for general documents, token for LLMs, sentence for Q&A
2. **Set appropriate overlap**: 10-20% of chunk_size is typical (e.g., 50-100 for size 512)
3. **Monitor chunk statistics**: Use the returned `avg_chunk_size` to tune parameters
4. **Batch process**: Use directory prefixes to process multiple files efficiently
5. **Cache results**: The component has caching enabled - identical inputs reuse outputs

## Troubleshooting

### Empty chunks returned
- Check that input files contain text (not just images)
- Verify S3 paths are correct and accessible
- For PDFs, ensure they contain extractable text (not scanned images)

### Chunks too large/small
- Adjust `chunk_size` parameter
- Try a different strategy (e.g., token for more consistent sizes)
- Check the `avg_chunk_size` output to understand actual sizes

### S3 access errors
- Ensure the Flyte execution role has S3 read/write permissions
- Verify bucket names and paths are correct
- Check that files exist at the input path

## Related Components

- `genai/embeddings.py` - Generate embeddings from chunked text
- `genai/rag.py` - Run RAG pipeline with chunked documents
- `data/ingest.py` - Download datasets from S3

## Dependencies

The component requires these packages (pre-installed in `data-cpu` image):
- `flytekit==1.14.3` - Flyte SDK
- `boto3==1.34.84` - AWS S3 access
- `tiktoken==0.8.0` - Token counting
- `pypdf==5.1.0` - PDF parsing
- `beautifulsoup4==4.12.3` - HTML parsing
- `lxml==5.3.0` - HTML parser backend
- `nltk==3.9.1` - Sentence tokenization

## Further Reading

- [Flyte Documentation](https://docs.flyte.org/)
- [RAG Best Practices](https://www.pinecone.io/learn/chunking-strategies/)
- [NLTK Tokenizers](https://www.nltk.org/api/nltk.tokenize.html)
- [Tiktoken Documentation](https://github.com/openai/tiktoken)
