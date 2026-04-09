"""
GenAI — batch-generate text embeddings with sentence-transformers.

Image: genai-gpu
"""

from flytekit import Resources, task
from flytekit.types.file import FlyteFile


@task(
    retries=2,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1"),
    cache=True,
    cache_version="1.0",
)
def generate_embeddings(
    texts: FlyteFile,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
) -> FlyteFile:
    """Batch-generate text embeddings.

    Args:
        texts: JSONL file where each line has a ``text`` field.
        model_name: Sentence-transformers model to use.
        batch_size: Number of texts to encode per batch.

    Returns:
        NumPy ``.npy`` file of shape ``(N, D)`` with the resulting embeddings.
    """
    import json

    import numpy as np
    from sentence_transformers import SentenceTransformer

    texts.download()
    corpus = []
    with open(texts.path) as fh:
        for line in fh:
            item = json.loads(line.strip())
            corpus.append(item["text"])

    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus, batch_size=batch_size, show_progress_bar=False)

    out_path = "/tmp/embeddings.npy"
    np.save(out_path, embeddings)
    return FlyteFile(path=out_path)
