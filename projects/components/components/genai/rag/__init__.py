"""
GenAI — Retrieval-Augmented Generation pipeline.

Image: genai-gpu
"""

from flytekit import Resources, task
from flytekit.types.file import FlyteFile


@task(
    retries=1,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1"),
    cache=False,
)
def run_rag_pipeline(
    query: str,
    corpus_embeddings: FlyteFile,
    corpus_texts: FlyteFile,
    llm_endpoint: str,
    top_k: int = 5,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> str:
    """Run a Retrieval-Augmented Generation pipeline.

    Args:
        query: Natural language question.
        corpus_embeddings: Pre-computed embeddings ``.npy`` file.
        corpus_texts: JSONL file with the original corpus texts.
        llm_endpoint: vLLM or OpenAI-compatible endpoint URL.
        top_k: Number of retrieved documents to include in the context.
        embedding_model: Sentence-transformers model for query encoding.
            Must match the model used to generate ``corpus_embeddings``.

    Returns:
        Generated answer grounded in the retrieved context.
    """
    import json

    import numpy as np
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    corpus_embeddings.download()
    corpus_texts.download()

    embeddings = np.load(corpus_embeddings.path)
    corpus = []
    with open(corpus_texts.path) as fh:
        for line in fh:
            item = json.loads(line.strip())
            corpus.append(item["text"])

    # Encode query and retrieve top-k documents
    model = SentenceTransformer(embedding_model)
    query_emb = model.encode([query])
    scores = (embeddings @ query_emb.T).squeeze()
    top_indices = scores.argsort()[-top_k:][::-1]
    context = "\n\n".join(corpus[i] for i in top_indices)

    # Generate answer
    client = OpenAI(base_url=llm_endpoint, api_key="not-required")
    response = client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. " "Answer based only on the provided context."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content
