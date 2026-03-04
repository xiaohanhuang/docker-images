# ============================================================
# genai-gpu — Layer 3: GenAI inference workloads
#
# What's inside (on top of flyte-gpu):
#   • vLLM                     — high-throughput LLM serving
#   • langchain + community    — LLM orchestration
#   • sentence-transformers    — embedding models
#   • openai (client)          — OpenAI-compatible API calls
#   • tiktoken                 — BPE tokenizer
#   • faiss-cpu                — similarity search (CPU)
#   • chromadb                 — in-process vector store
#   • psycopg2-binary + pgvector — PostgreSQL vector store
#
# Used by: components/genai/*  (vector_store_indexer, etc.)
#
# Pre-pull tag example:  genai-gpu:1.0.0
# Parent: flyte-gpu
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=fly1.14-cu12.9-1.0.0
FROM ${REGISTRY}/${REPO}/flyte-gpu:${BASE_TAG}

# GenAI inference dependencies (flytekit/boto3/s3fs inherited from flyte-gpu)
ARG VLLM_VERSION=0.5.0
ARG LANGCHAIN_VERSION=0.2.5
ARG LANGCHAIN_COMMUNITY_VERSION=0.2.5
ARG SENTENCE_TRANSFORMERS_VERSION=3.0.1
ARG OPENAI_VERSION=1.34.0
ARG TIKTOKEN_VERSION=0.7.0
ARG FAISS_VERSION=1.8.0
ARG CHROMADB_VERSION=0.4.24
ARG PSYCOPG2_VERSION=2.9.9
ARG PGVECTOR_VERSION=0.3.5

RUN pip install --no-cache-dir --ignore-installed \
    vllm==${VLLM_VERSION} \
    langchain==${LANGCHAIN_VERSION} \
    langchain-community==${LANGCHAIN_COMMUNITY_VERSION} \
    sentence-transformers==${SENTENCE_TRANSFORMERS_VERSION} \
    openai==${OPENAI_VERSION} \
    tiktoken==${TIKTOKEN_VERSION} \
    faiss-cpu==${FAISS_VERSION} \
    chromadb==${CHROMADB_VERSION} \
    psycopg2-binary==${PSYCOPG2_VERSION} \
    pgvector==${PGVECTOR_VERSION}

WORKDIR /root
