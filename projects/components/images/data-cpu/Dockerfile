# ============================================================
# data-cpu — Layer 3: Data ingestion / processing workloads
#
# What's inside (on top of flyte-cpu):
#   • pandas + pyarrow         — tabular data
#   • scikit-learn             — preprocessing / metrics
#   • tiktoken                 — BPE tokenizer for LLM text splits
#   • pypdf + beautifulsoup4 + lxml — document parsing
#   • nltk (punkt tokenizer baked in) — sentence splitting
#
# Used by: components/data/*  (text_chunker, loaders, etc.)
# No GPU required; keeps the image lean for fast cold starts.
#
# Pre-pull tag example:  data-cpu:1.0.0
# Parent: flyte-cpu
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=fly1.14-1.0.0
FROM ${REGISTRY}/${REPO}/flyte-cpu:${BASE_TAG}

# Data-processing dependencies (flytekit/boto3/s3fs/requests inherited from flyte-cpu)
ARG PANDAS_VERSION=2.2.2
ARG PYARROW_VERSION=16.0.0
ARG SKLEARN_VERSION=1.4.2
ARG TIKTOKEN_VERSION=0.8.0
ARG PYPDF_VERSION=5.1.0
ARG BS4_VERSION=4.12.3
ARG LXML_VERSION=5.3.0
ARG NLTK_VERSION=3.9.1

RUN pip install --no-cache-dir \
    pandas==${PANDAS_VERSION} \
    pyarrow==${PYARROW_VERSION} \
    scikit-learn==${SKLEARN_VERSION} \
    tiktoken==${TIKTOKEN_VERSION} \
    pypdf==${PYPDF_VERSION} \
    "beautifulsoup4==${BS4_VERSION}" \
    lxml==${LXML_VERSION} \
    nltk==${NLTK_VERSION}

# Pre-bake NLTK tokenizer data so tasks don't need network access at runtime
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

WORKDIR /root
