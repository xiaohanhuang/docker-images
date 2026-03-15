# ============================================================
# training-llm-flash — training-llm + pre-compiled flash-attn
#
# Identical to training-llm but includes flash-attn built from
# source with NVCC.  This requires the CUDA toolkit (nvcc) at
# build time, so we install it temporarily.
#
# Pre-pull tag example:  training-llm-flash:1.1.0
# Parent: training-llm
# ============================================================

ARG BASE_IMAGE=805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/training-llm:latest
FROM ${BASE_IMAGE}

ARG FLASH_ATTN_VERSION=2.7.4

# Install CUDA toolkit (nvcc) needed to compile flash-attn from source.
# The runtime base image only has CUDA libraries, not the compiler.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        cuda-nvcc-12-9 \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Build and install flash-attn.
# MAX_JOBS=4 limits parallel compilation to avoid OOM on CI runners.
# FLASH_ATTENTION_FORCE_BUILD=TRUE skips any pre-built wheel check.
ENV MAX_JOBS=4
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
RUN pip install --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# Clean up build tools to reduce image size (optional — they're small)
RUN apt-get purge -y --auto-remove cuda-nvcc-12-9 g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root
