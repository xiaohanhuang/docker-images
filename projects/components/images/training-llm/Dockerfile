# ============================================================
# training-llm — Layer 3: Distributed LLM training (Ray)
#
# What's inside (on top of ray-worker):
#   • transformers + peft + accelerate + datasets — fine-tuning
#   • bitsandbytes  — 4/8-bit quantization
#   • deepspeed     — ZeRO optimizer (JIT-compiled, not pre-built)
#   • einops + sentencepiece + protobuf — model utilities
#   • flytekitplugins-ray — Flyte ↔ Ray integration
#
# DeepSpeed is installed WITHOUT DS_BUILD_OPS=1 to avoid CUDA
# JIT compilation failures at build time on CUDA 12.9.x.
# Operators are compiled lazily at first use.
#
# Pre-pull tag example:  training-llm:1.0.0
# Parent: ray-worker
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=ray2.54-cu12.9-1.0.0
FROM ${REGISTRY}/${REPO}/ray-worker:${BASE_TAG}

ARG TRANSFORMERS_VERSION=4.36.2
ARG ACCELERATE_VERSION=0.25.0
ARG PEFT_VERSION=0.7.1
ARG DATASETS_VERSION=2.16.1
ARG BITSANDBYTES_VERSION=0.41.3
ARG DEEPSPEED_VERSION=0.12.6
ARG FLYTE_RAY_PLUGIN=1.14.3

RUN pip install --no-cache-dir --ignore-installed \
    transformers==${TRANSFORMERS_VERSION} \
    accelerate==${ACCELERATE_VERSION} \
    peft==${PEFT_VERSION} \
    datasets==${DATASETS_VERSION} \
    bitsandbytes==${BITSANDBYTES_VERSION} \
    scipy \
    einops \
    sentencepiece \
    protobuf

# DeepSpeed: install with lazy JIT compilation (DS_BUILD_OPS=1 breaks on CUDA 12.9.x at build time)
RUN pip install --no-cache-dir deepspeed==${DEEPSPEED_VERSION}

# Flyte ↔ Ray plugin so this image can be used in Flyte RayJobConfig tasks
RUN pip install --no-cache-dir flytekitplugins-ray==${FLYTE_RAY_PLUGIN}

WORKDIR /root
