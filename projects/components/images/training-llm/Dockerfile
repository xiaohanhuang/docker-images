# ============================================================
# training-llm — Layer 3: Distributed LLM training (Ray)
#
# What's inside (on top of ray-worker):
#   • transformers + peft + accelerate + datasets — fine-tuning
#   • bitsandbytes  — 4/8-bit quantization
#   • deepspeed     — ZeRO optimizer (JIT-compiled, not pre-built)
#   • mlflow        — experiment tracking
#   • openrlhf + vllm — distributed RLHF with vLLM inference
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

ARG BASE_IMAGE=805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/ray-worker:latest
FROM ${BASE_IMAGE}

# Ensure CUDA libraries are discoverable. The NVIDIA runtime base image sets
# LD_LIBRARY_PATH to /usr/local/nvidia/lib64 but omits /usr/local/cuda/lib64.
# bitsandbytes and vLLM need the unversioned libcudart.so symlink as well.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so

ARG TRANSFORMERS_VERSION=5.3.0
ARG ACCELERATE_VERSION=1.13.0
ARG PEFT_VERSION=0.18.1
ARG DATASETS_VERSION=4.6.1
ARG BITSANDBYTES_VERSION=0.49.2
ARG DEEPSPEED_VERSION=0.18.6
ARG FLYTE_RAY_PLUGIN=1.16.14
ARG MLFLOW_VERSION=3.10.0
ARG OPENRLHF_VERSION=0.9.5
ARG VLLM_VERSION=0.17.0

# Remove distutils-installed blinker 1.4 (Ubuntu system pkg) that blocks pip uninstall
RUN rm -rf /usr/lib/python3/dist-packages/blinker* || true

RUN pip install --no-cache-dir \
    transformers==${TRANSFORMERS_VERSION} \
    accelerate==${ACCELERATE_VERSION} \
    peft==${PEFT_VERSION} \
    datasets==${DATASETS_VERSION} \
    bitsandbytes==${BITSANDBYTES_VERSION} \
    mlflow==${MLFLOW_VERSION} \
    scipy \
    einops \
    sentencepiece \
    protobuf

# DeepSpeed: install with lazy JIT compilation (DS_BUILD_OPS=1 breaks on CUDA 12.9.x at build time)
RUN pip install --no-cache-dir deepspeed==${DEEPSPEED_VERSION}

# DeepSpeed's import unconditionally calls `nvcc -V` to check CUDA compatibility.
# The runtime base image doesn't ship nvcc, so we create a stub that reports the
# version matching our CUDA toolkit. DS only parses "release X.Y" — no compilation.
RUN mkdir -p /usr/local/cuda/bin && \
    printf '#!/bin/sh\necho "nvcc: NVIDIA (R) Cuda compiler driver"\necho "Cuda compilation tools, release 12.9, V12.9.41"\n' \
      > /usr/local/cuda/bin/nvcc && \
    chmod +x /usr/local/cuda/bin/nvcc

# Flyte ↔ Ray plugin so this image can be used in Flyte RayJobConfig tasks
RUN pip install --no-cache-dir flytekitplugins-ray==${FLYTE_RAY_PLUGIN}

# OpenRLHF with vLLM backend for distributed RLHF training
# vllm: pre-built wheel with bundled flash-attention kernels
# openrlhf: --no-deps skips flash-attn which needs NVCC (not in runtime image);
#           all major deps (torch, transformers, deepspeed, ray) already present
RUN pip install --no-cache-dir vllm==${VLLM_VERSION} && \
    pip install --no-cache-dir --no-deps openrlhf==${OPENRLHF_VERSION} && \
    pip install --no-cache-dir jsonlines pylatexenc

WORKDIR /root
