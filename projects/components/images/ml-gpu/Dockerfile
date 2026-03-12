# ============================================================
# ml-gpu — Layer 3: Model training and evaluation workloads
#
# What's inside (on top of flyte-gpu):
#   • transformers + peft + accelerate — fine-tuning
#   • datasets                         — HuggingFace data loading
#   • mlflow                           — experiment tracking
#   • scikit-learn + evaluate + rouge  — offline metrics
#   • sentencepiece                    — tokenizer dependency
#
# Used by: components/training/*, components/evaluation/*
#
# Pre-pull tag example:  ml-gpu:1.0.0
# Parent: flyte-gpu
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=fly1.14-cu12.9-1.0.0
FROM ${REGISTRY}/${REPO}/flyte-gpu:${BASE_TAG}

# Remove distutils-installed blinker 1.4 (Ubuntu system pkg) that blocks pip uninstall
RUN rm -rf /usr/lib/python3/dist-packages/blinker* || true

# Training and evaluation dependencies (flytekit/boto3/s3fs inherited from flyte-gpu)
ARG TRANSFORMERS_VERSION=4.41.2
ARG ACCELERATE_VERSION=0.30.1
ARG PEFT_VERSION=0.11.1
ARG DATASETS_VERSION=2.19.1
ARG MLFLOW_VERSION=2.13.0
ARG SKLEARN_VERSION=1.4.2
ARG EVALUATE_VERSION=0.4.2
ARG ROUGE_VERSION=0.1.2
ARG SENTENCEPIECE_VERSION=0.2.0
ARG OPENRLHF_VERSION=0.9.4

RUN pip install --no-cache-dir \
    transformers==${TRANSFORMERS_VERSION} \
    accelerate==${ACCELERATE_VERSION} \
    peft==${PEFT_VERSION} \
    datasets==${DATASETS_VERSION} \
    mlflow==${MLFLOW_VERSION} \
    scikit-learn==${SKLEARN_VERSION} \
    evaluate==${EVALUATE_VERSION} \
    rouge-score==${ROUGE_VERSION} \
    sentencepiece==${SENTENCEPIECE_VERSION}

# openrlhf: --no-deps skips flash-attn which needs NVCC (not in runtime image);
# all major deps (torch, transformers, deepspeed, ray) already present above.
RUN pip install --no-cache-dir --no-deps openrlhf==${OPENRLHF_VERSION}

# DeepSpeed (pulled in by openrlhf) probes nvcc at import time to check CUDA
# compatibility.  The *runtime* base image ships libcudart but not the CUDA
# toolkit, so nvcc is absent.  Provide a lightweight stub that returns the
# correct version string so DeepSpeed can complete its compatibility check.
RUN mkdir -p /usr/local/cuda/bin && \
    printf '#!/bin/bash\necho "nvcc: NVIDIA (R) Cuda compiler driver"\necho "Cuda compilation tools, release 12.9, V12.9.41"\n' \
        > /usr/local/cuda/bin/nvcc && \
    chmod +x /usr/local/cuda/bin/nvcc

WORKDIR /root
