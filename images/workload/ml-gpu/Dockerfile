# ============================================================
# ml-gpu — Layer 3: Model training and evaluation workloads
#
# What's inside (on top of flyte-gpu):
#   • transformers + peft + accelerate — fine-tuning
#   • datasets                         — HuggingFace data loading
#   • mlflow                           — experiment tracking
#   • rouge-score                      — offline metrics
#   • sentencepiece                    — tokenizer dependency
#
# Used by: components/training/*, components/evaluation/*
#
# Pre-pull tag example:  ml-gpu:1.0.0
# Parent: flyte-gpu
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=fly1.16.14-cu12.9.1-1.1.0
FROM ${REGISTRY}/${REPO}/flyte-gpu:${BASE_TAG}

# Remove distutils-installed blinker 1.4 (Ubuntu system pkg) that blocks pip uninstall
RUN rm -rf /usr/lib/python3/dist-packages/blinker* || true

# Training and evaluation dependencies (flytekit/boto3/s3fs inherited from flyte-gpu)
ARG TRANSFORMERS_VERSION=5.3.0
ARG ACCELERATE_VERSION=1.13.0
ARG PEFT_VERSION=0.18.1
ARG DATASETS_VERSION=4.6.1
ARG MLFLOW_VERSION=3.10.0
ARG ROUGE_VERSION=0.1.2
ARG SENTENCEPIECE_VERSION=0.2.1

RUN pip install --no-cache-dir \
    transformers==${TRANSFORMERS_VERSION} \
    accelerate==${ACCELERATE_VERSION} \
    peft==${PEFT_VERSION} \
    datasets==${DATASETS_VERSION} \
    mlflow==${MLFLOW_VERSION} \
    rouge-score==${ROUGE_VERSION} \
    sentencepiece==${SENTENCEPIECE_VERSION}

WORKDIR /root
