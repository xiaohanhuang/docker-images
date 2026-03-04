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

RUN pip install --no-cache-dir --ignore-installed \
    transformers==${TRANSFORMERS_VERSION} \
    accelerate==${ACCELERATE_VERSION} \
    peft==${PEFT_VERSION} \
    datasets==${DATASETS_VERSION} \
    mlflow==${MLFLOW_VERSION} \
    scikit-learn==${SKLEARN_VERSION} \
    evaluate==${EVALUATE_VERSION} \
    rouge-score==${ROUGE_VERSION} \
    sentencepiece==${SENTENCEPIECE_VERSION}

WORKDIR /root
