# ============================================================
# flyte-gpu — Layer 2: GPU Flyte runtime
#
# What's inside (on top of base-gpu):
#   • flytekit — Flyte SDK / task execution engine
#   • boto3 + s3fs — S3 access (needed by nearly every task)
#   • requests — common HTTP utility
#
# All GPU-accelerated Flyte tasks should build FROM this image,
# not from base-gpu directly.
#
# Pre-pull tag example:  flyte-gpu:fly1.14-cu12.9-1.0.0
# Parent: base-gpu       Child images: ml-gpu, genai-gpu
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=cu12.9-torch2.9-1.0.0
FROM ${REGISTRY}/${REPO}/base-gpu:${BASE_TAG}

ARG FLYTEKIT_VERSION=1.14.3
ARG S3FS_VERSION=2024.3.1
ARG REQUESTS_VERSION=2.31.0

RUN pip install --no-cache-dir \
    flytekit==${FLYTEKIT_VERSION} \
    boto3 \
    s3fs==${S3FS_VERSION} \
    requests==${REQUESTS_VERSION}

WORKDIR /root
