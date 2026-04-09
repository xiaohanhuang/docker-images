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
# Parent: base-gpu-torch    Child images: ml-gpu, genai-gpu
# ============================================================

ARG BASE_IMAGE=805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/base-gpu:latest
FROM ${BASE_IMAGE}

ARG FLYTEKIT_VERSION=1.16.14
ARG BOTO3_VERSION=1.42.61
ARG S3FS_VERSION=2026.2.0
ARG REQUESTS_VERSION=2.32.5

RUN pip install --no-cache-dir "setuptools<70" wheel && \
    pip install --no-cache-dir \
    flytekit==${FLYTEKIT_VERSION} \
    boto3==${BOTO3_VERSION} \
    s3fs==${S3FS_VERSION} \
    requests==${REQUESTS_VERSION}

# Install the ML Platform SDK so all downstream images can use
# `import ml_platform_sdk as mp` instead of raw flytekit APIs.
COPY projects/components/sdk /tmp/ml-platform-sdk
RUN pip install --no-cache-dir /tmp/ml-platform-sdk && rm -rf /tmp/ml-platform-sdk

WORKDIR /root
