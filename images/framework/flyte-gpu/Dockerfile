# ============================================================
# flyte-gpu — Layer 2: GPU Flyte runtime
#
# What's inside (on top of base-gpu):
#   • flytekit — Flyte SDK / task execution engine
#   • boto3 + s3fs — S3 access (needed by nearly every task)
#   • requests — common HTTP utility
#   • ml_platform_sdk                  — platform SDK
#
# All GPU-accelerated Flyte tasks should build FROM this image,
# not from base-gpu directly.
#
# Pre-pull tag example:  flyte-gpu:fly1.14-cu12.9-1.0.0
# Parent: base-gpu-torch    Child images: ml-gpu, genai-gpu
# ============================================================

FROM 805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/source:latest AS source

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=latest
FROM ${REGISTRY}/${REPO}/base-gpu:${BASE_TAG}

ARG FLYTEKIT_VERSION=1.16.14
ARG BOTO3_VERSION=1.42.61
ARG S3FS_VERSION=2026.2.0
ARG REQUESTS_VERSION=2.32.5

RUN pip install --no-cache-dir \
    flytekit==${FLYTEKIT_VERSION} \
    boto3==${BOTO3_VERSION} \
    s3fs==${S3FS_VERSION} \
    requests==${REQUESTS_VERSION}

# Install ML Platform SDK dynamically without leaving source layer bloat
RUN --mount=type=bind,from=source,source=/src/projects/components/sdk,target=/tmp/sdk \
    pip install --no-cache-dir /tmp/sdk

WORKDIR /root
