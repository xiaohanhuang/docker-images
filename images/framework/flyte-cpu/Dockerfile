# ============================================================
# flyte-cpu — Layer 2: CPU Flyte runtime
#
# What's inside (on top of base-cpu):
#   • flytekit — Flyte SDK / task execution engine
#   • boto3 + s3fs — S3 access (needed by nearly every task)
#   • requests — common HTTP utility
#   • ml_platform_sdk                  — platform SDK
#
# All data-processing and CPU-only Flyte tasks should build
# FROM this image, not from python:3.12-slim directly.
#
# Pre-pull tag example:  flyte-cpu:fly1.14-1.0.0
# Parent: base-cpu       Child images: data-cpu
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=py3.12-1.1.0

FROM 805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/source:latest AS source

FROM ${REGISTRY}/${REPO}/base-cpu:${BASE_TAG}

ARG FLYTEKIT_VERSION=1.16.14
ARG BOTO3_VERSION=1.42.61
ARG S3FS_VERSION=2026.2.0
ARG REQUESTS_VERSION=2.32.5

RUN pip install --no-cache-dir \
    flytekit==${FLYTEKIT_VERSION} \
    boto3==${BOTO3_VERSION} \
    s3fs==${S3FS_VERSION} \
    requests==${REQUESTS_VERSION}

# Install ML Platform SDK — copy from read-only bind mount to writable dir first
RUN --mount=type=bind,from=source,source=/src/projects/components/sdk,target=/tmp/sdk \
    cp -r /tmp/sdk /tmp/sdk-build && \
    pip install --no-cache-dir --ignore-installed blinker /tmp/sdk-build && \
    rm -rf /tmp/sdk-build

WORKDIR /root
