# ============================================================
# flyte-cpu — Layer 2: CPU Flyte runtime
#
# What's inside (on top of base-cpu):
#   • flytekit — Flyte SDK / task execution engine
#   • boto3 + s3fs — S3 access (needed by nearly every task)
#   • requests — common HTTP utility
#
# All data-processing and CPU-only Flyte tasks should build
# FROM this image, not from python:3.12-slim directly.
#
# Pre-pull tag example:  flyte-cpu:fly1.14-1.0.0
# Parent: base-cpu       Child images: data-cpu
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=py3.12-1.0.0
FROM ${REGISTRY}/${REPO}/base-cpu:${BASE_TAG}

ARG FLYTEKIT_VERSION=1.14.3
ARG S3FS_VERSION=2024.3.1
ARG REQUESTS_VERSION=2.31.0

RUN pip install --no-cache-dir \
    flytekit==${FLYTEKIT_VERSION} \
    boto3 \
    s3fs==${S3FS_VERSION} \
    requests==${REQUESTS_VERSION}

WORKDIR /root
