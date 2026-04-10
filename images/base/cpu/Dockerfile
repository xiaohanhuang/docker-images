# ============================================================
# base-cpu — Layer 1: Minimal CPU base
#
# What's inside:
#   • Python 3.12-slim (official image)
#   • System tools: git, curl, ca-certificates
#
# This image has NO framework deps (no flytekit, no boto3).
# It exists purely so higher layers get stable, cacheable OS
# layers without re-downloading them on every dep change.
#
# Pre-pull tag example:  base-cpu:py3.12-1.0.0
# Child images: flyte-cpu
# ============================================================

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip once here so every child layer inherits it
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /root
