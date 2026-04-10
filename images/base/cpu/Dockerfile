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
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for interactive pod access
RUN mkdir -p /var/run/sshd \
    && if grep -qE '^\s*#?\s*PermitRootLogin\b' /etc/ssh/sshd_config; then \
        sed -ri 's/^\s*#?\s*PermitRootLogin\s+.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config; \
    else \
        echo 'PermitRootLogin prohibit-password' >> /etc/ssh/sshd_config; \
    fi \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

EXPOSE 22

# Upgrade pip once here so every child layer inherits it
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /root
