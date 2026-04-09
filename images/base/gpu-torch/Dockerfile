# ============================================================
# base-gpu-torch — Layer 1: GPU + PyTorch base
#
# What's inside:
#   • CUDA 12.9.1 + cuDNN runtime (official NVIDIA image)
#   • Python 3.12 (via deadsnakes PPA)
#   • PyTorch 2.10 (CUDA 12.9 wheels)
#   • SSH server (for interactive debugging)
#   • cloudpickle (required for @remote decorator)
#
# This image has NO Flyte, NO Ray, NO ML framework deps.
# It is the cacheable GPU foundation every higher layer shares.
# Keep it to the slowest-changing deps only (CUDA ~yearly,
# PyTorch ~quarterly) so child images rarely need full rebuilds.
#
# Child images: flyte-gpu, ray-gpu
# ============================================================

ARG CUDA_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# The NVIDIA *runtime* image ships libcudart.so.12 (versioned) but not the
# unversioned libcudart.so symlink. bitsandbytes and vLLM probe for the
# unversioned name during CUDA auto-detection, so we create it here.
RUN ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    wget \
    curl \
    ca-certificates \
    openssh-server \
    htop \
    screen \
    vim \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python3.12 -m pip install --no-cache-dir setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for debugging (In prod, use non-root and properly managed keys)
RUN mkdir /var/run/sshd
# Explicitly set PermitRootLogin to prohibit-password for security
RUN if grep -qE '^\s*#?\s*PermitRootLogin\b' /etc/ssh/sshd_config; then \
        sed -ri 's/^\s*#?\s*PermitRootLogin\s+.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config; \
    else \
        echo 'PermitRootLogin prohibit-password' >> /etc/ssh/sshd_config; \
    fi
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

EXPOSE 22

# Symlink python3 to python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install PyTorch matching the CUDA version above.
# NOTE: This is the ONLY ML package in this image. transformers, peft,
# accelerate, etc. belong in Layer 3 workload images (ml-gpu, training-llm).
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/cu129

# Install cloudpickle for @remote decorator support
RUN pip install --no-cache-dir cloudpickle

# Verify install
RUN python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

