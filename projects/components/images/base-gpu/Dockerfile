FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    ca-certificates \
    openssh-server \
    htop \
    screen \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for debugging (In prod, use non-root and properly managed keys)
RUN mkdir /var/run/sshd
# Allow root login with password for simplicity in this demo (Use keys in real world!)
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

EXPOSE 22

# Symlink python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch (matching CUDA version)
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Verify install
RUN python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
