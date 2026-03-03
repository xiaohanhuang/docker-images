FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install cloudpickle for @remote decorator support
RUN pip install --no-cache-dir cloudpickle

# Verify install
RUN python -c "import cloudpickle; print('cloudpickle', cloudpickle.__version__)"
