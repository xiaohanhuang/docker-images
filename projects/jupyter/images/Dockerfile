FROM jupyter/scipy-notebook:python-3.11.6

USER root

# Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Install ML + Platform dependencies from pinned requirements
COPY projects/jupyter/images/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-install the platform SDK
COPY --chown=${NB_UID}:${NB_GID} projects/components/sdk /home/jovyan/ml-platform-sdk/
RUN pip install -e /home/jovyan/ml-platform-sdk/

WORKDIR /home/jovyan/work
