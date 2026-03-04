# ============================================================
# ray-worker — Layer 2: GPU Ray worker node
#
# What's inside (on top of base-gpu):
#   • ray[default] — Ray core + dashboard + client
#   • pandas, numpy, pyarrow — data interchange
#   • boto3, s3fs — S3 access
#   • torch-tb-profiler — TensorBoard training profiling
#
# Ray ports:
#   6379  — Redis / GCS (head node)
#   8265  — Dashboard
#   10001 — Ray client
#   8000  — Serve
#
# Pre-pull tag example:  ray-worker:ray2.54-cu12.9-1.0.0
# Parent: base-gpu       Child images: training-llm
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=cu12.9-torch2.9-1.0.0
FROM ${REGISTRY}/${REPO}/base-gpu:${BASE_TAG}

ARG RAY_VERSION=2.54.0

RUN pip install --no-cache-dir \
    "ray[default]==${RAY_VERSION}" \
    pandas \
    numpy \
    pyarrow \
    requests \
    boto3 \
    s3fs \
    torch-tb-profiler

EXPOSE 6379 8265 10001 8000

CMD ["ray", "start", "--block"]
