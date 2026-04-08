# ============================================================
# ray-gpu — Layer 2: GPU Ray runtime
#
# What's inside (on top of base-gpu-torch):
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
# Parent: base-gpu-torch    Child images: training-llm
# ============================================================

ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=cu12.9-torch2.10-1.1.0
FROM ${REGISTRY}/${REPO}/base-gpu-torch:${BASE_TAG}

ARG RAY_VERSION=2.54.0

RUN pip install --no-cache-dir \
    "ray[default]==${RAY_VERSION}" \
    pandas \
    numpy \
    pyarrow \
    requests \
    boto3 \
    s3fs \
    torch-tb-profiler \
    flytekit

# Install the ML Platform SDK so tasks can use `import ml_platform_sdk as mp`
COPY projects/components/sdk /tmp/ml-platform-sdk
RUN pip install --no-cache-dir /tmp/ml-platform-sdk && rm -rf /tmp/ml-platform-sdk

EXPOSE 6379 8265 10001 8000

CMD ["ray", "start", "--block"]
