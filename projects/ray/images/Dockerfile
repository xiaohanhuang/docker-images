ARG BASE_IMAGE=base-gpu
FROM ${BASE_IMAGE}

# Install Ray and common ML libs
# ray[default] includes dashboard, client, etc.
RUN pip install --no-cache-dir \
    "ray[default]==2.54.0" \
    pandas \
    numpy \
    pyarrow \
    requests \
    boto3 \
    s3fs \
    torch-tb-profiler

# Open Ray ports
EXPOSE 6379 8265 10001 8000

# Default command for a Ray node
CMD ["ray", "start", "--block"]
