ARG REGISTRY=805673386114.dkr.ecr.us-west-2.amazonaws.com
ARG REPO=ml-platform
ARG BASE_TAG=cu12.9.1-torch2.10.0-1.1.0
FROM ${REGISTRY}/${REPO}/base-gpu-torch:${BASE_TAG}

# Install Flask for HTTP server
RUN pip install --no-cache-dir flask==3.1.1

# Copy executor server
COPY executor_server.py /app/executor_server.py

WORKDIR /app

# Expose port
EXPOSE 8080

# Run executor server
CMD ["python", "/app/executor_server.py"]
