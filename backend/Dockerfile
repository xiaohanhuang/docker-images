# Backend API Dockerfile
# Public builder fetching proprietary source from ECR
FROM 805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/source:latest AS source

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast, reliable dependency resolution
RUN pip install --no-cache-dir uv

# Copy the full source tree directly from the source image
COPY --from=source /src/pyproject.toml /src/README.md /app/
COPY --from=source /src/backend/ /app/backend/
COPY --from=source /src/cli/ /app/cli/
COPY --from=source /src/projects/components/sdk/ml_platform_sdk/ /app/projects/components/sdk/ml_platform_sdk/
COPY --from=source /src/projects/components/components/ /app/projects/components/components/
COPY --from=source /src/projects/recipes/ /app/projects/recipes/

# Override backend from workspace (may be newer than source image)
COPY backend/ /app/backend/

# Install backend dependencies using uv
RUN uv pip install --no-cache --system ".[backend]"

# Expose the API port
EXPOSE 8000

# Run the backend API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
