# Backend API Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast, reliable dependency resolution
RUN pip install --no-cache-dir uv

# Copy the full source tree needed for packaging
COPY pyproject.toml README.md /app/
COPY backend/ /app/backend/
COPY cli/ /app/cli/
COPY projects/components/sdk/ml_platform_sdk/ /app/projects/components/sdk/ml_platform_sdk/

# Install backend dependencies using uv (handles complex dep graphs much better than pip)
RUN uv pip install --no-cache --system ".[backend]"

# Expose the API port
EXPOSE 8000

# Run the backend API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
