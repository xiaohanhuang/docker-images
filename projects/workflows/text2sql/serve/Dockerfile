FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "transformers>=4.41" \
    "torch>=2.1" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "boto3>=1.34" \
    "pydantic>=2.0" \
    "sentencepiece" \
    "protobuf"

COPY server.py /app/server.py

ENV PYTHONPATH=/app
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
