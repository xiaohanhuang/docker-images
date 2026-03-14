"""
ML Platform Registry Service.

FastAPI service that provides recipe registry CRUD operations (list, push, pull, status)
against PostgreSQL + S3.  This is a stateless HTTP API that the CLI can use without
direct Postgres/S3 access.
"""

import logging
import os

import uvicorn
from fastapi import FastAPI
from registry_api import router as registry_router

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ML Platform Registry Service",
    description="Recipe registry CRUD operations (PostgreSQL + S3)",
    version="0.1.0",
)

# Include registry API routes
app.include_router(registry_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "ml-platform-registry-service",
        "version": "0.1.0",
        "status": "healthy",
    }


@app.get("/health")
async def health():
    """Kubernetes health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8081"))
    print(f"[registry] Starting ML Platform Registry Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
