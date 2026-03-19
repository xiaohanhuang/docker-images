"""Main FastAPI application for ml-platform backend."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api import chat, cost, dashboard, jobs, kubernetes, mlflow, notebooks, pods, ray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting ml-platform backend API")
    yield
    # Shutdown
    logger.info("Shutting down ml-platform backend API")


app = FastAPI(
    title="ML Platform API",
    description="Backend API for ml-platform CLI operations",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS – restrict origins in production via CORS_ALLOWED_ORIGINS env var
_default_origins = (
    "http://localhost:3000,http://localhost:3001,"
    "http://ml-platform-dashboard.ml-platform.svc.cluster.local"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOWED_ORIGINS", _default_origins).split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-User"],
)


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    """Middleware to audit all requests."""
    # TODO: In production, replace X-User header with proper auth (e.g., Bearer token
    # validation, OIDC). The current approach trusts the header without verification.
    user = request.headers.get("X-User", "unknown")

    # Store user in request state for use in endpoints
    request.state.user = user

    response = await call_next(request)
    return response


# Include routers
app.include_router(pods.router, prefix="/pods", tags=["pods"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(notebooks.router, prefix="/notebooks", tags=["notebooks"])
app.include_router(cost.router, prefix="/cost", tags=["cost"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(mlflow.router, prefix="/mlflow", tags=["mlflow"])
app.include_router(ray.router, prefix="/ray", tags=["ray"])
app.include_router(kubernetes.router, prefix="/kubernetes", tags=["kubernetes"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ML Platform API", "version": "0.1.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
