"""Main FastAPI application for ml-platform backend."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api import (
    chat,
    components,
    cost,
    dashboard,
    desks,
    jobs,
    kubernetes,
    mlflow,
    notebooks,
    pods,
    ray,
    recipes,
    serving,
    settings,
    tensorboard,
)
from backend.auth import AuthMiddleware

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
    logger.info("Starting ML Platform API")
    yield
    # Shutdown
    from backend.api.svc_proxy import close_client

    await close_client()
    logger.info("Shutting down ML Platform API")


app = FastAPI(
    title="ML Platform API",
    description=(
        "Backend API for the ML Platform — desk management,"
        " cost tracking, model serving, and experiment orchestration."
    ),
    version="0.2.0",
    lifespan=lifespan,
    redirect_slashes=False,
)

# Configure CORS – restrict origins in production via CORS_ALLOWED_ORIGINS env var
_default_origins = (
    "http://localhost:3000,http://localhost:3001,"
    "http://ml-platform-dashboard.ml-platform.svc.cluster.local"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOWED_ORIGINS", _default_origins).split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-User", "Authorization"],
)

# Authentication middleware (disabled by default in development)
app.add_middleware(AuthMiddleware)


# ── API versioned router ─────────────────────────────────────────
# All domain endpoints live under /api/v1 for forward-compatible versioning.
# Root-level endpoints (/, /health, /auth/me) stay at the root.

api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(pods.router, prefix="/pods", tags=["pods"])
api_v1.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_v1.include_router(notebooks.router, prefix="/notebooks", tags=["notebooks"])
api_v1.include_router(cost.router, prefix="/cost", tags=["cost"])
api_v1.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_v1.include_router(mlflow.router, prefix="/mlflow", tags=["mlflow"])
api_v1.include_router(ray.router, prefix="/ray", tags=["ray"])
api_v1.include_router(kubernetes.router, prefix="/kubernetes", tags=["kubernetes"])
api_v1.include_router(chat.router, prefix="/chat", tags=["chat"])
api_v1.include_router(desks.router, prefix="/desks", tags=["desks"])
api_v1.include_router(serving.router, prefix="/serving", tags=["serving"])
api_v1.include_router(settings.router, prefix="/settings", tags=["settings"])
api_v1.include_router(recipes.router, prefix="/recipes", tags=["recipes"])
api_v1.include_router(components.router, prefix="/components", tags=["components"])
api_v1.include_router(tensorboard.router, prefix="/tensorboard", tags=["tensorboard"])

app.include_router(api_v1)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ML Platform API", "version": "0.2.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/auth/me")
@app.get("/api/auth")
async def auth_me(request: Request):
    """Return the authenticated user's profile.

    Used by the dashboard frontend (GET /api/auth) to get the current user.
    AuthMiddleware populates request.state.user from the OIDC token or
    X-User header (development mode).
    """
    username: str = getattr(request.state, "user", "unknown")
    email: str = getattr(request.state, "user_email", "")
    groups: list[str] = getattr(request.state, "user_groups", [])

    # Derive a human-readable display name from email or username
    display_name = email.split("@")[0].replace(".", " ").title() if email else username

    return {
        "sub": username,
        "name": display_name,
        "email": email,
        "role": "admin" if "admin" in groups else "ML Engineer",
        "groups": groups,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
