"""
Registry API endpoints for the ML Platform Registry Service.

Exposes recipe registry operations (list, push, pull, status) as HTTP endpoints
so the CLI can talk to the platform without direct Postgres/S3 access.

The registry service runs in-cluster and connects to Postgres at
``postgres.postgres.svc.cluster.local:5432`` and S3 via IAM role.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Security,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Security: API Token Authentication
API_TOKEN = os.getenv("REGISTRY_SERVICE_API_TOKEN")
API_TOKEN_HASH = hashlib.sha256(API_TOKEN.encode()).hexdigest() if API_TOKEN else None

_security = HTTPBearer(auto_error=False)


def verify_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_security),
) -> bool:
    """Verify Bearer token. No-op if REGISTRY_SERVICE_API_TOKEN is unset."""
    if not API_TOKEN:
        return True
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    if not secrets.compare_digest(provided_hash, API_TOKEN_HASH):
        raise HTTPException(
            status_code=401,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


router = APIRouter(
    prefix="/registry",
    tags=["registry"],
    dependencies=[Security(verify_api_token)],
)

# ── Postgres config (in-cluster defaults) ─────────────────────────────────

_PG_HOST = os.getenv("POSTGRES_HOST", "postgres.postgres.svc.cluster.local")
_PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
_PG_DB = os.getenv("POSTGRES_DB", "flyte")
_PG_USER = os.getenv("POSTGRES_USER", "flyte")
_PG_PASS = os.getenv("POSTGRES_PASSWORD", "")
_S3_BUCKET = os.getenv("S3_BUCKET", "")

if not _S3_BUCKET:
    raise SystemExit("[registry] S3_BUCKET env var must be set")
if not _PG_PASS:
    raise SystemExit("[registry] POSTGRES_PASSWORD env var must be set")


def _get_registry():
    """Lazy-import and return a RecipeRegistry configured for in-cluster use."""
    from registry_db import RecipeRegistry

    return RecipeRegistry(
        s3_bucket=_S3_BUCKET,
        pg_host=_PG_HOST,
        pg_port=_PG_PORT,
        pg_dbname=_PG_DB,
        pg_user=_PG_USER,
        pg_password=_PG_PASS,
    )


# ── Response models ───────────────────────────────────────────────────────


class RecipeEntry(BaseModel):
    recipe_name: str
    version: str
    s3_key: str = ""
    archive_name: str = ""
    verification_status: str = "experimental"
    verified_profile: Optional[str] = None
    verified_at: Optional[str] = None
    tags: List[str] = []
    pushed_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PushResult(BaseModel):
    recipe_name: str
    version: str
    s3_uri: str
    s3_key: str
    verification_status: str
    profile: Optional[str] = None
    tags: List[str] = []
    pushed_at: str
    archive_name: str


class StatusUpdate(BaseModel):
    recipe_name: str
    version: str
    profile: str
    status: str
    canary_results: Optional[Dict[str, Any]] = None


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.get("/list", response_model=List[RecipeEntry])
def list_recipes(
    tag: Optional[List[str]] = Query(None),
    status: Optional[str] = Query(None),
):
    """List recipes in the registry, optionally filtered by tags or status."""
    try:
        registry = _get_registry()
        recipes = registry.list_recipes(
            tags=tag or None,
            verification_status=status,
        )
        return recipes
    except Exception as exc:
        logger.exception("Failed to list recipes")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/push", response_model=PushResult)
async def push_recipe(
    archive: UploadFile = File(...),
    verification_status: str = Form("experimental"),
    profile: Optional[str] = Form(None),
    tags: str = Form("[]"),  # JSON-encoded list
    overwrite: bool = Form(False),
):
    """Upload a recipe archive to S3 and register it in PostgreSQL."""
    try:
        tags_list: List[str] = json.loads(tags) if tags else []
    except (json.JSONDecodeError, TypeError):
        tags_list = []

    # Stream upload into a per-request temp directory to avoid filename
    # collisions between concurrent uploads.
    tmpdir = tempfile.mkdtemp(prefix="registry-upload-")
    with tempfile.NamedTemporaryFile(suffix=".ml-plat", delete=False, dir=tmpdir) as tmp:
        shutil.copyfileobj(archive.file, tmp)
        tmp_path = Path(tmp.name)

    # Sanitise the client-supplied filename to a plain basename to prevent
    # path-traversal attacks (e.g. '../../../etc/passwd' as filename).
    safe_name = os.path.basename(archive.filename or "") or "upload.ml-plat"
    archive_path = Path(tmpdir) / safe_name
    tmp_path.rename(archive_path)

    try:
        registry = _get_registry()
        result = registry.push(
            archive_path=archive_path,
            verification_status=verification_status,
            profile=profile,
            tags=tags_list,
            overwrite=overwrite,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to push recipe")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


@router.get("/pull/{recipe_name}")
def pull_recipe(
    recipe_name: str,
    version: str = Query("latest"),
):
    """Download a recipe archive from the registry.

    Returns the .ml-plat file as a streaming binary download.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        registry = _get_registry()
        archive_path = registry.pull(
            recipe_name=recipe_name,
            version=version,
            output_path=Path(tmpdir),
        )
    except ValueError as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.exception("Failed to pull recipe")
        raise HTTPException(status_code=500, detail=str(exc))

    # Stream the file and clean up after delivery.
    def iterfile():
        try:
            with open(archive_path, "rb") as f:
                while chunk := f.read(64 * 1024):
                    yield chunk
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return StreamingResponse(
        iterfile(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{archive_path.name}"',
            "X-Recipe-Name": recipe_name,
            "X-Recipe-Version": version,
        },
    )


@router.get("/status/{recipe_name}/{version}")
def get_status(
    recipe_name: str,
    version: str,
    profile: str = Query(...),
):
    """Get verification status for a recipe version + profile."""
    try:
        registry = _get_registry()
        status = registry.get_verification_status(recipe_name, version, profile)
        if status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Recipe not found: {recipe_name} v{version}",
            )
        return {"recipe_name": recipe_name, "version": version, "status": status}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get status")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/status")
def set_status(body: StatusUpdate):
    """Update verification status for a recipe version + profile."""
    try:
        registry = _get_registry()
        registry.set_verification_status(
            recipe_name=body.recipe_name,
            version=body.version,
            profile=body.profile,
            status=body.status,
            canary_results=body.canary_results,
        )
        return {"ok": True}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to set status")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{recipe_name}/{version}")
def delete_recipe(recipe_name: str, version: str):
    """Delete a recipe version from the registry."""
    try:
        registry = _get_registry()
        registry.delete(recipe_name, version)
        return {"ok": True, "deleted": f"{recipe_name} v{version}"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to delete recipe")
        raise HTTPException(status_code=500, detail=str(exc))
