"""User settings and preferences API endpoints.

Stores per-user defaults (GPU type, namespace, editor preferences,
auto-suspend policy, cost alert thresholds).  Uses the pluggable
``backend.store`` abstraction (in-memory by default, Redis-ready).
"""

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend.store import get_store

logger = logging.getLogger(__name__)
router = APIRouter()

_STORE_PREFIX = "settings"


class UserSettings(BaseModel):
    """User preference model matching v3.12 Settings page."""

    # Defaults
    default_namespace: str = "ml-team"
    default_gpu_type: str = "A100"
    default_recipe_profile: str = "Medium"

    # Editor
    editor_font_size: int = 14
    editor_key_bindings: str = "default"  # default | vim | emacs

    # Cost preferences
    burn_rate_display: str = "badge"  # badge | inline | hidden
    budget_alerts_enabled: bool = True
    weekly_report_email: bool = True
    budget_limit_monthly: float = 50_000.0

    # Auto-suspend
    idle_timeout_minutes: int = 120
    auto_suspend_action: str = "suspend"  # suspend | terminate
    ghost_desk_protection: bool = True


def _get_user(request: Request) -> str:
    """Extract user from request state (set by AuthMiddleware)."""
    return getattr(request.state, "user", "unknown")


@router.get("")
async def get_settings(request: Request) -> UserSettings:
    """Get current user's settings. Returns defaults if not configured."""
    user = _get_user(request)
    store = get_store()
    data = await store.get(_STORE_PREFIX, user)
    if data is None:
        return UserSettings()
    return UserSettings(**data)


@router.put("")
async def update_settings(settings: UserSettings, request: Request) -> dict[str, Any]:
    """Update current user's settings."""
    user = _get_user(request)
    store = get_store()
    await store.set(_STORE_PREFIX, user, settings.model_dump())
    logger.info(f"Settings updated for user: {user}")
    return {"status": "updated", "user": user}


@router.patch("")
async def patch_settings(updates: dict[str, Any], request: Request) -> UserSettings:
    """Partially update settings (PATCH semantics)."""
    user = _get_user(request)
    store = get_store()
    data = await store.get(_STORE_PREFIX, user)
    current = UserSettings(**(data or {}))
    updated = current.model_copy(update=updates)
    await store.set(_STORE_PREFIX, user, updated.model_dump())
    logger.info(f"Settings patched for user: {user} (fields: {list(updates.keys())})")
    return updated
