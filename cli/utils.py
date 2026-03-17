"""
Shared utilities for the ml-plat CLI.

Platform configuration is loaded from ``~/.ml-plat/config.yaml``.  All
service endpoints (Flyte gRPC, PostgreSQL, Flyte Console, MLflow) are
resolved from this file so no manual ``kubectl port-forward`` is needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import grpc
from flytekit.configuration import Config
from flytekit.remote import FlyteRemote
from rich.console import Console

console = Console()

# ── Platform config ───────────────────────────────────────────────────────


def _load_platform_config() -> Dict[str, Any]:
    """Read ``~/.ml-plat/config.yaml`` and return the raw dict (or ``{}``)."""
    import yaml  # lazy – avoid import cost at CLI registration time

    cfg_path = Path.home() / ".ml-plat" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def platform_config() -> Dict[str, Any]:
    """Return the ``cluster`` section of the platform config."""
    return _load_platform_config().get("cluster", {})


# ── Flyte helpers ─────────────────────────────────────────────────────────


def flyte_console_url(project: str, domain: str, execution_id: str) -> str:
    """Build a Flyte Console URL for an execution.

    Resolution order:
    1. ``FLYTE_CONSOLE_URL`` env var
    2. ``cluster.flyte_console_url`` in ``~/.ml-plat/config.yaml``
    3. Empty string (URL will be relative)
    """
    base = os.getenv("FLYTE_CONSOLE_URL", "").rstrip("/")
    if not base:
        cfg = platform_config()
        base = cfg.get("flyte_console_url", "")
    return f"{base}/console/projects/{project}/domains/{domain}/executions/{execution_id}"


def _patch_auth_interceptor() -> None:
    """Patch flytekit auth interceptor for non-auth Flyte servers.

    When connecting via NLB to a Flyte server without authentication,
    the auth interceptor tries to call ``GetPublicClientConfig`` which
    fails with ``UNIMPLEMENTED`` on non-auth servers.  This patch wraps
    the entire interceptor so that auth failures are swallowed and the
    gRPC call proceeds without auth headers (correct for non-auth servers).
    """
    try:
        from flytekit.clients.grpc_utils.auth_interceptor import AuthUnaryInterceptor

        if getattr(AuthUnaryInterceptor, "_ml_plat_patched", False):
            return  # Already patched

        _orig = AuthUnaryInterceptor.intercept_unary_unary

        def _safe_intercept(self, continuation, client_call_details, request):  # type: ignore[override]
            try:
                return _orig(self, continuation, client_call_details, request)
            except grpc.RpcError as rpc_err:
                # Non-auth Flyte server returns UNIMPLEMENTED for GetPublicClientConfig
                if hasattr(rpc_err, "code") and rpc_err.code() == grpc.StatusCode.UNIMPLEMENTED:
                    return continuation(client_call_details, request)
                raise

        AuthUnaryInterceptor.intercept_unary_unary = _safe_intercept  # type: ignore[assignment]
        AuthUnaryInterceptor._ml_plat_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass  # If patching fails, proceed anyway


def flyte_remote() -> FlyteRemote:
    """Create a ``FlyteRemote`` using the current environment / config.

    Resolution order for the gRPC endpoint:
    1. ``FLYTE_ENDPOINT`` env var
    2. ``cluster.flyte_endpoint`` in ``~/.ml-plat/config.yaml``
    3. ``~/.flyte/config.yaml`` (flytekit native config)
    4. Fall back to ``dns:///localhost:8089``
    """
    _patch_auth_interceptor()
    cfg = platform_config()

    endpoint = os.getenv("FLYTE_ENDPOINT") or cfg.get("flyte_endpoint")
    if endpoint:
        fly_cfg = Config.for_endpoint(endpoint, insecure=True)
    elif Path.home().joinpath(".flyte", "config.yaml").exists():
        fly_cfg = Config.auto()
    else:
        fly_cfg = Config.for_endpoint("localhost:8089", insecure=True)

    return FlyteRemote(
        config=fly_cfg,
        default_project=os.getenv("FLYTE_PROJECT", cfg.get("flyte_project", "flytesnacks")),
        default_domain=os.getenv("FLYTE_DOMAIN", cfg.get("flyte_domain", "development")),
    )
