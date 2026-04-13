"""
Shared utilities for the ml-plat CLI.

Platform configuration is loaded from ``~/.ml-plat/config.yaml``.  All
service endpoints (Flyte gRPC, PostgreSQL, Flyte Console, MLflow) are
resolved from this file so no manual ``kubectl port-forward`` is needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
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
        import grpc
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
    from flytekit.configuration import Config
    from flytekit.remote import FlyteRemote

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


# ── Image version resolution ─────────────────────────────────────────────

_ECR_REGISTRY_DEFAULT = "805673386114.dkr.ecr.us-west-2.amazonaws.com"
_ECR_REPO = "ml-platform"


def _find_versions_env() -> Path | None:
    """Walk up from CWD to find images/versions.env."""
    start = Path.cwd().resolve()
    for parent in (start, *start.parents):
        candidate = parent / "images" / "versions.env"
        if candidate.exists():
            return candidate
    return None


def _parse_versions_env(path: Path) -> dict[str, str]:
    """Parse a Makefile-style KEY := VALUE file into a dict."""
    result: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in (":=", "="):
            if f" {sep} " in line or line.find(sep) > 0:
                key, _, val = line.partition(sep)
                result[key.strip()] = val.strip()
                break
    return result


def resolve_image_tag(image_short_name: str) -> str:
    """Resolve a semver tag for the given image from env or versions.env.

    Lookup order:
      1. ``IMAGE_TAG_<SHORT_NAME>`` environment variable
      2. ``IMAGE_TAG_<SHORT_NAME>`` in ``images/versions.env``
      3. Global ``IMAGE_TAG`` in ``images/versions.env``

    Args:
        image_short_name: e.g. "WORKFLOW_CPU", "RAY_WORKER", "ML_GPU"

    Returns:
        Semver tag string like "1.1.0".
    """
    env_key = f"IMAGE_TAG_{image_short_name.upper()}"
    tag = os.getenv(env_key)
    if tag:
        return tag

    versions_file = _find_versions_env()
    if versions_file:
        versions = _parse_versions_env(versions_file)
        tag = versions.get(env_key) or versions.get("IMAGE_TAG")
        if tag:
            return tag

    return "1.0.0"


def resolve_image(image_short_name: str, ecr_name: str | None = None) -> str:
    """Return a fully qualified ECR image reference with semver tag.

    Args:
        image_short_name: upper-case key suffix, e.g. "RAY_WORKER"
        ecr_name: ECR repository name, e.g. "ray-worker".
                  Defaults to lower-case of image_short_name with _ -> -.

    Returns:
        e.g. "805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/ray-worker:1.1.0"
    """
    registry = os.getenv("ECR_REGISTRY", _ECR_REGISTRY_DEFAULT)
    if ecr_name is None:
        ecr_name = image_short_name.lower().replace("_", "-")
    tag = resolve_image_tag(image_short_name)
    return f"{registry}/{_ECR_REPO}/{ecr_name}:{tag}"
