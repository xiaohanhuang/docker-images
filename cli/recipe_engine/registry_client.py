"""HTTP client for the recipe registry service.

All registry operations (list, push, pull, status) go through the registry-service's
``/registry/*`` endpoints so the CLI never needs direct Postgres or S3
access.

Registry service URL resolution (highest → lowest priority):
1. Explicit ``registry_url`` parameter
2. ``ML_PLAT_REGISTRY_URL`` env var
3. ``registry_url`` key in ``~/.ml-plat/config.yaml``
4. Service endpoint auto-discovered via ``kubectl get ingress``
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Status constants (mirror server-side) ─────────────────────────────────

STATUS_VERIFIED = "verified"
STATUS_EXPERIMENTAL = "experimental"
STATUS_FAILING = "failing"


def _load_platform_config() -> dict:
    """Read ``~/.ml-plat/config.yaml`` and return the raw dict (or ``{}``)."""
    try:
        import yaml

        cfg_path = Path.home() / ".ml-plat" / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as fh:
                return yaml.safe_load(fh) or {}
    except Exception:
        pass
    return {}


def _discover_registry_url_from_service() -> Optional[str]:
    """Try ``kubectl get ingress registry-service`` to find the ALB hostname."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "ingress",
                "registry-service",
                "-o",
                "jsonpath={.status.loadBalancer.ingress[0].hostname}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        hostname = result.stdout.strip()
        if hostname and result.returncode == 0:
            return f"http://{hostname}"
    except Exception:
        pass
    return None


def get_registry_url(explicit_url: Optional[str] = None) -> str:
    """Resolve the registry service base URL.

    Priority:
    1. ``explicit_url`` parameter
    2. ``ML_PLAT_REGISTRY_URL`` env var
    3. ``registry_url`` in ``~/.ml-plat/config.yaml``
    4. Auto-discover from ``kubectl get ingress registry-service``
    5. Fallback ``http://localhost:8766``
    """
    if explicit_url:
        return explicit_url.rstrip("/")

    env = os.getenv("ML_PLAT_REGISTRY_URL") or os.getenv("ML_PLAT_AGENT_URL")
    if env:
        return env.rstrip("/")

    cfg = _load_platform_config()
    cfg_url = cfg.get("registry_url") or cfg.get("agent_url")
    if cfg_url:
        return cfg_url.rstrip("/")

    discovered = _discover_registry_url_from_service()
    if discovered:
        return discovered.rstrip("/")

    return "http://localhost:8766"


class RegistryClient:
    """HTTP client that talks to the registry-service's ``/registry/*`` endpoints."""

    # Re-export for backward compat with code that references these
    STATUS_VERIFIED = STATUS_VERIFIED
    STATUS_EXPERIMENTAL = STATUS_EXPERIMENTAL
    STATUS_FAILING = STATUS_FAILING

    def __init__(self, registry_url: Optional[str] = None, timeout: int = 60, **kwargs):
        # Backward compat: accept old agent_url parameter
        if registry_url is None:
            registry_url = kwargs.get("agent_url")
        self.base_url = get_registry_url(registry_url)
        self.timeout = timeout
        self._api_token = os.getenv("REGISTRY_SERVICE_API_TOKEN")

    def _url(self, path: str) -> str:
        return f"{self.base_url}/registry{path}"

    def _headers(self) -> Dict[str, str]:
        if self._api_token:
            return {"Authorization": f"Bearer {self._api_token}"}
        return {}

    # ── list ──────────────────────────────────────────────────────────────

    def list_recipes(
        self,
        tags: Optional[List[str]] = None,
        verification_status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if tags:
            params["tag"] = tags
        if verification_status:
            params["status"] = verification_status

        resp = requests.get(
            self._url("/list"), params=params, headers=self._headers(), timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    # ── push ──────────────────────────────────────────────────────────────

    def push(
        self,
        archive_path: Path,
        verification_status: str = STATUS_EXPERIMENTAL,
        profile: Optional[str] = None,
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        with open(archive_path, "rb") as f:
            files = {"archive": (archive_path.name, f, "application/octet-stream")}
            data = {
                "verification_status": verification_status,
                "tags": json.dumps(tags or []),
                "overwrite": str(overwrite).lower(),
            }
            if profile:
                data["profile"] = profile

            resp = requests.post(
                self._url("/push"),
                files=files,
                data=data,
                headers=self._headers(),
                timeout=max(self.timeout, 300),  # uploads may be large
            )

        if resp.status_code == 409:
            detail = resp.json().get("detail", "Version already exists")
            raise ValueError(detail)
        resp.raise_for_status()
        return resp.json()

    # ── pull ──────────────────────────────────────────────────────────────

    def pull(
        self,
        recipe_name: str,
        version: str = "latest",
        output_path: Optional[Path] = None,
    ) -> Path:
        resp = requests.get(
            self._url(f"/pull/{recipe_name}"),
            params={"version": version},
            headers=self._headers(),
            stream=True,
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            detail = resp.json().get("detail", f"Recipe not found: {recipe_name}")
            raise ValueError(detail)
        resp.raise_for_status()

        # Determine output filename from Content-Disposition header
        cd = resp.headers.get("Content-Disposition", "")
        filename = ""
        if 'filename="' in cd:
            filename = cd.split('filename="')[1].rstrip('"')
        if not filename:
            filename = f"{recipe_name}-v{version}.ml-plat"

        if output_path is None:
            dest = Path.cwd() / filename
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                dest = output_path / filename
            else:
                dest = output_path

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                f.write(chunk)

        return dest

    # ── status ────────────────────────────────────────────────────────────

    def get_verification_status(
        self,
        recipe_name: str,
        version: str,
        profile: str,
    ) -> Optional[str]:
        resp = requests.get(
            self._url(f"/status/{recipe_name}/{version}"),
            params={"profile": profile},
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("status")

    def set_verification_status(
        self,
        recipe_name: str,
        version: str,
        profile: str,
        status: str,
        canary_results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        payload = {
            "recipe_name": recipe_name,
            "version": version,
            "profile": profile,
            "status": status,
        }
        if canary_results:
            payload["canary_results"] = canary_results

        resp = requests.post(
            self._url("/status"), json=payload, headers=self._headers(), timeout=self.timeout
        )
        if resp.status_code == 404:
            raise ValueError(f"Recipe not found: {recipe_name} v{version}")
        resp.raise_for_status()
        return True

    # ── delete ────────────────────────────────────────────────────────────

    def delete(self, recipe_name: str, version: str) -> bool:
        resp = requests.delete(
            self._url(f"/{recipe_name}/{version}"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            raise ValueError(f"Recipe not found: {recipe_name} v{version}")
        resp.raise_for_status()
        return True
