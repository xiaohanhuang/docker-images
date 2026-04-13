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

    env = os.getenv("ML_PLAT_REGISTRY_URL")
    if env:
        return env.rstrip("/")

    cfg = _load_platform_config()
    cfg_url = cfg.get("registry_url")
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

    def __init__(self, registry_url: Optional[str] = None, timeout: int = 60):
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
        presets: Optional[Dict[str, Any]] = None,
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
                "presets": json.dumps(presets or {}),
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
        profile: Optional[str] = None,
    ) -> Path:
        params: Dict[str, str] = {"version": version}
        if profile:
            params["profile"] = profile
        resp = requests.get(
            self._url(f"/pull/{recipe_name}"),
            params=params,
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

    # ── versions ─────────────────────────────────────────────────────────

    def list_versions(
        self,
        recipe_name: str,
    ) -> List[Dict[str, Any]]:
        """List all versions of a recipe in the registry.

        Returns a list of dicts (one per version) with keys like
        ``recipe_name``, ``version``, ``verification_status``, ``pushed_at``, etc.
        Falls back to filtering the full list when a dedicated endpoint is
        unavailable.
        """
        # Try a dedicated endpoint first; fall back to filtering /list.
        try:
            resp = requests.get(
                self._url(f"/versions/{recipe_name}"),
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # Fallback: fetch all recipes and filter by name
        all_recipes = self.list_recipes()
        return [r for r in all_recipes if r.get("recipe_name", r.get("name")) == recipe_name]

    # ── delete ────────────────────────────────────────────────────────────

    def delete(self, recipe_name: str, version: str, profile: Optional[str] = None) -> bool:
        params: Dict[str, str] = {}
        if profile:
            params["profile"] = profile
        resp = requests.delete(
            self._url(f"/{recipe_name}/{version}"),
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            raise ValueError(f"Recipe not found: {recipe_name} v{version}")
        resp.raise_for_status()
        return True
