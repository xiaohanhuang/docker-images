"""
Server-side recipe registry (PostgreSQL + S3).

This is the in-cluster version used by the remote-agent.  It connects to
Postgres at ``postgres.postgres.svc.cluster.local:5432`` (or env overrides)
and uses the node's IAM role for S3 access.

Mirrors the public API of ``cli.recipe_engine.registry.RecipeRegistry``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import psycopg2

logger = logging.getLogger(__name__)

# ── Status constants ──────────────────────────────────────────────────────

STATUS_VERIFIED = "verified"
STATUS_EXPERIMENTAL = "experimental"
STATUS_FAILING = "failing"

_VALID_STATUSES = {STATUS_VERIFIED, STATUS_EXPERIMENTAL, STATUS_FAILING}

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS recipe_registry (
    recipe_name          TEXT        NOT NULL,
    version              TEXT        NOT NULL,
    s3_key               TEXT        NOT NULL,
    archive_name         TEXT        NOT NULL,
    verification_status  TEXT        NOT NULL DEFAULT 'experimental',
    verified_profile     TEXT,
    verified_at          TIMESTAMPTZ,
    tags                 TEXT[]      DEFAULT '{}',
    pushed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata             JSONB       DEFAULT '{}',
    PRIMARY KEY (recipe_name, version)
);
"""


def _parse_archive_name(archive_name: str) -> tuple[str, str]:
    if not archive_name.endswith(".ml-plat"):
        raise ValueError(f"Invalid archive name: {archive_name}")
    base = archive_name.removesuffix(".ml-plat")
    parts = base.rsplit("-v", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid archive format: {archive_name}")
    return parts[0], parts[1]


def _parse_semver(version: str) -> tuple:
    try:
        return tuple(int(p) for p in version.split("."))
    except Exception:
        return (0, 0, 0)


def _validate_status(status: str) -> None:
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"Invalid status: {status!r}. Must be one of: {sorted(_VALID_STATUSES)}"
        )


class RecipeRegistry:
    """PostgreSQL + S3 recipe registry (server-side)."""

    def __init__(
        self,
        s3_bucket: str,
        pg_host: str = "postgres.postgres.svc.cluster.local",
        pg_port: int = 5432,
        pg_dbname: str = "flyte",
        pg_user: str = "flyte",
        pg_password: str = "ml-platform-pg-2026",
    ):
        self.s3_bucket = s3_bucket
        self._pg_host = pg_host
        self._pg_port = pg_port
        self._pg_dbname = pg_dbname
        self._pg_user = pg_user
        self._pg_password = pg_password
        self._conn = None
        self._s3_client = None

    @property
    def s3(self):
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=self._pg_host,
                port=self._pg_port,
                dbname=self._pg_dbname,
                user=self._pg_user,
                password=self._pg_password,
            )
            self._conn.autocommit = True
            with self._conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
        return self._conn

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

        _validate_status(verification_status)
        recipe_name, version = _parse_archive_name(archive_path.name)

        if not overwrite and self._version_exists(recipe_name, version):
            raise ValueError(
                f"Recipe {recipe_name} version {version} already exists. "
                "Use overwrite=True to replace it."
            )

        s3_key = f"recipes/{recipe_name}/{version}/{archive_path.name}"
        logger.info("Uploading %s → s3://%s/%s", archive_path, self.s3_bucket, s3_key)
        self.s3.upload_file(str(archive_path), self.s3_bucket, s3_key)

        now = datetime.now(timezone.utc)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO recipe_registry
                    (recipe_name, version, s3_key, archive_name,
                     verification_status, verified_profile, tags, pushed_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (recipe_name, version) DO UPDATE SET
                    s3_key = EXCLUDED.s3_key,
                    archive_name = EXCLUDED.archive_name,
                    verification_status = EXCLUDED.verification_status,
                    verified_profile = EXCLUDED.verified_profile,
                    tags = EXCLUDED.tags,
                    pushed_at = EXCLUDED.pushed_at,
                    metadata = EXCLUDED.metadata
                """,
                (
                    recipe_name, version, s3_key, archive_path.name,
                    verification_status, profile, tags or [], now, json.dumps({}),
                ),
            )

        return {
            "recipe_name": recipe_name,
            "version": version,
            "s3_uri": f"s3://{self.s3_bucket}/{s3_key}",
            "s3_key": s3_key,
            "verification_status": verification_status,
            "profile": profile,
            "tags": tags or [],
            "pushed_at": now.isoformat(),
            "archive_name": archive_path.name,
        }

    def pull(
        self,
        recipe_name: str,
        version: str = "latest",
        output_path: Optional[Path] = None,
    ) -> Path:
        if version == "latest":
            versions = self._list_versions(recipe_name)
            if not versions:
                raise ValueError(f"No versions found for recipe: {recipe_name}")
            version = versions[0]

        row = self._get_row(recipe_name, version)
        if row is None:
            raise ValueError(f"Recipe not found: {recipe_name} v{version}")

        s3_key = row["s3_key"]
        archive_name = row["archive_name"]

        if output_path is None:
            output_path = Path("/tmp") / archive_name
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / archive_name

        self.s3.download_file(self.s3_bucket, s3_key, str(output_path))
        return output_path

    def list_recipes(
        self,
        tags: Optional[List[str]] = None,
        verification_status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if verification_status:
            clauses.append("verification_status = %s")
            params.append(verification_status)
        if tags:
            clauses.append("tags && %s")
            params.append(tags)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM recipe_registry{where} ORDER BY pushed_at DESC"

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [self._row_to_dict(cols, r) for r in cur.fetchall()]

    def get_verification_status(
        self, recipe_name: str, version: str, profile: str,
    ) -> Optional[str]:
        row = self._get_row(recipe_name, version)
        if row is None:
            return None
        stored_profile = row.get("verified_profile")
        if stored_profile and stored_profile != profile:
            return STATUS_EXPERIMENTAL
        return row.get("verification_status")

    def set_verification_status(
        self,
        recipe_name: str,
        version: str,
        profile: str,
        status: str,
        canary_results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        _validate_status(status)
        row = self._get_row(recipe_name, version)
        if row is None:
            raise ValueError(f"Recipe not found: {recipe_name} v{version}")

        now = datetime.now(timezone.utc)
        meta = row.get("metadata") or {}
        if canary_results:
            meta["canary_results"] = canary_results

        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE recipe_registry
                   SET verification_status = %s,
                       verified_profile    = %s,
                       verified_at         = %s,
                       metadata            = %s
                 WHERE recipe_name = %s AND version = %s
                """,
                (status, profile, now, json.dumps(meta), recipe_name, version),
            )
        return True

    def delete(self, recipe_name: str, version: str) -> bool:
        row = self._get_row(recipe_name, version)
        if row is None:
            raise ValueError(f"Recipe not found: {recipe_name} v{version}")

        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM recipe_registry WHERE recipe_name = %s AND version = %s",
                (recipe_name, version),
            )
        return True

    # ── Private helpers ───────────────────────────────────────────────────

    def _version_exists(self, recipe_name: str, version: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM recipe_registry WHERE recipe_name = %s AND version = %s",
                (recipe_name, version),
            )
            return cur.fetchone() is not None

    def _list_versions(self, recipe_name: str) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT version FROM recipe_registry WHERE recipe_name = %s",
                (recipe_name,),
            )
            versions = [r[0] for r in cur.fetchall()]
        return sorted(versions, key=_parse_semver, reverse=True)

    def _get_row(self, recipe_name: str, version: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM recipe_registry WHERE recipe_name = %s AND version = %s",
                (recipe_name, version),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return self._row_to_dict(cols, row)

    @staticmethod
    def _row_to_dict(cols: list[str], row: tuple) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for col, val in zip(cols, row):
            if hasattr(val, "isoformat"):
                d[col] = val.isoformat()
            else:
                d[col] = val
        return d
