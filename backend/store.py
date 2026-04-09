"""Pluggable key-value store abstraction.

Provides a unified interface for per-user data storage used by
``settings.py``, ``chat.py``, and any future module that needs
simple per-user state.

Currently only ``MemoryStore`` is implemented (suitable for
development / single-process mode).  To add persistence, implement
a new ``Store`` subclass (e.g. Redis, DynamoDB) and register it in
``get_store()``.

All stores implement the same async interface so no call-site changes
are needed when switching backends.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class Store(ABC):
    """Abstract key-value store with per-prefix namespacing."""

    @abstractmethod
    async def get(self, prefix: str, key: str) -> Any | None:
        """Get a value by prefix + key.  Returns None if not present."""

    @abstractmethod
    async def set(self, prefix: str, key: str, value: Any) -> None:
        """Set a value by prefix + key."""

    @abstractmethod
    async def delete(self, prefix: str, key: str) -> bool:
        """Delete a value.  Returns True if it existed."""

    @abstractmethod
    async def list(self, prefix: str) -> dict[str, Any]:
        """List all key→value pairs under a prefix."""

    @abstractmethod
    async def list_by_field(self, prefix: str, field: str, expected_value: Any) -> list[Any]:
        """Return values where record[field] == expected_value."""


class MemoryStore(Store):
    """In-memory dict-based store.  Suitable for development."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    async def get(self, prefix: str, key: str) -> Any | None:
        return self._data.get(prefix, {}).get(key)

    async def set(self, prefix: str, key: str, value: Any) -> None:
        self._data.setdefault(prefix, {})[key] = value

    async def delete(self, prefix: str, key: str) -> bool:
        bucket = self._data.get(prefix, {})
        if key in bucket:
            del bucket[key]
            return True
        return False

    async def list(self, prefix: str) -> dict[str, Any]:
        return dict(self._data.get(prefix, {}))

    async def list_by_field(self, prefix: str, field: str, expected_value: Any) -> list[Any]:
        return [v for v in self._data.get(prefix, {}).values() if v.get(field) == expected_value]


# ── Singleton factory ──────────────────────────────────────────────

_store_instance: Store | None = None


def get_store() -> Store:
    """Return the global store singleton.

    The backend is selected by the ``STORE_BACKEND`` environment variable.
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    backend = os.getenv("STORE_BACKEND", "memory").lower()

    if backend == "memory":
        _store_instance = MemoryStore()
    else:
        logger.warning(f"Unknown STORE_BACKEND={backend!r}, falling back to memory")
        _store_instance = MemoryStore()

    logger.info(f"Store backend: {type(_store_instance).__name__}")
    return _store_instance
