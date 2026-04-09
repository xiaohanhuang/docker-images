"""Shared lazy Kubernetes client singleton.

Provides a single ``get_core_v1()`` function that loads kubeconfig on
first call and returns a cached ``CoreV1Api`` instance.  All backend
modules should use this instead of loading config at module level.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from kubernetes import client, config
    from kubernetes.config import ConfigException
except ImportError:
    client = None  # type: ignore[assignment]
    config = None  # type: ignore[assignment]
    ConfigException = Exception  # type: ignore[assignment]

_v1_cached: "client.CoreV1Api | None" = None  # type: ignore[name-defined]
_config_loaded = False


def ensure_config() -> None:
    """Load kubeconfig exactly once. Safe to call multiple times."""
    global _config_loaded
    if _config_loaded or config is None:
        return
    try:
        config.load_incluster_config()
        _config_loaded = True
    except ConfigException:
        try:
            config.load_kube_config()
            _config_loaded = True
        except ConfigException:
            logger.warning("Could not load Kubernetes config")
            _config_loaded = True  # mark as attempted


def get_core_v1() -> "client.CoreV1Api":
    """Return a (cached) CoreV1Api, loading config on first use.

    Raises:
        RuntimeError: If the ``kubernetes`` package is not installed.
    """
    global _v1_cached
    if client is None:
        raise RuntimeError("kubernetes package not installed")
    if _v1_cached is None:
        ensure_config()
        _v1_cached = client.CoreV1Api()
    return _v1_cached


def is_available() -> bool:
    """Return True if the kubernetes client library is installed."""
    return client is not None
