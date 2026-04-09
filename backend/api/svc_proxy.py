"""Service proxy utility.

Provides reliable access to cluster services via their public ingress URLs.
Each service is configured with:
  - cluster_url: internal K8s DNS (used when backend runs in-cluster)
  - ingress_url: public ALB/NLB URL (used when backend runs locally)

The proxy tries cluster_url first, then ingress_url.
"""

import asyncio
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ─── Ingress URLs (from `kubectl get ingress --all-namespaces`) ──────
# These are the real ALB/NLB endpoints provisioned by AWS.

_MLFLOW_INGRESS = os.getenv(
    "MLFLOW_INGRESS_URL",
    "http://k8s-monitori-mlflow-0985830759-1985412867.us-west-2.elb.amazonaws.com",
)
_PROMETHEUS_INGRESS = os.getenv(
    "PROMETHEUS_INGRESS_URL",
    "http://k8s-monitori-promethe-133c8f7ec6-1451950175.us-west-2.elb.amazonaws.com",
)
_FLYTE_INGRESS = os.getenv(
    "FLYTE_INGRESS_URL",
    "http://k8s-flyte-flytecon-a425d1f87c-1407955100.us-west-2.elb.amazonaws.com",
)
_FLYTE_GRPC_INGRESS = os.getenv(
    "FLYTE_GRPC_INGRESS_URL",
    "dns:///k8s-flyte-flytegrp-0908d4c3c6-dc2128ffa3e34d2f.elb.us-west-2.amazonaws.com:8089",
)
_KUBECOST_INGRESS = os.getenv(
    "KUBECOST_INGRESS_URL",
    "http://k8s-kubecost-kubecost-5b0bfdbe38-1593830500.us-west-2.elb.amazonaws.com",
)
_GRAFANA_INGRESS = os.getenv(
    "GRAFANA_INGRESS_URL",
    "http://k8s-monitori-grafana-8e443b47cf-1425963785.us-west-2.elb.amazonaws.com",
)
_JUPYTERHUB_INGRESS = os.getenv(
    "JUPYTERHUB_INGRESS_URL",
    "http://k8s-jupyter-jupyterh-827e6a6320-482154231.us-west-2.elb.amazonaws.com",
)
_TENSORBOARD_INGRESS = os.getenv(
    "TENSORBOARD_INGRESS_URL",
    "",
)


class ServiceEndpoint:
    """A cluster service with both in-cluster and ingress URLs."""

    def __init__(self, name: str, cluster_url: str, ingress_url: str):
        self.name = name
        self.cluster_url = cluster_url
        self.ingress_url = ingress_url

    def get_base_url(self) -> str:
        """Return the best reachable base URL (cached after first check)."""
        # We'll do the live check in the async methods below.
        # This is just for reference.
        return self.ingress_url


# ─── Service registry ────────────────────────────────────────────────

SERVICES: dict[str, ServiceEndpoint] = {
    "mlflow": ServiceEndpoint(
        name="mlflow",
        cluster_url=os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://mlflow.monitoring.svc.cluster.local:80",
        ),
        ingress_url=_MLFLOW_INGRESS,
    ),
    "prometheus": ServiceEndpoint(
        name="prometheus",
        cluster_url=os.getenv(
            "PROMETHEUS_URL",
            "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
        ),
        ingress_url=_PROMETHEUS_INGRESS,
    ),
    "flyte_http": ServiceEndpoint(
        name="flyte_http",
        cluster_url=os.getenv(
            "FLYTE_HTTP_URL",
            "http://flyte-binary-http.flyte.svc.cluster.local:8088",
        ),
        ingress_url=_FLYTE_INGRESS,
    ),
    "kubecost": ServiceEndpoint(
        name="kubecost",
        cluster_url=os.getenv(
            "KUBECOST_URL",
            "http://kubecost-cost-analyzer.kubecost.svc.cluster.local:9090",
        ),
        ingress_url=_KUBECOST_INGRESS,
    ),
    "grafana": ServiceEndpoint(
        name="grafana",
        cluster_url=os.getenv(
            "GRAFANA_URL",
            "http://kube-prometheus-stack-grafana.monitoring.svc.cluster.local:80",
        ),
        ingress_url=_GRAFANA_INGRESS,
    ),
    "jupyterhub": ServiceEndpoint(
        name="jupyterhub",
        cluster_url=os.getenv(
            "JUPYTERHUB_URL",
            "http://proxy-public.jupyter.svc.cluster.local:80",
        ),
        ingress_url=_JUPYTERHUB_INGRESS,
    ),
    "tensorboard": ServiceEndpoint(
        name="tensorboard",
        cluster_url=os.getenv(
            "TENSORBOARD_URL",
            "http://tensorboard.monitoring.svc.cluster.local:6006",
        ),
        ingress_url=_TENSORBOARD_INGRESS,
    ),
}

# Track which URL works for each service to avoid re-checking
_resolved_urls: dict[str, tuple[str, float]] = {}  # service -> (url, timestamp)
_resolve_lock = asyncio.Lock()
_RESOLVE_TTL = 300  # seconds — re-resolve after 5 minutes

# ─── Shared HTTP client ──────────────────────────────────────────────
# Module-level client with connection pooling.  Use ``close_client()``
# during application shutdown (registered in FastAPI lifespan).

_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return the shared httpx client, creating it lazily."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            follow_redirects=True,
        )
    return _http_client


async def close_client() -> None:
    """Close the shared httpx client.  Call during app shutdown."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


async def _resolve_url(service_name: str) -> str:
    """Determine the reachable base URL for a service.

    Tries cluster URL first (fast when in-cluster), then ingress URL.
    Caches the result with a TTL so stale entries don't persist forever.
    """
    cached = _resolved_urls.get(service_name)
    if cached is not None:
        url, ts = cached
        if time.monotonic() - ts < _RESOLVE_TTL:
            return url

    async with _resolve_lock:
        # Double-check after acquiring lock
        cached = _resolved_urls.get(service_name)
        if cached is not None:
            url, ts = cached
            if time.monotonic() - ts < _RESOLVE_TTL:
                return url

        svc = SERVICES[service_name]

        # Strategy 1: try cluster-internal URL
        try:
            client = _get_client()
            await client.get(f"{svc.cluster_url}/", timeout=3.0)
            # Any response (even 404) means it's reachable
            _resolved_urls[service_name] = (svc.cluster_url, time.monotonic())
            logger.info(f"Service {service_name}: using cluster URL {svc.cluster_url}")
            return svc.cluster_url
        except Exception:
            pass

        # Strategy 2: use ingress URL
        _resolved_urls[service_name] = (svc.ingress_url, time.monotonic())
        logger.info(f"Service {service_name}: using ingress URL {svc.ingress_url}")
        return svc.ingress_url


# ─── Public API ───────────────────────────────────────────────────────


async def svc_get(
    service_name: str,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: float = 3.0,
) -> dict:
    """GET request to a cluster service via ingress or cluster URL.

    Args:
        service_name: Key in SERVICES registry (e.g. 'mlflow', 'prometheus')
        path: URL path (e.g. '/api/v1/query')
        params: Query parameters
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response

    Raises:
        RuntimeError: If the service is unreachable
    """
    base_url = await _resolve_url(service_name)
    url = f"{base_url}/{path.lstrip('/')}"

    try:
        client = _get_client()
        resp = await client.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # If cached URL failed, clear cache and try the other URL
        _resolved_urls.pop(service_name, None)
        svc = SERVICES[service_name]
        alt_url = svc.ingress_url if base_url == svc.cluster_url else svc.cluster_url
        fallback_url = f"{alt_url}/{path.lstrip('/')}"
        try:
            resp = await client.get(fallback_url, params=params, timeout=timeout)
            resp.raise_for_status()
            _resolved_urls[service_name] = (alt_url, time.monotonic())
            return resp.json()
        except Exception as e2:
            raise RuntimeError(
                f"Service {service_name} unreachable via both cluster ({svc.cluster_url}) "
                f"and ingress ({svc.ingress_url}): {e2}"
            ) from e2


async def svc_post(
    service_name: str,
    path: str,
    body: dict | None = None,
    timeout: float = 3.0,
) -> dict:
    """POST request to a cluster service via ingress or cluster URL.

    Args:
        service_name: Key in SERVICES registry
        path: URL path
        body: JSON body
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response

    Raises:
        RuntimeError: If the service is unreachable
    """
    base_url = await _resolve_url(service_name)
    url = f"{base_url}/{path.lstrip('/')}"

    try:
        client = _get_client()
        resp = await client.post(
            url, json=body, headers={"Content-Type": "application/json"}, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        _resolved_urls.pop(service_name, None)
        svc = SERVICES[service_name]
        alt_url = svc.ingress_url if base_url == svc.cluster_url else svc.cluster_url
        fallback_url = f"{alt_url}/{path.lstrip('/')}"
        try:
            resp = await client.post(
                fallback_url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            _resolved_urls[service_name] = (alt_url, time.monotonic())
            return resp.json()
        except Exception as e2:
            raise RuntimeError(
                f"Service {service_name} unreachable via both cluster and ingress: {e2}"
            ) from e2


def get_flyte_grpc_endpoint() -> str:
    """Return the Flyte gRPC endpoint for FlyteRemote SDK.

    Uses the NLB endpoint when running locally, or cluster-internal DNS in-cluster.
    """
    return os.getenv(
        "FLYTE_ENDPOINT",
        # NLB endpoint (always reachable from local)
        "k8s-flyte-flytegrp-0908d4c3c6-dc2128ffa3e34d2f.elb.us-west-2.amazonaws.com:8089",
    )


def get_flyte_remote():
    """Get Flyte remote client using the public NLB endpoint."""
    from flytekit.configuration import Config, PlatformConfig
    from flytekit.remote import FlyteRemote

    endpoint = get_flyte_grpc_endpoint()
    project = os.getenv("FLYTE_PROJECT", "ml-platform")
    domain = os.getenv("FLYTE_DOMAIN", "development")

    config = Config(platform=PlatformConfig(endpoint=endpoint, insecure=True))
    return FlyteRemote(config=config, default_project=project, default_domain=domain)


def get_flyte_http_endpoint() -> str:
    """Return the Flyte HTTP API endpoint."""
    return _FLYTE_INGRESS
