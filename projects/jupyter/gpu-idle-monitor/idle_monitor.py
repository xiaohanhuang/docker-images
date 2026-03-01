#!/usr/bin/env python3
"""
GPU Idle-Shutdown Monitor

Sidecar container that monitors a GPU notebook pod for Jupyter kernel/terminal
activity and SSH sessions. After IDLE_THRESHOLD_SECONDS of detected inactivity
the pod is deleted via the Kubernetes API, allowing the cluster to reclaim
expensive GPU resources.

Configuration (environment variables):
  JUPYTER_URL             Jupyter server base URL (default: http://localhost:8888)
  JUPYTER_TOKEN_FILE      File containing the JupyterHub API token
                          (default: /var/idle-monitor/token)
  IDLE_THRESHOLD_SECONDS  Seconds of inactivity before pod deletion (default: 1800)
  CHECK_INTERVAL_SECONDS  Polling interval in seconds (default: 60)
  POD_NAME                Pod to delete — injected by Kubernetes Downward API
  POD_NAMESPACE           Namespace of the pod (default: jupyter)
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from kubernetes import client, config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

JUPYTER_URL = os.environ.get("JUPYTER_URL", "http://localhost:8888")
JUPYTER_TOKEN_FILE = os.environ.get("JUPYTER_TOKEN_FILE", "/var/idle-monitor/token")
IDLE_THRESHOLD_SECONDS = int(os.environ.get("IDLE_THRESHOLD_SECONDS", "1800"))
CHECK_INTERVAL_SECONDS = int(os.environ.get("CHECK_INTERVAL_SECONDS", "60"))
POD_NAME = os.environ.get("POD_NAME", "")
POD_NAMESPACE = os.environ.get("POD_NAMESPACE", "jupyter")

# SSH port in upper-case hex as it appears in /proc/net/tcp local_address field
_SSH_PORT_HEX = "0016"
# TCP state code for ESTABLISHED connections
_TCP_ESTABLISHED = "01"


def _read_token() -> str:
    """Read the JupyterHub API token from the shared token file.

    Returns an empty string when the file is missing or unreadable (e.g. before
    the main container has finished its postStart hook).
    """
    try:
        return Path(JUPYTER_TOKEN_FILE).read_text().strip()
    except OSError:
        return ""


def get_jupyter_activity(token: str = "") -> bool:
    """Return True if the Jupyter server reports busy kernels or open terminals.

    Args:
        token: API token to use in the ``Authorization`` header.  When empty
               the header is omitted (unauthenticated request).
    """
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"token {token}"

    for path in ("/api/kernels", "/api/terminals"):
        try:
            resp = requests.get(f"{JUPYTER_URL}{path}", headers=headers, timeout=5)
            if resp.status_code != 200:
                continue
            items = resp.json()
            if path == "/api/kernels":
                for kernel in items:
                    if kernel.get("execution_state") == "busy":
                        log.info("Busy kernel detected: %s", kernel.get("id"))
                        return True
            else:
                if items:
                    log.info("Open terminals: %d", len(items))
                    return True
        except requests.RequestException as exc:
            log.warning("Could not reach Jupyter API %s: %s", path, exc)

    return False


def get_ssh_connections(proc_net_tcp_path: str = "/proc/net/tcp") -> int:
    """Return the number of established SSH connections from ``/proc/net/tcp``.

    All containers in a Kubernetes pod share the same network namespace, so
    ``/proc/net/tcp`` on the sidecar reflects the pod-level TCP table.

    Args:
        proc_net_tcp_path: Path to the proc file (overridable for unit tests).
    """
    count = 0
    try:
        with open(proc_net_tcp_path) as fh:
            next(fh)  # skip header row
            for line in fh:
                parts = line.split()
                if len(parts) < 4:
                    continue
                local_addr = parts[1]
                state = parts[3]
                if ":" not in local_addr:
                    continue
                local_port = local_addr.split(":")[1].upper()
                if local_port == _SSH_PORT_HEX and state == _TCP_ESTABLISHED:
                    count += 1
    except OSError as exc:
        log.debug("Could not read %s: %s", proc_net_tcp_path, exc)
    return count


def is_active(token: str = "") -> bool:
    """Return True if any user activity is detected on the pod.

    Checks (in order):
    1. Jupyter busy kernels or open terminals via the REST API.
    2. Established SSH connections via ``/proc/net/tcp``.
    """
    if get_jupyter_activity(token):
        return True
    ssh_count = get_ssh_connections()
    if ssh_count > 0:
        log.info("Active SSH connections: %d", ssh_count)
        return True
    return False


def delete_pod() -> None:
    """Delete the current pod via the Kubernetes in-cluster API."""
    if not POD_NAME:
        log.error("POD_NAME is not set; cannot self-terminate")
        return
    try:
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        log.warning("Deleting idle pod %s/%s", POD_NAMESPACE, POD_NAME)
        v1.delete_namespaced_pod(name=POD_NAME, namespace=POD_NAMESPACE)
    except Exception as exc:
        log.error("Failed to delete pod %s/%s: %s", POD_NAMESPACE, POD_NAME, exc)


def run() -> None:
    """Main monitoring loop."""
    log.info(
        "GPU idle-shutdown monitor started — idle threshold: %ds, "
        "check interval: %ds, pod: %s/%s",
        IDLE_THRESHOLD_SECONDS,
        CHECK_INTERVAL_SECONDS,
        POD_NAMESPACE,
        POD_NAME,
    )

    idle_since: datetime | None = None

    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)

        token = _read_token()
        if is_active(token):
            if idle_since is not None:
                log.info("Activity resumed; idle timer reset")
            idle_since = None
        else:
            now = datetime.now(tz=timezone.utc)
            if idle_since is None:
                idle_since = now
                log.info("No activity detected; idle timer started")
            else:
                idle_seconds = (now - idle_since).total_seconds()
                log.info(
                    "Pod idle for %.0fs / threshold %ds",
                    idle_seconds,
                    IDLE_THRESHOLD_SECONDS,
                )
                if idle_seconds >= IDLE_THRESHOLD_SECONDS:
                    log.warning(
                        "Idle threshold reached (%.0fs). Terminating pod.",
                        idle_seconds,
                    )
                    delete_pod()
                    # Sleep so that Kubernetes has time to process the deletion
                    # request and terminate this container gracefully before the
                    # process exits on its own.
                    time.sleep(30)


if __name__ == "__main__":
    run()
