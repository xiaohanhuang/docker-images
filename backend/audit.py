"""Audit logging utilities for ml-platform backend."""

import datetime
import json
import logging
import os

import anyio

logger = logging.getLogger(__name__)

AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", "/var/log/ml-platform/audit.log")


async def log_audit_event(
    user: str,
    action: str,
    resource_type: str,
    resource_name: str,
    details: dict | None = None,
):
    """Log audit events for all API operations."""
    event = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user": user,
        "action": action,
        "resource_type": resource_type,
        "resource_name": resource_name,
        "details": details or {},
    }

    event_line = json.dumps(event) + "\n"

    # Log to stdout for container log aggregation (primary sink)
    logger.info(f"AUDIT: {event_line.rstrip()}")

    # Also append to file (non-blocking)
    def _write_audit_log():
        os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
        with open(AUDIT_LOG_FILE, "a") as f:
            f.write(event_line)

    try:
        await anyio.to_thread.run_sync(_write_audit_log)
    except OSError:
        pass  # File write failure should not break API requests
