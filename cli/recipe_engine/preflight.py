"""Preflight checks to verify endpoint targets before recipe submission."""

import socket
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


def judge_endpoint_from_model(judge_model: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract endpoint URL from judge_model when endpoint-backed mode is used.

    Supported endpoint-backed forms:
      - http://... / https://...
      - vllm://<model>@http://...

    Returns:
        (endpoint_url, error_message). Exactly one of the tuple values is non-None.
    """
    judge_model = (judge_model or "").strip()
    if not judge_model:
        return None, None

    if judge_model.startswith("vllm://"):
        spec = judge_model.removeprefix("vllm://")
        if "@" not in spec:
            return None, (
                "judge_model must use 'vllm://<model>@<endpoint>' format "
                "when using vLLM URI syntax"
            )
        _model_name, endpoint = spec.split("@", 1)
        endpoint = endpoint.strip()
        if not endpoint:
            return None, "judge_model vLLM endpoint must not be empty"
        return endpoint, None

    if judge_model.startswith("http://") or judge_model.startswith("https://"):
        return judge_model, None

    return None, None


def extract_preflight_targets(
    parameters: Dict[str, Any],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Collect endpoint targets to verify before recipe submission.

    Returns:
        (targets, validation_errors)
        targets: list of (parameter_name, endpoint_url)
    """
    targets: List[Tuple[str, str]] = []
    errors: List[str] = []

    is_non_colocated = parameters.get("distributed_colocate_critic_reward") is False
    if is_non_colocated:
        for key in ("reference_service_url", "reward_service_url", "redis_url"):
            value = str(parameters.get(key, "") or "").strip()
            if not value:
                errors.append(f"{key} is required when distributed_colocate_critic_reward=false")
            else:
                targets.append((key, value))

    judge_model = str(parameters.get("judge_model", "") or "")
    judge_endpoint, judge_error = judge_endpoint_from_model(judge_model)
    if judge_error:
        errors.append(judge_error)
    elif judge_endpoint:
        targets.append(("judge_model", judge_endpoint))

    return targets, errors


def check_cluster_service_exists(hostname: str, timeout_seconds: float = 5.0) -> Tuple[bool, str]:
    """Verify a '*.svc.cluster.local' hostname maps to an existing K8s Service."""
    parts = hostname.split(".")
    if len(parts) < 5 or parts[2:] != ["svc", "cluster", "local"]:
        return False, f"Invalid cluster service hostname format: {hostname}"

    service_name, namespace = parts[0], parts[1]
    try:
        result = subprocess.run(
            ["kubectl", "get", "service", service_name, "-n", namespace],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return False, "kubectl not found; cannot verify in-cluster service endpoints"
    except Exception as exc:
        return False, f"kubectl check failed for {namespace}/{service_name}: {exc}"

    if result.returncode == 0:
        return True, f"Kubernetes Service {namespace}/{service_name} exists"

    details = result.stderr.strip() or result.stdout.strip() or "unknown kubectl error"
    return False, f"Kubernetes Service {namespace}/{service_name} check failed: {details}"


def check_endpoint_target_reachable(url: str, timeout_seconds: float = 3.0) -> Tuple[bool, str]:
    """Check endpoint reachability using service existence or TCP connect."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https", "redis"}:
        return False, f"Unsupported URL scheme '{scheme}' in {url}"
    if not parsed.hostname:
        return False, f"Missing hostname in URL: {url}"

    hostname = parsed.hostname
    if hostname.endswith(".svc.cluster.local"):
        return check_cluster_service_exists(hostname)

    default_ports = {"http": 80, "https": 443, "redis": 6379}
    port = parsed.port or default_ports[scheme]
    try:
        with socket.create_connection((hostname, port), timeout=timeout_seconds):
            pass
        return True, f"TCP reachable at {hostname}:{port}"
    except OSError as exc:
        return False, f"TCP connect failed for {hostname}:{port}: {exc}"


def run_endpoint_preflight(
    parameters: Dict[str, Any],
) -> Tuple[List[Tuple[str, bool, str]], List[str]]:
    """Run endpoint preflight checks and return detailed results and failures."""
    targets, failures = extract_preflight_targets(parameters)
    checks: List[Tuple[str, bool, str]] = []
    for label, endpoint in targets:
        ok, detail = check_endpoint_target_reachable(endpoint)
        checks.append((label, ok, detail))
        if not ok:
            failures.append(f"{label}: {detail}")

    return checks, failures
