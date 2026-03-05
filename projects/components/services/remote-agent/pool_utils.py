"""Shared utility functions for the pod pool.

Extracted so they can be imported by both the agent and the test suite
without pulling in FastAPI/Kubernetes dependencies.
"""

import hashlib


def compute_config_hash(config: dict) -> str:
    """
    Compute a deterministic hash of an execution config for pod matching.

    Pods with the same config hash can be reused.
    Hash is based on: image, gpu count, gpu_type, and sorted packages.
    cpu/memory are intentionally excluded — those affect scheduling but
    a pod allocated for more resources can serve a request asking for less.
    """
    image = config.get("image", "")
    gpu = config.get("gpu", 0)
    gpu_type = config.get("gpu_type", "any")
    packages = sorted(config.get("packages", []))

    hash_input = f"{image}:{gpu}:{gpu_type}:{','.join(packages)}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
