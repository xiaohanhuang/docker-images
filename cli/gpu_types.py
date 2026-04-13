"""
Shared GPU type definitions for ML Platform CLI.

Duplicated from ml_platform_sdk.gpu_types to avoid importing the full SDK
(which pulls in flytekit/grpc/kubernetes/~460ms) at CLI startup.
Shell tab-completion runs the full import chain on every keypress, so
keeping this import lightweight is critical for responsiveness.
"""

from enum import Enum


class GpuType(str, Enum):
    """Supported GPU types mapped to Karpenter instance-gpu-name labels."""

    any = "any"  # No preference — Karpenter picks cheapest
    t4 = "t4"  # g4dn instances (NVIDIA T4, 16GB)
    a10g = "a10g"  # g5 instances (NVIDIA A10G, 24GB)
    a100 = "a100"  # p4d/p4de instances (NVIDIA A100, 40/80GB)
