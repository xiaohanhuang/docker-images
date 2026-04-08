"""
Shared GPU type definitions for ML Platform.

Used by the CLI, SDK (@remote decorator), and execution service
to ensure consistent GPU type options across all surfaces.
"""

from enum import Enum


class GpuType(str, Enum):
    """Supported GPU types mapped to Karpenter instance-gpu-name labels."""

    any = "any"  # No preference — Karpenter picks cheapest
    t4 = "t4"  # g4dn instances (NVIDIA T4, 16GB)
    a10g = "a10g"  # g5 instances (NVIDIA A10G, 24GB)
    a100 = "a100"  # p4d/p4de instances (NVIDIA A100, 40/80GB)


# Mapping from GpuType to human-readable descriptions (for help text)
GPU_TYPE_DESCRIPTIONS = {
    GpuType.any: "no preference",
    GpuType.t4: "g4dn, NVIDIA T4",
    GpuType.a10g: "g5, NVIDIA A10G",
    GpuType.a100: "p4d/p4de, NVIDIA A100",
}

# Valid gpu_type string values (for validation in services)
VALID_GPU_TYPES = {t.value for t in GpuType}
