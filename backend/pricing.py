"""Centralized AWS instance pricing data.

Single source of truth for all cost estimation across the platform.
Pricing is On-Demand for us-west-2 (Oregon) as of 2026.
Source: https://aws.amazon.com/ec2/pricing/on-demand/

Two views of the data:
- INSTANCE_PRICING: keyed by instance type (e.g. "g5.xlarge") with full specs
- GPU_TYPE_PRICING: keyed by GPU type (e.g. "A10G") for desk cost estimation
"""

import datetime

# ── Per-instance-type pricing (CLI cost tracking, FinOps) ────────

INSTANCE_PRICING: dict[str, dict] = {
    # GPU instances (g4dn = NVIDIA T4)
    "g4dn.xlarge": {"cost_hr": 0.526, "gpus": 1, "gpu_mem": 16, "vcpu": 4, "ram": 16},
    "g4dn.2xlarge": {"cost_hr": 0.752, "gpus": 1, "gpu_mem": 16, "vcpu": 8, "ram": 32},
    # GPU instances (g5 = NVIDIA A10G)
    "g5.xlarge": {"cost_hr": 1.006, "gpus": 1, "gpu_mem": 24, "vcpu": 4, "ram": 16},
    "g5.2xlarge": {"cost_hr": 1.212, "gpus": 1, "gpu_mem": 24, "vcpu": 8, "ram": 32},
    "g5.4xlarge": {"cost_hr": 1.624, "gpus": 1, "gpu_mem": 24, "vcpu": 16, "ram": 64},
    "g5.8xlarge": {"cost_hr": 2.448, "gpus": 1, "gpu_mem": 24, "vcpu": 32, "ram": 128},
    "g5.12xlarge": {"cost_hr": 5.672, "gpus": 4, "gpu_mem": 96, "vcpu": 48, "ram": 192},
    "g5.16xlarge": {"cost_hr": 4.096, "gpus": 1, "gpu_mem": 24, "vcpu": 64, "ram": 256},
    "g5.24xlarge": {"cost_hr": 8.144, "gpus": 4, "gpu_mem": 96, "vcpu": 96, "ram": 384},
    "g5.48xlarge": {"cost_hr": 16.288, "gpus": 8, "gpu_mem": 192, "vcpu": 192, "ram": 768},
    # GPU instances (p3 = NVIDIA V100)
    "p3.2xlarge": {"cost_hr": 3.06, "gpus": 1, "gpu_mem": 16, "vcpu": 8, "ram": 61},
    "p3.8xlarge": {"cost_hr": 12.24, "gpus": 4, "gpu_mem": 64, "vcpu": 32, "ram": 244},
    # GPU instances (p4d = NVIDIA A100)
    "p4d.24xlarge": {"cost_hr": 32.77, "gpus": 8, "gpu_mem": 320, "vcpu": 96, "ram": 1152},
    "p4de.24xlarge": {"cost_hr": 40.97, "gpus": 8, "gpu_mem": 640, "vcpu": 96, "ram": 1152},
    # GPU instances (p5 = NVIDIA H100)
    "p5.48xlarge": {"cost_hr": 55.04, "gpus": 8, "gpu_mem": 640, "vcpu": 192, "ram": 2048},
    # CPU instances
    "m5.large": {"cost_hr": 0.096, "gpus": 0, "gpu_mem": 0, "vcpu": 2, "ram": 8},
    "m5.xlarge": {"cost_hr": 0.192, "gpus": 0, "gpu_mem": 0, "vcpu": 4, "ram": 16},
    "m5.2xlarge": {"cost_hr": 0.384, "gpus": 0, "gpu_mem": 0, "vcpu": 8, "ram": 32},
    "m5.4xlarge": {"cost_hr": 0.768, "gpus": 0, "gpu_mem": 0, "vcpu": 16, "ram": 64},
    "m5.8xlarge": {"cost_hr": 1.536, "gpus": 0, "gpu_mem": 0, "vcpu": 32, "ram": 128},
    "m5a.xlarge": {"cost_hr": 0.172, "gpus": 0, "gpu_mem": 0, "vcpu": 4, "ram": 16},
    "m6i.xlarge": {"cost_hr": 0.192, "gpus": 0, "gpu_mem": 0, "vcpu": 4, "ram": 16},
    "m6a.xlarge": {"cost_hr": 0.173, "gpus": 0, "gpu_mem": 0, "vcpu": 4, "ram": 16},
}

# Spot discount estimate (actual spot pricing is dynamic)
SPOT_DISCOUNT = 0.60

# ── Per-GPU-type pricing (desk burn-rate estimation) ──────────────

GPU_TYPE_PRICING: dict[str, dict] = {
    "CPU": {
        "family": "m5",
        "gpus_per_instance": 0,
        "rate_per_gpu": 0.0,
        "rate_instance": 0.192 / 4,
    },
    "T4": {
        "family": "g4dn",
        "gpus_per_instance": 1,
        "rate_per_gpu": 0.526,
        "rate_instance": 0.526,
    },
    "A10G": {
        "family": "g5",
        "gpus_per_instance": 1,
        "rate_per_gpu": 1.006,
        "rate_instance": 1.006,
    },
    "A100": {
        "family": "p4d",
        "gpus_per_instance": 8,
        "rate_per_gpu": 4.10,
        "rate_instance": 32.77,
    },
    "H100": {
        "family": "p5",
        "gpus_per_instance": 8,
        "rate_per_gpu": 6.88,
        "rate_instance": 55.04,
    },
}

# ── Helpers ──────────────────────────────────────────────────────


def get_hourly_rate(instance_type: str) -> float:
    """Return the hourly cost for an instance type, or 0.0 if unknown."""
    info = INSTANCE_PRICING.get(instance_type)
    return info["cost_hr"] if info else 0.0


def get_cost_estimate(instance_type: str, duration: datetime.timedelta) -> float:
    """Calculate cost estimate for an instance type and duration."""
    return get_hourly_rate(instance_type) * (duration.total_seconds() / 3600.0)


def estimate_desk_burn_rate(gpu_type: str, gpu_count: float) -> str:
    """Estimate hourly burn rate for a desk based on GPU type and count."""
    info = GPU_TYPE_PRICING.get(gpu_type)
    if not info:
        return "$0.19/hr"
    if info["gpus_per_instance"] == 0:
        return f"${info['rate_instance']:.2f}/hr"
    rate = info["rate_per_gpu"] * gpu_count
    return f"${rate:.2f}/hr"
