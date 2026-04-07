"""
Benchmark script for Checkpoint I/O overhead.
"""

import os
import tempfile
import time


def benchmark_checkpoint_io():
    """Measure synchronous checkpoint save overhead."""
    import torch
    from ml_platform_sdk.checkpoint import CheckpointManager

    # Default to CI-safe size; scale up via env vars for real measurements.
    in_features = int(os.getenv("ML_PLATFORM_BENCHMARK_IN_FEATURES", "2048"))
    out_features = int(os.getenv("ML_PLATFORM_BENCHMARK_OUT_FEATURES", "2048"))
    model = torch.nn.Linear(in_features, out_features)
    model_state = model.state_dict()
    approx_mib = (in_features * out_features * 4) / (1024 * 1024)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Benchmarking in {tmpdir}...")
        print(f"Model: Linear({in_features}, {out_features}) (~{approx_mib:.1f} MiB)")

        # Sync baseline with non-I/O features disabled for accurate timing
        manager = CheckpointManager(
            checkpoint_dir=os.path.join(tmpdir, "sync"),
            save_interval_steps=1,
            mlflow_tracking=False,
        )
        start = time.perf_counter()
        manager.save_checkpoint(step=1, model_state=model_state)
        sync_time = time.perf_counter() - start
        print(f"Sync Save Time: {sync_time:.2f}s")


if __name__ == "__main__":
    try:
        benchmark_checkpoint_io()
    except ModuleNotFoundError as e:
        print(f"Benchmark skipped: {e}")
        print("Install missing dependencies (torch, ml_platform_sdk) to run this benchmark.")
