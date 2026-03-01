import contextlib
import os
from datetime import datetime

import torch


@contextlib.contextmanager
def profile(output_dir: str = "/tmp/traces"):
    """
    Context manager for PyTorch Profiler.
    Automatically saves traces to the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure PyTorch Profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        try:
            yield prof
        finally:
            print(f"Profiling complete. Traces saved to {output_dir}")
            # In a real scenario, you might upload to S3 here
            print(
                f"To upload to S3: aws s3 sync {output_dir} s3://my-bucket/profiles/{timestamp}"
            )
