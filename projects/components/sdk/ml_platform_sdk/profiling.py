import contextlib
import os
import shutil
import signal
import subprocess
from datetime import datetime

import torch

# ---------------------------------------------------------------------------
# S3 upload helper
# ---------------------------------------------------------------------------

S3_PROFILES_PREFIX = "profiles"

# Default data bucket for the ML Platform EKS cluster.  Override via
# S3_BUCKET or ML_PLATFORM_DATA_BUCKET environment variables.
_DEFAULT_DATA_BUCKET = "ml-platform-data-ml-platform-eks-805673386114"


def _upload_to_s3(local_dir: str) -> str | None:
    """Upload profiling traces from *local_dir* to S3.

    Reads ``S3_BUCKET`` (or ``ML_PLATFORM_DATA_BUCKET``) from the environment.
    Returns the S3 URI on success, or ``None`` if the bucket is not configured
    or the upload fails.
    """
    bucket = os.getenv("S3_BUCKET") or os.getenv("ML_PLATFORM_DATA_BUCKET", _DEFAULT_DATA_BUCKET)
    if not bucket:
        return None

    # Derive a meaningful key: profiles/<execution_id>/...
    # local_dir is typically /mnt/efs/profiles/<execution_id>
    basename = os.path.basename(local_dir.rstrip("/"))
    s3_uri = f"s3://{bucket}/{S3_PROFILES_PREFIX}/{basename}/"

    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, s3_uri, "--quiet"],
            check=True,
            timeout=300,
        )
        print(f"Traces uploaded to {s3_uri}")
        return s3_uri
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"Warning: S3 upload failed ({exc}). Traces remain on EFS at {local_dir}")
        return None


@contextlib.contextmanager
def profile(output_dir: str = "/tmp/traces"):
    """
    Context manager for PyTorch Profiler.
    Automatically saves traces to the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
            _upload_to_s3(output_dir)


@contextlib.contextmanager
def nsight_profile(output_dir: str | None = None):
    """Context manager that captures an Nsight Systems trace.

    Launches ``nsys profile`` as a background process that monitors the current
    process tree.  When the context exits, ``nsys`` is stopped and the
    ``.nsys-rep`` report is written to *output_dir*.

    Requires ``nsys`` to be on ``$PATH`` (provided automatically when using
    ``@gpu_task(nsight=True)``).

    Args:
        output_dir: Directory for the ``.nsys-rep`` file.  Defaults to
            ``/mnt/efs/profiles/<execution_id>/`` when running inside a Flyte
            task, or ``/tmp/nsight-traces/`` otherwise.

    Yields:
        The path to the output report file (without the ``.nsys-rep`` suffix
        that ``nsys`` appends automatically).

    Example::

        from ml_platform_sdk.profiling import nsight_profile

        with nsight_profile() as report_path:
            for batch in dataloader:
                train_step(model, batch)
        # report_path + ".nsys-rep" now contains the Nsight trace
    """
    nsys_bin = shutil.which("nsys") or "/opt/nsight/bin/nsys"

    if not os.path.isfile(nsys_bin):
        raise FileNotFoundError(
            f"nsys not found at {nsys_bin}. "
            "Use @gpu_task(nsight=True) to inject Nsight via init container, "
            "or install nsight-systems-cli in your image."
        )

    if output_dir is None:
        execution_id = os.getenv("FLYTE_INTERNAL_EXECUTION_ID", "unknown")
        output_dir = os.path.join("/mnt/efs/profiles", execution_id)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"nsight_{timestamp}")

    pid = os.getpid()
    cmd = [
        nsys_bin,
        "profile",
        "--trace",
        "cuda,nvtx,osrt,cudnn,cublas",
        "--sample",
        "process-tree",
        "--output",
        report_path,
        "--force-overwrite",
        "true",
        # Attach to the current process tree
        "--process-scope",
        "process-tree",
        "--target-processes",
        "all",
        # Wait — nsys will run until we signal it
        "--duration",
        "0",
        "-p",
        str(pid),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        yield report_path
    finally:
        # Signal nsys to stop and write the report
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        report_file = f"{report_path}.nsys-rep"
        if os.path.exists(report_file):
            print(f"Nsight profiling complete. Report: {report_file}")
            print(f"Analyze: nsys stats {report_file}")
            _upload_to_s3(output_dir)
        else:
            print(f"Nsight profiling finished but report not found at {report_file}")
