"""
TensorBoard integration for the ML Platform SDK.

Provides ``get_summary_writer()`` to create a ``SummaryWriter`` that writes
to the shared EFS log directory (``/mnt/efs/tensorboard/<execution_id>/``)
with automatic S3 backup on close.

Usage::

    from ml_platform_sdk.tasks.tensorboard import get_summary_writer

    writer = get_summary_writer()
    for step, loss in enumerate(losses):
        writer.add_scalar("train/loss", loss, step)
    writer.close()   # triggers S3 upload
"""

import os
import subprocess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TB_EFS_BASE: str = "/mnt/efs/tensorboard"
S3_TB_PREFIX: str = "tensorboard"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_execution_id() -> str:
    """Derive a unique run identifier from the Flyte execution context or env."""
    try:
        from flytekit import current_context

        ctx = current_context()
        exec_name = ctx.execution_id.name
        if exec_name:
            return exec_name
    except Exception:
        pass

    execution_id = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID")
    if execution_id:
        return execution_id

    return os.environ.get("EXECUTION_ID", f"local-{os.getpid()}")


def _upload_to_s3(local_dir: str) -> str | None:
    """Upload TensorBoard logs from *local_dir* to S3."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return None

    # Only upload directories under TB_EFS_BASE to avoid '..' segments
    # in S3 keys when log_dir is a custom path outside EFS.
    # Use trailing separator to prevent prefix matches like /mnt/efs/tensorboard2/.
    abs_local = os.path.abspath(local_dir)
    abs_base = os.path.abspath(TB_EFS_BASE)
    if not (abs_local == abs_base or abs_local.startswith(abs_base + os.sep)):
        return None

    # Build S3 key using S3_TB_PREFIX and the relative path within TB_EFS_BASE
    # so the S3 layout is explicit and stable regardless of TB_EFS_BASE path.
    rel = os.path.relpath(abs_local, abs_base)
    s3_key = f"{S3_TB_PREFIX}/{rel}" if rel != "." else S3_TB_PREFIX
    s3_uri = f"s3://{bucket}/{s3_key}/"

    try:
        subprocess.run(
            ["aws", "s3", "sync", abs_local, s3_uri, "--quiet"],
            check=True,
            timeout=300,
        )
        return s3_uri
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_summary_writer(
    sub_dir: str | None = None,
    log_dir: str | None = None,
):
    """Return a ``torch.utils.tensorboard.SummaryWriter`` for the current run.

    Writes to ``/mnt/efs/tensorboard/<execution_id>/[sub_dir]`` by default,
    making logs visible to the shared TensorBoard deployment.  On ``close()``,
    the writer performs a best-effort S3 upload of the execution log root
    only when ``S3_BUCKET`` is set, the ``aws`` CLI is available, and the
    effective log directory is under ``TB_EFS_BASE``.  If any of those
    conditions are not met, logs remain local/EFS-only and no S3 backup is
    performed.

    Args:
        sub_dir: Optional subdirectory under the execution log dir
            (e.g., ``"train"`` or ``"eval"``).
        log_dir: Override the full log directory path. When set,
            *sub_dir* is ignored.  S3 upload on ``close()`` is only attempted
            for paths under ``TB_EFS_BASE``; custom paths outside that tree
            will not be uploaded.

    Returns:
        A ``SummaryWriter`` instance.
    """
    from torch.utils.tensorboard import SummaryWriter

    if log_dir is None:
        execution_id = _get_execution_id()
        log_dir = os.path.join(TB_EFS_BASE, execution_id)
        if sub_dir:
            log_dir = os.path.join(log_dir, sub_dir)

    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # Wrap close() to trigger S3 upload of the execution root directory
    _original_close = writer.close
    abs_log = os.path.abspath(log_dir)
    abs_base = os.path.abspath(TB_EFS_BASE)
    is_under_efs = abs_log == abs_base or abs_log.startswith(abs_base + os.sep)

    if is_under_efs:
        # Derive exec_root from the actual log_dir path (first segment under
        # TB_EFS_BASE) so that custom log_dir paths pointing at older runs
        # upload the correct directory, not the current execution id.
        rel = os.path.relpath(abs_log, abs_base)
        first_segment = rel.split(os.sep)[0]
        exec_root = os.path.join(TB_EFS_BASE, first_segment)
    else:
        exec_root = log_dir

    def _close_and_upload():
        _original_close()
        _upload_to_s3(exec_root)

    writer.close = _close_and_upload  # type: ignore[assignment]
    return writer
