"""
Checkpoint Manager for Fault-Tolerant Training.

Provides automatic periodic checkpointing with EFS (fast) storage and
async S3 (durable) backup. Integrates with MLflow for checkpoint lineage tracking.

Key Features:
- Periodic checkpointing (every N steps or M minutes)
- EFS for fast local checkpoints (survives pod restarts)
- Async S3 backup for durable archival (manual restore if EFS is lost)
- MLflow integration for checkpoint lineage
- Automatic recovery from latest valid EFS checkpoint
- Atomic writes to prevent corruption during spot interruptions

Usage:
    from ml_platform_sdk.checkpoint import CheckpointManager

    # Initialize checkpoint manager
    ckpt_mgr = CheckpointManager(
        checkpoint_dir="/mnt/efs/checkpoints/my-training",
        s3_bucket="my-bucket",
        s3_prefix="checkpoints/my-training",
        save_interval_steps=100,
        save_interval_seconds=300,
        mlflow_tracking=True,
    )

    # In training loop
    for step in range(start_step, num_steps):
        # ... training logic ...

        # Save checkpoint periodically
        if ckpt_mgr.should_save(step):
            ckpt_mgr.save_checkpoint(
                step=step,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                metrics={"loss": loss, "accuracy": acc},
            )

    # Resume from latest checkpoint
    checkpoint = ckpt_mgr.load_latest_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
"""

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages periodic checkpointing with EFS + S3 backup and MLflow tracking.

    The checkpoint manager handles:
    1. Periodic saving based on steps or time interval
    2. Atomic writes to EFS to prevent corruption
    3. Async background upload to S3 for durability
    4. MLflow logging of checkpoint metadata
    5. Automatic recovery from latest valid checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        save_interval_steps: int = 100,
        save_interval_seconds: int = 300,
        max_checkpoints_to_keep: int = 3,
        mlflow_tracking: bool = True,
        execution_id: Optional[str] = None,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Local directory for checkpoints (typically on EFS: /mnt/efs/...)
            s3_bucket: S3 bucket name for durable backup (optional)
            s3_prefix: S3 key prefix for checkpoints (optional)
            save_interval_steps: Save checkpoint every N training steps
            save_interval_seconds: Save checkpoint every M seconds (whichever comes first)
            max_checkpoints_to_keep: Maximum number of checkpoints to retain
            mlflow_tracking: Whether to log checkpoint metadata to MLflow
            execution_id: Flyte execution ID (auto-detected from env if not provided)
        """
        self.checkpoint_dir = checkpoint_dir
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix or ""
        self.save_interval_steps = save_interval_steps
        self.save_interval_seconds = save_interval_seconds
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.mlflow_tracking = mlflow_tracking
        self.execution_id = execution_id or os.getenv("FLYTE_INTERNAL_EXECUTION_ID", "default")

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Track last save time – initialise so that step 0 triggers a save
        self.last_save_time = 0.0
        self.last_save_step = -self.save_interval_steps

        # Background upload queue
        self._upload_queue: deque = deque()
        self._upload_thread: Optional[threading.Thread] = None
        self._upload_lock = threading.Lock()
        self._shutdown = False

        # In-memory replication engine (e.g. nvidia-resiliency-ext)
        self._replication_engine = None

        # Start background uploader if S3 is configured
        if self.s3_bucket:
            self._start_background_uploader()

    def should_save(self, step: int) -> bool:
        """
        Determine if a checkpoint should be saved at this step.

        Args:
            step: Current training step

        Returns:
            True if checkpoint should be saved based on interval criteria
        """
        time_elapsed = time.time() - self.last_save_time
        steps_elapsed = step - self.last_save_step

        return (
            steps_elapsed >= self.save_interval_steps or time_elapsed >= self.save_interval_seconds
        )

    def save_checkpoint(
        self,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint atomically to EFS and queue for S3 upload.

        Uses atomic write pattern: save to .tmp file, fsync, rename to final path.
        This prevents corruption if pod is killed during checkpoint save.

        Args:
            step: Current training step
            model_state: Model state dict (from model.state_dict())
            optimizer_state: Optimizer state dict (optional)
            epoch: Current epoch number (optional)
            metrics: Training metrics to log (optional)
            extra_state: Additional state to save (optional)

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint-{step:08d}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        tmp_path = f"{checkpoint_path}.tmp"

        # Prepare checkpoint data
        checkpoint_data = {
            "step": step,
            "model_state_dict": model_state,
            "timestamp": time.time(),
            "execution_id": self.execution_id,
        }

        if optimizer_state is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer_state

        if epoch is not None:
            checkpoint_data["epoch"] = epoch

        if metrics is not None:
            checkpoint_data["metrics"] = metrics

        if extra_state is not None:
            checkpoint_data["extra_state"] = extra_state

        # Atomic write: save to .tmp, fsync, rename
        try:
            import torch

            torch.save(checkpoint_data, tmp_path)

            # Ensure data is written to disk before rename
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())

            # Atomic rename
            os.rename(tmp_path, checkpoint_path)

            # Fsync the directory to ensure the rename is durable
            dir_fd = os.open(self.checkpoint_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

            logger.info("Checkpoint saved: %s", checkpoint_path)

        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # Save metadata as JSON
        metadata_path = os.path.join(self.checkpoint_dir, f"checkpoint-{step:08d}.json")
        metadata = {
            "step": step,
            "epoch": epoch,
            "timestamp": checkpoint_data["timestamp"],
            "execution_id": self.execution_id,
            "metrics": metrics or {},
            "checkpoint_file": checkpoint_name,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update tracking
        self.last_save_time = time.time()
        self.last_save_step = step

        # Queue for S3 upload
        if self.s3_bucket:
            self._queue_s3_upload(checkpoint_path, metadata_path)

        # Log to MLflow
        if self.mlflow_tracking:
            self._log_to_mlflow(step, checkpoint_path, metrics)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest valid checkpoint from EFS.

        Walks checkpoints from newest to oldest and returns the first valid one.
        Handles incomplete checkpoints from spot interruptions.

        Returns:
            Checkpoint data dictionary, or None if no valid checkpoint found
        """
        if not os.path.exists(self.checkpoint_dir):
            return None

        # Find all checkpoint files
        checkpoint_files = [
            f
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint-") and f.endswith(".pt")
        ]

        if not checkpoint_files:
            return None

        # Sort by step number (descending)
        checkpoint_files.sort(
            key=lambda x: int(x.replace("checkpoint-", "").replace(".pt", "")),
            reverse=True,
        )

        # Try to load from newest to oldest
        for ckpt_file in checkpoint_files:
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_file)

            try:
                import torch

                # weights_only=False: checkpoints are self-created trusted
                # data on EFS and may contain non-tensor extra_state.
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                logger.info(
                    "Loaded checkpoint from: %s (step=%s, epoch=%s)",
                    ckpt_path,
                    checkpoint.get("step", "unknown"),
                    checkpoint.get("epoch", "unknown"),
                )
                return checkpoint

            except Exception as e:
                logger.warning("Skipping corrupted checkpoint %s: %s", ckpt_file, e)
                continue

        return None

    def _queue_s3_upload(self, checkpoint_path: str, metadata_path: str):
        """Queue checkpoint files for async S3 upload."""
        with self._upload_lock:
            self._upload_queue.append((checkpoint_path, metadata_path))

    def _start_background_uploader(self):
        """Start background thread for S3 uploads."""
        self._upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self._upload_thread.start()

    def _upload_worker(self):
        """Background worker that uploads queued checkpoints to S3."""
        import boto3

        s3 = boto3.client("s3")

        while True:
            work_item = None
            with self._upload_lock:
                if self._upload_queue:
                    work_item = self._upload_queue.popleft()
                elif self._shutdown:
                    break

            if work_item is not None:
                checkpoint_path, metadata_path = work_item
                self._do_upload(s3, checkpoint_path, metadata_path)
            else:
                time.sleep(1)

    def _do_upload(self, s3, checkpoint_path: str, metadata_path: str):
        """Upload a checkpoint + metadata pair to S3 using streaming upload."""
        for fpath in (checkpoint_path, metadata_path):
            try:
                fname = os.path.basename(fpath)
                s3_key = f"{self.s3_prefix}/{fname}".lstrip("/")
                s3.upload_file(Filename=fpath, Bucket=self.s3_bucket, Key=s3_key)
                if fpath == checkpoint_path:
                    logger.info("Uploaded to S3: s3://%s/%s", self.s3_bucket, s3_key)
            except Exception as e:
                logger.warning("S3 upload failed for %s: %s", fpath, e)

    def _log_to_mlflow(self, step: int, checkpoint_path: str, metrics: Optional[Dict[str, float]]):
        """Log checkpoint metadata to MLflow."""
        try:
            import mlflow

            if mlflow.active_run():
                mlflow.set_tag("latest_checkpoint_step", str(step))
                mlflow.set_tag("latest_checkpoint_path", checkpoint_path)

                if self.s3_bucket and self.s3_prefix:
                    s3_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/checkpoint-{step:08d}.pt"
                    mlflow.set_tag("latest_checkpoint_s3", s3_path)

                if metrics:
                    mlflow.log_metrics(
                        {f"checkpoint_{k}": v for k, v in metrics.items()},
                        step=step,
                    )

        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints_to_keep."""
        if not os.path.exists(self.checkpoint_dir):
            return

        # Find all checkpoint files
        checkpoint_files = [
            f
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint-") and f.endswith(".pt")
        ]

        if len(checkpoint_files) > self.max_checkpoints_to_keep:
            # Sort by step number (ascending)
            checkpoint_files.sort(
                key=lambda x: int(x.replace("checkpoint-", "").replace(".pt", ""))
            )

            # Remove oldest checkpoints
            num_to_remove = len(checkpoint_files) - self.max_checkpoints_to_keep
            for ckpt_file in checkpoint_files[:num_to_remove]:
                ckpt_path = os.path.join(self.checkpoint_dir, ckpt_file)
                metadata_path = ckpt_path.replace(".pt", ".json")

                try:
                    os.remove(ckpt_path)
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    logger.info("Removed old checkpoint: %s", ckpt_file)

                except Exception as e:
                    logger.warning("Failed to remove %s: %s", ckpt_file, e)

        # Always clean up stale .tmp files from interrupted writes
        for fname in os.listdir(self.checkpoint_dir):
            if fname.startswith("checkpoint-") and fname.endswith(".tmp"):
                tmp_path = os.path.join(self.checkpoint_dir, fname)
                try:
                    os.remove(tmp_path)
                    logger.info("Removed stale temp file: %s", fname)
                except Exception as e:
                    logger.warning("Failed to remove temp file %s: %s", fname, e)

        # Always clean up orphan .json metadata with no corresponding .pt
        for fname in os.listdir(self.checkpoint_dir):
            if fname.startswith("checkpoint-") and fname.endswith(".json"):
                ckpt_path = os.path.join(self.checkpoint_dir, fname.replace(".json", ".pt"))
                if not os.path.exists(ckpt_path):
                    try:
                        os.remove(os.path.join(self.checkpoint_dir, fname))
                        logger.info("Removed orphan metadata: %s", fname)
                    except Exception as e:
                        logger.warning("Failed to remove orphan %s: %s", fname, e)

    def shutdown(self, timeout_seconds: int = 120):
        """
        Gracefully shutdown the checkpoint manager.

        Waits for pending S3 uploads to complete, up to ``timeout_seconds``.
        Call this before training exits to ensure all checkpoints are uploaded.

        Args:
            timeout_seconds: Maximum seconds to wait for uploads (default: 120).
        """
        if self._upload_thread:
            # Signal the worker to drain the queue then exit
            with self._upload_lock:
                self._shutdown = True
            self._upload_thread.join(timeout=timeout_seconds)
            if self._upload_thread.is_alive():
                logger.warning(
                    "Upload thread did not finish within %ds; " "%d uploads may be incomplete",
                    timeout_seconds,
                    len(self._upload_queue),
                )
        else:
            self._shutdown = True
        logger.info("Checkpoint manager shutdown complete")

    def set_replication_engine(self, engine: Any):
        """Set an external in-memory replication engine (e.g. nvidia-resiliency-ext).

        When set, ``replicate()`` will forward state to the engine so it can
        maintain in-memory redundancy across nodes. Recovery from replicated
        state is handled by the engine and/or caller code.
        """
        self._replication_engine = engine

    def replicate(self, state: Dict[str, Any]):
        """Forward state to the configured in-memory replication engine.

        This is a no-op if no replication engine has been configured via
        ``set_replication_engine()``. Failures are logged as warnings so
        replication issues do not crash training. After the first failure,
        the engine is disabled to avoid repeated warnings.
        """
        if self._replication_engine is not None:
            try:
                self._replication_engine.replicate(state)
            except Exception:
                logger.warning(
                    "Replication engine failed; disabling in-memory replication",
                    exc_info=True,
                )
                self._replication_engine = None


class HuggingFaceCheckpointManager:
    """
    Checkpoint manager for HuggingFace Trainer with automatic S3 backup.

    Wraps the standard HuggingFace Trainer checkpoint behavior with:
    - Validation of checkpoint completeness
    - S3 backup (synchronous upload via ``backup_checkpoint_to_s3()``)
    - MLflow lineage tracking

    Usage:
        from ml_platform_sdk.checkpoint import HuggingFaceCheckpointManager

        ckpt_mgr = HuggingFaceCheckpointManager(
            checkpoint_dir="/mnt/efs/checkpoints/my-training",
            s3_bucket="my-bucket",
            s3_prefix="checkpoints/my-training",
        )

        # Standard HuggingFace training
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

        # Resume from latest valid checkpoint
        resume_ckpt = ckpt_mgr.find_latest_valid_checkpoint()
        trainer.train(resume_from_checkpoint=resume_ckpt)

        # Upload final checkpoint to S3
        ckpt_mgr.backup_checkpoint_to_s3(trainer.state.best_model_checkpoint)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        mlflow_tracking: bool = True,
    ):
        """
        Initialize HuggingFace checkpoint manager.

        Args:
            checkpoint_dir: Local checkpoint directory (on EFS)
            s3_bucket: S3 bucket for backup
            s3_prefix: S3 key prefix
            mlflow_tracking: Whether to log to MLflow
        """
        self.checkpoint_dir = checkpoint_dir
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix or ""
        self.mlflow_tracking = mlflow_tracking

    def find_latest_valid_checkpoint(self) -> Optional[str]:
        """
        Find the latest valid checkpoint directory.

        Validates that checkpoint contains required files (trainer_state.json).
        Handles incomplete checkpoints from spot interruptions.

        Returns:
            Path to latest valid checkpoint directory, or None
        """
        if not os.path.isdir(self.checkpoint_dir):
            return None

        # Find all checkpoint directories
        ckpt_dirs = sorted(
            [
                d
                for d in os.listdir(self.checkpoint_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(self.checkpoint_dir, d))
            ],
            key=lambda x: int(x.split("-")[-1]),
            reverse=True,
        )

        # Walk from newest to oldest
        for d in ckpt_dirs:
            candidate = os.path.join(self.checkpoint_dir, d)
            required = os.path.join(candidate, "trainer_state.json")

            if os.path.isfile(required):
                logger.info("Found valid checkpoint: %s", candidate)
                return candidate
            else:
                logger.warning("Skipping incomplete checkpoint: %s", candidate)

        return None

    def backup_checkpoint_to_s3(self, checkpoint_path: Optional[str]) -> Optional[str]:
        """
        Upload a checkpoint directory to S3.

        Args:
            checkpoint_path: Local path to checkpoint directory (may be None)

        Returns:
            S3 URI of uploaded checkpoint, or None if upload failed/skipped
        """
        if not checkpoint_path or not self.s3_bucket or not os.path.isdir(checkpoint_path):
            return None

        import boto3

        s3 = boto3.client("s3")
        checkpoint_name = os.path.basename(checkpoint_path)

        try:
            # Upload all files in checkpoint directory
            for fname in os.listdir(checkpoint_path):
                fpath = os.path.join(checkpoint_path, fname)
                if os.path.isfile(fpath):
                    s3_key = f"{self.s3_prefix}/{checkpoint_name}/{fname}".lstrip("/")

                    s3.upload_file(Filename=fpath, Bucket=self.s3_bucket, Key=s3_key)

            s3_uri = f"s3://{self.s3_bucket}/{self.s3_prefix}/{checkpoint_name}".replace(
                "//", "/"
            ).replace("s3:/", "s3://")
            logger.info("Checkpoint backed up to S3: %s", s3_uri)

            # Log to MLflow
            if self.mlflow_tracking:
                try:
                    import mlflow

                    if mlflow.active_run():
                        mlflow.log_param("checkpoint_s3_path", s3_uri)
                except Exception:
                    pass

            return s3_uri

        except Exception as e:
            logger.error("S3 backup failed: %s", e)
            return None
