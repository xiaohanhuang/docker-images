"""
train.py — Task 3: Fine-tune CodeT5+ on the Text2SQL dataset using
Ray Train + HuggingFace Seq2SeqTrainer on a GPU worker.

Logs all metrics to MLflow via autolog.
"""

import sys

from flytekit import PodTemplate, Resources, task
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

sys.path.insert(0, "/app")
from config import (
    BASE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_LR,
    DEFAULT_WARMUP_STEPS,
    GPU_IMAGE,
    MAX_TARGET_LEN,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    S3_BUCKET,
)

# ── Ray cluster pod templates (EFS mount + GPU toleration) ─────────────
# kubernetes.client is only needed at serialization time (local machine).
try:
    from kubernetes.client import (
        V1Container,
        V1PersistentVolumeClaimVolumeSource,
        V1PodSpec,
        V1Toleration,
        V1Volume,
        V1VolumeMount,
    )

    _efs_volume = V1Volume(
        name="efs-storage",
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
            claim_name="efs-claim"
        ),
    )
    _efs_mount = V1VolumeMount(name="efs-storage", mount_path="/workspace")

    _head_pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="ray-head", volume_mounts=[_efs_mount])],
            volumes=[_efs_volume],
        )
    )

    _worker_pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            # Resource requests/limits are NOT set here — the @task requests
            # are propagated to the worker container by the Ray plugin, so GPU
            # must be declared there (see requests=Resources(..., gpu="1") below).
            containers=[V1Container(name="ray-worker", volume_mounts=[_efs_mount])],
            volumes=[_efs_volume],
            tolerations=[
                V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Equal",
                    value="true",
                    effect="NoSchedule",
                )
            ],
        )
    )
except ImportError:
    _head_pod_template = None
    _worker_pod_template = None

# ── Ray cluster config ─────────────────────────────────────────────────
ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(
        ray_start_params={"dashboard-host": "0.0.0.0"},
        pod_template=_head_pod_template,
    ),
    worker_node_config=[
        WorkerNodeConfig(
            group_name="gpu-workers",
            replicas=1,
            min_replicas=1,
            max_replicas=1,
            pod_template=_worker_pod_template,
        )
    ],
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=3600,
)


@task(
    task_config=ray_config,
    # These resources are propagated to the Ray worker pod by the flytekit Ray
    # plugin. gpu="1" is required so Karpenter schedules the worker on a GPU
    # spot node (training-spot-gpu-nodepool), not a CPU node.
    requests=Resources(cpu="4", mem="14Gi", gpu="1"),
    limits=Resources(cpu="4", mem="14Gi", gpu="1"),
    container_image=GPU_IMAGE,
    retries=3,
    interruptible=True,
    environment={
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        "HF_DATASETS_CACHE": "/tmp/hf_cache",
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    },
)
def train(
    processed_s3_path: str,
    num_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
) -> str:
    """
    Launch Ray Train job: fine-tunes Salesforce/codet5p-220m.
    Returns S3 path to best checkpoint.
    """
    import ray
    from ray import train as ray_train
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    ray.init(address="auto", ignore_reinit_error=True)

    def train_loop(config):
        import os
        import tempfile

        import boto3
        import mlflow
        import torch
        from datasets import load_from_disk
        from ray.train.huggingface.transformers import (
            RayTrainReportCallback,
            prepare_trainer,
        )
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )

        mlflow.set_tracking_uri(config["mlflow_uri"])
        mlflow.set_experiment(config["experiment"])

        # ── Load tokenised splits from S3 ─────────────────────────────
        s3 = boto3.client("s3")
        bucket = config["bucket"]

        def load_split(name):
            tmpdir = tempfile.mkdtemp()
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=bucket, Prefix=f"text2sql/processed/{name}/"
            )
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    fname = key.split("/")[-1]
                    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                    with open(os.path.join(tmpdir, fname), "wb") as f:
                        f.write(body)
            return load_from_disk(tmpdir)

        print("Loading train split...")
        train_ds = load_split("train")
        print("Loading val split...")
        val_ds = load_split("val")

        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        model = AutoModelForSeq2SeqLM.from_pretrained(config["base_model"])

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

        # ── Checkpointing Configuration ───────────────────────────────
        # /workspace is a persistent EFS mount.
        run_id_path = os.getenv("FLYTE_INTERNAL_EXECUTION_ID", "default")
        checkpoint_dir = f"/workspace/checkpoints/text2sql/{run_id_path}"

        training_args = Seq2SeqTrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            learning_rate=config["lr"],
            warmup_steps=config["warmup_steps"],
            evaluation_strategy="steps",
            eval_steps=config["eval_steps"],
            save_strategy="steps",
            save_steps=config["eval_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            predict_with_generate=True,
            generation_max_length=MAX_TARGET_LEN,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",  # We use MLflow via RayTrainReportCallback
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.add_callback(RayTrainReportCallback())
        trainer = prepare_trainer(trainer)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "base_model": config["base_model"],
                    "epochs": config["epochs"],
                    "batch_size": config["batch_size"],
                    "lr": config["lr"],
                }
            )
            # Automatically resume from the latest checkpoint if spot instance was preempted
            trainer.train(resume_from_checkpoint=True)
            final_metrics = trainer.evaluate()
            mlflow.log_metrics({k: v for k, v in final_metrics.items()})

            # Save best checkpoint locally then upload to S3
            ckpt_dir = "/tmp/best_checkpoint"
            trainer.save_model(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

            # Upload checkpoint to S3
            run_id = mlflow.active_run().info.run_id
            ckpt_s3_prefix = f"text2sql/checkpoints/{run_id}"
            for fname in os.listdir(ckpt_dir):
                fpath = os.path.join(ckpt_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as f:
                        s3.put_object(
                            Bucket=bucket,
                            Key=f"{ckpt_s3_prefix}/{fname}",
                            Body=f.read(),
                        )
            print(f"✅ Checkpoint uploaded to s3://{bucket}/{ckpt_s3_prefix}")
            mlflow.log_param("checkpoint_s3_path", f"s3://{bucket}/{ckpt_s3_prefix}")
            mlflow.set_tag("run_id", run_id)

            ray_train.report(
                {
                    "eval_loss": final_metrics.get("eval_loss", float("inf")),
                    "run_id": run_id,
                }
            )

    trainer = TorchTrainer(
        train_loop,
        train_loop_config={
            "base_model": BASE_MODEL,
            "bucket": S3_BUCKET,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "warmup_steps": DEFAULT_WARMUP_STEPS,
            "eval_steps": DEFAULT_EVAL_STEPS,
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "experiment": MLFLOW_EXPERIMENT,
        },
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
    )

    result = trainer.fit()
    run_id = result.metrics.get("run_id", "unknown")
    checkpoint_path = f"s3://{S3_BUCKET}/text2sql/checkpoints/{run_id}"
    print(f"✅ Training complete. Best checkpoint: {checkpoint_path}")
    return checkpoint_path
