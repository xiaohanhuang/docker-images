"""
train.py — Task 3: Fine-tune CodeT5+ on the Text2SQL dataset using
HuggingFace Seq2SeqTrainer on a single GPU.

Runs as a plain Flyte task on a GPU node (no Ray overhead).
Logs metrics to MLflow. Checkpoints are persisted on EFS so that
training can resume automatically after a spot preemption.
"""

import sys

from flytekit import PodTemplate, Resources, task

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

# ── Pod template: EFS mount + GPU toleration ───────────────────────────
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

    _gpu_pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(name="primary", volume_mounts=[_efs_mount]),
            ],
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
    _gpu_pod_template = None


@task(
    requests=Resources(cpu="4", mem="14Gi", gpu="1"),
    limits=Resources(cpu="4", mem="14Gi", gpu="1"),
    container_image=GPU_IMAGE,
    pod_template=_gpu_pod_template,
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
    Fine-tune Salesforce/codet5p-220m on a single GPU.
    Returns S3 path to best checkpoint.
    """
    import os
    import tempfile

    import boto3
    import mlflow
    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ── Load tokenised splits from S3 ─────────────────────────────────
    s3 = boto3.client("s3")

    def load_split(name):
        tmpdir = tempfile.mkdtemp()
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=S3_BUCKET, Prefix=f"text2sql/processed/{name}/"
        )
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                fname = key.split("/")[-1]
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                with open(os.path.join(tmpdir, fname), "wb") as f:
                    f.write(body)
        return load_from_disk(tmpdir)

    print("Loading train split...")
    train_ds = load_split("train")
    print("Loading val split...")
    val_ds = load_split("val")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # ── Checkpointing on EFS ──────────────────────────────────────────
    # /workspace is a persistent EFS mount that survives spot preemptions.
    run_id_path = os.getenv("FLYTE_INTERNAL_EXECUTION_ID", "default")
    checkpoint_dir = f"/workspace/checkpoints/text2sql/{run_id_path}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=DEFAULT_WARMUP_STEPS,
        evaluation_strategy="steps",
        eval_steps=DEFAULT_EVAL_STEPS,
        save_strategy="steps",
        save_steps=DEFAULT_EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",  # We log to MLflow explicitly
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ── Resume-aware checkpoint: find the latest checkpoint on EFS ────
    resume_ckpt = None
    if os.path.isdir(checkpoint_dir):
        ckpt_dirs = [
            d
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        if ckpt_dirs:
            resume_ckpt = os.path.join(
                checkpoint_dir, sorted(ckpt_dirs, key=lambda x: int(x.split("-")[-1]))[-1]
            )
            print(f"Resuming from checkpoint: {resume_ckpt}")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "base_model": BASE_MODEL,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "lr": learning_rate,
            }
        )

        trainer.train(resume_from_checkpoint=resume_ckpt)

        final_metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in final_metrics.items()})

        # Save best checkpoint locally then upload to S3
        ckpt_dir = "/tmp/best_checkpoint"
        trainer.save_model(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        run_id = mlflow.active_run().info.run_id
        ckpt_s3_prefix = f"text2sql/checkpoints/{run_id}"
        for fname in os.listdir(ckpt_dir):
            fpath = os.path.join(ckpt_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, "rb") as f:
                    s3.put_object(
                        Bucket=S3_BUCKET,
                        Key=f"{ckpt_s3_prefix}/{fname}",
                        Body=f.read(),
                    )
        print(f"✅ Checkpoint uploaded to s3://{S3_BUCKET}/{ckpt_s3_prefix}")
        mlflow.log_param("checkpoint_s3_path", f"s3://{S3_BUCKET}/{ckpt_s3_prefix}")
        mlflow.set_tag("run_id", run_id)

    checkpoint_path = f"s3://{S3_BUCKET}/{ckpt_s3_prefix}"
    print(f"✅ Training complete. Best checkpoint: {checkpoint_path}")
    return checkpoint_path
