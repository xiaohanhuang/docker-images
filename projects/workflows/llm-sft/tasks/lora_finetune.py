"""
Training component — LoRA/QLoRA fine-tuning (FlyteFile interface).

This task uses FlyteFile-based inputs/outputs consistent with the LLM-SFT
workflow pipeline.  The shared ``components.training.lora_finetune`` component
has a different (S3-path-based) interface and is used by other workflows.

Image: ml-gpu
"""

import logging
from typing import Dict, Tuple

from flytekit import PodTemplate, Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

logger = logging.getLogger(__name__)

# ── GPU pod template with EFS mount and GPU toleration ─────────────────
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
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name="efs-claim"),
    )
    _efs_mount = V1VolumeMount(name="efs-storage", mount_path="/mnt/efs")

    _gpu_efs_pod_template = PodTemplate(
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
            node_selector={"role": "gpu-worker"},
        )
    )
except ImportError:
    _gpu_efs_pod_template = None


@task(
    retries=1,
    requests=Resources(cpu="8", mem="32Gi", gpu="1"),
    limits=Resources(cpu="16", mem="64Gi", gpu="1"),
    cache=False,
    pod_template=_gpu_efs_pod_template,
)
def lora_finetune(
    base_model: str,
    train_data: FlyteFile,
    val_data: FlyteFile,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    quantization: str = "none",
    gradient_accumulation_steps: int = 4,
    use_efs_checkpoints: bool = True,
) -> Tuple[FlyteDirectory, Dict[str, float]]:
    """LoRA/QLoRA fine-tuning for HuggingFace causal LM models.

    Args:
        base_model: HuggingFace model ID (e.g., ``meta-llama/Llama-3.1-8B``).
        train_data: Training dataset (Arrow format, tar.gz).
        val_data: Validation dataset (Arrow format, tar.gz).
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        batch_size: Per-device training batch size.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        quantization: ``"none"`` for standard LoRA, ``"4bit"`` for QLoRA.
        gradient_accumulation_steps: Gradient accumulation steps.
        use_efs_checkpoints: Save checkpoints to EFS for durability.

    Returns:
        Tuple of (checkpoint_directory, training_metrics).
    """
    import os
    import tarfile
    import uuid

    from datasets import load_from_disk
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    def _safe_extract(tar, dest_dir: str):
        """Extract tar members safely, preventing path traversal."""
        base_path = os.path.realpath(dest_dir)
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                continue
            member_path = os.path.realpath(os.path.join(dest_dir, member.name))
            if not member_path.startswith(base_path + os.sep) and member_path != base_path:
                raise ValueError(f"Attempted path traversal in tar file member: {member.name}")
        tar.extractall(path=dest_dir)

    # Download and extract train data
    train_data.download()
    train_extract_dir = "/tmp/train_extract"
    os.makedirs(train_extract_dir, exist_ok=True)
    with tarfile.open(train_data.path, "r:gz") as tar:
        _safe_extract(tar, train_extract_dir)
    train_dataset = load_from_disk(os.path.join(train_extract_dir, "train"))

    # Download and extract val data
    val_data.download()
    val_extract_dir = "/tmp/val_extract"
    os.makedirs(val_extract_dir, exist_ok=True)
    with tarfile.open(val_data.path, "r:gz") as tar:
        _safe_extract(tar, val_extract_dir)
    val_dataset = load_from_disk(os.path.join(val_extract_dir, "val"))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization for QLoRA
    import torch

    quantization_config = None
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if quantization == "none" else None,
        device_map="auto",
    )

    # Apply LoRA adapter
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Determine output directory (unique per execution to avoid collisions)
    run_id = uuid.uuid4().hex[:8]
    if use_efs_checkpoints:
        checkpoint_dir = f"/mnt/efs/checkpoints/lora_finetune_{run_id}"
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = f"/tmp/checkpoint_{run_id}"

    # Training arguments
    use_qlora = quantization == "4bit"
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=10,
        bf16=not use_qlora,
        fp16=use_qlora,
        gradient_checkpointing=use_qlora,
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    result = trainer.train()

    # Save final checkpoint atomically (adapter weights + tokenizer)
    final_checkpoint_tmp = checkpoint_dir + ".tmp"
    trainer.save_model(final_checkpoint_tmp)
    tokenizer.save_pretrained(final_checkpoint_tmp)

    final_checkpoint = checkpoint_dir + "_final"
    os.rename(final_checkpoint_tmp, final_checkpoint)

    # Collect metrics
    metrics: Dict[str, float] = {
        "train_loss": result.training_loss,
        "epochs": float(num_epochs),
        "learning_rate": learning_rate,
        "lora_r": float(lora_r),
        "lora_alpha": float(lora_alpha),
    }

    if hasattr(result, "metrics") and result.metrics:
        eval_loss = result.metrics.get("eval_loss")
        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss

    return FlyteDirectory(path=final_checkpoint), metrics
