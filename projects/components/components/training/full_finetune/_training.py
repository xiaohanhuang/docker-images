"""Full-parameter fine-tuning implementation."""

import os
from typing import Any, Dict


def _safe_extract(tar: Any, dest_dir: str) -> None:
    """Extract tar members safely, preventing path traversal."""
    base_path = os.path.realpath(dest_dir)
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            continue
        member_path = os.path.realpath(os.path.join(dest_dir, member.name))
        if not member_path.startswith(base_path + os.sep) and member_path != base_path:
            raise ValueError(f"Attempted path traversal in tar file member: {member.name}")
    tar.extractall(path=dest_dir)


def run_full_finetune(
    base_model: str,
    train_data_path: str,
    val_data_path: str,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    use_efs_checkpoints: bool,
) -> Dict[str, Any]:
    """Run full-parameter fine-tuning and return results.

    Returns:
        Dictionary with keys: checkpoint_path, metrics.
    """
    import tarfile
    import uuid

    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # Download and extract train data
    train_extract_dir = "/tmp/train_extract"
    os.makedirs(train_extract_dir, exist_ok=True)
    with tarfile.open(train_data_path, "r:gz") as tar:
        _safe_extract(tar, train_extract_dir)
    train_dataset = load_from_disk(os.path.join(train_extract_dir, "train"))

    # Download and extract val data
    val_extract_dir = "/tmp/val_extract"
    os.makedirs(val_extract_dir, exist_ok=True)
    with tarfile.open(val_data_path, "r:gz") as tar:
        _safe_extract(tar, val_extract_dir)
    val_dataset = load_from_disk(os.path.join(val_extract_dir, "val"))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    # Determine output directory
    run_id = uuid.uuid4().hex[:8]
    if use_efs_checkpoints:
        checkpoint_dir = f"/mnt/efs/checkpoints/full_finetune_{run_id}"
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = f"/tmp/checkpoint_{run_id}"

    # Training arguments
    training_kwargs = {
        "output_dir": checkpoint_dir,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_steps": 10,
        "bf16": True,
        "report_to": "none",
    }

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()

    # Save final checkpoint atomically
    final_checkpoint_tmp = checkpoint_dir + ".tmp"
    trainer.save_model(final_checkpoint_tmp)
    tokenizer.save_pretrained(final_checkpoint_tmp)

    final_checkpoint = checkpoint_dir + "_final"
    os.rename(final_checkpoint_tmp, final_checkpoint)

    # Metrics
    metrics: Dict[str, float] = {
        "train_loss": result.training_loss,
        "epochs": float(num_epochs),
        "learning_rate": learning_rate,
    }

    if hasattr(result, "metrics") and result.metrics:
        eval_loss = result.metrics.get("eval_loss")
        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss

    return {
        "checkpoint_path": final_checkpoint,
        "metrics": metrics,
    }
