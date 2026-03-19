"""
Training component — fine-tune a causal LM with LoRA/QLoRA via PEFT.

Image: ml-gpu
"""

from typing import Dict, Tuple

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile


@task(
    retries=1,
    requests=Resources(cpu="8", mem="32Gi", gpu="1"),
    limits=Resources(cpu="16", mem="64Gi", gpu="1"),
    cache=False,
)
def finetune_lm(
    base_model: str,
    train_data: FlyteFile,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
) -> Tuple[FlyteDirectory, Dict[str, float]]:
    """Fine-tune a causal LM using LoRA/QLoRA.

    Args:
        base_model: HuggingFace model ID or local path.
        train_data: JSONL training file (each line has a ``text`` or ``messages`` field).
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate for AdamW.
        lora_rank: LoRA rank *r*.

    Returns:
        Tuple of (checkpoint_path, training_metrics_dict).
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    train_data.download()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="/tmp/checkpoint",
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        save_strategy="epoch",
        logging_steps=10,
    )

    # Heavy import inside task body to avoid loading torch at import time
    from datasets import load_dataset
    from transformers import Trainer

    train_dataset = load_dataset("json", data_files=train_data.path, split="train")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()

    checkpoint_path = training_args.output_dir
    metrics: Dict[str, float] = {
        "train_loss": result.training_loss,
        "epochs": float(num_epochs),
        "learning_rate": learning_rate,
    }
    return FlyteDirectory(path=checkpoint_path), metrics
