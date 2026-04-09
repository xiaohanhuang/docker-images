"""LoRA/QLoRA fine-tuning implementation."""

from typing import Any, Dict, List, Optional


def run_lora_finetune(
    base_model: str,
    train_data_path: str,
    val_data_path: Optional[str],
    method: str,
    lora_r: int,
    lora_alpha: int,
    lora_target_modules: Optional[List[str]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    mlflow_experiment: Optional[str],
    trust_remote_code: bool,
) -> Dict[str, Any]:
    """Run LoRA/QLoRA fine-tuning and return results.

    Returns:
        Dictionary with keys: checkpoint_dir, mlflow_run_id, final_metrics.
    """
    import os
    import tempfile

    import mlflow
    import s3fs
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    s3 = s3fs.S3FileSystem()

    # Setup MLflow tracking
    if mlflow_experiment is not None:
        mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run() as run:
        mlflow_run_id = run.info.run_id

        mlflow.log_params(
            {
                "base_model": base_model,
                "method": method,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            }
        )

        # Download training data from S3
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            train_local_path = f.name
        s3.get(train_data_path, train_local_path)

        # Download validation data if provided
        val_local_path = None
        if val_data_path is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                val_local_path = f.name
            s3.get(val_data_path, val_local_path)

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        except Exception:
            print("[lora_finetune] Fast tokenizer failed, falling back to slow tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure quantization for QLoRA
        import torch

        quantization_config = None
        if method == "qlora":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load datasets
        train_dataset = load_dataset("json", data_files=train_local_path, split="train")
        eval_dataset = None
        if val_local_path is not None:
            eval_dataset = load_dataset("json", data_files=val_local_path, split="train")

        # ── Tokenize raw text if the dataset is not already tokenized ──────
        def _needs_tokenization(ds) -> bool:
            return "input_ids" not in ds.column_names

        def _pick_text_column(ds) -> str:
            for col in ("text", "chosen", "content", "prompt"):
                if col in ds.column_names:
                    return col
            return ds.column_names[0]

        def _tokenize_dataset(ds, text_col: str):
            max_length = 512

            def tokenize_fn(examples):
                tokenized = tokenizer(
                    examples[text_col],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized

            ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
            return ds

        if _needs_tokenization(train_dataset):
            text_col = _pick_text_column(train_dataset)
            train_dataset = _tokenize_dataset(train_dataset, text_col)
            if eval_dataset is not None:
                eval_dataset = _tokenize_dataset(eval_dataset, text_col)

        # Use local directory for training
        local_checkpoint_dir = "/tmp/lora_checkpoint"
        os.makedirs(local_checkpoint_dir, exist_ok=True)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=local_checkpoint_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset is not None else False,
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            report_to="mlflow",
            bf16=False,
            fp16=False,
            gradient_checkpointing=True if method == "qlora" else False,
            optim="paged_adamw_8bit" if method == "qlora" else "adamw_torch",
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        # Train the model
        result = trainer.train()

        # Merge LoRA adapter into base model and save the full model.
        # Downstream tasks (e.g. vLLM in OpenRLHF) need config.json and
        # full model weights — a bare adapter directory won't work.
        model = model.merge_and_unload()
        model.save_pretrained(local_checkpoint_dir)
        tokenizer.save_pretrained(local_checkpoint_dir)

        # Collect final metrics
        final_metrics: Dict[str, float] = {
            "train_loss": float(result.training_loss),
            "epochs": float(epochs),
            "learning_rate": learning_rate,
        }

        if eval_dataset is not None and hasattr(result, "metrics"):
            eval_metrics = trainer.evaluate()
            final_metrics["eval_loss"] = float(eval_metrics.get("eval_loss", 0.0))
            final_metrics["eval_perplexity"] = float(eval_metrics.get("eval_perplexity", 0.0))

        mlflow.log_metrics(final_metrics)

        # Clean up temporary files
        os.unlink(train_local_path)
        if val_local_path is not None:
            os.unlink(val_local_path)

    return {
        "checkpoint_dir": local_checkpoint_dir,
        "mlflow_run_id": mlflow_run_id,
        "final_metrics": final_metrics,
    }
