"""Reward model training implementation."""

from typing import Any, Dict


def run_reward_model_training(
    base_model: str,
    preference_data_path: str,
    prompt_column: str,
    chosen_column: str,
    rejected_column: str,
    modeling_type: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    use_lora: bool,
    lora_rank: int,
    lora_alpha: int,
    num_gpus: int,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
) -> Dict[str, Any]:
    """Run reward model training and return results.

    Returns:
        Dictionary with keys: checkpoint_path, run_id, accuracy,
        reward_margin, final_loss.
    """
    import os
    import tempfile

    import mlflow
    import torch
    import torch.nn.functional as F  # noqa: N812
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    # Validate modeling_type
    if modeling_type not in ("bradley_terry", "regression"):
        raise ValueError(
            f"Invalid modeling_type '{modeling_type}'. Must be 'bradley_terry' or 'regression'."
        )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params(
            {
                "base_model": base_model,
                "modeling_type": modeling_type,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_length": max_length,
                "use_lora": use_lora,
                "lora_rank": lora_rank if use_lora else None,
                "lora_alpha": lora_alpha if use_lora else None,
                "num_gpus": num_gpus,
            }
        )

        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        except Exception:
            print("[reward_model_trainer] Fast tokenizer failed, falling back to slow tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=1,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model.to(device)

        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Load and preprocess dataset — download from S3 if needed
        local_data_path = preference_data_path
        _s3_tmp_path = None
        if preference_data_path.startswith("s3://"):
            import s3fs

            s3 = s3fs.S3FileSystem()
            tmp_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
            local_data_path = tmp_file.name
            _s3_tmp_path = local_data_path
            tmp_file.close()
            s3.get(preference_data_path, local_data_path)
        dataset = load_dataset("json", data_files=local_data_path, split="train")

        def preprocess_function(examples):
            """Tokenize preference pairs."""
            chosen_texts = [
                p + " " + c for p, c in zip(examples[prompt_column], examples[chosen_column])
            ]
            chosen_encodings = tokenizer(
                chosen_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            rejected_texts = [
                p + " " + r for p, r in zip(examples[prompt_column], examples[rejected_column])
            ]
            rejected_encodings = tokenizer(
                rejected_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids_chosen": chosen_encodings["input_ids"],
                "attention_mask_chosen": chosen_encodings["attention_mask"],
                "input_ids_rejected": rejected_encodings["input_ids"],
                "attention_mask_rejected": rejected_encodings["attention_mask"],
            }

        dataset = dataset.map(
            preprocess_function, batched=True, remove_columns=dataset.column_names
        )
        dataset.set_format(type="torch")

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_optimizer_steps = (len(dataloader) // gradient_accumulation_steps) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_optimizer_steps),
            num_training_steps=total_optimizer_steps,
        )

        # Training loop
        model.train()
        global_step = 0
        final_loss = 0.0
        final_accuracy = 0.0
        final_reward_margin = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            total_reward_diff = 0.0

            for batch_idx, batch in enumerate(dataloader):
                # Forward pass for chosen responses
                chosen_outputs = model(
                    input_ids=batch["input_ids_chosen"].to(device),
                    attention_mask=batch["attention_mask_chosen"].to(device),
                )
                chosen_rewards = chosen_outputs.logits.squeeze(-1)

                # Forward pass for rejected responses
                rejected_outputs = model(
                    input_ids=batch["input_ids_rejected"].to(device),
                    attention_mask=batch["attention_mask_rejected"].to(device),
                )
                rejected_rewards = rejected_outputs.logits.squeeze(-1)

                # Compute reward difference once, reuse for loss and metrics
                with torch.no_grad():
                    reward_diff = chosen_rewards - rejected_rewards

                # Compute loss based on modeling type
                if modeling_type == "bradley_terry":
                    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                else:
                    target_scores = torch.ones_like(chosen_rewards)
                    loss = F.mse_loss(chosen_rewards, target_scores)

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # Track metrics
                epoch_loss += loss.item() * gradient_accumulation_steps
                correct_predictions += (reward_diff > 0).sum().item()
                total_predictions += reward_diff.size(0)
                total_reward_diff += reward_diff.sum().item()

                is_accum_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                is_last_batch = (batch_idx + 1) == len(dataloader)

                # Optimizer step with gradient accumulation
                if is_accum_step or is_last_batch:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 10 == 0:
                        mlflow.log_metric("train_loss", loss.item(), step=global_step)

            # Epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            reward_margin = total_reward_diff / total_predictions

            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("reward_margin", reward_margin, step=epoch)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {avg_loss:.4f}, "
                f"Accuracy: {accuracy:.4f}, "
                f"Reward Margin: {reward_margin:.4f}"
            )

            final_loss = avg_loss
            final_accuracy = accuracy
            final_reward_margin = reward_margin

        # Save checkpoint to EFS with atomic write
        efs_checkpoint_path = "/mnt/efs/reward-model-checkpoints"
        os.makedirs(efs_checkpoint_path, exist_ok=True)

        final_checkpoint = os.path.join(efs_checkpoint_path, "final")
        tmp_checkpoint = f"{final_checkpoint}.tmp"

        if os.path.exists(tmp_checkpoint):
            import shutil

            shutil.rmtree(tmp_checkpoint)

        # Merge LoRA adapter into base model so downstream consumers
        # (e.g. OpenRLHF RewardModelActor) get a standalone model with
        # a full config.json including model_type.
        if use_lora:
            model = model.merge_and_unload()

        model.save_pretrained(tmp_checkpoint)
        tokenizer.save_pretrained(tmp_checkpoint)

        os.sync()

        old_checkpoint = f"{final_checkpoint}.old"
        if os.path.exists(final_checkpoint):
            os.rename(final_checkpoint, old_checkpoint)
        os.rename(tmp_checkpoint, final_checkpoint)

        if os.path.exists(old_checkpoint):
            import shutil

            shutil.rmtree(old_checkpoint)

        mlflow.log_metrics(
            {
                "final_accuracy": final_accuracy,
                "final_reward_margin": final_reward_margin,
                "final_loss": final_loss,
            }
        )
        mlflow.log_param("efs_checkpoint_path", final_checkpoint)

        run_id = run.info.run_id

    # Clean up temporary S3 download file
    if _s3_tmp_path and os.path.exists(_s3_tmp_path):
        os.unlink(_s3_tmp_path)

    return {
        "checkpoint_path": final_checkpoint,
        "run_id": run_id,
        "accuracy": final_accuracy,
        "reward_margin": final_reward_margin,
        "final_loss": final_loss,
    }
