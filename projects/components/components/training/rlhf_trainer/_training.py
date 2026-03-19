"""Core RLHF training loop for the rlhf_trainer component.

Extracted from __init__.py to keep the task entry point thin.
"""

import os


def run_rlhf_training(
    *,
    sft_model_path: str,
    reward_model_path: str,
    dataset_path: str,
    algorithm: str,
    algorithm_map: dict,
    prompt_column: str,
    ppo_epochs: int,
    learning_rate: float,
    batch_size: int,
    kl_penalty: str,
    init_kl_coef: float,
    target_kl: float,
    max_new_tokens: int,
    gradient_checkpointing: bool,
    checkpoint_interval: int,
    num_training_steps: int,
) -> dict:
    """Execute the RLHF training loop and return results dict.

    Returns a dict with keys: checkpoint_path, mlflow_run_id,
    reward_stats, kl_divergence.
    """
    import copy
    import json
    import tempfile
    from pathlib import Path

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    # ── Validate algorithm ────────────────────────────────────────────────
    if algorithm not in algorithm_map:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: {list(algorithm_map.keys())}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rlhf] Device: {device}")
    print(f"[rlhf] Algorithm: {algorithm}")

    # ── Output directory ──────────────────────────────────────────────────
    save_dir = Path(tempfile.mkdtemp(prefix="rlhf-output-"))
    ckpt_dir = Path(tempfile.mkdtemp(prefix="rlhf-ckpt-"))

    # ── Download from S3 ─────────────────────────────────────────────────
    def _download_s3(uri: str, label: str) -> str:
        import s3fs

        s3 = s3fs.S3FileSystem()
        local = tempfile.mkdtemp(prefix=f"rlhf-{label}-")
        s3.get(uri.rstrip("/"), local, recursive=True)
        for root, _, files in os.walk(local):
            if "config.json" in files or "adapter_config.json" in files:
                return root
        return local

    if sft_model_path.startswith("s3://"):
        print(f"[rlhf] Downloading SFT model from {sft_model_path}")
        sft_model_path = _download_s3(sft_model_path, "sft")
    if reward_model_path.startswith("s3://"):
        print(f"[rlhf] Downloading reward model from {reward_model_path}")
        reward_model_path = _download_s3(reward_model_path, "reward")

    # ── Load policy model (SFT checkpoint) ────────────────────────────────
    def _sanitize_adapter_config(path: str):
        """Strip adapter_config.json fields unknown to the installed PEFT version."""
        import inspect

        cfg_path = os.path.join(path, "adapter_config.json")
        with open(cfg_path) as f:
            raw = json.load(f)
        peft_type = raw.get("peft_type", "LORA")
        from peft import PEFT_TYPE_TO_CONFIG_MAPPING

        config_cls = PEFT_TYPE_TO_CONFIG_MAPPING.get(peft_type)
        if config_cls is None:
            return
        valid_keys = set(inspect.signature(config_cls.__init__).parameters)
        filtered = {k: v for k, v in raw.items() if k in valid_keys}
        if len(filtered) < len(raw):
            removed = set(raw) - set(filtered)
            print(f"[peft-compat] Removed unknown fields from adapter_config.json: {removed}")
            with open(cfg_path, "w") as f:
                json.dump(filtered, f, indent=2)

    def _merge_lora_manually(base_model, adapter_path: str, config):
        """Merge LoRA adapter weights directly into base model without PeftModel."""
        print("[rlhf] Merging LoRA weights manually …")
        safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_path, "adapter_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            adapter_state = load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            adapter_state = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

        scaling = config.lora_alpha / config.r
        base_params = dict(base_model.named_parameters())
        merged = 0
        for key in list(adapter_state.keys()):
            if ".lora_A." not in key:
                continue
            key_b = key.replace(".lora_A.", ".lora_B.")
            if key_b not in adapter_state:
                continue
            base_key = key.replace(".lora_A.weight", ".weight")
            if base_key.startswith("base_model.model."):
                base_key = base_key[len("base_model.model.") :]
            if base_key in base_params:
                lora_a = adapter_state[key]
                lora_b = adapter_state[key_b]
                delta = (lora_b @ lora_a) * scaling
                base_params[base_key].data += delta.to(base_params[base_key].dtype)
                merged += 1
        print(f"[rlhf] Merged {merged} LoRA layers (scaling={scaling})")
        return base_model

    def _load_model(path: str, is_reward: bool = False):
        """Load a model, handling LoRA adapters transparently."""
        adapter_cfg = os.path.join(path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            _sanitize_adapter_config(path)
            config = PeftConfig.from_pretrained(path)
            base_name = str(config.base_model_name_or_path)
            print(f"[rlhf] Loading LoRA adapter from {path}, base: {base_name}")
            if is_reward:
                base = AutoModelForSequenceClassification.from_pretrained(
                    base_name,
                    num_labels=1,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    base_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            try:
                model = PeftModel.from_pretrained(base, path)
            except (RuntimeError, ImportError, OSError):
                print("[rlhf] PeftModel.from_pretrained failed, merging manually")
                model = _merge_lora_manually(base, path, config)
            tokenizer = AutoTokenizer.from_pretrained(base_name)
        else:
            print(f"[rlhf] Loading full model from {path}")
            if is_reward:
                model = AutoModelForSequenceClassification.from_pretrained(
                    path,
                    num_labels=1,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # required for decoder-only generation
        if hasattr(model, "config") and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    print("[rlhf] Loading policy model …")
    policy_model, tokenizer = _load_model(sft_model_path)
    policy_model.to(device)

    # Merge and unload LoRA for policy so we train full weights
    if isinstance(policy_model, PeftModel):
        print("[rlhf] Merging policy LoRA adapter for training")
        policy_model = policy_model.merge_and_unload()
        policy_model.to(device)

    # Enable gradient checkpointing *after* merge to avoid grad issues
    if gradient_checkpointing and hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable()

    # Ensure all policy parameters are trainable
    for p in policy_model.parameters():
        p.requires_grad = True

    print("[rlhf] Loading reference model …")
    # Reference model: frozen copy of policy
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("[rlhf] Loading reward model …")
    reward_model, _ = _load_model(reward_model_path, is_reward=True)
    if isinstance(reward_model, PeftModel):
        reward_model = reward_model.merge_and_unload()
    reward_model.to(device).eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    # ── Load prompts ──────────────────────────────────────────────────────
    import s3fs as _s3fs

    if dataset_path.startswith("s3://"):
        print(f"[rlhf] Downloading dataset from {dataset_path}")
        _s3 = _s3fs.S3FileSystem()
        local_data = tempfile.mktemp(suffix=".jsonl", prefix="rlhf-data-")
        _s3.get(dataset_path.rstrip("/"), local_data)
        data_file = local_data
    else:
        data_file = dataset_path
    with open(data_file) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    prompts = []
    for line in lines:
        try:
            obj = json.loads(line)
            # Try the configured prompt column first
            text = obj.get(prompt_column, "")
            if not text:
                # Fall back to common column names
                text = obj.get("text", "") or obj.get("input", "")
            if not text and "chosen" in obj:
                # Anthropic hh-rlhf format: extract the Human turn from chosen
                raw = obj["chosen"]
                parts = raw.split("\n\nAssistant:")
                if parts:
                    text = parts[0].replace("\n\nHuman:", "").strip()
            if text:
                prompts.append(text)
        except json.JSONDecodeError:
            if line:
                prompts.append(line)
    prompts = prompts[: num_training_steps * batch_size]
    print(f"[rlhf] Loaded {len(prompts)} prompts")
    if not prompts:
        raise ValueError("No prompts found in dataset")

    # ── Set up optimizer ──────────────────────────────────────────────────
    policy_model.train()
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    # ── Helper: compute per-token log probs ───────────────────────────────
    @torch.no_grad()
    def _compute_logprobs(model, input_ids, response_start, attention_mask=None):
        """Compute sum of log-probs for the response portion of input_ids."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather log probs of the actual tokens (shifted by 1)
        # response tokens are at positions response_start to end
        token_log_probs = torch.gather(
            log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        # Mask to only count response tokens
        mask = torch.zeros_like(token_log_probs)
        for i in range(mask.shape[0]):
            start = max(response_start[i] - 1, 0)  # shifted by 1
            mask[i, start:] = 1.0
        if attention_mask is not None:
            mask = mask * attention_mask[:, 1:]
        return (token_log_probs * mask).sum(dim=-1)  # (B,)

    def _compute_logprobs_grad(model, input_ids, response_start, attention_mask=None):
        """Same as above but preserves gradient."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        mask = torch.zeros_like(token_log_probs)
        for i in range(mask.shape[0]):
            start = max(response_start[i] - 1, 0)
            mask[i, start:] = 1.0
        if attention_mask is not None:
            mask = mask * attention_mask[:, 1:]
        return (token_log_probs * mask).sum(dim=-1)

    @torch.no_grad()
    def _compute_reward(texts):
        """Score a batch of texts with the reward model."""
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        logits = reward_model(**enc).logits.squeeze(-1)  # (B,)
        return logits.float()

    # ── Training loop ─────────────────────────────────────────────────────
    ppo_clip = 0.2
    all_rewards: list[float] = []
    all_kls: list[float] = []
    kl_coef = init_kl_coef
    global_step = 0
    max_steps = min(num_training_steps, len(prompts) // max(batch_size, 1))

    print(
        f"[rlhf] Starting training: {ppo_epochs} epoch(s), "
        f"{max_steps} steps/epoch, batch_size={batch_size}"
    )

    # Start MLflow
    try:
        import mlflow

        mlflow.set_experiment("rlhf-alignment")
        mlflow_run = mlflow.start_run(run_name=f"rlhf-{algorithm}-native")
        mlflow.log_params(
            {
                "algorithm": algorithm,
                "framework": "native-torch",
                "sft_model": sft_model_path,
                "reward_model": reward_model_path,
                "learning_rate": learning_rate,
                "ppo_epochs": ppo_epochs,
                "batch_size": batch_size,
                "init_kl_coef": init_kl_coef,
            }
        )
        mlflow_available = True
    except Exception as exc:
        print(f"[rlhf] MLflow unavailable: {exc}")
        mlflow_available = False
        mlflow_run = None

    try:
        for epoch in range(ppo_epochs):
            print(f"\n[rlhf] ═══ Epoch {epoch + 1}/{ppo_epochs} ═══")
            epoch_rewards: list[float] = []
            epoch_kls: list[float] = []

            for step in range(max_steps):
                global_step += 1
                start_idx = step * batch_size
                batch_prompts = prompts[start_idx : start_idx + batch_size]
                if not batch_prompts:
                    break

                # — 1. Generate responses —
                policy_model.eval()
                gen_inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                prompt_lengths = gen_inputs["attention_mask"].sum(dim=-1).tolist()

                with torch.no_grad():
                    gen_output = policy_model.generate(
                        **gen_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Build full sequences and decode
                full_ids = gen_output  # (B, prompt_len + response_len)
                full_mask = (full_ids != tokenizer.pad_token_id).long()
                response_starts = [int(pl) for pl in prompt_lengths]

                # Decode full texts for reward scoring
                full_texts = tokenizer.batch_decode(full_ids, skip_special_tokens=True)

                # — 2. Compute rewards —
                rewards = _compute_reward(full_texts)
                batch_reward_mean = rewards.mean().item()
                epoch_rewards.extend(rewards.tolist())

                # — 3. Compute reference log probs (for KL) —
                ref_logprobs = _compute_logprobs(
                    ref_model,
                    full_ids,
                    response_starts,
                    full_mask,
                )

                # — 4. Compute old policy log probs —
                old_logprobs = _compute_logprobs(
                    policy_model,
                    full_ids,
                    response_starts,
                    full_mask,
                )

                # — 5. KL divergence (approx) —
                kl = old_logprobs - ref_logprobs  # (B,)
                batch_kl = kl.mean().item()
                epoch_kls.append(batch_kl)

                # — 6. Compute advantages —
                kl_penalties = kl_coef * kl
                advantages = rewards - kl_penalties

                if algorithm == "grpo":
                    # Group normalize advantages
                    adv_mean = advantages.mean()
                    adv_std = advantages.std() + 1e-8
                    advantages = (advantages - adv_mean) / adv_std
                elif algorithm in ("rloo", "reinforce_baseline"):
                    # Subtract mean as baseline
                    advantages = advantages - advantages.mean()

                # — 7. Policy gradient update —
                policy_model.train()

                if algorithm == "ppo":
                    # PPO: multiple mini-epochs with clipping
                    for ppo_iter in range(4):
                        new_logprobs = _compute_logprobs_grad(
                            policy_model,
                            full_ids,
                            response_starts,
                            full_mask,
                        )
                        ratio = torch.exp(new_logprobs - old_logprobs.detach())
                        clipped_ratio = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip)
                        loss = -torch.min(
                            ratio * advantages.detach(),
                            clipped_ratio * advantages.detach(),
                        ).mean()

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            policy_model.parameters(),
                            max_norm=1.0,
                        )
                        optimizer.step()
                else:
                    # REINFORCE++ / GRPO / RLOO: single update
                    new_logprobs = _compute_logprobs_grad(
                        policy_model,
                        full_ids,
                        response_starts,
                        full_mask,
                    )
                    loss = -(new_logprobs * advantages.detach()).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy_model.parameters(),
                        max_norm=1.0,
                    )
                    optimizer.step()

                # — 8. Adaptive KL coefficient —
                if target_kl > 0 and batch_kl > target_kl * 1.5:
                    kl_coef = min(kl_coef * 1.5, 1.0)
                elif target_kl > 0 and batch_kl < target_kl * 0.5:
                    kl_coef = max(kl_coef * 0.5, 1e-6)

                if global_step % 5 == 0 or global_step == 1:
                    print(
                        f"[rlhf] step {global_step:4d} | "
                        f"reward {batch_reward_mean:.4f} | "
                        f"kl {batch_kl:.4f} | "
                        f"kl_coef {kl_coef:.6f} | "
                        f"loss {loss.item():.4f}"
                    )
                    if mlflow_available:
                        try:
                            mlflow.log_metrics(
                                {
                                    "train/reward": batch_reward_mean,
                                    "train/kl": batch_kl,
                                    "train/loss": loss.item(),
                                    "train/kl_coef": kl_coef,
                                },
                                step=global_step,
                            )
                        except Exception:
                            pass

                # Checkpointing
                if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                    ckpt = str(ckpt_dir / f"step-{global_step}")
                    policy_model.save_pretrained(ckpt, safe_serialization=False)
                    tokenizer.save_pretrained(ckpt)
                    print(f"[rlhf] Saved checkpoint: {ckpt}")

            # End of epoch summary
            if epoch_rewards:
                mean_r = sum(epoch_rewards) / len(epoch_rewards)
                mean_kl = sum(epoch_kls) / len(epoch_kls) if epoch_kls else 0.0
                print(
                    f"[rlhf] Epoch {epoch + 1} — mean_reward: {mean_r:.4f}, "
                    f"mean_kl: {mean_kl:.4f}"
                )
                all_rewards.extend(epoch_rewards)
                all_kls.extend(epoch_kls)

        # ── Save final model ──────────────────────────────────────────────
        final_path = str(save_dir / "final")
        os.makedirs(final_path, exist_ok=True)
        policy_model.save_pretrained(final_path, safe_serialization=False)
        tokenizer.save_pretrained(final_path)
        print(f"[rlhf] Final model saved to: {final_path}")

        # Compute final metrics
        reward_stats = {
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "std_reward": (
                (
                    sum((r - sum(all_rewards) / len(all_rewards)) ** 2 for r in all_rewards)
                    / len(all_rewards)
                )
                ** 0.5
                if len(all_rewards) > 1
                else 0.0
            ),
            "max_reward": max(all_rewards) if all_rewards else 0.0,
            "min_reward": min(all_rewards) if all_rewards else 0.0,
        }
        final_kl = all_kls[-1] if all_kls else 0.0

        if mlflow_available:
            try:
                mlflow.log_metrics(
                    {
                        "final/mean_reward": reward_stats["mean_reward"],
                        "final/kl_divergence": final_kl,
                        "final/total_steps": global_step,
                    }
                )
            except Exception:
                pass

    finally:
        run_id = mlflow_run.info.run_id if mlflow_run else "no-mlflow"
        if mlflow_available:
            try:
                mlflow.end_run()
            except Exception:
                pass

    print(f"[rlhf] Training complete — {global_step} steps")
    print(f"[rlhf] Mean reward: {reward_stats['mean_reward']:.4f}")
    print(f"[rlhf] Final KL: {final_kl:.4f}")

    return {
        "checkpoint_path": final_path,
        "mlflow_run_id": run_id,
        "reward_stats": reward_stats,
        "kl_divergence": final_kl,
    }
