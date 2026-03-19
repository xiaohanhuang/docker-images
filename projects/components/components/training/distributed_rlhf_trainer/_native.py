"""Native PyTorch RLHF training loop (no OpenRLHF / Ray dependency).

Used as a fallback when OpenRLHF is not installed. Mirrors ``rlhf_trainer``
logic but supports the distributed trainer's parameter interface.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from ._helpers import ALGORITHM_MAP


def run_native(
    sft_model_local: str,
    reward_model_local: str,
    dataset_local: str,
    save_dir: Path,
    *,
    algorithm: str,
    prompt_column: str,
    ppo_epochs: int,
    learning_rate: float,
    init_kl_coef: float,
    kl_target: Optional[float],
    cliprange: float,
    max_samples: int,
    generate_max_len: int,
    gradient_checkpointing: bool,
    train_batch_size: int,
    micro_train_batch_size: int,
    checkpoint_interval: int,
    mlflow_available: bool,
) -> tuple:
    """Native PyTorch training loop — mirrors ``rlhf_trainer`` logic.

    Returns (reward_stats, final_kl, training_metrics, final_path).
    """
    import copy

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    if algorithm not in ALGORITHM_MAP:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. " f"Choose from: {list(ALGORITHM_MAP.keys())}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[openrlhf-native] Device: {device}")
    print(f"[openrlhf-native] Algorithm: {algorithm}")

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _safe_save_model(model, path: str) -> None:
        """Save model, with fallback if deepspeed/nvcc is unavailable."""
        os.makedirs(path, exist_ok=True)
        try:
            model.save_pretrained(path, safe_serialization=False)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            if "nvcc" in str(exc) or "deepspeed" in str(exc).lower():
                print(
                    f"[openrlhf-native] save_pretrained failed ({exc}), "
                    "using torch.save fallback"
                )
                model.config.save_pretrained(path)
                torch.save(model.state_dict(), os.path.join(path, "pytorch_model.bin"))
            else:
                raise

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
        print("[openrlhf-native] Merging LoRA weights manually …")
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
        print(f"[openrlhf-native] Merged {merged} LoRA layers (scaling={scaling})")
        return base_model

    def _load_model(path: str, is_reward: bool = False):
        adapter_cfg = os.path.join(path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            _sanitize_adapter_config(path)
            config = PeftConfig.from_pretrained(path)
            base_name = str(config.base_model_name_or_path)
            print(f"[openrlhf-native] Loading LoRA adapter from {path}, base: {base_name}")
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
                print("[openrlhf-native] PeftModel.from_pretrained failed, merging manually")
                model = _merge_lora_manually(base, path, config)
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_name)
            except Exception:
                print("[openrlhf-native] Fast tokenizer failed, falling back to slow tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=False)
        else:
            print(f"[openrlhf-native] Loading full model from {path}")
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
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
            except Exception:
                print("[openrlhf-native] Fast tokenizer failed, falling back to slow tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # required for decoder-only generation
        if hasattr(model, "config") and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    print("[openrlhf-native] Loading policy model …")
    policy_model, tokenizer = _load_model(sft_model_local)
    policy_model.to(device)
    if isinstance(policy_model, PeftModel):
        print("[openrlhf-native] Merging policy LoRA adapter for training")
        policy_model = policy_model.merge_and_unload()
        policy_model.to(device)
    if gradient_checkpointing and hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable()
    for p in policy_model.parameters():
        p.requires_grad = True

    print("[openrlhf-native] Loading reference model …")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("[openrlhf-native] Loading reward model …")
    reward_model, _ = _load_model(reward_model_local, is_reward=True)
    if isinstance(reward_model, PeftModel):
        reward_model = reward_model.merge_and_unload()
    reward_model.to(device).eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    # ── Load prompts ──────────────────────────────────────────────────────
    with open(dataset_local) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    prompts = []
    for line in lines:
        try:
            obj = json.loads(line)
            text = obj.get(prompt_column, "")
            if not text:
                text = obj.get("text", "") or obj.get("input", "")
            if not text and "chosen" in obj:
                raw = obj["chosen"]
                parts = raw.split("\n\nAssistant:")
                if parts:
                    text = parts[0].replace("\n\nHuman:", "").strip()
            if text:
                prompts.append(text)
        except json.JSONDecodeError:
            if line:
                prompts.append(line)
    batch_size = max(micro_train_batch_size, 1)
    num_training_steps = min(max_samples, len(prompts) // batch_size)
    prompts = prompts[: num_training_steps * batch_size]
    print(f"[openrlhf-native] Loaded {len(prompts)} prompts")
    if not prompts:
        raise ValueError("No prompts found in dataset")

    # ── Optimiser ─────────────────────────────────────────────────────────
    policy_model.train()
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    # ── Helpers ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def _compute_logprobs(model, input_ids, response_start, attention_mask=None):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_lp = torch.gather(log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(token_lp)
        for i in range(mask.shape[0]):
            start = max(response_start[i] - 1, 0)
            mask[i, start:] = 1.0
        if attention_mask is not None:
            mask = mask * attention_mask[:, 1:]
        return (token_lp * mask).sum(dim=-1)

    def _compute_logprobs_grad(model, input_ids, response_start, attention_mask=None):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_lp = torch.gather(log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(token_lp)
        for i in range(mask.shape[0]):
            start = max(response_start[i] - 1, 0)
            mask[i, start:] = 1.0
        if attention_mask is not None:
            mask = mask * attention_mask[:, 1:]
        return (token_lp * mask).sum(dim=-1)

    @torch.no_grad()
    def _compute_reward(texts):
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        return reward_model(**enc).logits.squeeze(-1).float()

    # ── Training loop ─────────────────────────────────────────────────────
    all_rewards: list = []
    all_kls: list = []
    kl_coef = init_kl_coef
    global_step = 0
    max_steps = min(num_training_steps, len(prompts) // batch_size)
    ckpt_dir = Path(tempfile.mkdtemp(prefix="openrlhf-ckpt-"))

    print(
        f"[openrlhf-native] Starting training: {ppo_epochs} epoch(s), "
        f"{max_steps} steps/epoch, batch_size={batch_size}"
    )

    for epoch in range(ppo_epochs):
        print(f"\n[openrlhf-native] ═══ Epoch {epoch + 1}/{ppo_epochs} ═══")
        epoch_rewards: list = []
        epoch_kls: list = []

        for step in range(max_steps):
            global_step += 1
            batch_prompts = prompts[step * batch_size : (step + 1) * batch_size]
            if not batch_prompts:
                break

            # 1. Generate responses
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
                    max_new_tokens=generate_max_len,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            full_ids = gen_output
            full_mask = (full_ids != tokenizer.pad_token_id).long()
            response_starts = [int(pl) for pl in prompt_lengths]
            full_texts = tokenizer.batch_decode(full_ids, skip_special_tokens=True)

            # 2. Compute rewards
            rewards = _compute_reward(full_texts)
            batch_reward_mean = rewards.mean().item()
            epoch_rewards.extend(rewards.tolist())

            # 3. Reference log probs (KL)
            ref_logprobs = _compute_logprobs(ref_model, full_ids, response_starts, full_mask)

            # 4. Old policy log probs
            old_logprobs = _compute_logprobs(policy_model, full_ids, response_starts, full_mask)

            # 5. KL divergence
            kl = old_logprobs - ref_logprobs
            batch_kl = kl.mean().item()
            epoch_kls.append(batch_kl)

            # 6. Advantages
            kl_penalties = kl_coef * kl
            advantages = rewards - kl_penalties
            if algorithm == "grpo":
                adv_std = advantages.std() + 1e-8
                advantages = (advantages - advantages.mean()) / adv_std
            elif algorithm in ("rloo", "reinforce_baseline"):
                advantages = advantages - advantages.mean()

            # 7. Policy gradient update
            policy_model.train()
            if algorithm == "ppo":
                for _ in range(4):
                    new_lp = _compute_logprobs_grad(
                        policy_model,
                        full_ids,
                        response_starts,
                        full_mask,
                    )
                    ratio = torch.exp(new_lp - old_logprobs.detach())
                    clipped = ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
                    loss = -torch.min(
                        ratio * advantages.detach(),
                        clipped * advantages.detach(),
                    ).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    optimizer.step()
            else:
                new_lp = _compute_logprobs_grad(
                    policy_model,
                    full_ids,
                    response_starts,
                    full_mask,
                )
                loss = -(new_lp * advantages.detach()).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()

            # 8. Adaptive KL coefficient
            _kl_target = kl_target or 0.1
            if _kl_target > 0 and batch_kl > _kl_target * 1.5:
                kl_coef = min(kl_coef * 1.5, 1.0)
            elif _kl_target > 0 and batch_kl < _kl_target * 0.5:
                kl_coef = max(kl_coef * 0.5, 1e-6)

            if global_step % 5 == 0 or global_step == 1:
                print(
                    f"[openrlhf-native] step {global_step:4d} | "
                    f"reward {batch_reward_mean:.4f} | kl {batch_kl:.4f} | "
                    f"kl_coef {kl_coef:.6f} | loss {loss.item():.4f}"
                )
                if mlflow_available:
                    try:
                        import mlflow

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

            if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                ckpt = str(ckpt_dir / f"step-{global_step}")
                _safe_save_model(policy_model, ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"[openrlhf-native] Saved checkpoint: {ckpt}")

        if epoch_rewards:
            mean_r = sum(epoch_rewards) / len(epoch_rewards)
            mean_kl = sum(epoch_kls) / len(epoch_kls) if epoch_kls else 0.0
            print(
                f"[openrlhf-native] Epoch {epoch + 1} — "
                f"mean_reward: {mean_r:.4f}, mean_kl: {mean_kl:.4f}"
            )
            all_rewards.extend(epoch_rewards)
            all_kls.extend(epoch_kls)

    # ── Save final model ──────────────────────────────────────────────────
    final_path = str(save_dir / "final")
    _safe_save_model(policy_model, final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[openrlhf-native] Final model saved to: {final_path}")

    reward_stats: Dict[str, float] = {
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
    training_metrics: Dict[str, float] = {
        "total_steps": float(global_step),
        "total_episodes": float(ppo_epochs),
        "actor_loss": float(loss.item()) if global_step > 0 else 0.0,
        "critic_loss": 0.0,
    }

    if mlflow_available:
        try:
            import mlflow

            mlflow.log_metrics(
                {
                    "final/mean_reward": reward_stats["mean_reward"],
                    "final/kl_divergence": final_kl,
                    "final/total_steps": training_metrics["total_steps"],
                }
            )
        except Exception:
            pass

    return reward_stats, final_kl, training_metrics, final_path
