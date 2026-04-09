"""Model loading helpers for reward model benchmarking."""

import os


def _merge_lora_manually(base_model, adapter_path: str, config, torch):
    """Merge LoRA adapter weights directly into base model without PeftModel."""
    print("[reward_bench] Merging LoRA weights manually …")
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
    print(f"[reward_bench] Merged {merged} LoRA layers (scaling={scaling})")
    return base_model


def _find_local_model_root(search_dir: str) -> str:
    """Return the downloaded directory that actually contains model files."""
    tokenizer_root = None
    for root, _dirs, files in os.walk(search_dir):
        if "config.json" in files or "adapter_config.json" in files:
            return root
        if tokenizer_root is None and (
            "tokenizer.json" in files or "tokenizer_config.json" in files
        ):
            tokenizer_root = root
    return tokenizer_root or search_dir


def _load_tokenizer_and_model(model_path: str, device: str):
    """Load either a full reward model or a PEFT adapter checkpoint.

    Args:
        model_path: Local path to model or adapter checkpoint.
        device: Device to load the model onto ('cuda' or 'cpu').

    Returns:
        Tuple of (tokenizer, model).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        import inspect
        import json

        from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftConfig, PeftModel

        # Sanitize adapter_config.json so PeftModel.from_pretrained also works
        with open(adapter_config_path) as f:
            raw = json.load(f)
        peft_type = raw.get("peft_type", "LORA")
        config_cls = PEFT_TYPE_TO_CONFIG_MAPPING.get(peft_type)
        if config_cls is not None:
            valid_keys = set(inspect.signature(config_cls.__init__).parameters)
            filtered = {k: v for k, v in raw.items() if k in valid_keys}
            if len(filtered) < len(raw):
                removed = set(raw) - set(filtered)
                print(f"[peft-compat] Removed unknown fields: {removed}")
                with open(adapter_config_path, "w") as f:
                    json.dump(filtered, f, indent=2)

        peft_config = PeftConfig.from_pretrained(model_path)
        tokenizer_source = model_path
        if not (
            os.path.exists(os.path.join(model_path, "tokenizer.json"))
            or os.path.exists(os.path.join(model_path, "tokenizer_config.json"))
        ):
            tokenizer_source = peft_config.base_model_name_or_path

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except Exception:
            print("[reward_bench] Fast tokenizer failed, falling back to slow tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=1,
            torch_dtype=torch.float32,
        )
        try:
            model = PeftModel.from_pretrained(base_model, model_path).to(device)
        except (RuntimeError, ImportError, OSError):
            print("[reward_bench] PeftModel.from_pretrained failed, merging LoRA manually")
            model = _merge_lora_manually(base_model, model_path, peft_config, torch).to(device)
        return tokenizer, model

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        print("[reward_bench] Fast tokenizer failed, falling back to slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=torch.float32,
    ).to(device)
    return tokenizer, model
