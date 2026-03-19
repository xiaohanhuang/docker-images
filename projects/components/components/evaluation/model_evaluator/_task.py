"""Flyte task definition for model_evaluator component."""

from typing import Dict, List, Optional

from flytekit import PodTemplate, Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

# ── GPU pod template with toleration ───────────────────────────────────
try:
    from kubernetes.client import V1PodSpec, V1Toleration

    _gpu_pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[],
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
    _gpu_pod_template = None


_model_evaluator_task_kwargs = dict(
    retries=1,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1"),
    cache=True,
    cache_version="1.0",
)
if _gpu_pod_template is not None:
    _model_evaluator_task_kwargs["pod_template"] = _gpu_pod_template


@task(**_model_evaluator_task_kwargs)
def model_evaluator(
    checkpoint_path: FlyteDirectory,
    test_data: FlyteFile,
    base_model: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate a fine-tuned model on test data.

    Args:
        checkpoint_path: Model checkpoint or LoRA adapter directory.
        test_data: Test dataset (Arrow format, tar.gz).
        base_model: Base model ID (required if checkpoint is LoRA adapter).
        metrics: List of metrics to compute (default: ["perplexity", "rouge"]).

    Returns:
        Dictionary mapping metric names to values.
    """
    import os
    import tarfile

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ._metrics import compute_metrics

    if metrics is None:
        metrics = ["perplexity", "rouge"]

    # Download checkpoint
    checkpoint_path.download()

    # Download and extract test data safely
    test_data.download()
    test_extract_dir = "/tmp/test_extract"
    os.makedirs(test_extract_dir, exist_ok=True)

    with tarfile.open(test_data.path, "r:gz") as tar:
        base_path = os.path.realpath(test_extract_dir)
        for member in tar.getmembers():
            member_path = os.path.realpath(os.path.join(test_extract_dir, member.name))
            if not member_path.startswith(base_path + os.sep) and member_path != base_path:
                raise ValueError(f"Unsafe path detected in archive member: {member.name}")
        tar.extractall(path=test_extract_dir)

    test_dataset = load_from_disk(os.path.join(test_extract_dir, "test"))

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = checkpoint_path.path
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)

    tokenizer_source = base_model if base_model is not None else checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if is_lora:
        if base_model is None:
            raise ValueError("base_model required for LoRA adapter evaluation")
        from peft import PeftModel

        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32
        ).to(device)
        model = PeftModel.from_pretrained(base_model_obj, checkpoint_dir).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.float32).to(
            device
        )

    model.eval()

    return compute_metrics(model, tokenizer, test_dataset, metrics, device)
