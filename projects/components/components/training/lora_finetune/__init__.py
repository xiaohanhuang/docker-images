"""
Training component — LoRA/QLoRA fine-tuning for HuggingFace models.

Image: ml-gpu
"""

from ._task import lora_finetune  # noqa: F401

__all__ = ["lora_finetune"]
