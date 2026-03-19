"""
Training component — full-parameter fine-tuning with optional FSDP.

Image: ml-gpu
"""

from ._task import full_finetune  # noqa: F401

__all__ = ["full_finetune"]
