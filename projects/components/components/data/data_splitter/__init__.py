"""
Data component — split tokenized data into train/val/test sets.

Image: data-cpu
"""

from ._task import data_splitter  # noqa: F401

__all__ = ["data_splitter"]
