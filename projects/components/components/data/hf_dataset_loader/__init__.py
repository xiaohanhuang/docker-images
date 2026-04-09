"""
HuggingFace Dataset Loader component — loads datasets from HuggingFace Hub.

Supports loading datasets with optional splits, subsets, and sampling.
Converts to JSONL format for downstream tasks.

Image: data-cpu
"""

from typing import Optional

from flytekit import Resources, task
from flytekit.types.file import FlyteFile


@task(
    retries=3,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
    cache=True,
    cache_version="1.0",
)
def hf_dataset_loader(
    dataset_name: str,
    split: str = "train",
    subset: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> FlyteFile:
    """Load a dataset from HuggingFace Hub and convert to JSONL format.

    Args:
        dataset_name: HuggingFace dataset name (e.g. ``Anthropic/hh-rlhf``).
        split: Dataset split to load (e.g. ``train``, ``test``, ``validation``).
        subset: Optional dataset subset/configuration name.
        num_samples: Optional number of samples to load. If None, loads entire split.
                    Use this to limit dataset size for testing or resource constraints.

    Returns:
        FlyteFile pointing to the JSONL file with the dataset.
        Each line is a JSON object with fields from the original dataset.

    Examples:
        Load full training split:
            >>> result = hf_dataset_loader("Anthropic/hh-rlhf", split="train")

        Load 1000 samples from a specific subset:
            >>> result = hf_dataset_loader(
            ...     "openai/summarize_from_feedback",
            ...     split="train",
            ...     subset="comparisons",
            ...     num_samples=1000
            ... )
    """
    import json
    import os
    import tempfile

    # Lazy import to avoid loading at task registration time
    from datasets import load_dataset

    # Prepare load_dataset arguments
    load_kwargs = {"path": dataset_name, "split": split}
    if subset:
        load_kwargs["name"] = subset

    # Load dataset
    # Use streaming mode for efficiency if we're sampling
    if num_samples is not None:
        load_kwargs["streaming"] = True
        dataset = load_dataset(**load_kwargs)
        # Take only the requested number of samples
        dataset = dataset.take(num_samples)
        # Convert to list to materialize the stream
        samples = list(dataset)
    else:
        # Load full dataset into memory
        dataset = load_dataset(**load_kwargs)
        samples = dataset

    # Convert to JSONL format using a unique temp file to avoid collisions
    safe_name = dataset_name.replace("/", "_")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        prefix=f"{safe_name}_{split}_",
        delete=False,
    ) as tmp:
        output_path = tmp.name
        for sample in samples:
            # Convert sample to JSON and write as line
            json.dump(sample, tmp)
            tmp.write("\n")
        # Flush and fsync inside the with block while the file is still open
        tmp.flush()
        os.fsync(tmp.fileno())

    return FlyteFile(path=output_path)
