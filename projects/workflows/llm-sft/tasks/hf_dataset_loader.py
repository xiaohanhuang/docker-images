"""
HuggingFace Dataset Loader — adapted for LLM-SFT pipeline.

Uses ``dataset`` parameter name (vs ``dataset_name`` in the shared component).

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
    dataset: str,
    split: str = "train",
    subset: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> FlyteFile:
    """Load a dataset from HuggingFace Hub and convert to JSONL format.

    Args:
        dataset: HuggingFace dataset name (e.g. ``tatsu-lab/alpaca``).
        split: Dataset split to load.
        subset: Optional dataset subset/configuration name.
        num_samples: Optional limit on samples. If None, loads the entire split.

    Returns:
        FlyteFile pointing to the JSONL output.
    """
    import json
    import os
    import tempfile

    from datasets import load_dataset

    load_kwargs = {"path": dataset, "split": split}
    if subset:
        load_kwargs["name"] = subset

    if num_samples is not None:
        load_kwargs["streaming"] = True
        ds = load_dataset(**load_kwargs)
        samples = list(ds.take(num_samples))
    else:
        ds = load_dataset(**load_kwargs)
        samples = ds

    safe_name = dataset.replace("/", "_")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        prefix=f"{safe_name}_{split}_",
        delete=False,
    ) as tmp:
        output_path = tmp.name
        for sample in samples:
            json.dump(sample, tmp)
            tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())

    return FlyteFile(path=output_path)
