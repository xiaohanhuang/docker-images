"""Flyte task definition for data_splitter component."""

from typing import Tuple

from flytekit import Resources, task
from flytekit.types.file import FlyteFile


@task(
    retries=2,
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="4", mem="16Gi"),
    cache=True,
    cache_version="1.0",
)
def data_splitter(
    tokenized_data: FlyteFile,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[FlyteFile, FlyteFile, FlyteFile]:
    """Split tokenized data into train/val/test sets.

    Args:
        tokenized_data: Tokenized dataset (Arrow format, tar.gz archive).
        train_ratio: Fraction for training (default: 0.8).
        val_ratio: Fraction for validation (default: 0.1).
        test_ratio: Fraction for test (default: 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_data, val_data, test_data) as FlyteFile objects.
    """
    import os
    import tarfile

    from datasets import load_from_disk

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Download and extract
    tokenized_data.download()
    extract_dir = "/tmp/tokenized_extract"
    os.makedirs(extract_dir, exist_ok=True)

    def _safe_members(tar, dest_dir: str):
        """Yield only members that extract within dest_dir (prevent path traversal)."""
        base_path = os.path.realpath(dest_dir)
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                continue
            member_path = os.path.realpath(os.path.join(dest_dir, member.name))
            if not member_path.startswith(base_path + os.sep) and member_path != base_path:
                raise ValueError(f"Attempted path traversal in tar file member: {member.name}")
            yield member

    with tarfile.open(tokenized_data.path, "r:gz") as tar:
        tar.extractall(path=extract_dir, members=_safe_members(tar, extract_dir))

    # Find the dataset directory
    dataset_dir = os.path.join(extract_dir, "tokenized")
    dataset = load_from_disk(dataset_dir)

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=(val_ratio + test_ratio), seed=seed)
    train_ds = train_test_split["train"]
    temp_ds = train_test_split["test"]

    # Further split temp into val and test
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_test_split = temp_ds.train_test_split(test_size=val_test_ratio, seed=seed)
    val_ds = val_test_split["train"]
    test_ds = val_test_split["test"]

    # Save splits
    train_path = "/tmp/train"
    val_path = "/tmp/val"
    test_path = "/tmp/test"

    train_ds.save_to_disk(train_path)
    val_ds.save_to_disk(val_path)
    test_ds.save_to_disk(test_path)

    # Create tar archives
    def create_archive(data_path, archive_name):
        archive_path = f"/tmp/{archive_name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(data_path, arcname=os.path.basename(data_path))
        return archive_path

    train_archive = create_archive(train_path, "train")
    val_archive = create_archive(val_path, "val")
    test_archive = create_archive(test_path, "test")

    return (
        FlyteFile(path=train_archive),
        FlyteFile(path=val_archive),
        FlyteFile(path=test_archive),
    )
