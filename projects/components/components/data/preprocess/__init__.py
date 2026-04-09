"""
Data preprocessing component — normalize and split tabular datasets.

Image: data-cpu
"""

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
def preprocess_tabular(
    dataset: FlyteFile,
    target_column: str,
    test_size: float = 0.1,
    val_size: float = 0.1,
) -> Tuple[FlyteFile, FlyteFile, FlyteFile]:
    """Normalize and split a tabular dataset into train/val/test splits.

    Args:
        dataset: Input dataset as a FlyteFile (CSV or Parquet).
        target_column: Name of the label/target column.
        test_size: Fraction reserved for the test split.
        val_size: Fraction reserved for the validation split.

    Returns:
        Tuple of (train_data, val_data, test_data) as FlyteFile objects.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Validate split sizes
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1 (exclusive), got {test_size}")
    if not (0 < val_size < 1):
        raise ValueError(f"val_size must be between 0 and 1 (exclusive), got {val_size}")
    if test_size + val_size >= 1:
        raise ValueError(f"test_size + val_size must be < 1, got {test_size + val_size}")

    dataset.download()
    path = dataset.path
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    features = df.drop(columns=[target_column])
    labels = df[[target_column]]

    # Normalize numeric features (skip if none exist)
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

    # Split
    test_frac = test_size
    val_frac = val_size / (1 - test_frac) if (1 - test_frac) > 0 else val_size
    feat_tv, feat_test, lbl_tv, lbl_test = train_test_split(features, labels, test_size=test_frac)
    feat_train, feat_val, lbl_train, lbl_val = train_test_split(feat_tv, lbl_tv, test_size=val_frac)

    def _save(feat, lbl, out_path):
        pd.concat([feat, lbl], axis=1).to_parquet(out_path, index=False)

    train_path, val_path, test_path = (
        "/tmp/train.parquet",
        "/tmp/val.parquet",
        "/tmp/test.parquet",
    )
    _save(feat_train, lbl_train, train_path)
    _save(feat_val, lbl_val, val_path)
    _save(feat_test, lbl_test, test_path)

    return FlyteFile(train_path), FlyteFile(val_path), FlyteFile(test_path)
