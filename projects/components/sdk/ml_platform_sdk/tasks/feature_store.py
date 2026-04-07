"""Feast Feature Store tasks for Flyte workflows.

Provides Flyte tasks for materializing features, fetching historical
features, and serving online features — all wired to the platform's
Feast deployment (S3 registry + Redis online store).

Usage in a workflow::

    import ml_platform_sdk as mp
    from ml_platform_sdk.tasks.feature_store import (
        materialize_features,
        get_historical_features,
        get_online_features,
    )

    @mp.workflow
    def training_pipeline(feature_view: str = "user_stats"):
        materialize_features(feature_view_name=feature_view, days=7)
        training_df = get_historical_features(
            feature_view_name=feature_view,
            entity_source_path="s3://bucket/entities.parquet",
        )
        ...
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

_DEFAULT_REPO = os.getenv(
    "FEAST_REPO_PATH",
    "/opt/feast",  # default mount path in the container
)


@task(
    retries=2,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
    cache=False,
)
def materialize_features(
    feature_view_name: str = "",
    days: int = 7,
    incremental: bool = True,
    repo_path: str = _DEFAULT_REPO,
) -> str:
    """Materialize offline features to the Redis online store.

    Args:
        feature_view_name: Specific feature view to materialize (all if empty).
        days: Number of days to look back for materialization window.
        incremental: Whether to use incremental materialization.
        repo_path: Path to the Feast feature repo.

    Returns:
        Summary string with materialization result.
    """
    from feast import FeatureStore

    store = FeatureStore(repo_path=repo_path)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    feature_views = None
    if feature_view_name:
        feature_views = [store.get_feature_view(feature_view_name)]

    label = feature_view_name or "all views"

    if incremental:
        store.materialize_incremental(
            end_date=end_date,
            feature_views=feature_views,
        )
        return f"Materialized {label} incrementally up to {end_date:%Y-%m-%d}"

    store.materialize(
        start_date=start_date,
        end_date=end_date,
        feature_views=feature_views,
    )
    return f"Materialized {label} ({start_date:%Y-%m-%d} → {end_date:%Y-%m-%d})"


@task(
    retries=2,
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="4", mem="16Gi"),
    cache=True,
    cache_version="1.0",
)
def get_historical_features(
    feature_view_name: str,
    entity_source_path: str,
    repo_path: str = _DEFAULT_REPO,
) -> FlyteFile:
    """Retrieve historical features for training by joining against an entity DataFrame.

    Reads the entity DataFrame from ``entity_source_path`` (Parquet on S3),
    joins it against the offline store, and writes the enriched DataFrame
    to a Parquet file.

    Args:
        feature_view_name: Feature view to pull features from.
        entity_source_path: S3 path to the entity Parquet file (must have
            the join key column and ``event_timestamp``).
        repo_path: Path to the Feast feature repo.

    Returns:
        FlyteFile pointing to the enriched Parquet output.
    """
    import pandas as pd
    from feast import FeatureStore

    store = FeatureStore(repo_path=repo_path)
    fv = store.get_feature_view(feature_view_name)
    feature_refs = [f"{feature_view_name}:{f.name}" for f in fv.schema]

    entity_df = pd.read_parquet(entity_source_path)

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()

    output_path = "/tmp/training_features.parquet"
    training_df.to_parquet(output_path, index=False)

    return FlyteFile(path=output_path)


@task(
    retries=2,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
)
def get_online_features(
    feature_view_name: str,
    entity_keys: list[int],
    entity_key_column: str = "user_id",
    repo_path: str = _DEFAULT_REPO,
) -> dict:
    """Fetch online features for a list of entity keys.

    Suitable for inference-time feature retrieval within a Flyte task.

    Args:
        feature_view_name: Feature view to query.
        entity_keys: List of entity key values to look up.
        entity_key_column: Name of the entity join key column.
        repo_path: Path to the Feast feature repo.

    Returns:
        Dict mapping feature names to lists of values.
    """
    from feast import FeatureStore

    store = FeatureStore(repo_path=repo_path)
    fv = store.get_feature_view(feature_view_name)
    feature_refs = [f"{feature_view_name}:{f.name}" for f in fv.schema]

    entity_rows = [{entity_key_column: key} for key in entity_keys]

    result = store.get_online_features(
        features=feature_refs,
        entity_rows=entity_rows,
    ).to_dict()

    return result
