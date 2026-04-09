"""Components API endpoints.

Fetches existing task and workflow components from the active Flyte cluster.
Enriches them with metadata from component.yaml files (version, description,
category, tags).  Provides fallback to component.yaml-only data when the
cluster is unreachable.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException

from backend.api.svc_proxy import svc_get

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Component metadata from component.yaml files ─────────────────────────────

_COMPONENTS_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "projects" / "components" / "components"
)


def _load_component_registry() -> dict[str, dict[str, Any]]:
    """Scan component.yaml files and return a name → metadata dict.

    Keyed by directory name (the canonical component identifier).
    """
    registry: dict[str, dict[str, Any]] = {}
    if not _COMPONENTS_ROOT.is_dir():
        return registry
    for meta_file in sorted(_COMPONENTS_ROOT.rglob("component.yaml")):
        try:
            meta = yaml.safe_load(meta_file.read_text()) or {}
            dir_name = meta_file.parent.name
            registry[dir_name] = {
                "name": meta.get("name", dir_name),
                "version": meta.get("version", "0.0.0"),
                "desc": meta.get("description", ""),
                "category": meta.get("category", meta_file.parent.parent.name),
                "tags": meta.get("tags", []),
                "image": meta.get("image", ""),
                "image_tag": meta.get("image_tag", ""),
            }
        except Exception:
            continue
    return registry


def _resolve_component(full_name: str, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Look up a Flyte entity in the component registry.

    Flyte entity names follow the pattern
    ``components.<category>.<component_dir>.task.<function>``.
    The component directory (3rd segment) is the canonical registry key.
    For entities outside the ``components.`` namespace, fall back to the
    last segment (function name).
    """
    parts = full_name.split(".")
    # Primary: use the component directory from the path
    if len(parts) >= 4 and parts[0] == "components":
        meta = registry.get(parts[2])
        if meta:
            return meta
    # Fall back to function name for non-component entities
    return registry.get(parts[-1], {})


@router.get("")
async def list_components() -> dict[str, Any]:
    """List components from Flyte, enriched with component.yaml metadata."""
    yaml_registry = _load_component_registry()

    try:
        components: list[dict[str, Any]] = []

        # 1. Fetch task components from Flyte
        try:
            tasks_data = await svc_get(
                "flyte_http",
                "api/v1/task_ids/ml-platform/development",
                params={"limit": 100},
            )
            for entity in tasks_data.get("entities", []):
                full_name = entity.get("name", "")
                meta = _resolve_component(full_name, yaml_registry)
                func_name = full_name.split(".")[-1]
                components.append(
                    {
                        "name": meta.get("name") or func_name,
                        "version": meta.get("version", "latest"),
                        "desc": meta.get("desc") or full_name,
                        "type": "task",
                        "category": meta.get("category", ""),
                        "tags": meta.get("tags", []),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not fetch Flyte tasks: {e}")

        # 2. Fetch workflow components from Flyte
        try:
            wf_data = await svc_get(
                "flyte_http",
                "api/v1/workflow_ids/ml-platform/development",
                params={"limit": 100},
            )
            for entity in wf_data.get("entities", []):
                full_name = entity.get("name", "")
                meta = _resolve_component(full_name, yaml_registry)
                func_name = full_name.split(".")[-1]
                components.append(
                    {
                        "name": meta.get("name") or func_name,
                        "version": meta.get("version", "latest"),
                        "desc": meta.get("desc") or full_name,
                        "type": "workflow",
                        "category": meta.get("category", ""),
                        "tags": meta.get("tags", []),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not fetch Flyte workflows: {e}")

        if components:
            components.sort(key=lambda x: x["name"])
            return {"components": components}

        # If reachable but empty, fall through to yaml-only data
    except Exception as e:
        logger.error(f"Flyte unreachable for components: {e}")

    # Fallback: return all components from component.yaml files
    if yaml_registry:
        components = sorted(
            [
                {
                    "name": m["name"],
                    "version": m["version"],
                    "desc": m["desc"],
                    "type": "task",
                    "category": m["category"],
                    "tags": m["tags"],
                }
                for m in yaml_registry.values()
            ],
            key=lambda x: x["name"],
        )
        return {"components": components}

    # Last resort: hardcoded sample data
    return {
        "components": [
            {
                "name": "S3_to_DuckDB_Ingest",
                "version": "v1.2",
                "desc": "Batch loads JSON/CSV from S3 into local DuckDB table",
                "type": "task",
                "category": "data",
                "tags": ["data", "ingestion"],
            },
            {
                "name": "HuggingFace_Tokenizer",
                "version": "v2.0",
                "desc": "Pre-trained tokenizer wrapper for LLM datasets",
                "type": "task",
                "category": "data",
                "tags": ["data", "tokenization"],
            },
            {
                "name": "Feature_Norm_Pipeline",
                "version": "v1.1",
                "desc": "Performs standard scaling and outlier removal",
                "type": "workflow",
                "category": "data",
                "tags": ["data", "preprocessing"],
            },
            {
                "name": "Ray_XGBoost_Train",
                "version": "v3.1",
                "desc": "Distributed XGBoost training on Ray workers",
                "type": "task",
                "category": "training",
                "tags": ["training", "xgboost"],
            },
            {
                "name": "DDP_Torch_Loop",
                "version": "v2.5",
                "desc": "Boilerplate PyTorch DDP training loop",
                "type": "workflow",
                "category": "training",
                "tags": ["training", "pytorch"],
            },
            {
                "name": "Model_Evaluator",
                "version": "v1.0",
                "desc": "Computes F1, AUC, and logs to MLflow",
                "type": "task",
                "category": "evaluation",
                "tags": ["evaluation", "metrics"],
            },
        ]
    }


@router.get("/{name}")
async def get_component(name: str) -> dict[str, Any]:
    """Return detailed metadata for a single component."""
    yaml_registry = _load_component_registry()
    meta = yaml_registry.get(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Component '{name}' not found")

    return {
        "name": meta["name"],
        "version": meta["version"],
        "desc": meta["desc"],
        "category": meta["category"],
        "tags": meta["tags"],
        "image": meta["image"],
        "image_tag": meta["image_tag"],
    }
