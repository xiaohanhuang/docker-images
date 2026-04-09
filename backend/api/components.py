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


async def _get_flyte_version(kind: str, full_name: str) -> str:
    """Fetch the latest version of a Flyte task or workflow.

    ``kind`` is ``"tasks"`` or ``"workflows"``.
    Returns the git-sha version string, or ``"unknown"`` on failure.
    """
    try:
        data = await svc_get(
            "flyte_http",
            f"api/v1/{kind}/ml-platform/development/{full_name}",
            params={
                "limit": 1,
                "sort_by.key": "created_at",
                "sort_by.direction": "DESCENDING",
            },
        )
        items = data.get(kind, [])
        if items:
            return items[0]["id"]["version"]
    except Exception:
        pass
    return "unknown"


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
                version = meta.get("version") or await _get_flyte_version("tasks", full_name)
                components.append(
                    {
                        "name": meta.get("name") or func_name,
                        "version": version,
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
                version = meta.get("version") or await _get_flyte_version("workflows", full_name)
                components.append(
                    {
                        "name": meta.get("name") or func_name,
                        "version": version,
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


def _format_flyte_type(type_obj: dict[str, Any]) -> str:
    """Convert a Flyte LiteralType JSON object to a readable Python type string."""
    if "simple" in type_obj:
        simple_map = {
            "INTEGER": "int",
            "FLOAT": "float",
            "STRING": "str",
            "BOOLEAN": "bool",
            "DATETIME": "datetime",
            "DURATION": "timedelta",
            "BINARY": "bytes",
            "NONE": "None",
        }
        return simple_map.get(type_obj["simple"], type_obj["simple"])
    if "collection_type" in type_obj:
        inner = _format_flyte_type(type_obj["collection_type"])
        return f"List[{inner}]"
    if "map_value_type" in type_obj:
        inner = _format_flyte_type(type_obj["map_value_type"])
        return f"Dict[str, {inner}]"
    if "blob" in type_obj:
        dim = type_obj["blob"].get("dimensionality", "")
        return "FlyteDirectory" if dim == "MULTIPART" else "FlyteFile"
    if "union_type" in type_obj:
        union_variants = type_obj["union_type"].get("variants", [])
        variants = [_format_flyte_type(v["type"]) for v in union_variants]
        nones = [v for v in variants if v == "None"]
        others = [v for v in variants if v != "None"]
        if nones and len(others) == 1:
            return f"Optional[{others[0]}]"
        return " | ".join(variants) if variants else "Any"
    if "structured_dataset_type" in type_obj:
        return "StructuredDataset"
    return "Any"


async def _fetch_flyte_task_detail(flyte_name: str) -> dict[str, Any] | None:
    """Fetch the latest version of a Flyte task with full interface details."""
    try:
        data = await svc_get(
            "flyte_http",
            f"api/v1/tasks/ml-platform/development/{flyte_name}",
            params={
                "limit": 1,
                "sort_by.key": "created_at",
                "sort_by.direction": "DESCENDING",
            },
        )
        tasks = data.get("tasks", [])
        if not tasks:
            return None
        task = tasks[0]
        tmpl = task["closure"]["compiled_task"]["template"]
        task_id = task["id"]

        # Extract interface
        iface = tmpl.get("interface", {})
        inputs = []
        for pname, pval in iface.get("inputs", {}).get("variables", {}).items():
            inputs.append({"name": pname, "type": _format_flyte_type(pval.get("type", {}))})
        outputs = []
        for pname, pval in iface.get("outputs", {}).get("variables", {}).items():
            outputs.append({"name": pname, "type": _format_flyte_type(pval.get("type", {}))})

        # Extract image from container or k8s_pod
        image = ""
        container = tmpl.get("container", {})
        if container.get("image"):
            image = container["image"]
        elif tmpl.get("k8s_pod"):
            for c in tmpl["k8s_pod"].get("pod_spec", {}).get("containers", []):
                if c.get("image"):
                    image = c["image"]
                    break

        return {
            "version": task_id.get("version", ""),
            "task_type": tmpl.get("type", ""),
            "image": image,
            "inputs": inputs,
            "outputs": outputs,
        }
    except Exception as e:
        logger.warning(f"Could not fetch Flyte task detail for {flyte_name}: {e}")
        return None


async def _fetch_flyte_workflow_detail(flyte_name: str) -> dict[str, Any] | None:
    """Fetch the latest version of a Flyte workflow with interface details."""
    try:
        data = await svc_get(
            "flyte_http",
            f"api/v1/workflows/ml-platform/development/{flyte_name}",
            params={
                "limit": 1,
                "sort_by.key": "created_at",
                "sort_by.direction": "DESCENDING",
            },
        )
        workflows = data.get("workflows", [])
        if not workflows:
            return None
        wf = workflows[0]
        wf_id = wf["id"]
        iface = (
            wf.get("closure", {})
            .get("compiled_workflow", {})
            .get("primary", {})
            .get("template", {})
            .get("interface", {})
        )

        inputs = []
        for pname, pval in iface.get("inputs", {}).get("variables", {}).items():
            inputs.append({"name": pname, "type": _format_flyte_type(pval.get("type", {}))})
        outputs = []
        for pname, pval in iface.get("outputs", {}).get("variables", {}).items():
            outputs.append({"name": pname, "type": _format_flyte_type(pval.get("type", {}))})

        return {
            "version": wf_id.get("version", ""),
            "task_type": "workflow",
            "image": "",
            "inputs": inputs,
            "outputs": outputs,
        }
    except Exception as e:
        logger.warning(f"Could not fetch Flyte workflow detail for {flyte_name}: {e}")
        return None


@router.get("/{name}")
async def get_component(name: str) -> dict[str, Any]:
    """Return detailed metadata for a single component.

    Merges component.yaml metadata with live Flyte task/workflow details
    (inputs, outputs, container image, version).
    """
    yaml_registry = _load_component_registry()
    meta = yaml_registry.get(name)

    flyte_detail: dict[str, Any] | None = None

    # Build candidate Flyte entity names to query
    task_candidates: list[str] = []
    wf_candidates: list[str] = []

    if meta:
        category = meta.get("category", "")
        task_candidates.append(f"components.{category}.{name}.task.{name}")

    # Scan Flyte task_ids and workflow_ids for matches
    try:
        tasks_data = await svc_get(
            "flyte_http",
            "api/v1/task_ids/ml-platform/development",
            params={"limit": 100},
        )
        for entity in tasks_data.get("entities", []):
            full_name = entity.get("name", "")
            parts = full_name.split(".")
            func_name = parts[-1] if parts else ""
            # Match by component dir or by function name
            if len(parts) >= 4 and parts[0] == "components" and parts[2] == name:
                if full_name not in task_candidates:
                    task_candidates.insert(0, full_name)
            elif func_name == name and full_name not in task_candidates:
                task_candidates.append(full_name)
    except Exception:
        pass

    try:
        wf_data = await svc_get(
            "flyte_http",
            "api/v1/workflow_ids/ml-platform/development",
            params={"limit": 100},
        )
        for entity in wf_data.get("entities", []):
            full_name = entity.get("name", "")
            parts = full_name.split(".")
            func_name = parts[-1] if parts else ""
            if func_name == name and full_name not in wf_candidates:
                wf_candidates.append(full_name)
    except Exception:
        pass

    # Try tasks first, then workflows
    for candidate in task_candidates:
        flyte_detail = await _fetch_flyte_task_detail(candidate)
        if flyte_detail:
            break

    if not flyte_detail:
        for candidate in wf_candidates:
            flyte_detail = await _fetch_flyte_workflow_detail(candidate)
            if flyte_detail:
                break

    if not meta and not flyte_detail:
        raise HTTPException(status_code=404, detail=f"Component '{name}' not found")

    result: dict[str, Any] = {
        "name": meta["name"] if meta else name,
        "version": (meta or {}).get("version", ""),
        "desc": (meta or {}).get("desc", ""),
        "category": (meta or {}).get("category", ""),
        "tags": (meta or {}).get("tags", []),
        "image": "",
        "image_tag": (meta or {}).get("image_tag", ""),
        "inputs": [],
        "outputs": [],
        "task_type": "",
    }

    if flyte_detail:
        result["version"] = flyte_detail["version"] or result["version"]
        result["image"] = flyte_detail["image"]
        result["inputs"] = flyte_detail["inputs"]
        result["outputs"] = flyte_detail["outputs"]
        result["task_type"] = flyte_detail["task_type"]
    elif meta:
        result["image"] = f"{meta.get('image', '')}:{meta.get('image_tag', '')}"

    return result
