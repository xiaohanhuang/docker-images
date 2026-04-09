"""ML Platform SDK — shared utilities for tasks, workflows, and services."""

from __future__ import annotations

import functools
import os
from typing import Callable, Dict

# Suppress noisy flytekit type warnings in notebooks
os.environ["FLYTE_SDK_LOGGING_LEVEL"] = "40"

import flytekit
from flytekit import Resources, dynamic, eager, map_task
from flytekit.types.directory import FlyteDirectory  # noqa: F401
from flytekit.types.file import FlyteFile  # noqa: F401

from .catalog import (
    data_splitter,
    deploy_vllm,
    download_dataset,
    full_finetune,
    hf_dataset_loader,
    llm_judge,
    llm_sft_full_pipeline,
    llm_sft_lora_pipeline,
    lora_finetune,
    model_evaluator,
    preprocess_tabular,
    text2sql_pipeline,
    tokenizer,
    vllm_deployer,
)
from .checkpoint import CheckpointManager, HuggingFaceCheckpointManager
from .components import (
    Component,
    Pipeline,
    _workflow_versions,
)
from .tasks.accelerate import accelerate_task, platform
from .tasks.tensorboard import get_summary_writer


def workflow(
    fn: Callable | None = None,
    *,
    versions: Dict[str, str] | None = None,
):
    """Wrapper around ``flytekit.workflow`` that supports per-component versions.

    Usage::

        @mp.workflow
        def pipeline(...):
            ...

        @mp.workflow(versions={"tokenizer": "v1.3.0", "lora_finetune": "v2.0.0"})
        def pinned_pipeline(...):
            ...

    When ``versions`` is provided, the given component versions take priority
    over the default (latest) resolution during workflow compilation.
    """
    if fn is None:
        # Called with arguments: @mp.workflow(versions={...})
        return lambda f: workflow(f, versions=versions)

    if versions:
        original_fn = fn

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            token = _workflow_versions.set(versions)
            try:
                return original_fn(*args, **kwargs)
            finally:
                _workflow_versions.reset(token)

        return flytekit.workflow(wrapped)

    return flytekit.workflow(fn)


def submit(
    entity,
    inputs: Dict | None = None,
    *,
    version: str = "latest",
    project: str | None = None,
    domain: str | None = None,
    wait: bool = False,
    **kwargs,
):
    """Register and execute a workflow (or task) on the cluster in one call.

    Handles ``FlyteRemote`` setup, registration, and execution so notebook
    users don't need boilerplate.  Returns a ``FlyteWorkflowExecution``.

    Usage::

        execution = mp.submit(finetune_pipeline, {"dataset": "tatsu-lab/alpaca"})
        print(execution.id.name)

    Args:
        entity: A ``@mp.workflow``, ``@mp.eager``, or Flyte task to execute.
        inputs: Input dict (uses defaults if omitted).
        version: Version tag (default ``"latest"`` — auto-generates hash).
        project: Flyte project (default from ``~/.ml-plat/config.yaml``).
        domain: Flyte domain (default from ``~/.ml-plat/config.yaml``).
        wait: Block until execution completes.
        **kwargs: Extra args forwarded to ``FlyteRemote.execute()``.
    """
    import os

    # Auto-load .env (same pattern as remote.py) so notebooks get
    # FLYTE_ENDPOINT, FLYTE_PROJECT, etc. without manual setup.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Import from cli.utils if available; fall back to direct flytekit config
    # for SDK-only environments (e.g., notebooks without the CLI package).
    try:
        from cli.utils import flyte_remote, platform_config

        cfg = platform_config()
        remote = flyte_remote()
    except ImportError:
        from flytekit.configuration import Config
        from flytekit.remote import FlyteRemote

        endpoint = os.getenv("FLYTE_ENDPOINT", "localhost:8089")
        fly_cfg = Config.for_endpoint(endpoint, insecure=True)
        remote = FlyteRemote(
            config=fly_cfg,
            default_project=os.getenv("FLYTE_PROJECT", "ml-platform"),
            default_domain=os.getenv("FLYTE_DOMAIN", "development"),
        )
        cfg = {}

    proj = project or os.getenv("FLYTE_PROJECT", cfg.get("flyte_project", "ml-platform"))
    dom = domain or os.getenv("FLYTE_DOMAIN", cfg.get("flyte_domain", "development"))

    # (Version generation moved further down with error handling)

    # Get default image from config or use a sensible fallback
    default_image = cfg.get(
        "default_image", "805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/base-cpu:latest"
    )

    from flytekit.configuration import Image, ImageConfig

    # Parse image string (e.g., registry/repo:tag)
    if ":" in default_image:
        fqn, tag = default_image.rsplit(":", 1)
    else:
        fqn, tag = default_image, "latest"

    img = Image(name="default", fqn=fqn, tag=tag)
    ImageConfig(default_image=img)

    # Auto-generate a version hash
    import hashlib
    import inspect

    try:
        src = inspect.getsource(entity)
        digest = hashlib.md5(src.encode()).hexdigest()[:8]  # noqa: S324
        version = version if version != "latest" else "nb-" + digest
    except Exception:
        version = version if version != "latest" else "nb-interactive"

    # Explicitly register the entity before executing
    print(f"[mp.submit] Registering {entity.name} (version={version})...")
    try:
        from flytekit.core.python_function_task import EagerAsyncPythonFunctionTask
        from flytekit.core.workflow import WorkflowBase

        if isinstance(entity, EagerAsyncPythonFunctionTask):
            raise TypeError(
                f"{entity.name} is an @eager function. Use `await {entity.name}(...)` "
                f"in a notebook cell instead of mp.submit(). Eager functions run locally "
                f"and dispatch sub-tasks to the cluster via FlyteRemote."
            )
        elif isinstance(entity, WorkflowBase):
            remote.register_workflow(entity, version=version)
        else:
            remote.register_task(entity, version=version)
    except TypeError:
        raise
    except Exception as e:
        print(f"[mp.submit] Note: Registration warning: {e}")

    print(f"[mp.submit] Executing {entity.name}...")
    execution = remote.execute(
        entity,
        inputs=inputs or {},
        project=proj,
        domain=dom,
        version=version,
        wait=wait,
        **kwargs,
    )

    console_url = cfg.get("flyte_console_url", "")
    if console_url:
        print(
            f"Console: {console_url}/console/projects/{proj}"
            f"/domains/{dom}/executions/{execution.id.name}"
        )
    else:
        print(f"Execution: {execution.id.name}")

    return execution


__all__ = [
    # Flytekit re-exports
    "Resources",
    "dynamic",
    "eager",
    "map_task",
    "workflow",
    "submit",
    # Component/Pipeline classes
    "Component",
    "Pipeline",
    # Checkpoint managers
    "CheckpointManager",
    "HuggingFaceCheckpointManager",
    # Components
    "data_splitter",
    "deploy_vllm",
    "download_dataset",
    "full_finetune",
    "hf_dataset_loader",
    "llm_judge",
    "llm_sft_full_pipeline",
    "llm_sft_lora_pipeline",
    "lora_finetune",
    "model_evaluator",
    "preprocess_tabular",
    "text2sql_pipeline",
    "tokenizer",
    "vllm_deployer",
    "accelerate_task",
    "platform",
    "get_summary_writer",
]
