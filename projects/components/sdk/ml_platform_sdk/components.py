"""Reusable component stubs — pre-registered Flyte tasks callable from @workflow.

Each stub is a ``Component`` (subclass of ``ReferenceTask``) or ``Pipeline``
(subclass of ``ReferenceLaunchPlan``) that points to an already-registered
Flyte entity.  Inside ``@workflow``, they return Promises for automatic DAG wiring.
Inside ``@eager``, they return real values when ``await``-ed.  Standalone, pass them
to ``FlyteRemote.execute()``.

Version resolution (highest to lowest priority):
  1. Inline subscript: ``mp.tokenizer["v1.5"](...)``
  2. Workflow-level: ``@mp.workflow(versions={"tokenizer": "v1.5"})``
  3. ``ML_PLAT_COMPONENT_VERSION`` env var (exact version string)
  4. Latest from FlyteAdmin (default — raises if unreachable)

At registration time, "latest" resolves to actual versions — the compiled
workflow freezes those versions permanently.

Usage::

    import ml_platform_sdk as mp

    @mp.workflow
    def my_pipeline(dataset: str = "alpaca"):
        raw = mp.hf_dataset_loader(dataset_name=dataset)              # latest
        tokenized = mp.tokenizer["v1.3.0"](raw_data=raw, model_id="llama")  # pinned
        splits = mp.data_splitter(tokenized_data=tokenized)
        result = mp.lora_finetune(
            base_model="llama", train_data_path=splits.train_data, epochs=3,
        )
        return mp.model_evaluator(
            checkpoint_path=result.checkpoint_dir, test_data=splits.val_data,
        )

    # Or pin versions at the workflow level:
    @mp.workflow(versions={"tokenizer": "v1.3.0", "lora_finetune": "v2.0.0"})
    def pinned_pipeline(dataset: str = "alpaca"):
        ...
"""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Any, Dict, List, Optional

from flytekit.core.launch_plan import ReferenceLaunchPlan
from flytekit.core.task import ReferenceTask
from flytekit.models.core.identifier import ResourceType
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

logger = logging.getLogger(__name__)

# ── Environment-driven config ────────────────────────────────────────────────

_PROJECT = os.getenv("FLYTE_PROJECT", "ml-platform")
_DOMAIN = os.getenv("FLYTE_DOMAIN", "development")

# ── Workflow-level version context ───────────────────────────────────────────

_workflow_versions: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "_workflow_versions", default={}
)

_resolved_versions: Dict[str, str] = {}


def _resolve_version(
    name: str, resource_type: int = ResourceType.TASK, explicit: str | None = None
) -> str:
    """Multi-layer version resolution.

    Priority: explicit > workflow-level > env override > latest from FlyteAdmin.
    Raises if FlyteAdmin is unreachable and no explicit version is set.
    """
    # 1. Explicit (from Component["v1.5"])
    if explicit:
        return explicit

    # 2. Workflow-level versions
    wf_versions = _workflow_versions.get({})
    if wf_versions:
        short_name = name.rsplit(".", 1)[-1]
        if short_name in wf_versions:
            return wf_versions[short_name]
        if name in wf_versions:
            return wf_versions[name]

    # 3. Environment variable
    override = os.getenv("ML_PLAT_COMPONENT_VERSION")
    if override and override != "latest":
        return override

    # 4. Latest from FlyteAdmin (default)
    cache_key = f"{resource_type}:{name}"
    if cache_key not in _resolved_versions:
        _resolved_versions[cache_key] = _fetch_latest_version(name, resource_type)
    return _resolved_versions[cache_key]


def _fetch_latest_version(name: str, resource_type: int = ResourceType.TASK) -> str:
    """Query FlyteAdmin for the latest registered version of a task or launch plan."""
    from flytekit.models.admin.common import Sort
    from flytekit.models.common import NamedEntityIdentifier

    from cli.utils import flyte_remote

    remote = flyte_remote()
    latest_first = Sort(key="created_at", direction=Sort.Direction.DESCENDING)
    identifier = NamedEntityIdentifier(project=_PROJECT, domain=_DOMAIN, name=name)

    if resource_type == ResourceType.LAUNCH_PLAN:
        entities, _ = remote.client.list_launch_plans_paginated(
            identifier, limit=1, sort_by=latest_first
        )
    else:
        entities, _ = remote.client.list_tasks_paginated(identifier, limit=1, sort_by=latest_first)

    if entities:
        return entities[0].id.version
    raise ValueError(f"No registered versions for {name} (type {resource_type})")


# ── Lazy Reference Mixin ─────────────────────────────────────────────────────


class _LazyReferenceMixin:
    """Shared logic for lazy version resolution and remote execution."""

    _stub_name: str
    _stub_inputs: dict
    _stub_outputs: dict
    _defaults: dict
    _explicit_version: Optional[str]
    _resource_type: int

    def _ensure_resolved(self):
        """Resolve version from FlyteAdmin on first use (lazy)."""
        if self.reference.id.version != "unresolved":
            return
        version = _resolve_version(
            self._stub_name,
            resource_type=self._resource_type,
            explicit=self._explicit_version,
        )
        from flytekit.models.core.identifier import Identifier

        old = self.reference.id
        self.reference._id = Identifier(
            old.resource_type,
            old.project,
            old.domain,
            old.name,
            version,
        )

    def __call__(self, **kwargs):
        self._ensure_resolved()
        for key, default in self._defaults.items():
            if key not in kwargs:
                kwargs[key] = default

        from flytekit.core.promise import flyte_entity_call_handler

        return flyte_entity_call_handler(self, **kwargs)

    def execute(self, **kwargs):
        """Execute the entity via FlyteRemote (eager-local / notebook use)."""
        self._ensure_resolved()
        import typing

        from cli.utils import flyte_remote

        # Fill in None for missing Optional inputs
        for key, typ in self._stub_inputs.items():
            if key not in kwargs:
                args = typing.get_args(typ)
                if type(None) in args:
                    kwargs[key] = None
                elif key in self._defaults:
                    kwargs[key] = self._defaults[key]

        remote = flyte_remote()
        if self._resource_type == ResourceType.LAUNCH_PLAN:
            entity = remote.fetch_launch_plan(
                project=_PROJECT,
                domain=_DOMAIN,
                name=self._stub_name,
                version=self.reference.id.version,
            )
        else:
            entity = remote.fetch_task(
                project=_PROJECT,
                domain=_DOMAIN,
                name=self._stub_name,
                version=self.reference.id.version,
            )

        execution = remote.execute(entity, inputs=kwargs, wait=True)
        outputs = execution.outputs

        output_keys = list(self._stub_outputs.keys())
        if len(output_keys) == 1:
            return outputs[output_keys[0]]

        from collections import namedtuple

        OutputType = namedtuple(self._stub_name.split(".")[-1] + "_output", output_keys)
        return OutputType(*(outputs[k] for k in output_keys))


# ── Component & Pipeline classes ─────────────────────────────────────────────


class Component(ReferenceTask, _LazyReferenceMixin):
    """A versioned component stub with subscript syntax for version pinning."""

    def __init__(
        self,
        name: str,
        inputs: dict,
        outputs: dict,
        defaults: Optional[Dict] = None,
        explicit_version: Optional[str] = None,
    ):
        self._stub_name = name
        self._stub_inputs = inputs
        self._stub_outputs = outputs
        self._defaults = defaults or {}
        self._explicit_version = explicit_version
        self._resource_type = ResourceType.TASK
        version = explicit_version or "unresolved"
        super().__init__(
            project=_PROJECT,
            domain=_DOMAIN,
            name=name,
            version=version,
            inputs=inputs,
            outputs=outputs,
        )

    def __getitem__(self, version: str) -> "Component":
        """Return a Component pinned to a specific version."""
        return Component(
            name=self._stub_name,
            inputs=self._stub_inputs,
            outputs=self._stub_outputs,
            defaults=self._defaults,
            explicit_version=version,
        )


class Pipeline(ReferenceLaunchPlan, _LazyReferenceMixin):
    """A versioned pipeline stub (ReferenceLaunchPlan)."""

    def __init__(
        self,
        name: str,
        inputs: dict,
        outputs: dict,
        defaults: Optional[Dict] = None,
        explicit_version: Optional[str] = None,
    ):
        self._stub_name = name
        self._stub_inputs = inputs
        self._stub_outputs = outputs
        self._defaults = defaults or {}
        self._explicit_version = explicit_version
        self._resource_type = ResourceType.LAUNCH_PLAN
        version = explicit_version or "unresolved"
        super().__init__(
            project=_PROJECT,
            domain=_DOMAIN,
            name=name,
            version=version,
            inputs=inputs,
            outputs=outputs,
        )

    def __getitem__(self, version: str) -> "Pipeline":
        """Return a Pipeline pinned to a specific version."""
        return Pipeline(
            name=self._stub_name,
            inputs=self._stub_inputs,
            outputs=self._stub_outputs,
            defaults=self._defaults,
            explicit_version=version,
        )


def _comp(
    name: str,
    inputs: dict,
    outputs: dict,
    defaults: Optional[Dict] = None,
) -> Component:
    """Create a ``Component`` pointing to a pre-registered Flyte task."""
    return Component(name=name, inputs=inputs, outputs=outputs, defaults=defaults)


def _pipe(
    name: str,
    inputs: dict,
    outputs: dict,
    defaults: Optional[Dict] = None,
) -> Pipeline:
    """Create a ``Pipeline`` pointing to a pre-registered Flyte workflow."""
    return Pipeline(name=name, inputs=inputs, outputs=outputs, defaults=defaults)


# ── Data Components ──────────────────────────────────────────────────────────

hf_dataset_loader = _comp(
    "components.data.hf_dataset_loader.task.hf_dataset_loader",
    inputs={
        "dataset_name": str,
        "split": str,
        "subset": Optional[str],
        "num_samples": Optional[int],
    },
    outputs={"dataset": FlyteFile},
    defaults={"split": "train"},
)

download_dataset = _comp(
    "components.data.ingest.task.download_dataset",
    inputs={"s3_uri": str},
    outputs={"dataset": FlyteFile},
)

tokenizer = _comp(
    "components.data.tokenizer.task.tokenizer",
    inputs={
        "raw_data": FlyteFile,
        "model_id": str,
        "prompt_template": str,
        "max_length": int,
    },
    outputs={"tokenized_data": FlyteFile},
    defaults={"prompt_template": "alpaca", "max_length": 2048},
)

data_splitter = _comp(
    "components.data.data_splitter.task.data_splitter",
    inputs={
        "tokenized_data": FlyteFile,
        "train_ratio": float,
        "val_ratio": float,
        "test_ratio": float,
        "seed": int,
    },
    outputs={"train_data": FlyteFile, "val_data": FlyteFile, "test_data": FlyteFile},
    defaults={"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1, "seed": 42},
)

preprocess_tabular = _comp(
    "components.data.preprocess.task.preprocess_tabular",
    inputs={
        "dataset": FlyteFile,
        "target_column": str,
        "test_size": float,
        "val_size": float,
        "seed": int,
    },
    outputs={"train_data": FlyteFile, "val_data": FlyteFile, "test_data": FlyteFile},
    defaults={"test_size": 0.1, "val_size": 0.1, "seed": 42},
)

# ── Training Components ──────────────────────────────────────────────────────

lora_finetune = _comp(
    "components.training.lora_finetune.task.lora_finetune",
    inputs={
        "base_model": str,
        "train_data_path": FlyteFile,
        "val_data_path": Optional[str],
        "method": str,
        "lora_r": int,
        "lora_alpha": int,
        "lora_target_modules": Optional[List[str]],
        "epochs": int,
        "batch_size": int,
        "learning_rate": float,
        "gradient_accumulation_steps": int,
        "mlflow_experiment": Optional[str],
        "trust_remote_code": bool,
    },
    outputs={
        "checkpoint_dir": FlyteDirectory,
        "mlflow_run_id": str,
        "final_metrics": Dict[str, float],
    },
    defaults={
        "method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "gradient_accumulation_steps": 4,
        "trust_remote_code": False,
    },
)

full_finetune = _comp(
    "components.training.full_finetune.task.full_finetune",
    inputs={
        "base_model": str,
        "train_data": FlyteFile,
        "val_data": FlyteFile,
        "num_epochs": int,
        "learning_rate": float,
        "batch_size": int,
        "gradient_accumulation_steps": int,
        "use_efs_checkpoints": bool,
    },
    outputs={"checkpoint_dir": FlyteDirectory, "metrics": Dict[str, float]},
    defaults={
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "use_efs_checkpoints": True,
    },
)

# ── Evaluation Components ────────────────────────────────────────────────────

model_evaluator = _comp(
    "components.evaluation.model_evaluator.task.model_evaluator",
    inputs={
        "checkpoint_path": FlyteDirectory,
        "test_data": FlyteFile,
        "base_model": Optional[str],
        "metrics": Optional[List[str]],
    },
    outputs={"metrics": Dict[str, float]},
)

llm_judge = _comp(
    "components.evaluation.llm_judge.task.llm_judge",
    inputs={
        "predictions_path": FlyteDirectory,
        "ground_truth_path": Optional[FlyteFile],
        "judge_model": str,
        "scorers": Optional[List[str]],
        "custom_rubric": Optional[str],
        "sample_size": int,
        "thresholds": Optional[Dict[str, float]],
        "mlflow_experiment": Optional[str],
    },
    outputs={"metrics": dict},
    defaults={"judge_model": "gpt-4o", "sample_size": 100},
)

# ── Serving Components ───────────────────────────────────────────────────────

vllm_deployer = _comp(
    "components.serving.vllm_deployer.task.vllm_deployer",
    inputs={
        "model_path": str,
        "service_name": str,
        "gpu_count": int,
        "quantization": Optional[str],
        "max_model_len": Optional[int],
        "min_replicas": int,
        "max_replicas": int,
    },
    outputs={
        "endpoint_url": str,
        "service_name": str,
        "deployment_config": Dict[str, Any],
    },
    defaults={"gpu_count": 1, "min_replicas": 1, "max_replicas": 5},
)

deploy_vllm = _comp(
    "components.serving.vllm_deploy.task.deploy_vllm",
    inputs={
        "model_id": str,
        "port": int,
        "max_model_len": int,
        "gpu_memory_utilization": float,
    },
    outputs={"endpoint_url": str},
    defaults={"port": 8000, "max_model_len": 4096, "gpu_memory_utilization": 0.9},
)

# ── Pipelines ────────────────────────────────────────────────────────────────

llm_sft_lora_pipeline = _pipe(
    "pipeline.llm_sft_lora_pipeline",
    inputs={
        "base_model": str,
        "dataset": str,
        "epochs": int,
        "learning_rate": float,
        "batch_size": int,
        "lora_r": int,
        "lora_alpha": int,
        "quantization": str,
        "model_name": str,
        "teams_webhook": str,
    },
    outputs={"o0": str},
    defaults={
        "quantization": "none",
        "model_name": "llm-sft",
        "teams_webhook": "",
    },
)

llm_sft_full_pipeline = _pipe(
    "pipeline.llm_sft_full_pipeline",
    inputs={
        "base_model": str,
        "dataset": str,
        "epochs": int,
        "learning_rate": float,
        "batch_size": int,
        "model_name": str,
        "teams_webhook": str,
    },
    outputs={"o0": str},
    defaults={
        "model_name": "llm-sft",
        "teams_webhook": "",
    },
)

text2sql_pipeline = _pipe(
    "pipeline.text2sql_pipeline",
    inputs={
        "num_epochs": int,
        "batch_size": int,
        "learning_rate": float,
    },
    outputs={"model_version_uri": str},
)
