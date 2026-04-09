"""Generic component framework — ``Component`` and ``Pipeline`` base classes.

``Component`` (subclass of ``ReferenceTask``) and ``Pipeline`` (subclass of
``ReferenceLaunchPlan``) point to already-registered Flyte entities.  Inside
``@workflow``, they return Promises for automatic DAG wiring.  Inside ``@eager``,
they return real values.  Standalone, pass them to ``FlyteRemote.execute()``.

Version resolution (highest to lowest priority):
  1. Inline subscript: ``component["v1.5"](...)``
  2. Workflow-level: ``@mp.workflow(versions={"tokenizer": "v1.5"})``
  3. ``ML_PLAT_COMPONENT_VERSION`` env var (exact version string)
  4. Latest from FlyteAdmin (default — raises if unreachable)

Concrete component instances (``hf_dataset_loader``, ``lora_finetune``, etc.)
live in ``ml_platform_sdk.catalog``.
"""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Dict, Optional

from flytekit.core.launch_plan import ReferenceLaunchPlan
from flytekit.core.task import ReferenceTask
from flytekit.models.core.identifier import ResourceType

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


class Component(_LazyReferenceMixin, ReferenceTask):
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


class Pipeline(_LazyReferenceMixin, ReferenceLaunchPlan):
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
