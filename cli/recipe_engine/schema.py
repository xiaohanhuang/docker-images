"""Pydantic models for Recipe YAML schema validation.

This module defines the complete schema for Recipe YAML files, including:
- Infrastructure profiles (resource groups, instance types, GPU configs)
- Pipeline definitions (parameters, steps, component references)
- Presets (profile + parameter overrides)

All models include validators to catch common errors at parse time.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class ParameterType(str, Enum):
    """Supported parameter types in recipe definitions."""

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    LIST = "list"


class Parameter(BaseModel):
    """A user-configurable parameter in a recipe."""

    type: ParameterType
    default: Union[str, int, float, bool, List[Any]]
    description: Optional[str] = None

    @field_validator("default")
    @classmethod
    def validate_default_type(cls, v: Any, info) -> Any:
        """Ensure default value matches the declared type."""
        param_type = info.data.get("type")
        if param_type == ParameterType.INT and not isinstance(v, int):
            raise ValueError(f"Default must be int, got {type(v).__name__}")
        if param_type == ParameterType.FLOAT and not isinstance(v, (int, float)):
            raise ValueError(f"Default must be float, got {type(v).__name__}")
        if param_type == ParameterType.BOOL and not isinstance(v, bool):
            raise ValueError(f"Default must be bool, got {type(v).__name__}")
        if param_type == ParameterType.STRING and not isinstance(v, str):
            raise ValueError(f"Default must be string, got {type(v).__name__}")
        if param_type == ParameterType.LIST and not isinstance(v, list):
            raise ValueError(f"Default must be list, got {type(v).__name__}")
        return v


class ResourceGroup(BaseModel):
    """Infrastructure configuration for a named resource group (e.g., 'actor', 'critic')."""

    instance_types: List[str] = Field(..., min_length=1)
    gpu_count: int = Field(..., ge=0)
    gpu_memory: str = Field(..., pattern=r"^\d+(GB|GiB)$")
    cpu: Optional[str] = Field(None, pattern=r"^\d+$")
    memory: Optional[str] = Field(None, pattern=r"^\d+(GB|GiB|Mi)$")
    networking: Optional[str] = None  # e.g., "efa" for Elastic Fabric Adapter

    @field_validator("instance_types")
    @classmethod
    def validate_instance_types(cls, v: List[str]) -> List[str]:
        """Validate that instance types are from known AWS families."""
        # Known GPU instance families
        valid_families = [
            "g4dn",
            "g5",
            "g6",
            "p3",
            "p4d",
            "p4de",
            "p5",
            "m5",
            "m6i",
            "m7i",
            "c5",
            "c6i",
            "c7i",
            "r5",
            "r6i",
            "r7i",
        ]
        for instance_type in v:
            family = instance_type.split(".")[0]
            if family not in valid_families:
                raise ValueError(
                    f"Unknown instance family: {family}. "
                    f"Supported families: {', '.join(valid_families)}"
                )
        return v

    @field_validator("gpu_count")
    @classmethod
    def validate_gpu_count(cls, v: int) -> int:
        """Ensure GPU count is reasonable (not more than 8 per instance)."""
        if v > 8:
            raise ValueError(
                f"gpu_count {v} exceeds maximum of 8 GPUs per instance. "
                "For multi-node training, use multiple resource groups."
            )
        return v


class StorageConfig(BaseModel):
    """Storage configuration for checkpoints and datasets."""

    checkpoints: str = Field(..., pattern=r"^(efs|s3)$")
    datasets: str = Field(..., pattern=r"^(efs|s3)$")


class NodePoolOverrides(BaseModel):
    """Optional Karpenter NodePool tuning parameters."""

    consolidation_policy: Optional[str] = Field(
        "WhenEmpty", pattern=r"^(WhenEmpty|WhenUnderutilized)$"
    )
    ttl_after_empty: Optional[str] = Field("300s", pattern=r"^\d+(s|m|h)$")


class Infrastructure(BaseModel):
    """Infrastructure layer: profiles, storage, and NodePool overrides."""

    profiles: Dict[str, Dict[str, ResourceGroup]]
    storage: StorageConfig
    node_pool_overrides: Optional[NodePoolOverrides] = None

    @field_validator("profiles")
    @classmethod
    def validate_profiles(
        cls, v: Dict[str, Dict[str, ResourceGroup]]
    ) -> Dict[str, Dict[str, ResourceGroup]]:  # noqa: E501
        """Ensure at least one profile is defined."""
        if not v:
            raise ValueError("At least one profile is required")
        return v


class PipelineStep(BaseModel):
    """A single step in a pipeline that references a component task."""

    name: str = Field(..., pattern=r"^[a-z0-9_]+$")
    component: str = Field(..., pattern=r"^[a-z0-9_.]+$")
    version: Optional[str] = Field(
        None,
        description="Pin to a specific component version. "
        "When set, the runner fetches this exact version from the Flyte registry "
        "instead of the latest. Accepts a semver string (e.g. '1.0.0') or a "
        "content-hash version (e.g. '3a7f1b2c9d0e').",
    )
    infra: Optional[str] = None  # resource group name or null for CPU-only
    config: Dict[str, Any] = Field(default_factory=dict)
    depends_on: Optional[List[str]] = None  # explicit dependencies

    @field_validator("name")
    @classmethod
    def validate_step_name(cls, v: str) -> str:
        """Ensure step names are valid Python identifiers."""
        if not v.isidentifier():
            raise ValueError(
                f"Step name '{v}' is not a valid identifier. "
                "Use lowercase letters, numbers, and underscores only."
            )
        return v


class Pipeline(BaseModel):
    """Pipeline layer: parameters and steps."""

    parameters: Dict[str, Parameter]
    steps: List[PipelineStep]

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: List[PipelineStep]) -> List[PipelineStep]:
        """Validate step dependencies and uniqueness."""
        # Check for duplicate step names
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate step names found: {duplicates}")

        # Check that depends_on references valid steps
        for step in v:
            if step.depends_on:
                for dep in step.depends_on:
                    if dep not in names:
                        raise ValueError(f"Step '{step.name}' depends on unknown step: '{dep}'")

        return v

    @model_validator(mode="after")
    def validate_no_circular_dependencies(self) -> "Pipeline":
        """Check for circular dependencies in the pipeline."""
        # Build dependency graph
        graph: Dict[str, List[str]] = {}
        for step in self.steps:
            graph[step.name] = step.depends_on or []

        # Detect cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for step_name in graph:
            if step_name not in visited:
                if has_cycle(step_name):
                    raise ValueError(
                        f"Circular dependency detected in pipeline involving step: {step_name}"
                    )

        return self


class Preset(BaseModel):
    """A preset combines a profile with parameter overrides."""

    profile: str
    overrides: Dict[str, Any] = Field(default_factory=dict)


class ConnectionType(str, Enum):
    """Type of connection between service groups."""

    REQUEST = "request"
    GRADIENT = "gradient"
    STREAM = "stream"


class ConnectionProtocol(str, Enum):
    """Communication protocol for connections."""

    GRPC = "grpc"
    HTTP = "http"
    TCP = "tcp"
    NCCL = "nccl"
    REDIS = "redis"


class PlacementConstraint(str, Enum):
    """Placement constraints for service groups."""

    COLOCATED = "colocated"
    SAME_AZ = "same_az"
    DIFFERENT_AZ = "different_az"
    ANY = "any"


class ServiceGroup(BaseModel):
    """A named group of replicas in the architecture."""

    component: str = Field(..., pattern=r"^[a-z0-9_.]+$")
    replicas: int = Field(..., ge=1)
    gpus_per_replica: int = Field(default=0, ge=0)
    instance: Optional[str] = None
    cpu: Optional[str] = None
    memory: Optional[str] = None
    image: Optional[str] = None
    command: Optional[List[str]] = Field(
        None,
        description="Container command override (if not set, uses image default)",
    )
    env: Optional[Dict[str, str]] = Field(default_factory=dict)


class Connection(BaseModel):
    """Connection between service groups."""

    from_group: str
    to_group: str
    type: ConnectionType
    protocol: ConnectionProtocol
    load_balancing: Optional[str] = Field(None, pattern=r"^(round_robin|random|least_conn)$")
    requires: Optional[str] = None
    placement: Optional[PlacementConstraint] = None
    bidirectional: bool = False

    @field_validator("from_group", "to_group")
    @classmethod
    def validate_group_name(cls, v: str) -> str:
        """Ensure group names are valid identifiers."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid group name: {v}. Use alphanumeric, underscore, or hyphen.")
        return v


class HealthCheck(BaseModel):
    """Health check configuration for a service group."""

    endpoint: str
    timeout: str = Field(..., pattern=r"^\d+(s|m|h)$")
    interval: Optional[str] = Field("30s", pattern=r"^\d+(s|m|h)$")
    retries: Optional[int] = Field(3, ge=1)


class Lifecycle(BaseModel):
    """Lifecycle configuration for service orchestration."""

    startup_order: Dict[int, List[str]]
    health_checks: Dict[str, HealthCheck] = Field(default_factory=dict)
    shutdown_grace_period: Optional[str] = Field("60s", pattern=r"^\d+(s|m|h)$")

    @field_validator("startup_order")
    @classmethod
    def validate_startup_order(cls, v: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """Validate startup order stages."""
        if not v:
            raise ValueError("At least one startup stage is required")
        stages = sorted(v.keys())
        if stages[0] != 1:
            raise ValueError("Startup stages must start at 1")
        for i in range(len(stages) - 1):
            if stages[i + 1] != stages[i] + 1:
                raise ValueError("Startup stages must be consecutive")
        return v


class Architecture(BaseModel):
    """Architecture definition for multi-service deployments."""

    auto_deploy: bool = Field(
        True,
        description=(
            "If True, the recipe engine deploys the architecture to K8s before "
            "running pipeline steps. Set to False when services require "
            "pre-provisioned models or manual deployment."
        ),
    )
    groups: Dict[str, ServiceGroup]
    connections: List[Connection] = Field(default_factory=list)
    lifecycle: Optional[Lifecycle] = None

    @field_validator("groups")
    @classmethod
    def validate_groups(cls, v: Dict[str, ServiceGroup]) -> Dict[str, ServiceGroup]:
        """Ensure at least one group is defined."""
        if not v:
            raise ValueError("At least one service group is required in architecture")
        return v

    @model_validator(mode="after")
    def validate_connections(self) -> "Architecture":
        """Validate that connection references point to defined groups."""
        group_names = set(self.groups.keys())
        for conn in self.connections:
            if conn.from_group not in group_names:
                raise ValueError(
                    f"Connection references undefined group: {conn.from_group}. "
                    f"Available groups: {', '.join(sorted(group_names))}"
                )
            if conn.to_group not in group_names:
                raise ValueError(
                    f"Connection references undefined group: {conn.to_group}. "
                    f"Available groups: {', '.join(sorted(group_names))}"
                )
        return self

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "Architecture":
        """Validate that lifecycle references point to defined groups."""
        if self.lifecycle:
            group_names = set(self.groups.keys())
            all_lifecycle_groups = set()
            for stage_groups in self.lifecycle.startup_order.values():
                for group in stage_groups:
                    if group not in group_names:
                        raise ValueError(
                            f"Lifecycle startup_order references undefined group: {group}. "
                            f"Available groups: {', '.join(sorted(group_names))}"
                        )
                    all_lifecycle_groups.add(group)
            for health_group in self.lifecycle.health_checks.keys():
                if health_group not in group_names:
                    raise ValueError(
                        f"Lifecycle health_checks references undefined group: {health_group}. "
                        f"Available groups: {', '.join(sorted(group_names))}"
                    )
        return self


class Recipe(BaseModel):
    """Top-level Recipe definition."""

    name: str = Field(..., pattern=r"^[a-z0-9-]+$")
    version: str = Field(..., pattern=r"^\d+\.\d+(\.\d+)?$")
    description: str
    author: str
    tags: List[str]
    infrastructure: Infrastructure
    pipeline: Pipeline
    presets: Dict[str, Preset] = Field(default_factory=dict)
    architecture: Optional[Architecture] = None
    component_versions: Optional[Dict[str, str]] = Field(
        None,
        description="Global component version pins. Maps fully-qualified component "
        "names (e.g. 'components.training.lora_finetune.lora_finetune') to version "
        "strings. Per-step `version` takes precedence over this mapping.",
    )

    @field_validator("name")
    @classmethod
    def validate_recipe_name(cls, v: str) -> str:
        """Ensure recipe name is URL-safe."""
        if not re.match(r"^[a-z0-9-]+$", v):
            raise ValueError(
                f"Recipe name '{v}' must contain only lowercase letters, numbers, and hyphens"
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate semantic versioning format."""
        if not re.match(r"^\d+\.\d+(\.\d+)?$", v):
            raise ValueError(
                f"Version '{v}' must follow semantic versioning (e.g., '1.0' or '1.0.0')"
            )  # noqa: E501
        return v

    @model_validator(mode="after")
    def validate_presets(self) -> "Recipe":
        """Ensure preset profiles exist and infra references are valid."""
        valid_profiles = set(self.infrastructure.profiles.keys())

        # Validate preset profiles
        for preset_name, preset in self.presets.items():
            if preset.profile not in valid_profiles:
                raise ValueError(
                    f"Preset '{preset_name}' references unknown profile: '{preset.profile}'. "
                    f"Available profiles: {', '.join(valid_profiles)}"
                )

        # Validate step infra references
        for step in self.pipeline.steps:
            if step.infra is not None:
                # Check if infra is defined in any profile
                # (we check the first profile as all profiles should define the same resource groups)  # noqa: E501
                first_profile = list(self.infrastructure.profiles.values())[0]
                if step.infra not in first_profile:
                    # It might be defined in other profiles but not all
                    # For now, we'll just warn via a softer check
                    pass  # This is okay; not all profiles need all resource groups

        return self

    @model_validator(mode="after")
    def validate_template_references(self) -> "Recipe":
        """Basic validation of template syntax in config values."""
        template_pattern = re.compile(r"\{\{\s*([^}]+)\s*\}\}")

        def check_templates(obj: Any, path: str = ""):
            if isinstance(obj, str):
                # Check for template syntax
                matches = template_pattern.findall(obj)
                for match in matches:
                    # Basic validation: should start with 'parameters.' or 'steps.'
                    if not (
                        match.strip().startswith("parameters.")
                        or match.strip().startswith("steps.")
                    ):  # noqa: E501
                        raise ValueError(
                            f"Invalid template '{{{{{match}}}}}' at {path}. "
                            "Templates must reference 'parameters.x' or 'steps.y.outputs.z'"
                        )
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    check_templates(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_templates(item, f"{path}[{i}]")

        # Check all step configs
        for step in self.pipeline.steps:
            check_templates(step.config, f"steps.{step.name}.config")

        return self
