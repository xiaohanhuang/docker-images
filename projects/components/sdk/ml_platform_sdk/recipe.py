"""Python SDK for programmatic recipe definition.

This module provides a fluent API for building recipes programmatically,
especially for complex multi-service architectures that are difficult to
express in YAML.

Example:
    ```python
    from ml_platform_sdk.recipe import Recipe

    recipe = Recipe("custom-rlhf")
    recipe.set_metadata(
        version="1.0",
        description="Custom RLHF architecture",
        author="my-team",
        tags=["llm", "rlhf"]
    )

    # Define service groups
    actor = recipe.group("actor", "training.rlhf_actor", nodes=4, gpus=8)
    reward = recipe.group("reward", "serving.reward_model", nodes=1, gpus=1)

    # Connect them
    recipe.connect(actor, reward, protocol="http")

    # Define startup order
    recipe.ready_before(reward, [actor])

    # Export to YAML
    recipe.to_yaml("custom-rlhf.yaml")
    ```
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


class ServiceGroupBuilder:
    """Builder for service group configuration."""

    def __init__(self, name: str, component: str):
        """Initialize service group builder.

        Args:
            name: Group name
            component: Component reference (e.g., "training.rlhf_actor")
        """
        self.name = name
        self.component = component
        self.replicas = 1
        self.gpus_per_replica = 0
        self.instance: Optional[str] = None
        self.cpu: Optional[str] = None
        self.memory: Optional[str] = None
        self.env: Dict[str, str] = {}

    def with_replicas(self, count: int) -> ServiceGroupBuilder:
        """Set replica count.

        Args:
            count: Number of replicas

        Returns:
            Self for chaining
        """
        self.replicas = count
        return self

    def with_gpus(self, count: int) -> ServiceGroupBuilder:
        """Set GPU count per replica.

        Args:
            count: GPUs per replica

        Returns:
            Self for chaining
        """
        self.gpus_per_replica = count
        return self

    def with_instance(self, instance_type: str) -> ServiceGroupBuilder:
        """Set instance type.

        Args:
            instance_type: AWS instance type (e.g., "p4d.24xlarge")

        Returns:
            Self for chaining
        """
        self.instance = instance_type
        return self

    def with_resources(self, cpu: str, memory: str) -> ServiceGroupBuilder:
        """Set CPU and memory resources.

        Args:
            cpu: CPU count (e.g., "8")
            memory: Memory size (e.g., "32GB")

        Returns:
            Self for chaining
        """
        self.cpu = cpu
        self.memory = memory
        return self

    def with_env(self, key: str, value: str) -> ServiceGroupBuilder:
        """Add environment variable.

        Args:
            key: Environment variable name
            value: Environment variable value

        Returns:
            Self for chaining
        """
        self.env[key] = value
        return self

    def build(self) -> Dict[str, Any]:
        """Build service group config dict.

        Returns:
            Service group configuration dictionary
        """
        config = {
            "component": self.component,
            "replicas": self.replicas,
            "gpus_per_replica": self.gpus_per_replica,
        }

        if self.instance:
            config["instance"] = self.instance
        if self.cpu:
            config["cpu"] = self.cpu
        if self.memory:
            config["memory"] = self.memory
        if self.env:
            config["env"] = self.env

        return config


class ConnectionBuilder:
    """Builder for connection configuration."""

    def __init__(self, from_group: str, to_group: str):
        """Initialize connection builder.

        Args:
            from_group: Source group name
            to_group: Target group name
        """
        self.from_group = from_group
        self.to_group = to_group
        self.conn_type = "request"
        self.protocol = "grpc"
        self.load_balancing: Optional[str] = None
        self.requires: Optional[str] = None
        self.placement: Optional[str] = None
        self.bidirectional = False

    def as_type(self, conn_type: str) -> ConnectionBuilder:
        """Set connection type.

        Args:
            conn_type: Connection type (request, gradient, stream)

        Returns:
            Self for chaining
        """
        self.conn_type = conn_type
        return self

    def with_protocol(self, protocol: str) -> ConnectionBuilder:
        """Set connection protocol.

        Args:
            protocol: Protocol (grpc, http, tcp, nccl, redis)

        Returns:
            Self for chaining
        """
        self.protocol = protocol
        return self

    def with_load_balancing(self, strategy: str) -> ConnectionBuilder:
        """Set load balancing strategy.

        Args:
            strategy: Strategy (round_robin, random, least_conn)

        Returns:
            Self for chaining
        """
        self.load_balancing = strategy
        return self

    def requiring(self, requirement: str) -> ConnectionBuilder:
        """Set network requirement.

        Args:
            requirement: Requirement (efa, etc.)

        Returns:
            Self for chaining
        """
        self.requires = requirement
        return self

    def with_placement(self, constraint: str) -> ConnectionBuilder:
        """Set placement constraint.

        Args:
            constraint: Placement constraint (colocated, same_az, different_az, any)

        Returns:
            Self for chaining
        """
        self.placement = constraint
        return self

    def bidirectional_mode(self, enabled: bool = True) -> ConnectionBuilder:
        """Set bidirectional mode.

        Args:
            enabled: Whether connection is bidirectional

        Returns:
            Self for chaining
        """
        self.bidirectional = enabled
        return self

    def build(self) -> Dict[str, Any]:
        """Build connection config dict.

        Returns:
            Connection configuration dictionary
        """
        config = {
            "from_group": self.from_group,
            "to_group": self.to_group,
            "type": self.conn_type,
            "protocol": self.protocol,
        }

        if self.load_balancing:
            config["load_balancing"] = self.load_balancing
        if self.requires:
            config["requires"] = self.requires
        if self.placement:
            config["placement"] = self.placement
        if self.bidirectional:
            config["bidirectional"] = self.bidirectional

        return config


class Recipe:
    """Programmatic recipe builder."""

    def __init__(self, name: str):
        """Initialize recipe builder.

        Args:
            name: Recipe name
        """
        self.name = name
        self.version = "1.0"
        self.description = ""
        self.author = ""
        self.tags: List[str] = []
        self.groups: Dict[str, ServiceGroupBuilder] = {}
        self.connections: List[ConnectionBuilder] = []
        self.startup_stages: Dict[int, List[str]] = {}
        self.health_checks: Dict[str, Dict[str, Any]] = {}

    def set_metadata(
        self,
        version: str,
        description: str,
        author: str,
        tags: Optional[List[str]] = None,
    ) -> Recipe:
        """Set recipe metadata.

        Args:
            version: Recipe version (semver)
            description: Recipe description
            author: Recipe author
            tags: Recipe tags

        Returns:
            Self for chaining
        """
        self.version = version
        self.description = description
        self.author = author
        if tags:
            self.tags = tags
        return self

    def group(
        self,
        name: str,
        component: str,
        nodes: int = 1,
        gpus: int = 0,
        instance: Optional[str] = None,
    ) -> ServiceGroupBuilder:
        """Define a service group.

        Args:
            name: Group name
            component: Component reference
            nodes: Number of replicas
            gpus: GPUs per replica
            instance: Instance type

        Returns:
            ServiceGroupBuilder for further configuration
        """
        builder = ServiceGroupBuilder(name, component)
        builder.with_replicas(nodes).with_gpus(gpus)
        if instance:
            builder.with_instance(instance)

        self.groups[name] = builder
        return builder

    def connect(
        self,
        from_group: Union[str, ServiceGroupBuilder],
        to_group: Union[str, ServiceGroupBuilder],
        protocol: str = "grpc",
        conn_type: str = "request",
    ) -> ConnectionBuilder:
        """Connect two service groups.

        Args:
            from_group: Source group name or builder
            to_group: Target group name or builder
            protocol: Connection protocol
            conn_type: Connection type

        Returns:
            ConnectionBuilder for further configuration
        """
        from_name = from_group.name if isinstance(from_group, ServiceGroupBuilder) else from_group
        to_name = to_group.name if isinstance(to_group, ServiceGroupBuilder) else to_group

        builder = ConnectionBuilder(from_name, to_name)
        builder.with_protocol(protocol).as_type(conn_type)

        self.connections.append(builder)
        return builder

    def ready_before(
        self,
        group: Union[str, ServiceGroupBuilder],
        dependencies: List[Union[str, ServiceGroupBuilder]],
    ) -> Recipe:
        """Define that a group must be ready before dependent groups start.

        Args:
            group: Group that must be ready first
            dependencies: Groups that depend on this group

        Returns:
            Self for chaining
        """
        group_name = group.name if isinstance(group, ServiceGroupBuilder) else group

        # Find or assign stage for group
        group_stage = None
        for stage, groups in self.startup_stages.items():
            if group_name in groups:
                group_stage = stage
                break

        if group_stage is None:
            # Assign to stage 1
            if 1 not in self.startup_stages:
                self.startup_stages[1] = []
            self.startup_stages[1].append(group_name)
            group_stage = 1

        # Assign dependencies to later stage
        dep_stage = group_stage + 1
        if dep_stage not in self.startup_stages:
            self.startup_stages[dep_stage] = []

        for dep in dependencies:
            dep_name = dep.name if isinstance(dep, ServiceGroupBuilder) else dep
            if dep_name not in self.startup_stages[dep_stage]:
                self.startup_stages[dep_stage].append(dep_name)

        return self

    def health_check(
        self,
        group: Union[str, ServiceGroupBuilder],
        endpoint: str,
        timeout: str = "30s",
        interval: str = "30s",
        retries: int = 3,
    ) -> Recipe:
        """Add health check for a group.

        Args:
            group: Group name or builder
            endpoint: Health check endpoint path
            timeout: Health check timeout
            interval: Health check interval
            retries: Number of retries

        Returns:
            Self for chaining
        """
        group_name = group.name if isinstance(group, ServiceGroupBuilder) else group

        self.health_checks[group_name] = {
            "endpoint": endpoint,
            "timeout": timeout,
            "interval": interval,
            "retries": retries,
        }

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Export recipe to dictionary format.

        Returns:
            Recipe as dictionary
        """
        recipe_dict: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
        }

        # Add architecture if groups are defined
        if self.groups:
            architecture: Dict[str, Any] = {
                "groups": {name: builder.build() for name, builder in self.groups.items()}
            }

            if self.connections:
                architecture["connections"] = [conn.build() for conn in self.connections]

            if self.startup_stages or self.health_checks:
                lifecycle: Dict[str, Any] = {}
                if self.startup_stages:
                    lifecycle["startup_order"] = self.startup_stages
                if self.health_checks:
                    lifecycle["health_checks"] = self.health_checks
                architecture["lifecycle"] = lifecycle

            recipe_dict["architecture"] = architecture

        return recipe_dict

    def to_yaml(self, path: Optional[str] = None) -> str:
        """Export recipe to YAML format.

        Args:
            path: Optional file path to write YAML

        Returns:
            YAML string
        """
        import yaml

        recipe_dict = self.to_dict()
        yaml_str = yaml.dump(recipe_dict, default_flow_style=False, sort_keys=False)

        if path:
            with open(path, "w") as f:
                f.write(yaml_str)

        return yaml_str
