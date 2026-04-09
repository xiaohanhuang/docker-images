"""Lifecycle orchestration for multi-service architectures.

This module provides:
- Startup ordering coordination
- Health check management
- Readiness gate enforcement
- Service dependency resolution
- Graceful shutdown handling
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cli.recipe_engine.architecture import TopologyGraph
from cli.recipe_engine.schema import Architecture

logger = logging.getLogger(__name__)


class LifecycleOrchestrator:
    """Orchestrator for managing service lifecycle."""

    def __init__(
        self,
        architecture: Architecture,
        topology: TopologyGraph,
    ):
        """Initialize lifecycle orchestrator.

        Args:
            architecture: Architecture definition
            topology: Topology graph
        """
        self.architecture = architecture
        self.topology = topology
        self.lifecycle = architecture.lifecycle

    def get_startup_plan(self) -> List[Dict[str, Any]]:
        """Generate a startup plan with ordered stages.

        Returns:
            List of stage dicts with:
            - stage: Stage number (1-indexed)
            - groups: List of group names to start
            - dependencies: List of groups from previous stages
            - health_checks: Health check configs for groups in this stage
        """
        stages = self.topology.get_startup_stages()
        plan = []

        for stage_idx, stage_groups in enumerate(stages, 1):
            # Collect dependencies (all groups from earlier stages)
            dependencies = []
            for i in range(stage_idx - 1):
                dependencies.extend(stages[i])

            # Collect health checks for groups in this stage
            health_checks = {}
            if self.lifecycle:
                for group in stage_groups:
                    if group in self.lifecycle.health_checks:
                        health_checks[group] = self.lifecycle.health_checks[group]

            plan.append(
                {
                    "stage": stage_idx,
                    "groups": stage_groups,
                    "dependencies": dependencies,
                    "health_checks": health_checks,
                }
            )

        return plan

    def generate_readiness_gates(self) -> Dict[str, List[str]]:
        """Generate readiness gates for each group.

        A group's readiness gate includes:
        - All groups it depends on (from dependencies)
        - All groups in earlier lifecycle stages

        Returns:
            Dict mapping group name to list of dependency group names
        """
        readiness_gates = {}

        for group_name in self.architecture.groups.keys():
            gates = set(self.topology.get_dependencies(group_name))
            readiness_gates[group_name] = sorted(gates)

        return readiness_gates

    def validate_health_check_config(self, group_name: str) -> Optional[str]:
        """Validate health check configuration for a group.

        Args:
            group_name: Name of the service group

        Returns:
            Error message if invalid, None if valid
        """
        if not self.lifecycle:
            return None

        if group_name not in self.lifecycle.health_checks:
            return None

        health_check = self.lifecycle.health_checks[group_name]

        # Validate endpoint format
        if not health_check.endpoint.startswith("/"):
            return f"Health check endpoint must start with '/': {health_check.endpoint}"

        # Validate timeout format
        if not self._parse_duration(health_check.timeout):
            return f"Invalid timeout format: {health_check.timeout}"

        if health_check.interval and not self._parse_duration(health_check.interval):
            return f"Invalid interval format: {health_check.interval}"

        return None

    def _parse_duration(self, duration: str) -> Optional[int]:
        """Parse duration string to seconds.

        Args:
            duration: Duration string (e.g., "30s", "5m", "1h")

        Returns:
            Duration in seconds, or None if invalid
        """
        if not duration:
            return None

        try:
            if duration.endswith("s"):
                return int(duration[:-1])
            elif duration.endswith("m"):
                return int(duration[:-1]) * 60
            elif duration.endswith("h"):
                return int(duration[:-1]) * 3600
        except (ValueError, IndexError):
            return None

        return None

    def generate_kubernetes_probes(self, group_name: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Generate Kubernetes liveness and readiness probes for a group.

        Args:
            group_name: Name of the service group

        Returns:
            Dict with 'livenessProbe' and 'readinessProbe' configs,
            or None if no health checks defined
        """
        if not self.lifecycle:
            return None

        if group_name not in self.lifecycle.health_checks:
            return None

        health_check = self.lifecycle.health_checks[group_name]
        timeout_seconds = self._parse_duration(health_check.timeout) or 30
        interval_seconds = self._parse_duration(health_check.interval) or 30

        probes = {
            "livenessProbe": {
                "httpGet": {
                    "path": health_check.endpoint,
                    "port": 8080,
                },
                "initialDelaySeconds": 30,
                "periodSeconds": interval_seconds,
                "timeoutSeconds": timeout_seconds,
                "failureThreshold": health_check.retries or 3,
            },
            "readinessProbe": {
                "httpGet": {
                    "path": health_check.endpoint,
                    "port": 8080,
                },
                "initialDelaySeconds": 10,
                "periodSeconds": interval_seconds,
                "timeoutSeconds": timeout_seconds,
                "failureThreshold": 2,
            },
        }

        return probes

    def estimate_startup_time(self) -> int:
        """Estimate total startup time for the architecture.

        Returns:
            Estimated startup time in seconds
        """
        if not self.lifecycle:
            # Without lifecycle, estimate based on number of groups
            return len(self.architecture.groups) * 60

        total_time = 0
        stages = self.topology.get_startup_stages()

        for stage_groups in stages:
            # Each stage waits for all its groups to be ready
            # Estimate: max health check timeout for groups in stage
            stage_time = 60  # Default 1 minute

            for group in stage_groups:
                if group in self.lifecycle.health_checks:
                    health_check = self.lifecycle.health_checks[group]
                    timeout = self._parse_duration(health_check.timeout)
                    if timeout and timeout > stage_time:
                        stage_time = timeout

            total_time += stage_time

        return total_time


def generate_startup_order(
    architecture: Architecture,
    topology: TopologyGraph,
) -> List[Dict[str, Any]]:
    """Generate startup order plan from architecture and topology.

    Args:
        architecture: Architecture definition
        topology: Topology graph

    Returns:
        List of startup stages with groups and dependencies
    """
    orchestrator = LifecycleOrchestrator(architecture, topology)
    return orchestrator.get_startup_plan()


def generate_health_checks(
    architecture: Architecture,
    topology: TopologyGraph,
) -> Dict[str, Dict[str, Any]]:
    """Generate health check configurations for all groups.

    Args:
        architecture: Architecture definition
        topology: Topology graph

    Returns:
        Dict mapping group name to health check config
    """
    orchestrator = LifecycleOrchestrator(architecture, topology)
    health_configs = {}

    for group_name in architecture.groups.keys():
        probes = orchestrator.generate_kubernetes_probes(group_name)
        if probes:
            health_configs[group_name] = probes

    return health_configs
