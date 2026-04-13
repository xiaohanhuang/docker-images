"""Architecture topology graph builder for multi-service recipes.

This module provides:
- Parsing of architecture blocks from recipes
- Building topology graphs of service groups and connections
- Dependency resolution for startup ordering
- Validation of network topologies and placement constraints
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from cli.recipe_engine.schema import (
    Architecture,
    Connection,
    ConnectionProtocol,
    PlacementConstraint,
)

logger = logging.getLogger(__name__)


class TopologyGraph:
    """Graph representation of service architecture topology."""

    def __init__(self, architecture: Architecture):
        """Initialize topology graph from architecture definition.

        Args:
            architecture: Architecture definition from recipe
        """
        self.architecture = architecture
        self.groups = architecture.groups
        self.connections = architecture.connections
        self.lifecycle = architecture.lifecycle

        # Build adjacency lists
        self._outgoing: Dict[str, List[Connection]] = {}
        self._incoming: Dict[str, List[Connection]] = {}
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Build adjacency lists for quick lookup."""
        for group_name in self.groups.keys():
            self._outgoing[group_name] = []
            self._incoming[group_name] = []

        for conn in self.connections:
            self._outgoing[conn.from_group].append(conn)
            self._incoming[conn.to_group].append(conn)

            # Handle bidirectional connections
            if conn.bidirectional:
                reverse_conn = Connection(
                    from_group=conn.to_group,
                    to_group=conn.from_group,
                    type=conn.type,
                    protocol=conn.protocol,
                    load_balancing=conn.load_balancing,
                    requires=conn.requires,
                    placement=conn.placement,
                    bidirectional=False,
                )
                self._outgoing[conn.to_group].append(reverse_conn)
                self._incoming[conn.from_group].append(reverse_conn)

    def get_dependencies(self, group_name: str) -> List[str]:
        """Get groups that the specified group depends on.

        A group depends on another if:
        1. It has an incoming connection from that group (for REQUEST types)
        2. It's in a later startup stage in the lifecycle

        Args:
            group_name: Name of the service group

        Returns:
            List of group names that are dependencies
        """
        deps = set()

        # Add incoming request connections as dependencies
        for conn in self._incoming[group_name]:
            if conn.type.value == "request":
                deps.add(conn.from_group)

        # Add lifecycle dependencies
        if self.lifecycle:
            group_stage = None
            for stage, groups in self.lifecycle.startup_order.items():
                if group_name in groups:
                    group_stage = stage
                    break

            if group_stage:
                # Groups in earlier stages are dependencies
                for stage in range(1, group_stage):
                    if stage in self.lifecycle.startup_order:
                        deps.update(self.lifecycle.startup_order[stage])

        return sorted(deps)

    def get_startup_stages(self) -> List[List[str]]:
        """Get groups organized by startup stage.

        Returns:
            List of lists, where each inner list contains group names
            for that startup stage (1-indexed)
        """
        if not self.lifecycle or not self.lifecycle.startup_order:
            # No explicit lifecycle - return all groups in one stage
            return [list(self.groups.keys())]

        stages = []
        max_stage = max(self.lifecycle.startup_order.keys())
        for stage in range(1, max_stage + 1):
            if stage in self.lifecycle.startup_order:
                stages.append(self.lifecycle.startup_order[stage])
            else:
                stages.append([])

        return stages

    def get_colocated_groups(self) -> List[Tuple[str, str]]:
        """Get pairs of groups that must be colocated.

        Returns:
            List of (group_a, group_b) tuples requiring colocation
        """
        colocated_pairs = []
        for conn in self.connections:
            if conn.placement == PlacementConstraint.COLOCATED:
                colocated_pairs.append((conn.from_group, conn.to_group))
        return colocated_pairs

    def get_same_az_groups(self) -> List[Tuple[str, str]]:
        """Get pairs of groups that must be in the same availability zone.

        Returns:
            List of (group_a, group_b) tuples requiring same AZ
        """
        same_az_pairs = []
        for conn in self.connections:
            if conn.placement == PlacementConstraint.SAME_AZ:
                same_az_pairs.append((conn.from_group, conn.to_group))
        return same_az_pairs

    def requires_efa(self, group_name: str) -> bool:
        """Check if a group requires EFA networking.

        Args:
            group_name: Name of the service group

        Returns:
            True if group has any connection requiring EFA
        """
        for conn in self._outgoing[group_name] + self._incoming[group_name]:
            if conn.requires == "efa" or conn.protocol == ConnectionProtocol.NCCL:
                return True
        return False

    def get_connection_endpoints(self, group_name: str) -> Dict[str, List[Dict[str, str]]]:
        """Get all connection endpoints for a group.

        Returns mapping of direction to list of endpoint info:
        {
            "outgoing": [{"target": "group_name", "protocol": "grpc", ...}],
            "incoming": [{"source": "group_name", "protocol": "grpc", ...}]
        }

        Args:
            group_name: Name of the service group

        Returns:
            Dictionary with outgoing and incoming connection details
        """
        endpoints = {"outgoing": [], "incoming": []}

        for conn in self._outgoing[group_name]:
            endpoints["outgoing"].append(
                {
                    "target": conn.to_group,
                    "protocol": conn.protocol.value,
                    "type": conn.type.value,
                    "load_balancing": conn.load_balancing,
                }
            )

        for conn in self._incoming[group_name]:
            endpoints["incoming"].append(
                {
                    "source": conn.from_group,
                    "protocol": conn.protocol.value,
                    "type": conn.type.value,
                }
            )

        return endpoints

    def validate_topology(self) -> List[str]:
        """Validate the topology for common issues.

        Returns:
            List of warning messages (empty if valid)
        """
        warnings = []

        # Check for isolated groups (no connections)
        for group_name in self.groups.keys():
            if not self._outgoing[group_name] and not self._incoming[group_name]:
                warnings.append(f"Group '{group_name}' has no connections (isolated)")

        # Check for NCCL without placement constraints
        for conn in self.connections:
            if conn.protocol == ConnectionProtocol.NCCL:
                if conn.placement not in [
                    PlacementConstraint.COLOCATED,
                    PlacementConstraint.SAME_AZ,
                ]:
                    warnings.append(
                        f"NCCL connection from {conn.from_group} to {conn.to_group} "
                        "should have 'colocated' or 'same_az' placement constraint"
                    )

        # Check for groups in lifecycle but not in architecture
        if self.lifecycle:
            all_lifecycle_groups = set()
            for groups in self.lifecycle.startup_order.values():
                all_lifecycle_groups.update(groups)

            defined_groups = set(self.groups.keys())
            for group in all_lifecycle_groups:
                if group not in defined_groups:
                    warnings.append(f"Lifecycle references undefined group: {group}")

            # Check for groups not in lifecycle
            missing_from_lifecycle = defined_groups - all_lifecycle_groups
            if missing_from_lifecycle:
                warnings.append(
                    f"Groups not in lifecycle startup_order: {', '.join(missing_from_lifecycle)}"
                )

        return warnings


def build_topology_graph(architecture: Architecture) -> TopologyGraph:
    """Build a topology graph from an architecture definition.

    Args:
        architecture: Architecture definition from recipe

    Returns:
        TopologyGraph instance

    Raises:
        ValueError: If topology is invalid
    """
    graph = TopologyGraph(architecture)
    warnings = graph.validate_topology()

    if warnings:
        logger.warning("Topology validation warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    return graph
