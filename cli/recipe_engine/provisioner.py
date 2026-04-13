"""Kubernetes manifest provisioner for multi-service architectures.

This module provides:
- Translation of architecture graphs to Kubernetes manifests
- Kubernetes Service generation for inter-group discovery
- Environment injection for service discovery and NCCL configuration
- ConfigMap/Secret generation for configuration management

NodePool generation is intentionally omitted: recipes use the
cluster's pre-existing Karpenter nodepools (cpu-nodepool, gpu-nodepool,
etc.) via nodeSelector and tolerations on Deployments.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from cli.recipe_engine.architecture import TopologyGraph
from cli.recipe_engine.schema import (
    Architecture,
    ServiceGroup,
)

logger = logging.getLogger(__name__)


class KubernetesProvisioner:
    """Provisioner for generating Kubernetes manifests from architecture."""

    def __init__(
        self,
        architecture: Architecture,
        topology: TopologyGraph,
        namespace: str = "default",
        recipe_name: str = "unnamed",
    ):
        """Initialize provisioner.

        Args:
            architecture: Architecture definition
            topology: Topology graph
            namespace: Kubernetes namespace
            recipe_name: Name of the recipe (for labeling)
        """
        self.architecture = architecture
        self.topology = topology
        self.namespace = namespace
        self.recipe_name = recipe_name

    @staticmethod
    def _sanitize_k8s_name(name: str) -> str:
        """Sanitize a name for use as a Kubernetes resource name.

        Kubernetes resource names must comply with RFC 1035:
        - lowercase alphanumeric + hyphens only
        - must start with a letter
        - must end with alphanumeric

        Args:
            name: Raw name (e.g. 'trajectory_buffer')

        Returns:
            Sanitized name (e.g. 'trajectory-buffer')
        """
        # Replace underscores (and other non-alphanumeric) with hyphens
        sanitized = re.sub(r"[^a-z0-9-]", "-", name.lower())
        # Collapse multiple hyphens
        sanitized = re.sub(r"-+", "-", sanitized)
        # Strip leading/trailing hyphens
        sanitized = sanitized.strip("-")
        return sanitized

    @staticmethod
    def _to_k8s_quantity(value: str) -> str:
        """Convert human-readable resource quantities to K8s format.

        K8s uses binary SI suffixes (Ki, Mi, Gi, Ti) not decimal (KB, MB, GB, TB).

        Args:
            value: Human-readable value (e.g. '32GB', '16384MB', '4')

        Returns:
            K8s-compatible quantity (e.g. '32Gi', '16384Mi', '4')
        """
        if not value:
            return value
        value = value.strip()
        # Map common human suffixes to K8s binary suffixes
        suffix_map = {
            "TB": "Ti",
            "GB": "Gi",
            "MB": "Mi",
            "KB": "Ki",
            "tb": "Ti",
            "gb": "Gi",
            "mb": "Mi",
            "kb": "Ki",
            "T": "Ti",
            "G": "Gi",
            "M": "Mi",
            "K": "Ki",
        }
        for suffix, k8s_suffix in suffix_map.items():
            if value.endswith(suffix):
                numeric = value[: -len(suffix)]
                return f"{numeric}{k8s_suffix}"
        # Already valid (e.g. '32Gi', '4', '500m')
        return value

    def generate_all_manifests(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all Kubernetes manifests for the architecture.

        Returns:
            Dictionary mapping manifest type to list of manifests:
            {
                "services": [...],
                "deployments": [...],
                "configmaps": [...],
            }
        """
        manifests: Dict[str, List[Dict[str, Any]]] = {
            "services": [],
            "deployments": [],
            "configmaps": [],
        }

        # Generate Services for inter-group communication
        services = self._generate_services()
        manifests["services"].extend(services)

        # Generate Deployments for each service group
        deployments = self._generate_deployments()
        manifests["deployments"].extend(deployments)

        # Generate ConfigMaps for configuration
        configmaps = self._generate_configmaps()
        manifests["configmaps"].extend(configmaps)

        return manifests

    def _generate_services(self) -> List[Dict[str, Any]]:
        """Generate Kubernetes Service manifests for inter-group discovery.

        Creates a Service for each group that has incoming connections,
        enabling service discovery via DNS.

        Returns:
            List of Service manifest dicts
        """
        services = []

        for group_name, group in self.architecture.groups.items():
            # Check if this group has incoming connections
            endpoints = self.topology.get_connection_endpoints(group_name)
            if not endpoints["incoming"]:
                continue

            # Determine service ports based on protocols
            ports = []
            protocols_seen = set()

            for conn_info in endpoints["incoming"]:
                protocol = conn_info["protocol"]
                if protocol in protocols_seen:
                    continue
                protocols_seen.add(protocol)

                # Map protocol to port
                port_map = {
                    "grpc": 50051,
                    "http": 8080,
                    "tcp": 9000,
                    "redis": 6379,
                }
                port = port_map.get(protocol, 8080)

                ports.append(
                    {
                        "name": protocol,
                        "port": port,
                        "targetPort": port,
                        "protocol": "TCP",
                    }
                )

            service = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{self.recipe_name}-{self._sanitize_k8s_name(group_name)}",
                    "namespace": self.namespace,
                    "labels": {
                        "app.kubernetes.io/name": self.recipe_name,
                        "app.kubernetes.io/component": group_name,
                    },
                },
                "spec": {
                    "selector": {
                        "app.kubernetes.io/name": self.recipe_name,
                        "app.kubernetes.io/component": group_name,
                    },
                    "ports": ports,
                    "type": "ClusterIP",
                },
            }

            services.append(service)

        return services

    def _generate_deployments(self) -> List[Dict[str, Any]]:
        """Generate Kubernetes Deployment manifests for service groups.

        Creates a Deployment for each service group with:
        - Replica count
        - Resource requests/limits
        - GPU allocation
        - Environment variables for service discovery
        - Tolerations and node selectors

        Returns:
            List of Deployment manifest dicts
        """
        deployments = []

        for group_name, group in self.architecture.groups.items():
            container_spec = {
                "name": self._sanitize_k8s_name(group_name),
                "image": group.image or f"{self.recipe_name}-{group_name}:1.0.0",
                "resources": {
                    "requests": {},
                    "limits": {},
                },
                "env": self._build_env_vars(group_name, group),
            }

            # Add command override if specified
            if group.command:
                container_spec["command"] = group.command

            deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": f"{self.recipe_name}-{self._sanitize_k8s_name(group_name)}",
                    "namespace": self.namespace,
                    "labels": {
                        "app.kubernetes.io/name": self.recipe_name,
                        "app.kubernetes.io/component": group_name,
                    },
                },
                "spec": {
                    "replicas": group.replicas,
                    "selector": {
                        "matchLabels": {
                            "app.kubernetes.io/name": self.recipe_name,
                            "app.kubernetes.io/component": group_name,
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app.kubernetes.io/name": self.recipe_name,
                                "app.kubernetes.io/component": group_name,
                            }
                        },
                        "spec": {
                            "containers": [container_spec],
                        },
                    },
                },
            }

            # Add resource requirements
            resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]
            if group.cpu:
                resources["requests"]["cpu"] = group.cpu
                resources["limits"]["cpu"] = group.cpu
            if group.memory:
                k8s_mem = self._to_k8s_quantity(group.memory)
                resources["requests"]["memory"] = k8s_mem
                resources["limits"]["memory"] = k8s_mem
            if group.gpus_per_replica > 0:
                resources["requests"]["nvidia.com/gpu"] = str(group.gpus_per_replica)
                resources["limits"]["nvidia.com/gpu"] = str(group.gpus_per_replica)

            # Add GPU tolerations
            if group.gpus_per_replica > 0:
                deployment["spec"]["template"]["spec"]["tolerations"] = [
                    {"key": "nvidia.com/gpu", "value": "true", "effect": "NoSchedule"}
                ]
                deployment["spec"]["template"]["spec"]["nodeSelector"] = {"role": "gpu-worker"}

            # Add EFA requirements if needed
            if self.topology.requires_efa(group_name):
                if "nodeSelector" not in deployment["spec"]["template"]["spec"]:
                    deployment["spec"]["template"]["spec"]["nodeSelector"] = {}
                # EFA nodes would be labeled separately
                # deployment["spec"]["template"]["spec"]["nodeSelector"]["efa"] = "true"

            deployments.append(deployment)

        return deployments

    def _build_env_vars(self, group_name: str, group: ServiceGroup) -> List[Dict[str, str]]:
        """Build environment variables for a service group.

        Includes:
        - Service discovery endpoints
        - NCCL configuration
        - Custom environment variables

        Args:
            group_name: Name of the service group
            group: Service group definition

        Returns:
            List of env var dicts
        """
        env_vars = []

        # Add custom env vars from group definition
        for key, value in group.env.items():
            env_vars.append({"name": key, "value": value})

        # Add service discovery endpoints
        endpoints = self.topology.get_connection_endpoints(group_name)
        for idx, conn_info in enumerate(endpoints["outgoing"]):
            target = conn_info["target"]
            protocol = conn_info["protocol"]
            service_name = f"{self.recipe_name}-{self._sanitize_k8s_name(target)}"
            env_vars.append(
                {
                    "name": f"{re.sub(r'[^A-Za-z0-9_]', '_', target).upper()}_SERVICE_URL",
                    "value": f"{protocol}://{service_name}.{self.namespace}.svc.cluster.local",
                }
            )

        # Add NCCL configuration if needed
        if self.topology.requires_efa(group_name):
            env_vars.extend(
                [
                    {"name": "NCCL_DEBUG", "value": "INFO"},
                    {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
                    {"name": "NCCL_IB_DISABLE", "value": "0"},
                    {"name": "NCCL_NET_GDR_LEVEL", "value": "PHB"},
                ]
            )

        # Add replica-related vars
        env_vars.extend(
            [
                {"name": "REPLICA_COUNT", "value": str(group.replicas)},
                {"name": "GROUP_NAME", "value": group_name},
            ]
        )

        return env_vars

    def _generate_configmaps(self) -> List[Dict[str, Any]]:
        """Generate ConfigMaps for configuration management.

        Returns:
            List of ConfigMap manifest dicts
        """
        configmaps = []

        # Create a ConfigMap with architecture metadata
        config_data = {
            "recipe_name": self.recipe_name,
            "groups": ",".join(self.architecture.groups.keys()),
        }

        # Add startup order if defined
        if self.topology.lifecycle:
            stages = self.topology.get_startup_stages()
            for idx, stage_groups in enumerate(stages, 1):
                config_data[f"startup_stage_{idx}"] = ",".join(stage_groups)

        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.recipe_name}-config",
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": self.recipe_name,
                    "app.kubernetes.io/component": "config",
                },
            },
            "data": config_data,
        }

        configmaps.append(configmap)

        return configmaps


def generate_kubernetes_manifests(
    architecture: Architecture,
    topology: TopologyGraph,
    namespace: str = "default",
    recipe_name: str = "unnamed",
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate Kubernetes manifests from architecture and topology.

    Args:
        architecture: Architecture definition
        topology: Topology graph
        namespace: Kubernetes namespace
        recipe_name: Name of the recipe

    Returns:
        Dictionary of manifests by type
    """
    provisioner = KubernetesProvisioner(architecture, topology, namespace, recipe_name)
    return provisioner.generate_all_manifests()
