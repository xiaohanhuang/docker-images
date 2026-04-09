"""Architecture deployer for multi-service recipe execution.

This module provides:
- Kubernetes manifest generation from architecture definitions
- kubectl-based deployment of NodePools, Services, and Deployments
- Staged lifecycle orchestration with health check polling
- Labeled resource cleanup on teardown

The deployer bridges the gap between the architecture engine (topology,
provisioner, lifecycle) and the actual K8s cluster.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from typing import Any, Dict, List

import yaml

from cli.recipe_engine.architecture import TopologyGraph, build_topology_graph
from cli.recipe_engine.lifecycle import LifecycleOrchestrator
from cli.recipe_engine.provisioner import generate_kubernetes_manifests
from cli.recipe_engine.schema import Architecture

logger = logging.getLogger(__name__)

# Labels applied to all resources for easy cleanup
_MANAGED_BY = "ml-plat-recipe-engine"


class DeploymentError(Exception):
    """Raised when architecture deployment fails."""

    pass


class ArchitectureDeployer:
    """Deploy and manage multi-service architectures on Kubernetes.

    Lifecycle:
        1. deploy()   — generate manifests, kubectl apply, wait for health
        2. status()   — check running state of deployed resources
        3. teardown() — kubectl delete all managed resources

    All resources are labeled with ``app.kubernetes.io/managed-by: ml-plat-recipe-engine``
    and ``app.kubernetes.io/instance: <recipe_name>`` for targeted cleanup.
    """

    def __init__(
        self,
        recipe_name: str,
        architecture: Architecture,
        namespace: str = "default",
    ):
        """Initialize deployer.

        Args:
            recipe_name: Name of the recipe (used for labels/naming)
            architecture: Architecture definition from recipe schema
            namespace: Kubernetes namespace to deploy into
        """
        self.recipe_name = recipe_name
        self.architecture = architecture
        self.namespace = namespace

        # Build graph + orchestrator eagerly so errors surface early
        self.topology: TopologyGraph = build_topology_graph(architecture)
        self.orchestrator = LifecycleOrchestrator(architecture, self.topology)
        self.manifests: Dict[str, List[Dict[str, Any]]] = {}
        self._applied = False

    # ── Public API ────────────────────────────────────────────────────────────

    def deploy(self, dry_run: bool = False) -> Dict[str, Any]:
        """Deploy the architecture to Kubernetes.

        Args:
            dry_run: If True, generate manifests but don't apply them.

        Returns:
            Dict with deployment details:
            - manifests: generated K8s manifests by type
            - topology: topology summary (groups, connections, EFA, placement)
            - lifecycle: startup plan with stages
            - estimated_startup_time: seconds
            - status: "dry_run" | "deployed"

        Raises:
            DeploymentError: If kubectl apply fails or health checks time out.
        """
        # 1. Generate manifests
        self.manifests = generate_kubernetes_manifests(
            self.architecture,
            self.topology,
            namespace=self.namespace,
            recipe_name=self.recipe_name,
        )

        # Inject management labels into all manifests
        self._inject_labels(self.manifests)

        # 2. Build summary
        startup_plan = self.orchestrator.get_startup_plan()
        result = {
            "manifests": self.manifests,
            "topology": self._topology_summary(),
            "lifecycle": {
                "startup_plan": startup_plan,
                "estimated_startup_time": self.orchestrator.estimate_startup_time(),
            },
            "manifest_counts": {k: len(v) for k, v in self.manifests.items()},
        }

        if dry_run:
            result["status"] = "dry_run"
            result["yaml"] = self._manifests_to_yaml()
            return result

        # 3. Verify kubectl is available
        self._check_kubectl()

        # 4. Apply manifests in order: NodePools → Services → Deployments → ConfigMaps
        apply_order = ["nodepools", "services", "configmaps", "deployments"]
        for resource_type in apply_order:
            manifests_list = self.manifests.get(resource_type, [])
            if manifests_list:
                logger.info("Applying %d %s...", len(manifests_list), resource_type)
                self._kubectl_apply(manifests_list)

        self._applied = True

        # 5. Lifecycle orchestration: staged startup with health check polling
        logger.info("Waiting for architecture services to become ready...")
        self._wait_for_lifecycle(startup_plan)

        result["status"] = "deployed"
        return result

    def teardown(self) -> None:
        """Delete all Kubernetes resources created by this deployer.

        Uses label selectors to find and delete managed resources.
        Safe to call even if deploy() was not called or partially completed.
        """
        label_selector = (
            f"app.kubernetes.io/managed-by={_MANAGED_BY},"
            f"app.kubernetes.io/instance={self.recipe_name}"
        )

        # Namespaced resources deleted with -n; cluster-scoped ones without.
        namespaced_types = ["deployment", "service", "configmap"]
        cluster_scoped_types = ["nodepool"]

        for rtype in namespaced_types + cluster_scoped_types:
            try:
                cmd = [
                    "kubectl",
                    "delete",
                    rtype,
                    "-l",
                    label_selector,
                ]
                if rtype in namespaced_types:
                    cmd.extend(["-n", self.namespace])
                cmd.append("--ignore-not-found=true")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0 and result.stdout.strip():
                    logger.info("Deleted %s: %s", rtype, result.stdout.strip())
            except subprocess.TimeoutExpired:
                logger.warning("Timeout deleting %s resources", rtype)
            except Exception as exc:
                logger.warning("Failed to delete %s resources: %s", rtype, exc)

    def status(self) -> Dict[str, Any]:
        """Check status of deployed architecture resources.

        Returns:
            Dict mapping group names to their pod/deployment status.
        """
        label_selector = (
            f"app.kubernetes.io/managed-by={_MANAGED_BY},"
            f"app.kubernetes.io/instance={self.recipe_name}"
        )

        result = {}
        try:
            cmd = [
                "kubectl",
                "get",
                "pods",
                "-l",
                label_selector,
                "-n",
                self.namespace,
                "-o",
                "json",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                pods = json.loads(proc.stdout)
                for pod in pods.get("items", []):
                    name = pod["metadata"]["name"]
                    phase = pod["status"].get("phase", "Unknown")
                    component = (
                        pod["metadata"]
                        .get("labels", {})
                        .get("app.kubernetes.io/component", "unknown")
                    )
                    result[name] = {
                        "component": component,
                        "phase": phase,
                        "ready": phase == "Running",
                    }
        except Exception as exc:
            logger.warning("Failed to get pod status: %s", exc)

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_kubectl(self) -> None:
        """Verify kubectl is available and the cluster is reachable."""
        if not shutil.which("kubectl"):
            raise DeploymentError(
                "kubectl not found. Install: https://kubernetes.io/docs/tasks/tools/"
            )

        try:
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise DeploymentError(f"Cannot reach Kubernetes cluster: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            raise DeploymentError("kubectl cluster-info timed out — is the cluster reachable?")

    def _inject_labels(self, manifests: Dict[str, List[Dict[str, Any]]]) -> None:
        """Add management labels to all manifests for cleanup."""
        for resource_type, manifest_list in manifests.items():
            for manifest in manifest_list:
                labels = manifest.setdefault("metadata", {}).setdefault("labels", {})
                labels["app.kubernetes.io/managed-by"] = _MANAGED_BY
                labels["app.kubernetes.io/instance"] = self.recipe_name

                # Also label the pod template spec for deployments
                if manifest.get("kind") == "Deployment":
                    template_labels = (
                        manifest.setdefault("spec", {})
                        .setdefault("template", {})
                        .setdefault("metadata", {})
                        .setdefault("labels", {})
                    )
                    template_labels["app.kubernetes.io/managed-by"] = _MANAGED_BY
                    template_labels["app.kubernetes.io/instance"] = self.recipe_name

    def _kubectl_apply(self, manifests: List[Dict[str, Any]]) -> None:
        """Apply a list of manifests via kubectl apply -f -.

        Pipes YAML to stdin to avoid temp files.

        Args:
            manifests: List of K8s manifest dicts.

        Raises:
            DeploymentError: If kubectl apply fails.
        """
        yaml_str = yaml.dump_all(manifests, default_flow_style=False)

        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-", "-n", self.namespace],
                input=yaml_str,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise DeploymentError(f"kubectl apply failed:\n{result.stderr.strip()}")
            if result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    logger.info("  %s", line)
        except subprocess.TimeoutExpired:
            raise DeploymentError("kubectl apply timed out")

    def _wait_for_lifecycle(self, startup_plan: List[Dict[str, Any]]) -> None:
        """Wait for services to become ready according to the startup plan.

        For each stage:
        1. Wait for all pods in this stage's groups to be Running
        2. If health checks are defined, poll the health endpoints

        Args:
            startup_plan: List of stage dicts from LifecycleOrchestrator.

        Raises:
            DeploymentError: If a stage times out.
        """
        for stage in startup_plan:
            stage_num = stage["stage"]
            groups = stage["groups"]
            health_checks = stage.get("health_checks", {})

            logger.info("Stage %d: waiting for %s...", stage_num, groups)

            # Wait for pods to reach Running state
            for group_name in groups:
                self._wait_for_pods_ready(
                    group_name,
                    timeout=self._get_health_timeout(group_name, health_checks),
                )

            logger.info("Stage %d: all groups ready ✓", stage_num)

    def _wait_for_pods_ready(self, group_name: str, timeout: int = 300) -> None:
        """Wait for a group's pods to reach Running state.

        Args:
            group_name: Name of the service group.
            timeout: Maximum seconds to wait.

        Raises:
            DeploymentError: If timeout exceeded.
        """
        label = (
            f"app.kubernetes.io/component={group_name},"
            f"app.kubernetes.io/instance={self.recipe_name}"
        )
        deadline = time.time() + timeout
        poll_interval = 5

        expected_replicas = self.architecture.groups[group_name].replicas

        while time.time() < deadline:
            try:
                result = subprocess.run(
                    [
                        "kubectl",
                        "get",
                        "pods",
                        "-l",
                        label,
                        "-n",
                        self.namespace,
                        "-o",
                        "jsonpath={.items[*].status.phase}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    phases = result.stdout.strip().split()
                    running = sum(1 for p in phases if p == "Running")
                    if running >= expected_replicas:
                        logger.info(
                            "  %s: %d/%d pods Running ✓",
                            group_name,
                            running,
                            expected_replicas,
                        )
                        return
                    logger.info(
                        "  %s: %d/%d pods Running (waiting...)",
                        group_name,
                        running,
                        expected_replicas,
                    )
            except Exception:
                pass

            time.sleep(poll_interval)

        raise DeploymentError(
            f"Timeout waiting for {group_name} pods to become Running "
            f"(waited {timeout}s, expected {expected_replicas} replicas)"
        )

    def _get_health_timeout(self, group_name: str, health_checks: Dict[str, Any]) -> int:
        """Get timeout for a group's health check, defaulting to 300s."""
        if group_name in health_checks:
            hc = health_checks[group_name]
            # hc can be a HealthCheck Pydantic model or a dict
            if hasattr(hc, "timeout"):
                timeout_str = hc.timeout or "300s"
            elif isinstance(hc, dict):
                timeout_str = hc.get("timeout", "300s")
            else:
                timeout_str = "300s"
            return self._parse_duration(timeout_str)
        return 300

    @staticmethod
    def _parse_duration(duration_str: str) -> int:
        """Parse a duration string (e.g., '300s', '5m', '1h') to seconds."""
        duration_str = duration_str.strip().lower()
        if duration_str.endswith("s"):
            return int(duration_str[:-1])
        if duration_str.endswith("m"):
            return int(duration_str[:-1]) * 60
        if duration_str.endswith("h"):
            return int(duration_str[:-1]) * 3600
        return int(duration_str)

    def _topology_summary(self) -> Dict[str, Any]:
        """Build a human-readable topology summary."""
        groups_summary = {}
        for name, group in self.architecture.groups.items():
            groups_summary[name] = {
                "component": group.component,
                "replicas": group.replicas,
                "gpus_per_replica": group.gpus_per_replica,
                "instance": group.instance,
                "efa_required": self.topology.requires_efa(name),
            }

        connections = []
        for conn in self.architecture.connections:
            connections.append(
                {
                    "from": conn.from_group,
                    "to": conn.to_group,
                    "protocol": conn.protocol.value,
                    "type": conn.type.value,
                    "bidirectional": conn.bidirectional,
                }
            )

        return {
            "groups": groups_summary,
            "connections": connections,
            "colocated_pairs": self.topology.get_colocated_groups(),
            "same_az_pairs": self.topology.get_same_az_groups(),
        }

    def _manifests_to_yaml(self) -> str:
        """Convert all manifests to a single YAML string for dry-run output."""
        all_manifests = []
        for resource_type in ["nodepools", "services", "configmaps", "deployments"]:
            all_manifests.extend(self.manifests.get(resource_type, []))
        return yaml.dump_all(all_manifests, default_flow_style=False)
