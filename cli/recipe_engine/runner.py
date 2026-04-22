"""Recipe runner for executing recipes via FlyteRemote.

This module provides:
- Loading and parsing recipes
- Applying preset and CLI overrides
- Resolving template variables (with type preservation)
- Topological ordering of pipeline steps by ``depends_on``
- Submitting pipeline steps to Flyte with latest-version lookup
- Multi-service architecture deployment via ArchitectureDeployer
- Dry-run mode for config inspection
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from flytekit import Resources
    from flytekit.remote import FlyteRemote

from cli.recipe_engine.deployer import ArchitectureDeployer
from cli.recipe_engine.parser import RecipeParser
from cli.recipe_engine.schema import Recipe

logger = logging.getLogger(__name__)

# Pattern matching a string that is *only* a single {{ ... }} reference
_BARE_TEMPLATE = re.compile(r"^\{\{\s*([^}]+)\s*\}\}$")


def _normalize_memory_quantity(value: Optional[str]) -> Optional[str]:
    """Normalize memory quantities to Flyte/K8s-compatible suffixes.

    Recipe profiles commonly use decimal-style units like ``GB`` while Flyte
    resources expect Kubernetes binary units (``Gi``/``Mi``).
    """
    if not value:
        return None

    memory = value.strip()
    suffix_map = {
        "TiB": "Ti",
        "TB": "Ti",
        "GiB": "Gi",
        "GB": "Gi",
        "MiB": "Mi",
        "MB": "Mi",
        "KiB": "Ki",
        "KB": "Ki",
    }

    for suffix, normalized in suffix_map.items():
        if memory.endswith(suffix):
            return f"{memory[:-len(suffix)]}{normalized}"

    return memory


def _build_node_resources(resource_group: Any) -> Optional[Resources]:
    """Build a Flyte ``Resources`` object from a recipe profile resource group."""
    from flytekit import Resources

    resource_kwargs: Dict[str, str] = {}

    cpu = getattr(resource_group, "cpu", None)
    if cpu:
        resource_kwargs["cpu"] = str(cpu)

    memory = _normalize_memory_quantity(getattr(resource_group, "memory", None))
    if memory:
        resource_kwargs["mem"] = memory

    gpu_count = getattr(resource_group, "gpu_count", 0)
    if gpu_count and int(gpu_count) > 0:
        resource_kwargs["gpu"] = str(gpu_count)

    if not resource_kwargs:
        return None

    return Resources(**resource_kwargs)


def _is_not_found_error(exc: Exception) -> bool:
    """Return True if *exc* indicates a missing entity (not a transport/auth error)."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "notfound" in name
        or "notexist" in name
        or "not found" in msg
        or "does not exist" in msg
        or "no matching" in msg
    )


class RecipeRunner:
    """Runner for executing recipe pipelines via Flyte."""

    def __init__(self, recipes_dir: str = ""):
        """Initialize runner with recipes directory.

        Args:
            recipes_dir: Path to directory containing recipe subdirectories.
                         Defaults to projects/recipes/ relative to repo root.
        """
        self.parser = RecipeParser(recipes_dir)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _name_variants(name: str) -> list[str]:
        """Generate name variants for component lookup.

        Components migrated from ``foo.py`` to ``foo/__init__.py`` are
        registered under ``pkg.foo.__init__.func`` instead of ``pkg.foo.func``.
        Components with tasks in ``task.py`` are registered as
        ``pkg.foo.task.func``.  This helper yields all variants so the runner
        can find the component regardless of which naming convention the recipe
        uses.

        Also tries a ``components.`` prefix since ``pyflyte register`` uses
        the on-disk package structure which starts with ``components/``.
        """
        variants = [name]
        parts = name.rsplit(".", 2)
        if len(parts) >= 2:
            pkg, func = parts[-2], parts[-1]
            prefix = ".".join(parts[:-2])
            # foo.func → foo.__init__.func
            init_name = f"{prefix}.{pkg}.__init__.{func}" if prefix else f"{pkg}.__init__.{func}"
            if init_name != name:
                variants.append(init_name)
            # foo.func → foo.task.func
            task_name = f"{prefix}.{pkg}.task.{func}" if prefix else f"{pkg}.task.{func}"
            if task_name != name:
                variants.append(task_name)
            # foo.func → foo._task.func  (legacy)
            legacy_name = f"{prefix}.{pkg}._task.{func}" if prefix else f"{pkg}._task.{func}"
            if legacy_name != name:
                variants.append(legacy_name)
            # foo.__init__.func → foo.func  (reverse)
            if pkg == "__init__" and len(parts) >= 3:
                no_init_name = f"{prefix}.{func}" if prefix else func
                if no_init_name != name:
                    variants.append(no_init_name)
            # foo.task.func or foo._task.func → foo.func  (reverse)
            if pkg in ("task", "_task") and len(parts) >= 3:
                no_task_name = f"{prefix}.{func}" if prefix else func
                if no_task_name != name:
                    variants.append(no_task_name)

        # pyflyte register may use a "components." prefix from the on-disk
        # package layout.  Add prefixed variants for all existing names.
        prefixed = []
        for v in variants:
            if not v.startswith("components."):
                prefixed.append(f"components.{v}")
        variants.extend(prefixed)

        return variants

    def _fetch_component(
        self,
        remote: FlyteRemote,
        name: str,
        project: str,
        domain: str,
        version: Optional[str] = None,
    ):
        """Fetch a task or workflow by name, optionally pinned to a specific version.

        When *version* is provided the runner fetches that exact version from
        the Flyte registry.  Otherwise it falls back to the latest version
        (sorted by ``created_at`` descending).

        Also tries ``__init__``-based name variants for components that were
        migrated from flat files to package directories.

        Args:
            remote: FlyteRemote instance.
            name: Fully-qualified component name (e.g. ``training.finetune``).
            project: Flyte project.
            domain: Flyte domain.
            version: Optional exact version string to fetch.

        Returns:
            FlyteTask or FlyteWorkflow.

        Raises:
            ValueError: If the component cannot be found as a task or workflow.
            Exception: Re-raised for transport/auth/permission errors.
        """
        from flytekit.models.admin.common import Sort
        from flytekit.models.common import NamedEntityIdentifier

        # Sort by created_at descending so the first entry is the newest version
        latest_first = Sort(key="created_at", direction=Sort.Direction.DESCENDING)

        names_to_try = self._name_variants(name)

        # ── Try as task first ────────────────────────────────────────────────
        for candidate_name in names_to_try:
            try:
                if version:
                    # Directly fetch the pinned version — no need to list
                    try:
                        task = remote.fetch_task(
                            project=project,
                            domain=domain,
                            name=candidate_name,
                            version=version,
                        )
                        if candidate_name != name:
                            logger.info("  Resolved '%s' → '%s'", name, candidate_name)
                        logger.info("  Pinned to version %s", version)
                        return task
                    except Exception as exc:
                        if not _is_not_found_error(exc):
                            raise
                        continue
                else:
                    # Latest-version lookup
                    identifier = NamedEntityIdentifier(
                        project=project, domain=domain, name=candidate_name
                    )
                    tasks, _ = remote.client.list_tasks_paginated(
                        identifier,
                        limit=1,
                        sort_by=latest_first,
                    )
                    if tasks:
                        task_id = tasks[0].id
                        if candidate_name != name:
                            logger.info("  Resolved '%s' → '%s'", name, candidate_name)
                        return remote.fetch_task(
                            project=task_id.project,
                            domain=task_id.domain,
                            name=task_id.name,
                            version=task_id.version,
                        )
            except Exception as exc:
                if not _is_not_found_error(exc):
                    raise

        # ── Fall back to workflow ────────────────────────────────────────────
        for candidate_name in names_to_try:
            try:
                if version:
                    try:
                        wf = remote.fetch_workflow(
                            project=project,
                            domain=domain,
                            name=candidate_name,
                            version=version,
                        )
                        if candidate_name != name:
                            logger.info("  Resolved '%s' → '%s'", name, candidate_name)
                        logger.info("  Pinned to version %s", version)
                        return wf
                    except Exception as exc:
                        if not _is_not_found_error(exc):
                            raise
                        continue
                else:
                    identifier = NamedEntityIdentifier(
                        project=project, domain=domain, name=candidate_name
                    )
                    workflows, _ = remote.client.list_workflows_paginated(
                        identifier,
                        limit=1,
                        sort_by=latest_first,
                    )
                    if workflows:
                        wf_id = workflows[0].id
                        if candidate_name != name:
                            logger.info("  Resolved '%s' → '%s'", name, candidate_name)
                        return remote.fetch_workflow(
                            project=wf_id.project,
                            domain=wf_id.domain,
                            name=wf_id.name,
                            version=wf_id.version,
                        )
            except Exception as exc:
                if not _is_not_found_error(exc):
                    raise

        raise ValueError(
            f"Could not fetch component '{name}' as a task or workflow " f"in {project}/{domain}"
        )

    def _topo_sort_steps(self, recipe: Recipe) -> List:
        """Return recipe steps sorted by ``depends_on`` (topological order).

        Steps without dependencies come first. Within the same dependency
        level the original declared order is preserved (Kahn's algorithm).

        Args:
            recipe: Recipe whose steps should be sorted.

        Returns:
            Ordered list of PipelineStep objects.

        Raises:
            ValueError: If a ``depends_on`` references an unknown step name, or
                        if a circular dependency is detected.
        """
        steps_by_name = {step.name: step for step in recipe.pipeline.steps}

        # Validate all depends_on references up-front
        for step in recipe.pipeline.steps:
            for dep in step.depends_on or []:
                if dep not in steps_by_name:
                    raise ValueError(f"Step '{step.name}' depends_on unknown step '{dep}'")

        in_degree: Dict[str, int] = {
            step.name: len(step.depends_on or []) for step in recipe.pipeline.steps
        }
        adjacency: Dict[str, List[str]] = {step.name: [] for step in recipe.pipeline.steps}
        for step in recipe.pipeline.steps:
            for dep in step.depends_on or []:
                adjacency[dep].append(step.name)

        queue = [step.name for step in recipe.pipeline.steps if in_degree[step.name] == 0]
        sorted_names: List[str] = []
        while queue:
            name = queue.pop(0)
            sorted_names.append(name)
            for dependent in adjacency[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(sorted_names) != len(recipe.pipeline.steps):
            raise ValueError("Circular dependency detected in pipeline steps")

        return [steps_by_name[n] for n in sorted_names]

    def _resolve_step_config_typed(
        self,
        step_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve template strings in *step_config* while preserving value types.

        For a config value that is a *bare* ``{{ parameters.x }}`` or
        ``{{ steps.y.outputs.z }}`` reference (i.e. the entire string is a
        single template), the resolved value keeps the same Python type as the
        referenced variable (int, float, bool, list) rather than being coerced
        to ``str`` by the parser.

        Compound strings with embedded templates (e.g.
        ``"prefix-{{ parameters.run_name }}"``) are still returned as ``str``.

        Args:
            step_config: Raw step config dict (may contain template strings).
            context: Template context with ``parameters`` and ``steps`` keys.

        Returns:
            Step config with templates resolved and types preserved.
        """

        def resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                m = _BARE_TEMPLATE.match(value.strip())
                if m:
                    key = m.group(1).strip()
                    parts = key.split(".")
                    result: Any = context
                    try:
                        for part in parts:
                            result = result[part]

                        # In dynamic workflow compilation, `result` might be a Flyte Promise
                        from flytekit.core.promise import Promise

                        if isinstance(result, Promise):
                            return result

                        # Convert FlyteFile/FlyteDirectory to their remote path
                        # so they can be passed as str inputs to downstream tasks
                        if hasattr(result, "remote_source"):
                            return str(result.remote_source)
                        if hasattr(result, "path"):
                            return str(result.path)
                        # Convert dict outputs to JSON strings for cross-task
                        # compatibility (avoids Flyte STRUCT vs BINARY mismatch)
                        if isinstance(result, dict):
                            import json as _json

                            return _json.dumps(result)
                        return result  # raw typed value (int/float/bool/list/...)
                    except (KeyError, TypeError):
                        return value  # unresolved -- keep verbatim
                # Compound string or plain string: fall back to string resolver
                return self.parser._resolve_template_string(value, context)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(v) for v in value]
            return value

        return {k: resolve_value(v) for k, v in step_config.items()}

    # ── public API ────────────────────────────────────────────────────────────

    def resolve_config(
        self,
        recipe: Recipe,
        preset_name: Optional[str] = None,
        param_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve complete configuration for a recipe.

        Args:
            recipe: Recipe object
            preset_name: Optional preset to apply
            param_overrides: Optional parameter overrides (dict of param_name -> value)

        Returns:
            Dict with keys: ``profile``, ``parameters``, ``steps`` (all resolved)
        """
        # Start with preset or defaults
        if preset_name:
            config = self.parser.resolve_preset(recipe, preset_name)
        else:
            first_profile = next(iter(recipe.infrastructure.profiles))
            config = {
                "profile": first_profile,
                "parameters": {k: v.default for k, v in recipe.pipeline.parameters.items()},
                "steps": {step.name: dict(step.config) for step in recipe.pipeline.steps},
            }

        # Apply CLI parameter overrides
        if param_overrides:
            unknown = [k for k in param_overrides if k not in recipe.pipeline.parameters]
            if unknown:
                logger.warning(
                    "Unknown parameter override(s) will be ignored: %s. " "Valid parameters: %s",
                    unknown,
                    list(recipe.pipeline.parameters.keys()),
                )
            for param_name, param_value in param_overrides.items():
                if param_name in recipe.pipeline.parameters:
                    param_def = recipe.pipeline.parameters[param_name]
                    config["parameters"][param_name] = self._convert_param_value(
                        param_value, param_def.type.value
                    )

        # Resolve templates in step configs with type preservation.
        # context["steps"] is empty here because no steps have run yet;
        # {{ steps.<x>.outputs.<y> }} refs that can't be resolved are kept
        # verbatim and re-resolved inside run() as each step completes.
        context = {"parameters": config["parameters"], "steps": {}}
        config["steps"] = {
            step.name: self._resolve_step_config_typed(config["steps"].get(step.name, {}), context)
            for step in recipe.pipeline.steps
        }

        return config

    def _convert_param_value(self, value: Any, param_type: str) -> Any:
        """Convert a parameter value to the appropriate type.

        Args:
            value: The value to convert (typically a string from CLI)
            param_type: The parameter type (string, int, float, bool, list)

        Returns:
            Converted value
        """
        if param_type == "int":
            return int(value)
        elif param_type == "float":
            return float(value)
        elif param_type == "bool":
            if isinstance(value, bool):
                return value
            return value.lower() in ("true", "yes", "1", "y")
        elif param_type == "list":
            if isinstance(value, list):
                return value
            return [v.strip() for v in str(value).split(",")]
        else:  # string
            return str(value)

    def run(
        self,
        recipe_name: str,
        remote: FlyteRemote,
        preset_name: Optional[str] = None,
        param_overrides: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        project: Optional[str] = None,
        domain: Optional[str] = None,
        submission_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        auto_teardown: bool = False,
        overwrite_cache: bool = False,
    ) -> Dict[str, Any]:
        """Execute a recipe pipeline.

        Steps are executed in topological order (respecting ``depends_on``).
        Steps that are prerequisites for later steps are awaited (``wait=True``)
        so their outputs can be captured and substituted into downstream step
        configs via ``{{ steps.<name>.outputs.<x> }}`` references.

        Args:
            recipe_name: Name of the recipe to run
            remote: FlyteRemote instance for workflow submission
            preset_name: Optional preset to use
            param_overrides: Optional parameter overrides
            dry_run: If True, only resolve and display config without submitting
            project: Optional Flyte project override
            domain: Optional Flyte domain override

        Returns:
            Dict with:
                - dry_run mode: ``{"config": resolved_config, "recipe": recipe}``
                - execution mode: ``{"executions": [...], "recipe": recipe,
                                    "config": resolved_config}``

        Raises:
            FileNotFoundError: If recipe not found
            ValueError: If recipe validation fails or a component cannot be located
        """
        recipe = self.parser.load(recipe_name)
        config = self.resolve_config(recipe, preset_name, param_overrides)

        # ── Architecture deployment ──────────────────────────────────────
        deployer = None
        arch_result = None
        proj = project or (remote.default_project if remote else "ml-platform")
        dom = domain or (remote.default_domain if remote else "development")
        has_architecture = recipe.architecture is not None and recipe.architecture.groups
        should_deploy = has_architecture and getattr(recipe.architecture, "auto_deploy", True)

        # Skip architecture deployment when running in colocated mode —
        # external services (reference, reward, buffer) are not needed.
        resolved_params = config.get("parameters", {})
        colocated = resolved_params.get("distributed_colocate_critic_reward", True)
        if should_deploy and colocated:
            logger.info(
                "Colocated mode (distributed_colocate_critic_reward=True); "
                "skipping architecture deployment."
            )
            should_deploy = False

        if should_deploy:
            namespace = f"{proj}-{dom}"
            assert recipe.architecture is not None  # Type narrowing for mypy
            deployer = ArchitectureDeployer(
                recipe_name=recipe_name,
                architecture=recipe.architecture,
                namespace=namespace,
            )
            arch_result = deployer.deploy(dry_run=dry_run)
        elif has_architecture:
            logger.info("Architecture defined but auto_deploy=False; skipping deployment.")

        if dry_run:
            result = {"config": config, "recipe": recipe}
            if arch_result:
                result["architecture"] = arch_result
            return result

        # Topologically sort steps so dependencies always execute first
        ordered_steps = self._topo_sort_steps(recipe)

        import typing

        from flytekit.core.type_engine import TypeEngine
        from flytekit.core.workflow import ImperativeWorkflow

        wf_name = re.sub(r"[^a-zA-Z0-9_]", "_", f"recipe_{recipe_name}")

        # ── Pre-fetch all component tasks from the Flyte registry ────────
        fetched_entities: dict[str, Any] = {}
        for step in ordered_steps:
            comp = step.component
            if comp not in fetched_entities:
                # Resolve version: per-step pin > recipe-level component_versions > latest
                comp_version = step.version
                if comp_version is None and recipe.component_versions:
                    comp_version = recipe.component_versions.get(comp)

                logger.info("Fetching component %s from registry …", comp)
                if comp_version:
                    logger.info("  Version pin: %s", comp_version)
                fetched_entities[comp] = self._fetch_component(
                    remote, comp, proj, dom, version=comp_version
                )
                logger.info(
                    "  → %s (v=%s)",
                    type(fetched_entities[comp]).__name__,
                    getattr(fetched_entities[comp].id, "version", "?"),
                )

        executions = []
        try:
            wf = ImperativeWorkflow(name=wf_name)

            dynamic_wf_inputs_values: dict[str, Any] = {}
            # track workflow inputs created: param_name -> workflow_input_ref
            wf_input_refs: dict[str, Any] = {}

            # 1. Promote explicit recipe parameters to Workflow Inputs
            for param_name, param_def in recipe.pipeline.parameters.items():
                py_type: type = str
                if param_def.type.value == "int":
                    py_type = int
                elif param_def.type.value == "float":
                    py_type = float
                elif param_def.type.value == "bool":
                    py_type = bool
                elif param_def.type.value == "list":
                    py_type = typing.List[typing.Any]  # type: ignore

                safe_name = f"param_{param_name.replace('-', '_')}"
                wf_input_refs[safe_name] = wf.add_workflow_input(safe_name, py_type)
                dynamic_wf_inputs_values[safe_name] = config["parameters"].get(param_name)

            # 2. Iterate through steps, fetch remote entities, and bind Nodes
            ordered_nodes: dict[str, typing.Any] = {}  # step_name -> node_ref

            for step_idx, step in enumerate(ordered_steps):
                task_entity = fetched_entities[step.component]

                # Build a lookup from input name → Python type via TypeEngine
                # so we can promote literal config values as typed workflow inputs.
                task_input_py_types: dict[str, type] = {}
                if hasattr(task_entity, "interface") and task_entity.interface.inputs:
                    for inp_name, var in task_entity.interface.inputs.items():
                        try:
                            task_input_py_types[inp_name] = TypeEngine.guess_python_type(var.type)
                        except Exception:
                            task_input_py_types[inp_name] = str  # safe fallback

                raw_config = dict(step.config)
                node_inputs: dict[str, Any] = {}

                for k, v in raw_config.items():
                    if isinstance(v, str):
                        m = _BARE_TEMPLATE.match(v.strip())
                        if m:
                            key = m.group(1).strip()
                            if key.startswith("steps."):
                                # "steps.load_dataset.outputs.dataset" -> bind to node Promise
                                ref_parts = key.split(".")
                                ref_node_name = ref_parts[1]
                                ref_out = ref_parts[3] if len(ref_parts) > 3 else None
                                ref_node = ordered_nodes[ref_node_name]
                                if ref_out is None:
                                    # No explicit key — use first output
                                    ref_out = list(ref_node.outputs.keys())[0]
                                elif ref_out not in ref_node.outputs:
                                    raise KeyError(
                                        f"Step '{step.name}' references output "
                                        f"'{ref_out}' from step '{ref_node_name}', "
                                        f"but available outputs are: "
                                        f"{list(ref_node.outputs.keys())}"
                                    )
                                node_inputs[k] = ref_node.outputs[ref_out]
                                continue
                            elif key.startswith("parameters."):
                                # "parameters.base_model" -> bind to workflow input Promise
                                param_key = key.split(".")[1]
                                safe_param_name = f"param_{param_key.replace('-', '_')}"
                                node_inputs[k] = wf_input_refs[safe_param_name]
                                continue

                    # Normal primitive value - promote to a Workflow Input with EXACT type
                    resolved_val = self._resolve_step_config_typed(
                        {k: v}, {"parameters": config["parameters"], "steps": {}}
                    )[k]
                    dyn_param_name = f"{step.name.replace('-', '_')}_{k}"

                    # Use the task's registered interface to get the expected Python type.
                    # This avoids type mismatches (e.g. passing int when task expects float).
                    py_type = task_input_py_types.get(k, type(resolved_val))

                    # For Optional[X] types, use the inner type for the workflow input
                    # and allow None values through.
                    origin = getattr(py_type, "__origin__", None)
                    if origin is typing.Union:
                        args = [a for a in typing.get_args(py_type) if a is not type(None)]
                        if args and resolved_val is None:
                            # Keep Optional type so None is accepted
                            pass
                        elif args:
                            py_type = args[0]

                    # Coerce the resolved value to match the expected type
                    if resolved_val is not None and not isinstance(resolved_val, type):
                        try:
                            if py_type is int and not isinstance(resolved_val, int):
                                resolved_val = int(resolved_val)
                            elif py_type is float and not isinstance(resolved_val, float):
                                resolved_val = float(resolved_val)
                            elif py_type is bool and not isinstance(resolved_val, bool):
                                resolved_val = str(resolved_val).lower() in (
                                    "true",
                                    "1",
                                    "yes",
                                )
                            elif py_type is str and not isinstance(resolved_val, str):
                                resolved_val = str(resolved_val)
                        except (ValueError, TypeError):
                            pass  # keep original value

                    # Register as a formal Workflow Input
                    wf_input_refs[dyn_param_name] = wf.add_workflow_input(dyn_param_name, py_type)
                    dynamic_wf_inputs_values[dyn_param_name] = resolved_val

                    # Bind the promise to the new node
                    node_inputs[k] = wf_input_refs[dyn_param_name]

                # Add the remote entity as a DAG node with a meaningful name.
                node = wf.add_entity(task_entity, **node_inputs)
                # ImperativeWorkflow.add_entity() does not accept a node_name
                # parameter (flytekit as of v1.x only accepts **kwargs for
                # task inputs).  The only way to assign a human-readable name
                # is to mutate the private ``_id`` attribute.  This is safe
                # today because ``_id`` is a plain ``str`` on ``Node`` and is
                # only used for display/serialization — it does not affect
                # output bindings (which use the ``Node`` object reference).
                # The broad except clause guards against future flytekit
                # releases that make ``_id`` read-only or change its type.
                try:
                    node._id = step.name.replace("_", "-")
                except Exception:  # noqa: BLE001
                    pass

                # Apply infrastructure profile resources (if configured for this step)
                if step.infra:
                    selected_profile = config.get("profile")
                    profile_groups = recipe.infrastructure.profiles.get(selected_profile, {})
                    resource_group = profile_groups.get(step.infra)
                    if resource_group is None:
                        logger.warning(
                            "Infra group '%s' not found in profile '%s' for step '%s'; "
                            "using task default resources.",
                            step.infra,
                            selected_profile,
                            step.name,
                        )
                    else:
                        node_resources = _build_node_resources(resource_group)
                        if node_resources is not None:
                            node.with_overrides(requests=node_resources, limits=node_resources)
                            logger.info(
                                "  Applied infra '%s' to step '%s' " "(cpu=%s, mem=%s, gpu=%s)",
                                step.infra,
                                step.name,
                                node_resources.cpu,
                                node_resources.mem,
                                node_resources.gpu,
                            )

                ordered_nodes[step.name] = node

                # If this is the last step, set its output as the workflow output
                if step_idx == len(ordered_steps) - 1 and len(node.outputs) > 0:
                    output_key = list(node.outputs.keys())[0]  # first output
                    # For remote entities, python_interface is None, so we must
                    # pass python_type explicitly to add_workflow_output.
                    out_py_type: type = str  # safe default
                    if (
                        hasattr(task_entity, "interface")
                        and task_entity.interface.outputs
                        and output_key in task_entity.interface.outputs
                    ):
                        try:
                            out_py_type = TypeEngine.guess_python_type(
                                task_entity.interface.outputs[output_key].type
                            )
                        except Exception:
                            pass
                    wf.add_workflow_output(
                        output_key, node.outputs[output_key], python_type=out_py_type
                    )

            # 3. Execute the single compiled DAG workflow
            #
            # Human-readable version: same recipe version + profile = same
            # workflow registration.  Flyte reuses an existing version if it
            # matches, so repeated runs skip re-registration.
            profile = config.get("profile", "default")
            wf_version = f"v{recipe.version}-{profile}"

            exec_labels = {
                "mlp-recipe": recipe_name,
                "mlp-version": recipe.version,
                "mlp-profile": profile,
                "mlp-preset": preset_name or "default",
            }

            execute_kwargs = {
                "inputs": dynamic_wf_inputs_values,
                "wait": False,
                "execution_name_prefix": re.sub(r"[^a-z0-9-]", "-", recipe_name)[:24].rstrip("-"),
                "project": proj,
                "domain": dom,
                "version": wf_version,
                "overwrite_cache": overwrite_cache,
            }
            try:
                execution = remote.execute(wf, labels=exec_labels, **execute_kwargs)
            except TypeError as exc:
                if "labels" not in str(exc):
                    raise
                logger.warning(
                    "FlyteRemote.execute() does not support labels;"
                    " submitting recipe without labels."
                )
                execution = remote.execute(wf, **execute_kwargs)

            executions.append(
                {
                    "step_name": "entire_workflow",
                    "execution_id": execution.id.name,
                    "workflow": wf_name,
                    "status": "SUBMITTED",
                }
            )

            if submission_callback:
                submission_callback(
                    {
                        "step_name": wf_name,
                        "execution": execution,
                        "execution_id": execution.id.name,
                        "execution_project": proj,
                        "execution_domain": dom,
                        "remote": remote,
                    }
                )

        finally:
            # NOTE: Architecture resources are NOT torn down here because
            # the Flyte execution runs asynchronously (wait=False). Tearing
            # down services now would break running tasks that depend on them.
            # Use `mlp recipe teardown <name>` after the execution
            # reaches a terminal phase, or set auto_teardown=True to poll.
            if deployer and auto_teardown:
                self._wait_and_teardown(deployer, executions, remote, proj, dom)
            elif deployer:
                logger.info(
                    "Architecture resources left running. "
                    "Run 'mlp recipe teardown %s' when the execution completes.",
                    recipe_name,
                )

        result = {
            "executions": executions,
            "recipe": recipe,
            "config": config,
        }
        if arch_result:
            result["architecture"] = arch_result
        if deployer:
            result["deployer"] = deployer
        return result

    @staticmethod
    def _wait_and_teardown(
        deployer: Any,
        executions: List[Dict[str, Any]],
        remote: Any,
        project: str,
        domain: str,
    ) -> None:
        """Poll until the Flyte execution reaches a terminal phase, then teardown."""
        import time

        if not executions:
            return
        exec_id = executions[0].get("execution_id")
        if not exec_id:
            return

        logger.info("Waiting for execution %s to complete before teardown...", exec_id)
        terminal_phases = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"}
        while True:
            try:
                execution = remote.fetch_execution(name=exec_id, project=project, domain=domain)
                phase = str(execution.closure.phase)
                # Phase may be int enum or string
                phase_name = phase.split(".")[-1] if "." in phase else phase
                if phase_name in terminal_phases:
                    logger.info("Execution %s reached %s — tearing down.", exec_id, phase_name)
                    break
            except Exception as e:
                logger.warning("Error polling execution status: %s", e)
            time.sleep(30)

        try:
            deployer.teardown()
            logger.info("Architecture resources torn down.")
        except Exception as e:
            logger.warning("Failed to teardown architecture: %s", e)

    def get_execution_steps(self, recipe: Recipe) -> List[Dict[str, Any]]:
        """Get ordered list of steps with dependencies resolved.

        Args:
            recipe: Recipe object

        Returns:
            List of step dicts with name, component, infra, depends_on
        """
        steps = []
        for step in recipe.pipeline.steps:
            steps.append(
                {
                    "name": step.name,
                    "component": step.component,
                    "infra": step.infra,
                    "depends_on": step.depends_on or [],
                }
            )
        return steps
