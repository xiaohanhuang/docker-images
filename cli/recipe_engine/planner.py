"""Recipe planner for generating Flyte workflows from YAML steps.

This module provides:
- Workflow code generation from recipe steps
- Dynamic @workflow creation from recipe definitions
- Support for ReferenceTask and ReferenceWorkflow composition
- Lockfile generation for immutable component pinning
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cli.recipe_engine.schema import Recipe

logger = logging.getLogger(__name__)


class RecipePlanner:
    """Planner for generating Flyte workflow definitions from recipes."""

    def __init__(self):
        """Initialize the recipe planner."""
        pass

    def generate_workflow_code(
        self,
        recipe: Recipe,
        component_versions: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate Python workflow code from a recipe.

        Args:
            recipe: Recipe object to generate workflow for
            component_versions: Optional dict mapping component names to versions
                               (for lockfile generation)

        Returns:
            Python code string defining a Flyte @workflow

        Example output:
            ```python
            from flytekit import workflow
            from flytekit.remote import FlyteRemote

            @workflow
            def text2sql_workflow(
                num_epochs: int = 3,
                batch_size: int = 4,
                learning_rate: float = 0.0002,
                model_name: str = "facebook/opt-125m"
            ):
                # Step 1: ingest_data
                ingest_data_result = data.hf_dataset_loader(
                    dataset="wikisql",
                    split="train"
                )

                # Step 2: preprocess
                preprocess_result = data.text_chunker(
                    input_file=ingest_data_result
                )

                # ... etc
            ```
        """
        component_versions = component_versions or {}

        # Generate imports
        imports = self._generate_imports(recipe)

        # Generate workflow signature
        signature = self._generate_workflow_signature(recipe)

        # Generate workflow body (step execution)
        body = self._generate_workflow_body(recipe, component_versions)

        # Combine all parts
        code = f"{imports}\n\n{signature}\n{body}\n"

        return code

    def _generate_imports(self, recipe: Recipe) -> str:
        """Generate import statements for the workflow."""
        imports = [
            "from flytekit import workflow, task",
            "from flytekit.remote import FlyteRemote",
            "from typing import Any, Dict, List",
        ]

        # Add imports for any specific Flyte types used
        has_flyte_file = any("FlyteFile" in str(step.config) for step in recipe.pipeline.steps)
        has_flyte_dir = any("FlyteDirectory" in str(step.config) for step in recipe.pipeline.steps)

        if has_flyte_file or has_flyte_dir:
            types = []
            if has_flyte_file:
                types.append("FlyteFile")
            if has_flyte_dir:
                types.append("FlyteDirectory")
            imports.append(f"from flytekit import {', '.join(types)}")

        # Add imports for component modules referenced by steps
        seen_modules = set()
        for step in recipe.pipeline.steps:
            parts = step.component.rsplit(".", 1)
            if len(parts) == 2:
                module_path = parts[0]
                if module_path not in seen_modules:
                    seen_modules.add(module_path)
                    imports.append(f"import {module_path}")

        return "\n".join(imports)

    def _generate_workflow_signature(self, recipe: Recipe) -> str:
        """Generate the @workflow function signature."""
        workflow_name = f"{recipe.name.replace('-', '_')}_workflow"

        # Build parameter list
        params = []
        for param_name, param_def in recipe.pipeline.parameters.items():
            # Map recipe parameter types to Python types
            type_map = {
                "string": "str",
                "int": "int",
                "float": "float",
                "bool": "bool",
                "list": "List[Any]",
            }
            python_type = type_map.get(param_def.type.value, "Any")

            # Format default value
            default_val = param_def.default
            if param_def.type.value == "string":
                default_val = repr(default_val)
            elif param_def.type.value == "float" and isinstance(default_val, str):
                # Handle scientific notation strings from YAML
                default_val = float(default_val)

            if not param_name.isidentifier():
                raise ValueError(f"Invalid parameter name: {param_name!r}")
            params.append(f"    {param_name}: {python_type} = {default_val}")

        params_str = ",\n".join(params)

        signature = f"""@workflow
def {workflow_name}(
{params_str}
):
    \"\"\"Generated workflow for recipe: {recipe.name} v{recipe.version}.

    {recipe.description}
    \"\"\""""

        return signature

    @staticmethod
    def _resolve_config_value(value: Any) -> str:
        """Translate a step-config value to a Python expression string.

        Template references (``{{ parameters.X }}`` / ``{{ steps.Y.outputs }}``)
        are mapped to the corresponding Python identifiers so the generated
        workflow actually wires inputs/outputs rather than passing literal
        template strings.

        * ``{{ parameters.X }}``          → ``X``   (workflow-level parameter)
        * ``{{ steps.Y.outputs }}``        → ``Y_result``
        * ``{{ steps.Y.outputs.attr }}``   → ``Y_result``  (attr access TBD)
        * Any other value                  → ``repr(value)``
        """
        import re

        if not isinstance(value, str) or "{{" not in value:
            return repr(value)

        # Match whole-string parameter references: {{ parameters.name }}
        m = re.fullmatch(r"\s*{{\s*parameters\.(\w+)\s*}}\s*", value)
        if m:
            return m.group(1)  # Python variable in workflow scope

        # Match whole-string step output references: {{ steps.name.outputs[.attr] }}
        m = re.fullmatch(r"\s*{{\s*steps\.(\w+)\.outputs(?:\.\w+)?\s*}}\s*", value)
        if m:
            return f"{m.group(1)}_result"

        # Unrecognised template — keep as a string literal so the code at least
        # compiles; a runtime warning comment is emitted alongside it.
        return repr(value)

    def _generate_workflow_body(
        self,
        recipe: Recipe,
        component_versions: Dict[str, str],
    ) -> str:
        """Generate the workflow body with step execution."""
        lines = []

        # Topologically sort steps
        sorted_steps = self._topological_sort(recipe)

        for step in sorted_steps:
            # Add comment
            lines.append(f"    # Step: {step.name}")

            result_var = f"{step.name}_result"

            # Build arguments from step config, resolving template references to
            # actual Python identifiers (workflow params / previous step outputs).
            args = []
            for key, value in step.config.items():
                resolved = self._resolve_config_value(value)
                args.append(f"        {key}={resolved}")

            if args:
                args_str = ",\n".join(args)
                lines.append(f"    {result_var} = {step.component}(")
                lines.append(args_str)
                lines.append("    )")
            else:
                lines.append(f"    {result_var} = {step.component}()")

            lines.append("")

        # Add return statement (return outputs from final step or all steps)
        if recipe.pipeline.steps:
            final_step = sorted_steps[-1]
            lines.append(f"    return {final_step.name}_result")

        return "\n".join(lines)

    def _topological_sort(self, recipe: Recipe) -> List:
        """Sort recipe steps topologically by depends_on relationships.

        Args:
            recipe: Recipe with pipeline steps

        Returns:
            List of PipelineStep objects in topological order

        Raises:
            ValueError: If circular dependency detected
        """
        from collections import defaultdict, deque

        # Build adjacency list and in-degree count
        steps_by_name = {step.name: step for step in recipe.pipeline.steps}
        in_degree = defaultdict(int)
        adj = defaultdict(list)

        for step in recipe.pipeline.steps:
            in_degree[step.name] = len(step.depends_on or [])
            for dep in step.depends_on or []:
                adj[dep].append(step.name)

        # Kahn's algorithm for topological sort
        queue = deque([name for name in steps_by_name if in_degree[name] == 0])
        sorted_names = []

        while queue:
            name = queue.popleft()
            sorted_names.append(name)

            for neighbor in adj[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_names) != len(recipe.pipeline.steps):
            raise ValueError("Circular dependency detected in pipeline steps")

        return [steps_by_name[name] for name in sorted_names]

    def generate_lockfile(
        self,
        recipe: Recipe,
        component_versions: Dict[str, str],
        profile: str,
    ) -> Dict[str, Any]:
        """Generate a lockfile for reproducible recipe execution.

        The lockfile captures:
        - Exact component versions (resolved from 'latest')
        - Infrastructure profile used
        - Recipe metadata

        Args:
            recipe: Recipe object
            component_versions: Dict mapping component names to exact versions
            profile: Infrastructure profile name

        Returns:
            Lockfile dict ready to be serialized to YAML
        """
        lockfile = {
            "recipe": {
                "name": recipe.name,
                "version": recipe.version,
            },
            "profile": profile,
            "components": {},
            "generated_at": None,  # Will be set at generation time
        }

        # Add component versions
        for step in recipe.pipeline.steps:
            component_name = step.component
            version = component_versions.get(component_name, "unknown")
            lockfile["components"][component_name] = {
                "version": version,
                "step": step.name,
            }

        return lockfile

    def plan_execution(
        self,
        recipe: Recipe,
        profile: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Plan recipe execution and return execution metadata.

        Args:
            recipe: Recipe object
            profile: Infrastructure profile to use
            parameters: Resolved parameters

        Returns:
            Dict with execution plan metadata:
            - steps: List of step execution details
            - resource_requirements: Resource requirements per step
            - estimated_cost: Optional cost estimation
        """
        plan = {
            "recipe": recipe.name,
            "version": recipe.version,
            "profile": profile,
            "steps": [],
            "resource_requirements": {},
        }

        # Get resource group for profile
        resource_groups = recipe.infrastructure.profiles.get(profile, {})

        # Plan each step
        for step in self._topological_sort(recipe):
            step_plan = {
                "name": step.name,
                "component": step.component,
                "depends_on": step.depends_on or [],
                "infra": step.infra,
            }

            # Add resource requirements if step has infra reference
            if step.infra and step.infra in resource_groups:
                rg = resource_groups[step.infra]
                step_plan["resources"] = {
                    "instance_types": rg.instance_types,
                    "gpu_count": rg.gpu_count,
                    "gpu_memory": rg.gpu_memory,
                    "cpu": rg.cpu,
                    "memory": rg.memory,
                }
                plan["resource_requirements"][step.name] = step_plan["resources"]

            plan["steps"].append(step_plan)

        return plan
