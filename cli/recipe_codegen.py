"""
Recipe engine core logic for the ML Platform.

Handles template resolution (Jinja2), workflow generation, and execution.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

from jinja2 import Environment, StrictUndefined

from cli.recipe_parser import Recipe, RecipeStep

# Map JSON Schema / YAML schema type names to Python type annotations
_SCHEMA_TYPE_MAP: Dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "float": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


class RecipeEngine:
    """
    Core engine for recipe execution.

    Responsibilities:
    - Resolve Jinja2 templates in recipe definitions
    - Generate Flyte workflow code dynamically
    - Submit workflows to Flyte for execution
    """

    def __init__(self, recipe: Optional[Recipe] = None):
        """
        Initialize the recipe engine.

        Args:
            recipe: Optional recipe definition for default parameter resolution
        """
        # Use StrictUndefined so missing variables raise an error immediately at render time
        self.jinja_env = Environment(undefined=StrictUndefined)
        self.recipe = recipe

    def resolve_templates(
        self, recipe: Recipe, params: Dict[str, Any], step_outputs: Optional[Dict[str, Any]] = None
    ) -> List[RecipeStep]:
        """
        Resolve Jinja2 templates in recipe step inputs.

        Args:
            recipe: Recipe definition
            params: User-provided parameters
            step_outputs: Outputs from previous steps (for multi-step resolution)

        Returns:
            List of steps with resolved input values
        """
        if step_outputs is None:
            step_outputs = {}

        # Build template context - include defaults from recipe parameters
        context = {
            "params": {},
            "parameters": {},
            "steps": step_outputs,
        }

        # Add all parameters with their defaults
        for param_name, param_def in recipe.parameters.items():
            if param_name in params:
                context["params"][param_name] = params[param_name]
                context["parameters"][param_name] = params[param_name]
            elif param_def.default is not None:
                context["params"][param_name] = param_def.default
                context["parameters"][param_name] = param_def.default
            else:
                # Use None for optional parameters without defaults
                context["params"][param_name] = None
                context["parameters"][param_name] = None

        resolved_steps = []

        for step in recipe.steps:
            resolved_step = RecipeStep(
                id=step.id,
                component=step.component,
                description=step.description,
                depends_on=step.depends_on,
                outputs=step.outputs,
                profile=step.profile,
                inputs={},
            )

            # Resolve each input template
            for input_name, input_value in step.inputs.items():
                if isinstance(input_value, str):
                    # Check if it references a previous step output
                    if input_value.startswith("{{") and "steps." in input_value:
                        # Keep the template for workflow generation phase
                        resolved_step.inputs[input_name] = input_value
                    # Check if it contains Jinja2 template syntax for parameters
                    elif "{{" in input_value and "}}" in input_value:
                        try:
                            template = self.jinja_env.from_string(input_value)
                            resolved_value = template.render(context)

                            # Try to convert to appropriate type
                            resolved_value = self._coerce_type(resolved_value)

                            # Skip if resolved to None (optional parameter not provided)
                            if resolved_value is not None and resolved_value != "None":
                                resolved_step.inputs[input_name] = resolved_value
                        except Exception:
                            # If template resolution fails, keep the original value
                            # (might be an optional parameter)
                            resolved_step.inputs[input_name] = input_value
                    else:
                        resolved_step.inputs[input_name] = input_value
                else:
                    resolved_step.inputs[input_name] = input_value

            resolved_steps.append(resolved_step)

        return resolved_steps

    def _coerce_type(self, value: str) -> Any:
        """
        Try to convert string value to appropriate Python type.

        Args:
            value: String value to convert

        Returns:
            Converted value (int, float, bool, or str)
        """
        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try bool
        if value.lower() in ("true", "yes", "1"):
            return True
        elif value.lower() in ("false", "no", "0"):
            return False

        # Return as string
        return value

    def generate_workflow_code(
        self, recipe: Recipe, resolved_steps: List[RecipeStep], params: Dict[str, Any]
    ) -> str:
        """
        Generate Python code for a Flyte workflow from the recipe.

        Args:
            recipe: Recipe definition
            resolved_steps: Steps with resolved input templates
            params: User-provided parameters

        Returns:
            Python code as a string
        """
        # Start building the workflow code
        lines = []

        # Imports
        lines.append('"""')
        lines.append(f"Auto-generated Flyte workflow for recipe: {recipe.name}")
        lines.append(f"Description: {recipe.description}")
        lines.append('"""')
        lines.append("")
        lines.append("from flytekit import workflow")
        lines.append("")

        # Import components
        component_imports = set()
        for step in resolved_steps:
            # Parse component name: "data.hf_dataset_loader" -> (data, hf_dataset_loader)
            parts = step.component.split(".")
            if len(parts) == 2:
                module, func = parts
                component_imports.add((module, func))

        lines.append("# Component imports")
        for module, func in sorted(component_imports):
            lines.append(f"from projects.components.components.{module}.{func} import {func}")

        lines.append("")
        lines.append("")

        # Generate workflow function
        lines.append("@workflow")
        lines.append(f'def {recipe.name.replace("-", "_")}_workflow(')

        # Add workflow parameters
        py_type = _SCHEMA_TYPE_MAP  # local alias for brevity
        param_lines = []
        for param_name, param_def in recipe.parameters.items():
            type_hint = py_type.get(param_def.type, param_def.type)
            if param_name in params:
                # Use provided value as default; use repr() so strings with
                # quotes or backslashes produce valid Python literals.
                # If the schema type is numeric but the CLI sent a string
                # (e.g. "2e-4"), coerce it so the generated code is valid.
                value = params[param_name]
                if isinstance(value, str) and type_hint in ("float", "int"):
                    coerced = self._coerce_type(value)
                    if isinstance(coerced, (int, float)):
                        value = coerced
                if isinstance(value, bool):
                    param_lines.append(f"    {param_name}: bool = {value},")
                elif isinstance(value, int):
                    param_lines.append(f"    {param_name}: int = {value},")
                elif isinstance(value, float):
                    param_lines.append(f"    {param_name}: float = {value},")
                elif isinstance(value, str):
                    param_lines.append(f"    {param_name}: str = {repr(value)},")
                else:
                    param_lines.append(f"    {param_name} = {repr(value)},")
            elif param_def.default is not None:
                # Use default from schema.  PyYAML cannot parse bare scientific
                # notation (e.g. "2e-4") as a float, so string defaults that
                # belong to a numeric type must be coerced before emitting code.
                default_val = param_def.default
                if isinstance(default_val, str) and type_hint in ("float", "int"):
                    coerced = self._coerce_type(default_val)
                    if isinstance(coerced, (int, float)):
                        default_val = coerced
                if isinstance(default_val, str):
                    param_lines.append(f"    {param_name}: {type_hint} = {repr(default_val)},")
                else:
                    param_lines.append(f"    {param_name}: {type_hint} = {default_val},")

        lines.extend(param_lines)
        lines.append("):")
        lines.append('    """')
        lines.append(f"    {recipe.description}")
        lines.append('    """')

        # Build a map for quick step lookup (used for output-count checks)
        step_output_map = {s.id: s.outputs for s in resolved_steps}

        # Generate workflow body
        for step in resolved_steps:
            lines.append("")
            lines.append(f"    # Step: {step.id}")
            if step.description:
                lines.append(f"    # {step.description}")

            # Get component function name
            component_func = step.component.split(".")[-1]

            # Build function call
            func_call = f"    {step.id}_result = {component_func}("

            # Add inputs
            input_strs = []
            for input_name, input_value in step.inputs.items():
                # Check if input references a previous step output
                is_step_ref = (
                    isinstance(input_value, str)
                    and input_value.startswith("{{")
                    and "steps." in input_value
                )
                if is_step_ref:
                    # Extract step reference: {{ steps.load_dataset.instruction_data }}
                    # Remove {{ }} and extract the path
                    ref = input_value.strip().replace("{{", "").replace("}}", "").strip()
                    parts = ref.split(".")

                    if len(parts) >= 2 and parts[0] == "steps":
                        prev_step_id = parts[1]

                        # If no output field specified, reference the result directly
                        if len(parts) == 2:
                            input_strs.append(f"{input_name}={prev_step_id}_result")
                        else:
                            # Determine how to access the output: tasks with a single
                            # return value (e.g. FlyteFile) return the value directly,
                            # while multi-output tasks return a named tuple.
                            prev_outputs = step_output_map.get(prev_step_id, {})
                            output_field = parts[2]
                            if len(prev_outputs) == 1:
                                # Single-output task — result is the value itself
                                input_strs.append(f"{input_name}={prev_step_id}_result")
                            else:
                                # Multi-output task — access by attribute name
                                input_strs.append(
                                    f"{input_name}={prev_step_id}_result.{output_field}"
                                )
                else:
                    # Use the resolved value; use repr() for strings to get a
                    # valid Python literal even when the value contains quotes.
                    if isinstance(input_value, str):
                        input_strs.append(f"{input_name}={repr(input_value)}")
                    elif isinstance(input_value, list):
                        input_strs.append(f"{input_name}={input_value}")
                    else:
                        input_strs.append(f"{input_name}={input_value}")

            if input_strs:
                func_call += "\n        " + ",\n        ".join(input_strs) + "\n    "

            func_call += ")"

            lines.append(func_call)

        # Return final result
        if resolved_steps:
            final_step = resolved_steps[-1]
            lines.append("")
            lines.append(f"    return {final_step.id}_result")

        lines.append("")

        return "\n".join(lines)

    def build_dependency_graph(self, steps: List[RecipeStep]) -> Dict[str, List[str]]:
        """
        Build a dependency graph from recipe steps.

        Args:
            steps: List of recipe steps

        Returns:
            Dictionary mapping step ID to list of dependent step IDs
        """
        graph = {step.id: step.depends_on for step in steps}
        return graph

    def topological_sort(self, steps: List[RecipeStep]) -> List[RecipeStep]:
        """
        Sort steps in topological order (respecting dependencies).

        Args:
            steps: List of recipe steps

        Returns:
            Steps sorted in execution order

        Raises:
            ValueError: If circular dependencies detected
        """
        graph = self.build_dependency_graph(steps)
        step_map = {step.id: step for step in steps}

        # Kahn's algorithm for topological sort
        in_degree = {step_id: len(deps) for step_id, deps in graph.items()}

        queue = deque(step_id for step_id, degree in in_degree.items() if degree == 0)
        sorted_steps = []

        while queue:
            step_id = queue.popleft()
            sorted_steps.append(step_map[step_id])

            # Find all steps that depend on this step
            for other_id, deps in graph.items():
                if step_id in deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(sorted_steps) != len(steps):
            raise ValueError("Circular dependency detected in recipe steps")

        return sorted_steps
