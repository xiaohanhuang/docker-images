"""Recipe parser for loading and validating YAML recipe files.

This module provides:
- YAML loading with Pydantic validation
- Recipe discovery (scanning recipes/ directory)
- Template resolution ({{ parameters.x }} and {{ steps.y.outputs.z }})
- Preset resolution (profile + overrides)
"""

import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from cli.recipe_engine.schema import Recipe


def _default_recipes_dir() -> Path:
    """Return the recipes directory, preferring projects/recipes/ relative to repo root."""
    # Walk up from this file to find repo root (contains pyproject.toml)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            candidate = parent / "projects" / "recipes"
            if candidate.exists():
                return candidate
            break
    return Path("./recipes")


class RecipeParser:
    """Parser for recipe YAML files."""

    def __init__(self, recipes_dir: str = ""):
        """Initialize parser with recipes directory.

        Args:
            recipes_dir: Path to directory containing recipe files/subdirectories.
                         Defaults to projects/recipes/ relative to repo root.
        """
        if recipes_dir:
            p = Path(recipes_dir)
            # If the caller passed the old default and it doesn't exist, auto-resolve
            if not p.exists() and recipes_dir == "./recipes":
                p = _default_recipes_dir()
            self.recipes_dir = p
        else:
            self.recipes_dir = _default_recipes_dir()

    def resolve_path(self, recipe_name: str) -> Path:
        """Resolve the path to a recipe's YAML file.

        Supports both <name>/recipe.yaml and <name>.yaml layouts.
        """
        recipe_path = self.recipes_dir / recipe_name / "recipe.yaml"
        if not recipe_path.exists():
            flat_path = self.recipes_dir / f"{recipe_name}.yaml"
            if flat_path.exists():
                recipe_path = flat_path
            else:
                raise FileNotFoundError(
                    f"Recipe not found: {recipe_name}\n"
                    f"Searched: {self.recipes_dir / recipe_name / 'recipe.yaml'} "
                    f"and {self.recipes_dir / recipe_name}.yaml\n"
                    f"Run 'mlp recipe list' to see available recipes."
                )
        return recipe_path

    def load(self, recipe_name: str) -> Recipe:
        """Load and validate a recipe by name.

        Args:
            recipe_name: Name of the recipe (directory name)

        Returns:
            Validated Recipe object

        Raises:
            FileNotFoundError: If recipe.yaml not found
            ValueError: If YAML is invalid or doesn't match schema
        """
        recipe_path = self.resolve_path(recipe_name)

        try:
            with open(recipe_path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {recipe_path}: {e}")

        if data is None:
            raise ValueError(f"Empty recipe file: {recipe_path}")

        try:
            return Recipe(**data)
        except Exception as e:
            raise ValueError(f"Recipe validation failed for {recipe_name}: {e}")

    def list_recipes(self) -> List[Dict[str, Any]]:
        """List all available recipes in the recipes directory.

        Returns:
            List of recipe metadata dicts with keys: name, version, description, author, tags
        """
        if not self.recipes_dir.exists():
            return []

        recipes = []
        seen: set = set()

        for entry in sorted(self.recipes_dir.iterdir()):
            # Subdirectory layout: <name>/recipe.yaml
            if entry.is_dir() and (entry / "recipe.yaml").exists():
                name = entry.name
            # Flat layout: <name>.yaml
            elif entry.is_file() and entry.suffix == ".yaml":
                name = entry.stem
            else:
                continue

            if name in seen:
                continue
            seen.add(name)

            try:
                recipe = self.load(name)
                recipes.append(
                    {
                        "name": recipe.name,
                        "version": recipe.version,
                        "description": recipe.description,
                        "author": recipe.author,
                        "tags": recipe.tags,
                    }
                )
            except Exception:
                # Fall back to raw YAML metadata for non-engine-schema recipes
                try:
                    yaml_path = entry / "recipe.yaml" if entry.is_dir() else entry
                    with open(yaml_path) as f:
                        raw = yaml.safe_load(f) or {}
                    if "name" in raw:
                        recipes.append(
                            {
                                "name": raw.get("name", name),
                                "version": str(raw.get("version", "?")),
                                "description": raw.get("description", ""),
                                "author": raw.get("author", ""),
                                "tags": raw.get("tags", []),
                            }
                        )
                except Exception:
                    continue

        return recipes

    def resolve_preset(self, recipe: Recipe, preset_name: str) -> Dict[str, Any]:
        """Resolve a preset into a concrete configuration.

        Args:
            recipe: Recipe object
            preset_name: Name of the preset to resolve

        Returns:
            Dict with keys: profile, parameters, steps (resolved config)

        Raises:
            ValueError: If preset not found
        """
        if preset_name not in recipe.presets:
            available = ", ".join(recipe.presets.keys())
            raise ValueError(f"Unknown preset: {preset_name}\n" f"Available presets: {available}")

        preset = recipe.presets[preset_name]
        profile = preset.profile

        # Start with default parameters
        config = {
            "profile": profile,
            "parameters": {k: v.default for k, v in recipe.pipeline.parameters.items()},
            "steps": {step.name: dict(step.config) for step in recipe.pipeline.steps},
        }

        # Apply preset overrides
        for key, value in preset.overrides.items():
            parts = key.split(".")
            target = config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value

        return config

    def resolve_templates(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve {{ parameters.x }} and {{ steps.y.outputs.z }} templates.

        Args:
            config: Configuration dict with template strings
            context: Context dict with 'parameters' and 'steps' keys

        Returns:
            Configuration with resolved templates
        """

        def resolve_value(value: Any) -> Any:
            """Recursively resolve templates in values."""
            if isinstance(value, str):
                return self._resolve_template_string(value, context)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(v) for v in value]
            else:
                return value

        return resolve_value(config)

    def _resolve_template_string(self, template: str, context: Dict[str, Any]) -> str:
        """Resolve a single template string like '{{ parameters.x }}'.

        Args:
            template: String potentially containing {{ }} templates
            context: Context dict for variable lookup

        Returns:
            String with templates resolved
        """
        pattern = r"\{\{\s*([^}]+)\s*\}\}"

        def replace(match):
            key = match.group(1).strip()
            parts = key.split(".")
            value = context
            try:
                for part in parts:
                    value = value[part]
                return str(value)
            except (KeyError, TypeError):
                # If template can't be resolved, leave it as-is
                # (will be resolved at runtime by Flyte)
                return match.group(0)

        return re.sub(pattern, replace, template)

    def apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter overrides to a configuration.

        Args:
            config: Base configuration
            overrides: Dict of dotted-path keys to override values

        Returns:
            Configuration with overrides applied

        Example:
            overrides = {"parameters.epochs": 10, "steps.train.config.lr": 0.001}
        """
        result = dict(config)
        for key, value in overrides.items():
            parts = key.split(".")
            target = result
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        return result
