"""Pydantic models for the legacy recipe YAML format.

These models define the structure of recipe YAML files used by the
recipe codegen engine, including infrastructure profiles, pipeline
steps with Jinja2 template support, and parameter validation rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class RecipeProfile(BaseModel):
    """Infrastructure profile for a recipe."""

    model_size: Optional[str] = None
    actor_gpus: Optional[int] = None
    critic_gpus: Optional[int] = None
    reference_gpus: Optional[int] = None
    reference_shared: bool = False
    reward_shared: bool = False
    instance_type: str
    total_gpus: int
    estimated_cost_per_hour: float
    enable_efa: bool = False


class RecipeStep(BaseModel):
    """A single step in the recipe pipeline."""

    id: str
    component: str
    description: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)
    profile: Optional[str] = None


class RecipeParameter(BaseModel):
    """Parameter definition with validation rules."""

    type: str
    required: bool = False
    default: Optional[Any] = None
    description: Optional[str] = None
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None
    examples: Optional[List[Any]] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        valid_types = ["string", "integer", "float", "boolean", "array", "object"]
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}")
        return v


class Recipe(BaseModel):
    """Complete recipe definition."""

    name: str
    version: str
    description: str
    complexity: str = Field(default="medium")
    min_gpus: int = Field(default=0)
    max_gpus: int = Field(default=1)
    profiles: Dict[str, RecipeProfile]
    steps: List[RecipeStep]
    parameters: Dict[str, RecipeParameter]
    resources: Optional[Dict[str, Any]] = None
    estimated_runtime: Optional[Dict[str, int]] = None
    success_criteria: Optional[Dict[str, float]] = None
    tags: List[str] = Field(default_factory=list)

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v):
        valid = ["low", "medium", "high"]
        if v not in valid:
            raise ValueError(f"complexity must be one of {valid}")
        return v


class RecipeParser:
    """Parser for recipe YAML files."""

    def __init__(self, recipes_dir: Optional[str] = None):
        if recipes_dir is None:
            cli_dir = Path(__file__).parent
            repo_root = cli_dir.parent
            recipes_dir = repo_root / "projects" / "recipes"
        self.recipes_dir = Path(recipes_dir)

    def parse(self, recipe_name: str) -> Recipe:
        recipe_path = self.recipes_dir / f"{recipe_name}.yaml"
        if not recipe_path.exists():
            raise FileNotFoundError(
                f"Recipe '{recipe_name}' not found at {recipe_path}. "
                f"Available recipes: {self.list_recipes()}"
            )
        with open(recipe_path, "r") as f:
            raw_data = yaml.safe_load(f)
        try:
            recipe = Recipe(**raw_data)
        except Exception as e:
            raise ValueError(f"Failed to parse recipe '{recipe_name}': {e}")
        return recipe

    def list_recipes(self) -> List[str]:
        if not self.recipes_dir.exists():
            return []
        recipes = []
        for path in self.recipes_dir.glob("*.yaml"):
            if path.name != "README.md":
                recipes.append(path.stem)
        return sorted(recipes)

    def validate_parameters(self, recipe: Recipe, params: Dict[str, Any]) -> List[str]:
        errors = []
        for param_name, param_def in recipe.parameters.items():
            if param_def.required and param_name not in params:
                errors.append(f"Required parameter '{param_name}' is missing")

        for param_name, param_value in params.items():
            if param_name not in recipe.parameters:
                errors.append(f"Unknown parameter '{param_name}'")
                continue
            param_def = recipe.parameters[param_name]
            if param_def.type == "string" and not isinstance(param_value, str):
                errors.append(f"Parameter '{param_name}' must be a string")
            elif param_def.type == "integer" and not isinstance(param_value, int):
                errors.append(f"Parameter '{param_name}' must be an integer")
            elif param_def.type == "float" and not isinstance(param_value, (int, float)):
                errors.append(f"Parameter '{param_name}' must be a number")
            elif param_def.type == "boolean" and not isinstance(param_value, bool):
                errors.append(f"Parameter '{param_name}' must be a boolean")
            if param_def.enum and param_value not in param_def.enum:
                errors.append(
                    f"Parameter '{param_name}' must be one of {param_def.enum}, "
                    f"got '{param_value}'"
                )
            if param_def.minimum is not None and isinstance(param_value, (int, float)):
                if param_value < param_def.minimum:
                    errors.append(f"Parameter '{param_name}' must be >= {param_def.minimum}")
            if param_def.maximum is not None and isinstance(param_value, (int, float)):
                if param_value > param_def.maximum:
                    errors.append(f"Parameter '{param_name}' must be <= {param_def.maximum}")
            if param_def.pattern and isinstance(param_value, str):
                import re

                if not re.match(param_def.pattern, param_value):
                    errors.append(
                        f"Parameter '{param_name}' must match pattern '{param_def.pattern}'"
                    )
        return errors
