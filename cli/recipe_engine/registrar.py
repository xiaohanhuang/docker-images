"""Engine for registering all components referenced by a recipe."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console

from cli.commands.component import COMPONENTS_ROOT, _ensure_project, _register_one
from cli.recipe_engine.parser import RecipeParser

console = Console()


class ComponentRegistrar:
    """Orchestrates registration of all components referenced by a recipe."""

    def __init__(self, recipes_dir: str):
        self.parser = RecipeParser(recipes_dir)

    def register_all(
        self,
        recipe_name: str,
        project: str,
        domain: str,
        auto_bump: bool = True,
    ) -> Tuple[int, int, Dict[str, str]]:
        """
        Register all components out of a recipe.

        Returns:
            (successes, failures, bumped_versions_dict)
        """
        try:
            recipe = self.parser.load(recipe_name)
        except Exception as exc:
            raise ValueError(f"Failed to load recipe {recipe_name}: {exc}") from exc

        if not recipe.component_versions:
            raise ValueError(
                f"Recipe '{recipe_name}' has no component_versions defined. Nothing to register."
            )

        try:
            _ensure_project(project)
        except Exception as exc:
            raise RuntimeError(f"Flyte project verification failed: {exc}") from exc

        targets: List[Tuple[Path, str, str]] = []
        missing: List[str] = []

        for comp_key, comp_version in recipe.component_versions.items():
            normalized_key = comp_key
            if normalized_key.startswith("components."):
                normalized_key = normalized_key[len("components.") :]
            parts = normalized_key.split(".")
            if len(parts) >= 2:
                comp_dir = COMPONENTS_ROOT / parts[0] / parts[1]
            else:
                comp_dir = COMPONENTS_ROOT / parts[0]

            if comp_dir.is_dir():
                targets.append((comp_dir, comp_key, comp_version))
            else:
                missing.append(f"{comp_key} -> {comp_dir}")

        if missing:
            console.print("[yellow]Warning: some component directories not found:[/yellow]")
            for m in missing:
                console.print(f"  [dim]{m}[/dim]")
            console.print()

        if not targets:
            raise ValueError("No component directories found to register.")

        successes = 0
        failures = 0
        bumped: Dict[str, str] = {}

        for comp_dir, comp_name, comp_version in targets:
            ok, resolved_ver = _register_one(
                comp_dir,
                comp_name,
                project,
                domain,
                comp_version,
                image=None,
                auto_bump=auto_bump,
            )
            if ok:
                successes += 1
                if resolved_ver and resolved_ver != comp_version:
                    bumped[comp_name] = resolved_ver
            else:
                failures += 1

        return successes, failures, bumped

    def update_recipe_yaml(self, recipe_name: str, bumped: Dict[str, str]) -> None:
        """Update the component_versions section of a recipe YAML file on disk."""
        if not bumped:
            return

        recipe = self.parser.load(recipe_name)
        try:
            recipe_path = self.parser.resolve_path(recipe_name)
        except FileNotFoundError:
            return

        text = recipe_path.read_text()
        for comp_key, new_ver in bumped.items():
            old_ver = recipe.component_versions.get(comp_key)
            if old_ver:
                # Replace the value safely, preserving quotes if any
                # Matches: comp_key: <spaces> "old_ver" or 'old_ver' or old_ver
                pattern = rf"({re.escape(comp_key)}:[\s]*)[\"']?{re.escape(old_ver)}[\"']?"
                text = re.sub(pattern, rf"\g<1>\"{new_ver}\"", text, count=1)
        recipe_path.write_text(text)
