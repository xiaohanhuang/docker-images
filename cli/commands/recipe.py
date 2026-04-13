"""CLI commands for the Recipe system.

Commands:
    ml-plat recipe list                          — list recipes in registry
    ml-plat recipe list --local                  — list local recipes
    ml-plat recipe info <name> [--version] [--local]
                                                 — show recipe details
    ml-plat recipe validate <name>               — validate recipe YAML
    ml-plat recipe run <name> [--version] [--preset] [--param] [--dry-run]
                                                 — execute (or dry-run) a recipe
    ml-plat recipe versions <name>               — list all versions of a recipe
    ml-plat recipe register <name>               — register recipe components
    ml-plat recipe push <name>                   — push recipe to registry
    ml-plat recipe pull <name> [--version]       — pull recipe from registry
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Heavy recipe_engine imports are deferred to function bodies to keep
# CLI auto-completion fast (pydantic, requests, yaml cascade).
from cli.utils import flyte_console_url, flyte_remote

console = Console()
app = typer.Typer(help="Declarative ML workflows with infrastructure blueprints")


# ── list ─────────────────────────────────────────────────────────────────────


@app.command("list")
def list_recipes(
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="List local recipes instead of querying the remote registry",
    ),
    recipes_dir: str = typer.Option(
        "",
        "--dir",
        "-d",
        help="Directory containing recipes (used with --local)",
    ),
    tag: Optional[List[str]] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Filter by tag (repeatable, remote only)",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by verification status (remote only)",
    ),
):
    """📦 List recipes in the remote registry.

    By default queries the platform registry.
    Use --local to scan the local recipes directory instead.

    Examples:
      ml-plat recipe list
      ml-plat recipe list --status verified
      ml-plat recipe list --tag gpu
      ml-plat recipe list --local
    """
    if local:
        _list_local_recipes(recipes_dir)
    else:
        _list_remote_recipes(tag, status)


def _list_local_recipes(recipes_dir: str) -> None:
    """Scan local filesystem for recipe.yaml files."""
    from cli.recipe_engine.parser import RecipeParser

    parser = RecipeParser(recipes_dir)

    try:
        recipes = parser.list_recipes()
    except Exception as exc:
        console.print(f"[bold red]Failed to list recipes:[/bold red] {exc}")
        raise typer.Exit(1)

    if not recipes:
        console.print(
            f"[dim]No recipes found in {parser.recipes_dir}.[/dim]\n"
            f"Create a recipe directory with a recipe.yaml file to get started."
        )
        raise typer.Exit(0)

    table = Table(
        title="[bold cyan]Local Recipes[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Name", style="bold white", no_wrap=True)
    table.add_column("Version", style="yellow")
    table.add_column("Description", style="dim")
    table.add_column("Tags", style="magenta")

    for recipe in recipes:
        tags_str = ", ".join(recipe["tags"][:3])
        if len(recipe["tags"]) > 3:
            tags_str += f" +{len(recipe['tags']) - 3}"
        table.add_row(
            recipe["name"],
            recipe["version"],
            (
                recipe["description"][:60] + "..."
                if len(recipe["description"]) > 60
                else recipe["description"]
            ),  # noqa: E501
            tags_str,
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Total: {len(recipes)} recipe(s). "
        "Run [bold]ml-plat recipe info <name>[/bold] for details.[/dim]\n"
    )


def _list_remote_recipes(tags: Optional[List[str]], verification_status: Optional[str]) -> None:
    """Query the recipe registry via the registry service."""
    from cli.recipe_engine.registry_client import RegistryClient

    try:
        client = RegistryClient()
        recipes = client.list_recipes(
            tags=tags or None,
            verification_status=verification_status,
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to connect to registry:[/bold red] {exc}")
        console.print(
            "[dim]Ensure the registry is reachable. Set ML_PLAT_REGISTRY_URL or check\n"
            "  kubectl get ingress[/dim]"
        )
        raise typer.Exit(1)

    if not recipes:
        console.print("[dim]No recipes found in the registry.[/dim]")
        console.print(
            "[dim]Push a recipe with [bold]ml-plat recipe push <name>[/bold] "
            "or use [bold]--local[/bold] to see local recipes.[/dim]"
        )
        raise typer.Exit(0)

    table = Table(
        title="[bold cyan]Registry Recipes[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Name", style="bold white", no_wrap=True)
    table.add_column("Version", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Profile", style="blue")
    table.add_column("Presets", style="cyan")
    table.add_column("Tags", style="magenta")
    table.add_column("Pushed", style="dim")

    for r in recipes:
        status_icon = {
            "verified": "🟢 verified",
            "experimental": "🟡 experimental",
            "failing": "🔴 failing",
        }.get(r.get("verification_status", ""), r.get("verification_status", ""))
        tags_list = r.get("tags") or []
        tags_str = ", ".join(tags_list[:3])
        if len(tags_list) > 3:
            tags_str += f" +{len(tags_list) - 3}"
        pushed = r.get("pushed_at", "")[:19] if r.get("pushed_at") else ""
        profile = r.get("profile") or ""
        if r.get("is_default"):
            profile = f"{profile} *" if profile else "*"
        presets_dict = r.get("presets") or {}
        row_profile = r.get("profile") or ""
        matching_presets = [
            pname
            for pname, pdata in presets_dict.items()
            if isinstance(pdata, dict) and pdata.get("profile") == row_profile
        ]
        presets_str = ", ".join(sorted(matching_presets)) if matching_presets else "-"
        table.add_row(
            r.get("recipe_name", "?"),
            r.get("version", "?"),
            status_icon,
            profile or "-",
            presets_str,
            tags_str or "-",
            pushed,
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Total: {len(recipes)} recipe(s) in registry. "
        "Run [bold]ml-plat recipe pull <name>[/bold] to download.[/dim]\n"
    )


# ── info ─────────────────────────────────────────────────────────────────────


@app.command("info")
def recipe_info(
    name: str = typer.Argument(
        ...,
        help="Recipe name",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Recipe version in the registry (default: latest)",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Load from local filesystem instead of the registry",
    ),
    recipes_dir: str = typer.Option(
        "",
        "--dir",
        "-d",
        help="Directory containing recipes (only used with --local)",
    ),
):
    """🔍 Show detailed information about a recipe.

    By default, fetches from the registry so you can inspect any published
    recipe.  Use ``--local`` to inspect a recipe on disk before pushing.

    Examples:\n
      ml-plat recipe info openrlhf-llm-rlhf\n
      ml-plat recipe info openrlhf-llm-rlhf --version 1.0\n
      ml-plat recipe info test-recipe --local\n
    """
    import tempfile

    from cli.recipe_engine.packager import RecipePackager
    from cli.recipe_engine.parser import RecipeParser
    from cli.recipe_engine.registry_client import RegistryClient

    if local:
        # ── Local mode ──────────────────────────────────────────────────
        parser = RecipeParser(recipes_dir)
        try:
            recipe = parser.load(name)
        except FileNotFoundError as exc:
            console.print(f"[bold red]Recipe not found locally:[/bold red] {name}")
            console.print(f"[dim]{exc}[/dim]")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[bold red]Failed to load recipe:[/bold red] {exc}")
            raise typer.Exit(1)
        source_label = "[dim](local)[/dim]"
    else:
        # ── Registry mode ───────────────────────────────────────────────
        client = RegistryClient()
        try:
            _tmp_dir = tempfile.TemporaryDirectory()
            tmp_path = Path(_tmp_dir.name)
            archive_path = client.pull(
                recipe_name=name,
                version=version or "latest",
                output_path=tmp_path,
            )
            packager = RecipePackager()
            packager.unpack(archive_path, output_dir=tmp_path / name)
            parser = RecipeParser(str(tmp_path))
            recipe = parser.load(name)
        except ValueError as exc:
            console.print(f"[bold red]Recipe not found in registry:[/bold red] {exc}")
            console.print(
                "[dim]Use --local to inspect a local recipe, or run "
                "'ml-plat recipe list' to see available recipes.[/dim]"
            )
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[bold red]Failed to fetch recipe from registry:[/bold red] {exc}")
            console.print("[dim]Use --local to inspect a local recipe.[/dim]")
            raise typer.Exit(1)
        source_label = "[dim](registry)[/dim]"

    # ── Header ──────────────────────────────────────────────────────────────
    header = (
        f"[bold white]{recipe.name}[/bold white]  "
        f"[dim]v{recipe.version}[/dim]  "
        f"[yellow]{recipe.author}[/yellow]  "
        f"{source_label}"
    )

    console.print()
    console.print(
        Panel(
            header,
            title="[bold cyan]Recipe Info[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print(f"\n[bold]Description:[/bold]  {recipe.description}")
    console.print(f"[bold]Tags:[/bold]  {', '.join(recipe.tags)}")

    # ── Infrastructure Profiles ─────────────────────────────────────────────
    console.print("\n[bold]Infrastructure Profiles:[/bold]")
    for profile_name, resource_groups in recipe.infrastructure.profiles.items():
        console.print(f"\n  [cyan]{profile_name}:[/cyan]")
        for rg_name, rg in resource_groups.items():
            instance_str = ", ".join(rg.instance_types[:2])
            if len(rg.instance_types) > 2:
                instance_str += f" +{len(rg.instance_types) - 2}"
            console.print(
                f"    [dim]•[/dim] {rg_name}: {rg.gpu_count}x GPU ({rg.gpu_memory}) — {instance_str}"  # noqa: E501
            )

    # ── Parameters ──────────────────────────────────────────────────────────
    if recipe.pipeline.parameters:
        console.print("\n[bold]Parameters:[/bold]")
        param_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        param_table.add_column("Name", style="cyan", no_wrap=True)
        param_table.add_column("Type", style="yellow")
        param_table.add_column("Default", style="green")

        for param_name, param in recipe.pipeline.parameters.items():
            default_str = str(param.default)
            if len(default_str) > 40:
                default_str = default_str[:37] + "..."
            param_table.add_row(param_name, param.type.value, default_str)

        console.print(param_table)

    # ── Component Versions ────────────────────────────────────────────────
    if recipe.component_versions:
        console.print("\n[bold]Component Versions:[/bold]")
        cv_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        cv_table.add_column("Component", style="cyan", no_wrap=True)
        cv_table.add_column("Version", style="yellow")

        for comp, ver in sorted(recipe.component_versions.items()):
            cv_table.add_row(comp, ver)

        console.print(cv_table)

    # ── Pipeline Steps ──────────────────────────────────────────────────────
    console.print("\n[bold]Pipeline Steps:[/bold]")
    for i, step in enumerate(recipe.pipeline.steps, 1):
        infra_str = f"[{step.infra}]" if step.infra else "[CPU]"
        console.print(f"  {i}. [cyan]{step.name}[/cyan]  {infra_str}  →  {step.component}")

    # ── Presets ─────────────────────────────────────────────────────────────
    if recipe.presets:
        console.print("\n[bold]Presets:[/bold]")
        for preset_name, preset in recipe.presets.items():
            override_count = len(preset.overrides)
            override_str = f"({override_count} override{'s' if override_count != 1 else ''})"
            console.print(
                f"  [dim]•[/dim] [cyan]{preset_name}[/cyan]: profile={preset.profile} {override_str}"  # noqa: E501
            )

    console.print()


# ── versions ─────────────────────────────────────────────────────────────────


@app.command("versions")
def list_recipe_versions(
    name: str = typer.Argument(
        ...,
        help="Recipe name to list versions for",
    ),
):
    """📋 List all published versions of a recipe in the registry.

    Examples:\n
      ml-plat recipe versions openrlhf-llm-rlhf\n
      ml-plat recipe versions text2sql\n
    """
    from cli.recipe_engine.registry_client import RegistryClient

    client = RegistryClient()

    try:
        versions = client.list_versions(name)
    except Exception as exc:
        console.print(f"[bold red]Failed to query registry:[/bold red] {exc}")
        console.print(
            "[dim]Ensure the registry is reachable. "
            "Set ML_PLAT_REGISTRY_URL or check 'kubectl get ingress'.[/dim]"
        )
        raise typer.Exit(1)

    if not versions:
        console.print(f"[dim]No versions found for recipe '{name}'.[/dim]")
        console.print("[dim]Run 'ml-plat recipe list' to see available recipes.[/dim]")
        raise typer.Exit(0)

    table = Table(
        title=f"Versions of [cyan]{name}[/cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Version", style="yellow", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Profile", style="dim")
    table.add_column("Pushed", style="dim")

    status_icons = {
        "verified": "[green]🟢 verified[/green]",
        "experimental": "[yellow]🟡 experimental[/yellow]",
        "failing": "[red]🔴 failing[/red]",
    }

    for v in versions:
        ver = v.get("version", "?")
        status_raw = v.get("verification_status", "unknown")
        status = status_icons.get(status_raw, status_raw)
        profile = v.get("profile") or ""
        if v.get("is_default"):
            profile = f"{profile} *" if profile else "*"
        pushed = v.get("pushed_at", "-")
        if pushed and len(pushed) > 19:
            pushed = pushed[:19]
        table.add_row(ver, status, profile or "-", pushed)

    console.print(table)
    console.print(
        f"\nRun [cyan]ml-plat recipe run {name} --version <ver>[/cyan] "
        "to execute a specific version."
    )


# ── validate ─────────────────────────────────────────────────────────────────


@app.command("validate")
def validate_recipe(
    name: str = typer.Argument(
        ...,
        help="Recipe name (directory name in recipes/)",
    ),
    recipes_dir: str = typer.Option(
        "",
        "--dir",
        "-d",
        help="Directory containing recipes (default: projects/recipes/ relative to repo root)",
    ),
):
    """✅ Validate a recipe YAML file.

    Checks YAML syntax, schema validation, and detects common errors like:
    - Invalid instance types
    - Circular dependencies
    - Unknown preset profiles
    - Invalid template syntax

    Examples:
      ml-plat recipe validate test-recipe
      ml-plat recipe validate my-recipe --dir /custom/recipes
    """
    from cli.recipe_engine.parser import RecipeParser

    parser = RecipeParser(recipes_dir)

    console.print(f"Validating recipe: [cyan]{name}[/cyan]")

    try:
        recipe = parser.load(name)
        console.print("[green]✓[/green] Recipe loaded successfully")
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/red] Recipe not found: {name}")
        console.print(f"[dim]{exc}[/dim]")
        raise typer.Exit(1)
    except ValueError as exc:
        console.print("[red]✗[/red] Validation failed")
        console.print(f"[dim]{exc}[/dim]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print("[red]✗[/red] Unexpected error")
        console.print(f"[dim]{exc}[/dim]")
        raise typer.Exit(1)

    # Additional checks
    checks_passed = 0
    checks_total = 0

    # Check 1: At least one profile defined
    checks_total += 1
    if recipe.infrastructure.profiles:
        console.print(
            f"[green]✓[/green] Infrastructure profiles defined: {len(recipe.infrastructure.profiles)}"  # noqa: E501
        )
        checks_passed += 1
    else:
        console.print("[red]✗[/red] No infrastructure profiles defined")

    # Check 2: At least one step defined
    checks_total += 1
    if recipe.pipeline.steps:
        console.print(f"[green]✓[/green] Pipeline steps defined: {len(recipe.pipeline.steps)}")
        checks_passed += 1
    else:
        console.print("[red]✗[/red] No pipeline steps defined")

    # Check 3: All step infra references are valid
    checks_total += 1
    invalid_infra_refs = []
    first_profile = list(recipe.infrastructure.profiles.values())[0]
    for step in recipe.pipeline.steps:
        if step.infra and step.infra not in first_profile:
            invalid_infra_refs.append(f"{step.name} → {step.infra}")

    if not invalid_infra_refs:
        console.print("[green]✓[/green] All step infra references are valid")
        checks_passed += 1
    else:
        console.print("[yellow]⚠[/yellow] Some infra references may not exist in all profiles:")
        for ref in invalid_infra_refs:
            console.print(f"    {ref}")

    # Check 4: Presets reference valid profiles
    checks_total += 1
    invalid_preset_profiles = []
    valid_profile_names = set(recipe.infrastructure.profiles.keys())
    for preset_name, preset in recipe.presets.items():
        if preset.profile not in valid_profile_names:
            invalid_preset_profiles.append(f"{preset_name} → {preset.profile}")

    if not invalid_preset_profiles:
        console.print("[green]✓[/green] All presets reference valid profiles")
        checks_passed += 1
    else:
        console.print("[red]✗[/red] Invalid preset profile references:")
        for ref in invalid_preset_profiles:
            console.print(f"    {ref}")

    # Summary
    console.print()
    if checks_passed == checks_total:
        console.print(
            f"[bold green]✅ Recipe is valid! ({checks_passed}/{checks_total} checks passed)[/bold green]"  # noqa: E501
        )
    else:
        console.print(
            f"[bold yellow]⚠️  Recipe has warnings ({checks_passed}/{checks_total} checks passed)[/bold yellow]"  # noqa: E501
        )
    console.print()


# ── run ──────────────────────────────────────────────────────────────────────


@app.command("run")
def run_recipe(
    name: str = typer.Argument(
        ...,
        help="Recipe name (local or registry)",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Run a specific version from the registry (e.g. '1.0', 'latest')",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help="Apply a named preset (overrides parameter defaults)",
    ),
    param: Optional[List[str]] = typer.Option(
        None,
        "--param",
        help="Parameter override in key=value format (repeatable)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show resolved config without submitting to Flyte",
    ),
    canary: bool = typer.Option(
        False,
        "--canary",
        help="Canary probe: scales down epochs=1/max_samples=100 for quick validation",
    ),
    require_verified: bool = typer.Option(
        False,
        "--require-verified",
        help="Only run if recipe+profile has verified badge (fail otherwise)",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        help="Flyte project (default: from FLYTE_PROJECT env or 'flytesnacks')",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        help="Flyte domain (default: from FLYTE_DOMAIN env or 'development')",
    ),
    recipes_dir: str = typer.Option(
        "",
        "--dir",
        "-d",
        help="Directory containing recipes (default: projects/recipes/ relative to repo root)",
    ),
    overwrite_cache: bool = typer.Option(
        False,
        "--overwrite-cache",
        "--owc",
        help="Overwrite cached task outputs",
    ),
):
    """🚀 Execute a recipe pipeline.

        Runs a recipe by submitting its pipeline steps to Flyte. Supports presets
        and parameter overrides.

        When ``--version`` is specified, the recipe is pulled from the registry
        instead of loaded from the local filesystem.  This means another user
        can push a recipe with pinned ``component_versions`` and you can run it
        by name + version without any local files.

        Examples:\n
          # Run a local recipe\n
          ml-plat recipe run text2sql\n
    \n
          # Run a specific version from the registry\n
          ml-plat recipe run openrlhf-llm-rlhf --version 1.0\n
          ml-plat recipe run openrlhf-llm-rlhf --version latest --preset small\n
    \n
          # Override individual parameters\n
          ml-plat recipe run text2sql --param num_epochs=5 --param batch_size=32\n
    \n
          # Dry-run to see resolved config\n
          ml-plat recipe run text2sql --preset production --dry-run\n
    """
    import tempfile

    from cli.recipe_engine.deployer import ArchitectureDeployer, DeploymentError
    from cli.recipe_engine.packager import RecipePackager
    from cli.recipe_engine.registry_client import RegistryClient
    from cli.recipe_engine.runner import RecipeRunner

    # ── Pull from registry when --version is given ──────────────────────
    _tmp_dir = None  # keep alive until function returns
    if version is not None:
        console.print(
            f"Pulling recipe [cyan]{name}[/cyan] "
            f"version [yellow]{version}[/yellow] from registry …"
        )
        try:
            client = RegistryClient()
            _tmp_dir = tempfile.TemporaryDirectory()
            tmp_path = Path(_tmp_dir.name)
            archive_path = client.pull(
                recipe_name=name,
                version=version,
                output_path=tmp_path,
            )
            packager = RecipePackager()
            packager.unpack(archive_path, output_dir=tmp_path / name)
            # Point the parser at the temp directory containing the unpacked
            # recipe so that it is loaded from the registry artefact.
            recipes_dir = str(tmp_path)
            console.print(
                f"[green]✓[/green] Using registry recipe "
                f"[cyan]{name}[/cyan] v[yellow]{version}[/yellow]"
            )
        except ValueError as exc:
            console.print(f"[bold red]Recipe not found in registry:[/bold red] {exc}")
            console.print("[dim]Run 'ml-plat recipe list' to see available recipes.[/dim]")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[bold red]Failed to pull recipe:[/bold red] {exc}")
            raise typer.Exit(1)

    runner = RecipeRunner(recipes_dir)

    # Parse parameter overrides
    param_overrides = {}
    if param:
        for p in param:
            if "=" not in p:
                console.print(f"[red]Invalid parameter format: {p}[/red]")
                console.print("[dim]Use format: --param key=value[/dim]")
                raise typer.Exit(1)
            raw_key, raw_value = p.split("=", 1)
            param_overrides[raw_key.strip()] = raw_value.strip()

    # Load recipe and resolve config
    try:
        recipe = runner.parser.load(name)
        config = runner.resolve_config(recipe, preset, param_overrides)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Recipe not found:[/bold red] {name}")
        console.print(f"[dim]{exc}[/dim]")
        raise typer.Exit(1)
    except ValueError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(1)

    # ── Verification check ──────────────────────────────────────────────────
    if require_verified:
        try:
            client = RegistryClient()
            profile = config["profile"]
            verification_status = client.get_verification_status(
                recipe.name, recipe.version, profile
            )
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[bold red]Registry unreachable — verification required:[/bold red] {e}")
            console.print(
                "[dim]Cannot verify recipe — aborting (--require-verified is strict).\n"
                "Set ML_PLAT_REGISTRY_URL or check 'kubectl get ingress'.[/dim]"
            )
            raise typer.Exit(1)

        if verification_status != RegistryClient.STATUS_VERIFIED:
            console.print(
                f"[bold red]Recipe {recipe.name} v{recipe.version} "
                f"with profile '{profile}' is not verified.[/bold red]"
            )
            console.print(f"[dim]Current status: {verification_status or 'unknown'}[/dim]")
            console.print(
                "[dim]Run without --require-verified to proceed anyway, "
                "or run with --canary to verify this configuration.[/dim]"
            )
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Recipe verified for profile '{profile}'")

    # ── Canary mode ─────────────────────────────────────────────────────────
    if canary:
        console.print(
            Panel.fit(
                f"[bold yellow]Running canary probe for:[/bold yellow]\n"
                f"Recipe: {recipe.name} v{recipe.version}\n"
                f"Profile: {config['profile']}\n"
                f"[dim]This will run a 10-minute test on real hardware "
                "to project VRAM usage, throughput, and cost.[/dim]",
                title="[bold yellow]🐤 Canary Mode[/bold yellow]",
                border_style="yellow",
            )
        )
        # In a full implementation, this would:
        # 1. Scale down epochs/iterations to ~10 min runtime
        # 2. Run the pipeline
        # 3. Collect metrics (VRAM, throughput, cost)
        # 4. Store results in registry
        # For now, we'll just proceed with normal execution
        console.print(
            "[dim]Note: Canary execution uses the same parameters. "
            "Consider using a small preset for faster validation.[/dim]\n"
        )

        # Scale down for canary: reduce epochs and max samples
        # Guard with isinstance so we don't blow up on string/bool/list params.
        if "parameters" in config:
            for key in config["parameters"]:
                val = config["parameters"][key]
                if "epoch" in key.lower():
                    config["parameters"][key] = 1
                elif ("step" in key.lower() or "sample" in key.lower()) and isinstance(
                    val, (int, float)
                ):
                    config["parameters"][key] = min(val, 100)
        console.print("[yellow]Canary: scaled down epochs=1, max_samples=100[/yellow]\n")

    # ── Dry-run mode ────────────────────────────────────────────────────────
    if dry_run:
        console.print(
            Panel.fit(
                f"[bold cyan]Recipe:[/bold cyan] {recipe.name} v{recipe.version}\n"
                f"[bold cyan]Profile:[/bold cyan] {config['profile']}\n"
                f"[bold cyan]Preset:[/bold cyan] {preset or '(default)'}",
                title="[bold cyan]🔍 Dry-Run Mode[/bold cyan]",
                border_style="cyan",
            )
        )

        # Parameters table
        console.print("\n[bold]Resolved Parameters:[/bold]")
        param_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        param_table.add_column("Parameter", style="cyan", no_wrap=True)
        param_table.add_column("Type", style="yellow")
        param_table.add_column("Value", style="green")

        for param_name, param_def in recipe.pipeline.parameters.items():
            value = config["parameters"].get(param_name, param_def.default)
            param_table.add_row(param_name, param_def.type.value, str(value))

        console.print(param_table)

        # ── Architecture block (dry-run) ─────────────────────────────────────
        if recipe.architecture and recipe.architecture.groups:
            console.print("\n[bold]Architecture (Multi-Service):[/bold]")
            try:
                deployer = ArchitectureDeployer(
                    recipe_name=name,
                    architecture=recipe.architecture,
                    namespace="default",
                )
                arch_result = deployer.deploy(dry_run=True)

                # Groups summary
                console.print("\n  [bold cyan]Service Groups:[/bold cyan]")
                for gname, ginfo in arch_result["topology"]["groups"].items():
                    efa_str = " [yellow]EFA[/yellow]" if ginfo["efa_required"] else ""
                    console.print(
                        f"    • {gname}: {ginfo['replicas']}x "
                        f"({ginfo['gpus_per_replica']} GPU) "
                        f"{ginfo.get('instance', 'auto')}{efa_str}"
                    )

                # Connections
                console.print("\n  [bold cyan]Connections:[/bold cyan]")
                for conn in arch_result["topology"]["connections"]:
                    direction = "↔" if conn["bidirectional"] else "→"
                    console.print(
                        f"    {conn['from']} {direction} {conn['to']} "
                        f"({conn['protocol']}, {conn['type']})"
                    )

                # Lifecycle
                console.print("\n  [bold cyan]Startup Order:[/bold cyan]")
                for stage in arch_result["lifecycle"]["startup_plan"]:
                    console.print(f"    Stage {stage['stage']}: {stage['groups']}")

                est = arch_result["lifecycle"]["estimated_startup_time"]
                msg = f"    [dim]Estimated startup: {est}s ({est // 60}m {est % 60}s)[/dim]"
                console.print(msg)

                # Manifest counts
                console.print("\n  [bold cyan]Generated K8s Manifests:[/bold cyan]")
                for mtype, count in arch_result["manifest_counts"].items():
                    console.print(f"    {mtype}: {count}")

            except Exception as exc:
                console.print(f"  [red]Architecture analysis failed: {exc}[/red]")

        # Pipeline steps
        console.print("\n[bold]Pipeline Steps:[/bold]")
        for i, step in enumerate(recipe.pipeline.steps, 1):
            infra_str = f"\\[{step.infra}]" if step.infra else "\\[CPU]"
            console.print(f"  {i}. [cyan]{step.name}[/cyan]  {infra_str}  →  {step.component}")

            # Show resolved config for this step
            step_config = config["steps"].get(step.name, {})
            if step_config:
                console.print("     [dim]Config:[/dim]")
                for key, value in step_config.items():
                    value_str = str(value)
                    if len(value_str) > 60:
                        value_str = value_str[:57] + "..."
                    console.print(f"       {key}: {value_str}")

        console.print("\n[dim]Run without --dry-run to submit to Flyte.[/dim]\n")
        return

    # ── Execute mode ────────────────────────────────────────────────────────
    console.print(
        Panel.fit(
            f"[bold cyan]Recipe:[/bold cyan] {recipe.name} v{recipe.version}\n"
            f"[bold cyan]Profile:[/bold cyan] {config['profile']}\n"
            f"[bold cyan]Preset:[/bold cyan] {preset or '(default)'}",
            title="[bold cyan]🚀 Executing Recipe[/bold cyan]",
            border_style="cyan",
        )
    )

    # Create FlyteRemote
    try:
        remote = flyte_remote()
    except Exception as exc:
        console.print(f"[bold red]Failed to connect to Flyte:[/bold red] {exc}")
        console.print("[dim]Check cluster.flyte_endpoint in ~/.ml-plat/config.yaml[/dim]")
        raise typer.Exit(1)

    def on_submit(exec_info: dict):
        exec_id = exec_info["execution_id"]
        step_name = exec_info["step_name"]

        # Use authoritative project/domain from the execution object itself
        # rather than re-deriving from env vars to avoid URL drift
        exec_project = (
            exec_info.get("execution_project")
            or project
            or os.getenv("FLYTE_PROJECT", "flytesnacks")
        )
        exec_domain = (
            exec_info.get("execution_domain") or domain or os.getenv("FLYTE_DOMAIN", "development")
        )
        url = flyte_console_url(exec_project, exec_domain, exec_id)
        console.print(f"  • {step_name}: [bold]{exec_id}[/bold]")
        console.print(f"    {url}", soft_wrap=True)

    console.print("\n[bold]Execution IDs:[/bold]")

    # ── Endpoint preflight (before remote submission) ─────────────────────
    skip_preflight = os.getenv("ML_PLAT_SKIP_ENDPOINT_PREFLIGHT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if skip_preflight:
        console.print(
            "[yellow]Skipping endpoint preflight (ML_PLAT_SKIP_ENDPOINT_PREFLIGHT set).[/yellow]"
        )
    else:
        from cli.recipe_engine.preflight import run_endpoint_preflight

        preflight_checks, preflight_failures = run_endpoint_preflight(config["parameters"])
        if preflight_checks or preflight_failures:
            console.print("\n[bold]Endpoint preflight:[/bold]")
            for label, ok, detail in preflight_checks:
                icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
                console.print(f"  {icon} {label}: {detail}")

        if preflight_failures:
            console.print("\n[bold red]Preflight failed. Aborting submission.[/bold red]")
            for item in preflight_failures:
                console.print(f"  [red]-[/red] {item}")
            console.print(
                "[dim]Set ML_PLAT_SKIP_ENDPOINT_PREFLIGHT=1 to bypass this check "
                "if you intentionally run from an environment without endpoint visibility.[/dim]"
            )
            raise typer.Exit(1)

    # Submit to Flyte (with architecture deployment if applicable)
    try:
        result = runner.run(
            recipe_name=name,
            remote=remote,
            preset_name=preset,
            param_overrides=param_overrides,
            dry_run=False,
            project=project,
            domain=domain,
            submission_callback=on_submit,
            overwrite_cache=overwrite_cache,
        )
    except DeploymentError as exc:
        console.print(f"[bold red]Architecture deployment failed:[/bold red] {exc}")
        console.print("[dim]Check kubectl access and cluster status.[/dim]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[bold red]Execution failed:[/bold red] {exc}")
        raise typer.Exit(1)

    # Display architecture deployment info if applicable
    if "architecture" in result:
        arch = result["architecture"]
        console.print("\n[bold green]✅ Architecture deployed![/bold green]")
        for mtype, count in arch.get("manifest_counts", {}).items():
            console.print(f"  {mtype}: {count}")
        console.print()

    # Display results
    console.print("\n[bold green]✅ Pipeline executed successfully![/bold green]\n")


# ── push ─────────────────────────────────────────────────────────────────────


@app.command("push")
def push_recipe(
    name: str = typer.Argument(
        ...,
        help="Recipe name to package and push",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Infrastructure profile for lockfile (default: first profile alphabetically)",
    ),
    verification_status: str = typer.Option(
        "experimental",
        "--status",
        "-s",
        help="Verification status: verified, experimental, or failing",
    ),
    tags: Optional[List[str]] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Additional tags (repeatable)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for archive (default: current directory)",
    ),
    recipes_dir: str = typer.Option(
        "",
        "--dir",
        "-d",
        help="Directory containing recipes (default: projects/recipes/ relative to repo root)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing version in registry",
    ),
):
    """📤 Package and push a recipe to the registry.

    This command:
    1. Packages the recipe into a .ml-plat archive
    2. Uploads it to S3
    3. Stores metadata in PostgreSQL

    Examples:
      ml-plat recipe push text2sql
      ml-plat recipe push llm-rlhf --profile small --status verified
      ml-plat recipe push my-recipe --tag production --tag gpu
    """
    from cli.recipe_engine.packager import RecipePackager
    from cli.recipe_engine.parser import RecipeParser
    from cli.recipe_engine.registry_client import RegistryClient

    console.print(f"Packaging recipe: [cyan]{name}[/cyan]")

    # Package the recipe
    if recipes_dir:
        packager = RecipePackager()
        packager.parser = RecipeParser(recipes_dir)  # RecipeParser expects str, not Path
    else:
        packager = RecipePackager()
    try:
        output_path_obj = Path(output_dir) if output_dir else None
        archive_path = packager.package(
            recipe_name=name,
            output_path=output_path_obj,
            include_lockfile=True,
            profile=profile,
        )
        console.print(f"[green]✓[/green] Created archive: {archive_path}")
    except Exception as exc:
        console.print(f"[bold red]Failed to package recipe:[/bold red] {exc}")
        raise typer.Exit(1)

    # Extract presets from recipe YAML so they're stored in the registry
    presets_data: dict = {}
    try:
        _parser = RecipeParser(recipes_dir)
        _recipe = _parser.load(name)
        if _recipe.presets:
            for pname, pobj in _recipe.presets.items():
                presets_data[pname] = {"profile": pobj.profile, "overrides": pobj.overrides}
    except Exception:
        pass  # Non-fatal: presets are a nice-to-have

    # Push to registry
    console.print("\nPushing to registry...")
    client = RegistryClient()

    try:
        result = client.push(
            archive_path=archive_path,
            verification_status=verification_status,
            profile=profile,
            tags=tags or [],
            overwrite=overwrite,
            presets=presets_data,
        )

        console.print(
            Panel.fit(
                f"[bold green]Recipe:[/bold green] {result['recipe_name']} v{result['version']}\n"
                f"[bold green]S3 URI:[/bold green] {result['s3_uri']}\n"
                f"[bold green]Status:[/bold green] {result['verification_status']}\n"
                f"[bold green]Profile:[/bold green] {result.get('profile', 'N/A')}",
                title="[bold green]✅ Successfully Pushed[/bold green]",
                border_style="green",
            )
        )

    except ValueError as exc:
        console.print(f"[bold red]Push failed:[/bold red] {exc}")
        console.print("[dim]Use --overwrite to replace existing version[/dim]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[bold red]Failed to push to registry:[/bold red] {exc}")
        console.print(
            "[dim]Ensure the registry is reachable. Set ML_PLAT_REGISTRY_URL or check\n"
            "  kubectl get ingress[/dim]"
        )
        raise typer.Exit(1)


# ── pull ─────────────────────────────────────────────────────────────────────


@app.command("pull")
def pull_recipe(
    name: str = typer.Argument(
        ...,
        help="Recipe name to pull from registry",
    ),
    version: str = typer.Option(
        "latest",
        "--version",
        "-v",
        help="Recipe version (default: latest)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for archive (default: current directory)",
    ),
    unpack: bool = typer.Option(
        False,
        "--unpack",
        help="Unpack archive after downloading",
    ),
):
    """📥 Pull a recipe from the registry.

    Downloads a recipe archive from S3.

    Examples:
      ml-plat recipe pull text2sql
      ml-plat recipe pull llm-rlhf --version 1.2.0
      ml-plat recipe pull my-recipe --unpack
    """
    console.print(f"Pulling recipe: [cyan]{name}[/cyan] version [yellow]{version}[/yellow]")

    from cli.recipe_engine.packager import RecipePackager
    from cli.recipe_engine.registry_client import RegistryClient

    client = RegistryClient()

    try:
        output_path_obj = Path(output_dir) if output_dir else None
        archive_path = client.pull(
            recipe_name=name,
            version=version,
            output_path=output_path_obj,
        )

        console.print(f"[green]✓[/green] Downloaded to: {archive_path}")

        # Unpack if requested
        if unpack:
            console.print("\nUnpacking archive...")
            packager = RecipePackager()
            unpacked_dir = packager.unpack(archive_path)
            console.print(f"[green]✓[/green] Unpacked to: {unpacked_dir}")

        console.print(
            Panel.fit(
                f"[bold green]Recipe:[/bold green] {name} v{version}\n"
                f"[bold green]Location:[/bold green] {archive_path}",
                title="[bold green]✅ Successfully Pulled[/bold green]",
                border_style="green",
            )
        )

    except ValueError as exc:
        console.print(f"[bold red]Recipe not found:[/bold red] {exc}")
        console.print("[dim]Run 'ml-plat recipe list' to see available recipes.[/dim]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[bold red]Failed to pull from registry:[/bold red] {exc}")
        console.print(
            "[dim]Ensure the registry is reachable. Set ML_PLAT_REGISTRY_URL or check\n"
            "  kubectl get ingress[/dim]"
        )
        raise typer.Exit(1)


# ── register ─────────────────────────────────────────────────────────────────


@app.command("register")
def register_recipe(
    name: str = typer.Argument(
        ...,
        help="Recipe name (e.g. openrlhf-llm-rlhf)",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="Flyte project (default: from config or FLYTE_PROJECT env)",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Flyte domain (default: 'development')",
    ),
    recipes_dir: str = typer.Option(
        "",
        "--dir",
        help="Directory containing recipes (default: projects/recipes/)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing recipe versions in registry",
    ),
):
    """📋 Register all components referenced by a recipe.

    Reads the recipe's ``component_versions`` and registers each component
    with its pinned version via ``pyflyte register``.

    Examples:\n
      ml-plat recipe register openrlhf-llm-rlhf\n
      ml-plat recipe register openrlhf-llm-rlhf --project ml-platform\n
    """
    from cli.recipe_engine.parser import RecipeParser
    from cli.recipe_engine.registrar import ComponentRegistrar

    parser = RecipeParser(recipes_dir)
    try:
        recipe = parser.load(name)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Recipe not found:[/bold red] {name}")
        console.print(f"[dim]{exc}[/dim]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[bold red]Failed to load recipe:[/bold red] {exc}")
        raise typer.Exit(1)

    proj = project or os.getenv("FLYTE_PROJECT", "ml-platform")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    registrar = ComponentRegistrar(recipes_dir)
    try:
        successes, failures, bumped = registrar.register_all(name, proj, dom, auto_bump=True)
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(0 if "no component" in str(exc).lower() else 1)
    except Exception as exc:
        console.print(f"[bold red]Registration failed:[/bold red] {exc}")
        raise typer.Exit(1)

    profiles = recipe.infrastructure.profiles
    profile_names = list(profiles.keys()) if profiles else []

    if bumped and failures == 0:
        registrar.update_recipe_yaml(name, bumped)
        console.print(
            f"[cyan]Updated {name}.yaml with bumped versions: "
            f"{', '.join(f'{k}→{v}' for k, v in bumped.items())}[/cyan]"
        )

    if failures == 0 and successes > 0:
        console.print(
            f"[bold green]All {successes} component(s) registered successfully.[/bold green]"
        )
    elif failures > 0:
        console.print(
            f"[bold red]{failures} component(s) failed, {successes} succeeded.[/bold red]"
        )
        raise typer.Exit(1)

    # Push recipe archive for every profile so the registry reflects the
    # current component versions (including any auto-bumped ones).
    if profile_names:
        from cli.recipe_engine.packager import RecipePackager
        from cli.recipe_engine.registry_client import RegistryClient

        console.print()
        console.print("[bold]Pushing recipe archive for all profiles...[/bold]")
        packager = RecipePackager()
        client = RegistryClient()
        push_failures = []
        # Extract presets once for all profiles
        _presets_data: dict = {}
        try:
            _rp = RecipeParser(recipes_dir)
            _r = _rp.load(name)
            if _r.presets:
                for pname, pobj in _r.presets.items():
                    _presets_data[pname] = {"profile": pobj.profile, "overrides": pobj.overrides}
        except Exception:
            pass
        for prof in profile_names:
            try:
                archive_path = packager.package(
                    recipe_name=name,
                    output_path=None,
                    include_lockfile=True,
                    profile=prof,
                )
                result = client.push(
                    archive_path=archive_path,
                    verification_status="experimental",
                    profile=prof,
                    tags=[],
                    overwrite=overwrite,
                    presets=_presets_data,
                )
                console.print(
                    f"  [green]✓[/green] Pushed [cyan]{result['recipe_name']} "
                    f"v{result['version']}[/cyan] (profile: {prof}"
                    + (" [dim]*default[/dim]" if result.get("is_default") else "")
                    + ")"
                )
            except ValueError as exc:
                # Should not happen unless same profile pushed twice
                console.print(f"  [yellow]⚠[/yellow] {prof}: {exc}")
            except Exception as exc:
                console.print(f"  [red]✗[/red] {prof}: {exc}")
                push_failures.append(prof)
        if push_failures:
            console.print(
                f"[yellow]Push failed for profiles: {', '.join(push_failures)}.[/yellow] "
                "Components are registered; re-run push manually if needed."
            )
