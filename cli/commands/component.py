"""
component.py — CLI commands for the ML Platform shared component library.

Components are Flyte tasks registered in FlyteAdmin.  This CLI queries the
Flyte registry (FlyteAdmin) as the single source of truth.

Each component lives in its own directory with a ``component.yaml`` meta file
that declares the component's image, description, and tags.  The CLI reads
this file to resolve images and compute content-hash versions.

Commands:
    ml-plat component list                            — list registered components
    ml-plat component info <name>                     — show inputs, outputs, image
    ml-plat component register training/lora_finetune — register one component
    ml-plat component register training               — register all in a category
    ml-plat component register --all                  — register everything
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.utils import flyte_remote

console = Console()
app = typer.Typer(help="Browse and manage the shared component library (backed by Flyte)")

# ── Component directory layout ────────────────────────────────────────────────
#
# Each component lives in its own directory with a ``component.yaml``:
#
#   projects/components/components/<category>/<component>/
#       component.yaml      <- meta: name, image, tags, description
#       __init__.py          <- Flyte @task code
#       (optional extras)    <- configs, READMEs, helpers
#
# The category directories (data/, training/, evaluation/, ...) are
# organisational groupings only; image and version live in component.yaml.

# Root directory containing component sub-packages
COMPONENTS_ROOT = Path("projects/components/components")

COMPONENT_META_FILE = "component.yaml"


def _load_component_yaml(path: Path) -> Dict[str, Any]:
    """Parse a component.yaml file and return its contents as a dict."""
    import yaml  # lazy - only needed at registration time

    return yaml.safe_load(path.read_text()) or {}


def _discover_components(
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Scan *root* for component directories (those containing component.yaml).

    Returns a sorted list of dicts, each with at least:
        path     - Path to the component directory
        name     - component name from component.yaml (fallback: dir name)
        category - parent directory name
        image    - container image short name (e.g. "ml-gpu")
    """
    root = root or COMPONENTS_ROOT
    if not root.is_dir():
        return []

    components: List[Dict[str, Any]] = []
    for meta_file in sorted(root.rglob(COMPONENT_META_FILE)):
        comp_dir = meta_file.parent
        meta = _load_component_yaml(meta_file)
        meta["path"] = comp_dir
        meta.setdefault("name", comp_dir.name)
        meta.setdefault("category", comp_dir.parent.name)
        components.append(meta)
    return components


def _category_names(root: Optional[Path] = None) -> List[str]:
    """Return sorted category names (top-level dirs with an __init__.py)."""
    root = root or COMPONENTS_ROOT
    if not root.is_dir():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "__init__.py").exists() and d.name != "__pycache__"
    )


def _content_hash(directory: Path) -> str:
    """Compute a short deterministic hash of **all** files in *directory*.

    Covers .py, .yaml, configs, READMEs - any file change bumps the hash.
    Unchanged content produces the same hash, so re-registering with the
    same hash is a no-op in Flyte (version already exists).
    """
    h = hashlib.sha256()
    for fpath in sorted(directory.rglob("*")):
        if fpath.is_file() and "__pycache__" not in fpath.parts:
            h.update(str(fpath.relative_to(directory)).encode())
            h.update(fpath.read_bytes())
    return h.hexdigest()[:12]


def _parse_versions_env() -> Dict[str, str]:
    """Read ECR_REGISTRY, ECR_REPO, IMAGE_TAG from versions.env."""
    versions_env = Path("projects/components/images/versions.env")
    vals: Dict[str, str] = {}
    if not versions_env.exists():
        return vals
    for line in versions_env.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        for sep in (":=", "="):
            if sep in line:
                k, v = line.split(sep, 1)
                vals[k.strip()] = v.strip()
                break
    return vals


def _resolve_image(
    image_short: str,
    explicit_image: Optional[str] = None,
) -> Optional[str]:
    """Return the full container image URI.

    Priority: explicit ``--image`` flag  >  ECR URI from versions.env  >  None.
    ``image_short`` is the short name from component.yaml (e.g. "ml-gpu").
    """
    if explicit_image:
        return explicit_image
    if not image_short:
        return None
    env = _parse_versions_env()
    registry = env.get("ECR_REGISTRY")
    repo = env.get("ECR_REPO")
    tag = env.get("IMAGE_TAG")
    if registry and repo and tag:
        return f"{registry}/{repo}/{image_short}:{tag}"
    return None


# ── helpers ──────────────────────────────────────────────────────────────────


def _get_remote():
    """Return a FlyteRemote instance.  Wraps the shared helper so we can mock
    it in tests without touching ``cli.utils``."""
    return flyte_remote()


def _list_task_ids(
    project: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = 500,
):
    """Return all task names registered in FlyteAdmin for the given project/domain."""
    remote = _get_remote()
    proj = project or os.getenv("FLYTE_PROJECT", "flytesnacks")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    all_ids = []
    token = None
    while True:
        page, token = remote.client.list_task_ids_paginated(
            project=proj,
            domain=dom,
            limit=limit,
            token=token,
        )
        all_ids.extend(page)
        if not token:
            break
    return all_ids


def _ensure_project(project: str):
    """Create the Flyte project if it does not already exist.

    This lets users run ``ml-plat component register --project <name>``
    without needing ``flytectl`` to pre-create the project.
    Prompts for confirmation before creating.
    """
    remote = _get_remote()
    try:
        existing = remote.client.list_projects()
        for p in existing.projects:
            if p.id == project:
                return  # already exists
    except Exception:
        pass  # if listing fails, try to create anyway

    from flytekit.models.project import Project

    create = typer.confirm(f"Project '{project}' does not exist in Flyte. Create it now?")
    if not create:
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(0)

    try:
        remote.client.register_project(
            Project(id=project, name=project, description="Auto-created by ml-plat")
        )
        console.print(f"[green]Created project '{project}'.[/green]")
    except Exception as exc:
        # Ignore "already exists" races; surface anything else
        if "already exists" not in str(exc).lower():
            console.print(f"[bold red]Failed to create project:[/bold red] {exc}")
            raise typer.Exit(1)


def _fetch_task(
    name: str,
    version: Optional[str] = None,
    project: Optional[str] = None,
    domain: Optional[str] = None,
):
    """Fetch a single task from FlyteAdmin by name (and optionally version)."""
    remote = _get_remote()
    proj = project or os.getenv("FLYTE_PROJECT", "flytesnacks")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    if version:
        return remote.fetch_task(project=proj, domain=dom, name=name, version=version)

    # No version specified — list versions and pick the latest
    from flytekit.models.common import NamedEntityIdentifier

    identifier = NamedEntityIdentifier(project=proj, domain=dom, name=name)
    tasks, _ = remote.client.list_tasks_paginated(
        identifier,
        limit=1,
    )
    if not tasks:
        return None
    # Re-fetch through FlyteRemote so we get a proper FlyteTask
    task_id = tasks[0].id
    return remote.fetch_task(
        project=task_id.project,
        domain=task_id.domain,
        name=task_id.name,
        version=task_id.version,
    )


# ── list ─────────────────────────────────────────────────────────────────────


@app.command("list")
def list_components(
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="Flyte project (default: $FLYTE_PROJECT)",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Flyte domain (default: $FLYTE_DOMAIN)",
    ),
):
    """📦 List all registered components (tasks) in the Flyte registry.

    Queries FlyteAdmin for all tasks in the given project/domain.

    Examples:
      ml-plat component list
      ml-plat component list --project ml-platform --domain production
    """
    try:
        task_ids = _list_task_ids(project=project, domain=domain)
    except Exception as exc:
        console.print(
            f"[bold red]Failed to connect to Flyte:[/bold red] {exc}\n"
            "Check FLYTE_ENDPOINT and cluster connectivity."
        )
        raise typer.Exit(1)

    if not task_ids:
        console.print("[dim]No components registered yet.[/dim]")
        raise typer.Exit(0)

    table = Table(
        title="[bold cyan]ML Platform Component Library[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Name", style="bold white", no_wrap=True)
    table.add_column("Project", style="yellow")
    table.add_column("Domain", style="dim")

    for tid in task_ids:
        table.add_row(tid.name, tid.project, tid.domain)

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Total: {len(task_ids)} component(s).  "
        "Run [bold]ml-plat component info <name>[/bold] for details.[/dim]\n"
    )


# ── info ──────────────────────────────────────────────────────────────────────


@app.command("info")
def component_info(
    name: str = typer.Argument(
        ...,
        help="Task name (e.g. components.data.ingest.download_dataset)",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Task version (default: latest)",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="Flyte project (default: $FLYTE_PROJECT)",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Flyte domain (default: $FLYTE_DOMAIN)",
    ),
):
    """🔍 Show detailed info about a component: inputs, outputs, image, and resources.

    Fetches task metadata directly from FlyteAdmin.

    Examples:
      ml-plat component info components.training.finetune.finetune_lm
      ml-plat component info download_dataset --version v1
    """
    try:
        task = _fetch_task(
            name=name,
            version=version,
            project=project,
            domain=domain,
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to fetch task:[/bold red] {exc}")
        raise typer.Exit(1)

    if task is None:
        console.print(
            f"[bold red]Component not found:[/bold red] {name}\n"
            "Run [bold]ml-plat component list[/bold] to see available components."
        )
        raise typer.Exit(1)

    # ── Header ──────────────────────────────────────────────────────────────
    task_id = task.id
    header = (
        f"[bold white]{task_id.name}[/bold white]  "
        f"[dim]v{task_id.version}[/dim]  "
        f"[yellow]{task_id.project}/{task_id.domain}[/yellow]"
    )

    # ── Inputs table ────────────────────────────────────────────────────────
    inputs_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
    )
    inputs_table.add_column("Parameter", style="cyan", no_wrap=True)
    inputs_table.add_column("Type", style="yellow")

    if task.interface and task.interface.inputs:
        for var_name, var in task.interface.inputs.items():
            type_str = str(var.type) if hasattr(var, "type") else str(var)
            inputs_table.add_row(var_name, type_str)

    # ── Outputs table ───────────────────────────────────────────────────────
    outputs_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
    )
    outputs_table.add_column("Parameter", style="cyan", no_wrap=True)
    outputs_table.add_column("Type", style="yellow")

    if task.interface and task.interface.outputs:
        for var_name, var in task.interface.outputs.items():
            type_str = str(var.type) if hasattr(var, "type") else str(var)
            outputs_table.add_row(var_name, type_str)

    # ── Container / Image ───────────────────────────────────────────────────
    image_str = "[dim]N/A[/dim]"
    if task.container and task.container.image:
        image_str = f"[magenta]{task.container.image}[/magenta]"

    console.print()
    console.print(
        Panel(
            header,
            title="[bold cyan]Component Info[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print(f"\n[bold]Image:[/bold]  {image_str}")

    console.print("\n[bold]Inputs:[/bold]")
    console.print(inputs_table)

    console.print("[bold]Outputs:[/bold]")
    console.print(outputs_table)
    console.print()


# ── register ─────────────────────────────────────────────────────────────────


def _register_one(
    component_dir: Path,
    component_name: str,
    project: str,
    domain: str,
    version: Optional[str],
    image: Optional[str],
) -> bool:
    """Register a single component directory. Returns True on success."""
    import shutil

    # Read image from component.yaml if available
    meta_path = component_dir / COMPONENT_META_FILE
    image_short = ""
    if meta_path.exists():
        meta = _load_component_yaml(meta_path)
        image_short = meta.get("image", "")

    resolved_image = _resolve_image(image_short, image)
    ver = version or _content_hash(component_dir)

    pyflyte_bin = shutil.which("pyflyte") or os.path.join(sys.prefix, "bin", "pyflyte")
    cmd = [pyflyte_bin, "register", "--project", project, "--domain", domain]
    cmd.extend(["--version", ver])
    if resolved_image:
        cmd.extend(["--image", resolved_image])
    cmd.append(str(component_dir))

    console.print(
        f"  [bold]{component_name}[/bold]  "
        f"version=[cyan]{ver}[/cyan]  "
        f"image=[dim]{resolved_image or 'default'}[/dim]"
    )
    console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            console.print(f"    {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            console.print(f"    [dim]{line}[/dim]")

    if result.returncode == 0:
        console.print(f"  [green]✅ {component_name} registered[/green]\n")
        return True
    else:
        console.print(f"  [red]❌ {component_name} failed (exit {result.returncode})[/red]\n")
        return False


@app.command("register")
def register_component(
    name: Optional[str] = typer.Argument(
        None,
        help=(
            "Component path: 'training/lora_finetune' (single component), "
            "'training' (all in category), or a filesystem path. "
            "Omit when using --all."
        ),
    ),
    all_components: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Register every component independently",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="Flyte project (default: $FLYTE_PROJECT)",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Flyte domain (default: $FLYTE_DOMAIN)",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Version tag (default: content-hash of all files in component dir)",
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image",
        "-i",
        help="Container image override (default: auto from component.yaml + versions.env)",
    ),
):
    """Register component tasks with FlyteAdmin via ``pyflyte register``.

    Each component is registered independently with its own content-hash
    version derived from ALL files in its directory (not just .py).  Unchanged
    components keep the same version so Flyte skips re-registration.

    The component image is read from ``component.yaml`` and resolved to
    a full ECR URI via ``versions.env``.

    Examples:\n
      ml-plat component register training/lora_finetune\n
      ml-plat component register training\n
      ml-plat component register --all\n
      ml-plat component register --all --version v2.0\n
      ml-plat component register ./my/path --image img:v1\n
    """
    proj = project or os.getenv("FLYTE_PROJECT", "flytesnacks")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    # Auto-create the Flyte project if it doesn't exist yet
    try:
        _ensure_project(proj)
    except SystemExit:
        raise
    except Exception as exc:
        console.print(f"[bold red]Failed to verify project:[/bold red] {exc}")
        raise typer.Exit(1)

    # ── Determine which components to register ───────────────────────────
    targets: list[tuple[Path, str]] = []  # (directory, display_name)

    if all_components:
        comps = _discover_components()
        if not comps:
            console.print("[red]No component directories found.[/red]")
            raise typer.Exit(1)
        for c in comps:
            targets.append((c["path"], f"{c['category']}/{c['name']}"))
    elif name:
        # Try: category/component  (e.g. "training/lora_finetune")
        if "/" in name:
            candidate = COMPONENTS_ROOT / name
            meta = candidate / COMPONENT_META_FILE
            if meta.exists():
                targets.append((candidate, name))
            else:
                # Fall back to raw filesystem path
                p = Path(name)
                if p.exists():
                    targets.append((p, p.name))
                else:
                    console.print(
                        f"[red]'{name}' is not a known component and "
                        f"the path does not exist.[/red]"
                    )
                    raise typer.Exit(1)
        else:
            # Try: category name  (e.g. "training" → all components in it)
            cat_dir = COMPONENTS_ROOT / name
            if cat_dir.is_dir() and (cat_dir / "__init__.py").exists():
                comps = _discover_components(cat_dir)
                if comps:
                    for c in comps:
                        targets.append((c["path"], f"{name}/{c['name']}"))
                else:
                    console.print(
                        f"[yellow]Category '{name}' has no " f"component.yaml files.[/yellow]"
                    )
                    raise typer.Exit(1)
            else:
                # Check if it's a bare component name (search all categories)
                comps = _discover_components()
                matches = [c for c in comps if c["name"] == name]
                if len(matches) == 1:
                    c = matches[0]
                    targets.append((c["path"], f"{c['category']}/{c['name']}"))
                elif len(matches) > 1:
                    cats = ", ".join(f"{c['category']}/{c['name']}" for c in matches)
                    console.print(
                        f"[red]Ambiguous: '{name}' found in "
                        f"multiple categories: {cats}[/red]\n"
                        "[dim]Use category/name to disambiguate.[/dim]"
                    )
                    raise typer.Exit(1)
                else:
                    # Raw path fallback
                    p = Path(name)
                    if p.exists():
                        targets.append((p, p.name))
                    else:
                        available = _category_names()
                        console.print(
                            f"[red]'{name}' is not a known component "
                            f"or category.[/red]\n"
                            f"[dim]Categories: "
                            f"{', '.join(available)}[/dim]"
                        )
                        raise typer.Exit(1)
    else:
        categories = _category_names()
        console.print(
            "[red]Provide a component name or use --all.[/red]\n"
            f"[dim]Categories: {', '.join(categories)}[/dim]"
        )
        raise typer.Exit(1)

    # ── Register each target ─────────────────────────────────────────────
    console.print(
        f"\n[bold]Registering {len(targets)} component(s) to Flyte " f"({proj}/{dom}):[/bold]\n"
    )

    successes = 0
    failures = 0
    for comp_dir, comp_name in targets:
        ok = _register_one(comp_dir, comp_name, proj, dom, version, image)
        if ok:
            successes += 1
        else:
            failures += 1

    # ── Summary ──────────────────────────────────────────────────────────
    if failures == 0:
        console.print(
            f"[bold green]All {successes} component(s) registered " f"successfully.[/bold green]"
        )
    else:
        console.print(
            f"[bold red]{failures} component(s) failed, " f"{successes} succeeded.[/bold red]"
        )
        raise typer.Exit(1)
