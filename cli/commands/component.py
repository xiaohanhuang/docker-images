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
    ml-plat component versions <name>                 — list registered versions
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
_CLI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CLI_DIR.parent.parent
COMPONENTS_ROOT = _REPO_ROOT / "projects" / "components" / "components"

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
    candidates = [
        Path("projects/components/images/versions.env"),
        Path("images/versions.env"),
    ]
    versions_env = None
    for c in candidates:
        if c.exists():
            versions_env = c
            break
    vals: Dict[str, str] = {}
    if versions_env is None:
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
    image_tag: Optional[str] = None,
) -> Optional[str]:
    """Return the full container image URI.

    Priority: explicit ``--image`` flag  >  component.yaml ``image_tag``
              >  per-image tag in versions.env  >  global IMAGE_TAG.
    ``image_short`` is the short name from component.yaml (e.g. "data-cpu").
    """
    if explicit_image:
        return explicit_image
    if not image_short:
        return None
    env = _parse_versions_env()
    registry = env.get("ECR_REGISTRY")
    repo = env.get("ECR_REPO")

    # Priority: component.yaml image_tag > per-image env > global
    per_image_key = f"IMAGE_TAG_{image_short.upper().replace('-', '_')}"
    tag = image_tag or env.get(per_image_key) or env.get("IMAGE_TAG")

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


def _normalize_task_name(name: str) -> List[str]:
    """Return candidate Flyte task names to try.

    Old registrations used e.g. ``components.data.text_chunker.chunk_documents``
    while current registrations use ``components.data.text_chunker.task.chunk_documents``
    (the ``.task.`` segment comes from the ``task.py`` module).  Return both
    variants so ``component info`` finds the latest regardless of what the user types.
    """
    candidates = [name]
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        prefix, func = parts
        if ".task." not in name:
            candidates.append(f"{prefix}.task.{func}")
        else:
            # User gave the .task. form — also try without it
            candidates.append(name.replace(".task.", "."))
    return candidates


def _fetch_task(
    name: str,
    version: Optional[str] = None,
    project: Optional[str] = None,
    domain: Optional[str] = None,
):
    """Fetch a single task from FlyteAdmin by name (and optionally version).

    Tries name variants with and without the ``.task.`` module segment so that
    both legacy and current registrations are found.
    """
    remote = _get_remote()
    proj = project or os.getenv("FLYTE_PROJECT", "flytesnacks")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    candidates = _normalize_task_name(name)

    if version:
        last_exc = None
        for candidate in candidates:
            try:
                return remote.fetch_task(project=proj, domain=dom, name=candidate, version=version)
            except Exception as exc:
                last_exc = exc
                continue
        # Re-raise the last exception so callers see "Failed to fetch"
        if last_exc is not None:
            raise last_exc
        return None

    # No version specified — try each candidate and pick the latest overall
    from flytekit.models.admin.common import Sort
    from flytekit.models.common import NamedEntityIdentifier

    latest_first = Sort(key="created_at", direction=Sort.Direction.DESCENDING)
    best = None

    for candidate in candidates:
        identifier = NamedEntityIdentifier(project=proj, domain=dom, name=candidate)
        try:
            tasks, _ = remote.client.list_tasks_paginated(
                identifier,
                limit=1,
                sort_by=latest_first,
            )
        except Exception:
            continue
        if tasks:
            task_id = tasks[0].id
            if best is None or task_id.version > best.version:
                best = task_id

    if best is None:
        return None

    return remote.fetch_task(
        project=best.project,
        domain=best.domain,
        name=best.name,
        version=best.version,
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

    # Filter out stale registrations (archived but still returned by API)
    task_ids = [
        tid for tid in task_ids if ".__init__." not in tid.name and "._task." not in tid.name
    ]

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


# ── versions ─────────────────────────────────────────────────────────────────


@app.command("versions")
def list_versions(
    name: str = typer.Argument(
        ...,
        help="Task name (e.g. components.data.ingest.task.download_dataset)",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum number of versions to show",
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
    """📋 List all registered versions of a component.

    Queries FlyteAdmin for all versions of a task, newest first.

    Examples:
      ml-plat component versions components.data.ingest.task.download_dataset
      ml-plat component versions download_dataset --limit 5
    """
    from flytekit.models.common import NamedEntityIdentifier

    remote = _get_remote()
    proj = project or os.getenv("FLYTE_PROJECT", "flytesnacks")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    candidates = _normalize_task_name(name)
    all_tasks = []
    last_exc = None

    for candidate in candidates:
        identifier = NamedEntityIdentifier(project=proj, domain=dom, name=candidate)
        try:
            tasks, _ = remote.client.list_tasks_paginated(identifier, limit=limit)
            all_tasks.extend(tasks)
        except Exception as exc:
            last_exc = exc
            continue

    if not all_tasks and last_exc is not None:
        console.print(f"[bold red]Failed to fetch versions:[/bold red] {last_exc}")
        raise typer.Exit(1)

    if not all_tasks:
        console.print(f"[yellow]No versions found for '{name}'.[/yellow]")
        raise typer.Exit(0)

    table = Table(
        title=f"[bold cyan]Versions: {name}[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("#", style="dim", no_wrap=True)
    table.add_column("Version", style="bold white", no_wrap=True)
    table.add_column("Created", style="yellow")

    for idx, task in enumerate(all_tasks, 1):
        created = ""
        if hasattr(task, "closure") and hasattr(task.closure, "created_at"):
            ts = task.closure.created_at
            if ts:
                created = ts.strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(str(idx), task.id.version, created)

    console.print()
    console.print(table)
    console.print(f"\n[dim]Showing {len(all_tasks)} version(s).[/dim]\n")


# ── Type & Literal formatting ───────────────────────────────────────────────


def _format_literal_type(lt, escape_rich: bool = False) -> str:
    """Convert a Flyte LiteralType to a human-readable Python type string."""
    if not hasattr(lt, "simple"):
        return str(lt)
    simple_types = {
        0: "None",
        1: "int",
        2: "float",
        3: "str",
        4: "bool",
        5: "datetime",
        6: "timedelta",
        7: "bytes",
        8: "FlyteError",
        9: "dict",
    }
    if lt.simple is not None and lt.simple in simple_types:
        return simple_types[lt.simple]
    if lt.blob is not None:
        return "FlyteDirectory" if lt.blob.dimensionality == 1 else "FlyteFile"
    if lt.collection_type is not None:
        inner = _format_literal_type(lt.collection_type, escape_rich=escape_rich)
        res = f"List[{inner}]"
        return res.replace("[", "\\[") if escape_rich else res
    if lt.map_value_type is not None:
        inner = _format_literal_type(lt.map_value_type, escape_rich=escape_rich)
        res = f"Dict[str, {inner}]"
        return res.replace("[", "\\[") if escape_rich else res
    if lt.enum_type is not None:
        return "Enum"
    if lt.structured_dataset_type is not None:
        return "StructuredDataset"
    if lt.union_type is not None:
        variants = [
            _format_literal_type(v, escape_rich=escape_rich) for v in lt.union_type.variants
        ]
        # Filter out None if there are other variants, making it Optional
        if "None" in variants and len(variants) > 1:
            other = [v for v in variants if v != "None"]
            if len(other) == 1:
                res = f"Optional[{other[0]}]"
            else:
                res = f"Optional[Union[{', '.join(other)}]]"
            return res.replace("[", "\\[") if escape_rich else res
        return " | ".join(variants)
    return str(lt)


def _format_literal_value(lit, _sentinel=None) -> Any:
    """Convert a Flyte Literal to a Python literal value.

    When *_sentinel* is provided, return it instead of ``None`` for
    unsupported literal types so callers can distinguish "unsupported" from
    an explicit ``None`` default.
    """
    if lit.scalar:
        s = lit.scalar
        if s.primitive:
            p = s.primitive
            if p.string_value is not None:
                return p.string_value
            if p.integer is not None:
                return p.integer
            if p.float_value is not None:
                return p.float_value
            if p.boolean is not None:
                return p.boolean
        if s.none_type is not None:
            return None
    if lit.collection:
        return [_format_literal_value(i, _sentinel=_sentinel) for i in lit.collection.literals]
    if lit.map:
        return {
            k: _format_literal_value(v, _sentinel=_sentinel) for k, v in lit.map.literals.items()
        }
    return _sentinel if _sentinel is not None else None


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
        f"[dim]{task_id.version}[/dim]  "
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
            type_str = (
                _format_literal_type(var.type, escape_rich=True)
                if hasattr(var, "type")
                else str(var)
            )
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
            type_str = (
                _format_literal_type(var.type, escape_rich=True)
                if hasattr(var, "type")
                else str(var)
            )
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


def _bump_semver(version: str) -> str:
    """Increment the patch segment of a semver-like version string.

    'v1.3.0' -> 'v1.4.0', 'v2.0.0' -> 'v2.1.0', 'v1' -> 'v2'.
    Non-semver strings get a '-1' suffix.
    """
    prefix = ""
    v = version
    if v.startswith("v"):
        prefix = "v"
        v = v[1:]

    parts = v.split(".")
    if len(parts) >= 2:
        try:
            parts[-2] = str(int(parts[-2]) + 1)
            return prefix + ".".join(parts)
        except ValueError:
            pass
    elif len(parts) == 1:
        try:
            return prefix + str(int(parts[0]) + 1)
        except ValueError:
            pass
    return version + "-1"


def _check_interface_compatibility(
    old_interface,
    new_interface,
    task_name: str,
) -> list[str]:
    """Compare two task interfaces and return breaking change descriptions."""
    violations = []
    old_inputs = old_interface.inputs or {}
    new_inputs = new_interface.inputs or {}
    old_outputs = old_interface.outputs or {}
    new_outputs = new_interface.outputs or {}

    # Check removed inputs
    for name in old_inputs:
        if name not in new_inputs:
            violations.append(f"{task_name}: removed input '{name}' — breaks callers")

    # Check changed input types
    for name in old_inputs:
        if name in new_inputs:
            old_type = _format_literal_type(old_inputs[name].type)
            new_type = _format_literal_type(new_inputs[name].type)
            if old_type != new_type:
                violations.append(
                    f"{task_name}: input '{name}' type changed " f"({old_type} → {new_type})"
                )

    # Check new required inputs (non-Optional)
    for name in new_inputs:
        if name not in old_inputs:
            type_str = _format_literal_type(new_inputs[name].type)
            is_optional = type_str.startswith("Optional")
            has_default = hasattr(new_inputs[name], "required") and not new_inputs[name].required
            if not is_optional and not has_default:
                violations.append(
                    f"{task_name}: new required input '{name}' " f"({type_str}) — must be Optional"
                )

    # Check removed outputs
    for name in old_outputs:
        if name not in new_outputs:
            violations.append(f"{task_name}: removed output '{name}' — breaks callers")

    # Check changed output types
    for name in old_outputs:
        if name in new_outputs:
            old_type = _format_literal_type(old_outputs[name].type)
            new_type = _format_literal_type(new_outputs[name].type)
            if old_type != new_type:
                violations.append(
                    f"{task_name}: output '{name}' type changed " f"({old_type} → {new_type})"
                )

    return violations


def _serialize_local_tasks(
    component_dir: Path,
    pyflyte_bin: str,
    image: str,
) -> Dict[str, Any]:
    """Serialize a component locally and return task name → proto interface map.

    Uses ``pyflyte package`` to produce protobuf specs without contacting
    FlyteAdmin, then parses the .pb files to extract task interfaces.
    Returns an empty dict if serialization fails.
    """
    import tarfile
    import tempfile

    from flyteidl.admin.task_pb2 import TaskSpec

    with tempfile.TemporaryDirectory() as tmpdir:
        tgz_path = os.path.join(tmpdir, "out.tgz")
        cmd = [
            pyflyte_bin,
            "package",
            "-f",
            "-o",
            tgz_path,
        ]
        if image:
            cmd.extend(["--image", image])
        cmd.append(str(component_dir))

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(component_dir.parent.parent)
        )
        if result.returncode != 0:
            return {}

        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir)
        try:
            with tarfile.open(tgz_path) as tar:
                tar.extractall(extract_dir, filter="data")
        except Exception:
            return {}

        interfaces: Dict[str, Any] = {}
        for pb_file in Path(extract_dir).glob("*.pb"):
            try:
                spec = TaskSpec()
                with open(pb_file, "rb") as f:
                    spec.ParseFromString(f.read())
                task_name = spec.template.id.name
                if spec.template.HasField("interface"):
                    interfaces[task_name] = spec.template.interface
            except Exception:
                continue

        return interfaces


def _convert_proto_interface(proto_iface) -> Any:
    """Wrap a protobuf TypedInterface into a dict-like object for comparison.

    Returns a mock-like object with .inputs and .outputs dicts whose values
    have a .type attribute compatible with ``_format_literal_type``.
    """
    from flytekit.models.interface import TypedInterface

    try:
        return TypedInterface.from_flyte_idl(proto_iface)
    except Exception:
        return None


def _pre_register_compat_check(
    component_dir: Path,
    component_name: str,
    project: str,
    domain: str,
    pyflyte_bin: str,
    image: str,
) -> list[str]:
    """Check compatibility before registration. Returns violations list.

    1. Serialize the component locally (``pyflyte package``) to get new interfaces
    2. Fetch old interfaces from FlyteAdmin
    3. Compare and return any breaking changes
    """
    all_violations: list[str] = []

    # Step 1: Serialize locally to get new task interfaces
    new_interfaces = _serialize_local_tasks(component_dir, pyflyte_bin, image)
    if not new_interfaces:
        return []  # can't serialize — skip check

    # Step 2: For each new task, fetch the latest registered version
    for task_name, new_proto_iface in new_interfaces.items():
        try:
            old_task = _fetch_task(name=task_name, project=project, domain=domain)
        except Exception:
            continue  # not registered yet — nothing to compare

        if not old_task or not old_task.interface:
            continue

        new_iface = _convert_proto_interface(new_proto_iface)
        if not new_iface:
            continue

        violations = _check_interface_compatibility(old_task.interface, new_iface, task_name)
        all_violations.extend(violations)

    return all_violations


def _register_one(
    component_dir: Path,
    component_name: str,
    project: str,
    domain: str,
    version: Optional[str],
    image: Optional[str],
    auto_bump: bool = False,
    skip_compat_check: bool = False,
) -> tuple[bool, Optional[str]]:
    """Register a single component directory.

    Returns (success, resolved_version).  When *auto_bump* is True and Flyte
    rejects the version because the task structure changed, the version is
    automatically incremented and the registration retried once.
    """
    import shutil

    # Read image and version from component.yaml if available
    meta_path = component_dir / COMPONENT_META_FILE
    image_short = ""
    meta_version = ""
    meta_image_tag = ""
    if meta_path.exists():
        meta = _load_component_yaml(meta_path)
        image_short = meta.get("image", "")
        meta_version = meta.get("version", "")
        meta_image_tag = meta.get("image_tag", "")

    resolved_image = _resolve_image(image_short, image, image_tag=meta_image_tag or None)
    ver = (
        version
        or os.getenv("ML_PLAT_COMPONENT_VERSION")
        or (f"v{meta_version}" if meta_version else "")
        or _content_hash(component_dir)
    )

    pyflyte_bin = shutil.which("pyflyte") or os.path.join(sys.prefix, "bin", "pyflyte")

    # Pre-registration backward compatibility check
    try:
        violations = _pre_register_compat_check(
            component_dir, component_name, project, domain, pyflyte_bin, resolved_image or ""
        )
        if violations:
            if skip_compat_check:
                console.print(
                    f"  [bold yellow]⚠ {component_name}: backward-incompatible "
                    f"changes detected, but check was skipped:[/bold yellow]"
                )
                for v in violations:
                    console.print(f"    [yellow]• {v}[/yellow]")
                console.print(
                    "  [dim]Registration will continue because "
                    "--skip-compat-check was provided.[/dim]\n"
                )
            else:
                console.print(
                    f"  [bold red]❌ {component_name}: blocked — "
                    f"backward-incompatible changes detected:[/bold red]"
                )
                for v in violations:
                    console.print(f"    [red]• {v}[/red]")
                console.print(
                    "  [dim]New inputs must be Optional with defaults. "
                    "Existing inputs/outputs must not be removed or change type.\n"
                    "  Use --skip-compat-check to bypass.[/dim]\n"
                )
                return False, None
    except Exception:
        pass  # best-effort — don't block if check itself fails

    def _try_register(v: str) -> subprocess.CompletedProcess[str]:
        cmd = [pyflyte_bin, "register", "--project", project, "--domain", domain]
        cmd.extend(["--version", v])
        if resolved_image:
            cmd.extend(["--image", resolved_image])
        cmd.append(str(component_dir))

        console.print(
            f"  [bold]{component_name}[/bold]  "
            f"version=[cyan]{v}[/cyan]  "
            f"image=[dim]{resolved_image or 'default'}[/dim]"
        )
        console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")

        return subprocess.run(cmd, capture_output=True, text=True)

    result = _try_register(ver)
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            console.print(f"    {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            console.print(f"    [dim]{line}[/dim]")

    # Auto-bump on "different structure already exists" conflict
    if result.returncode != 0 and auto_bump:
        combined = (result.stdout or "") + (result.stderr or "")
        if "different structure already exists" in combined:
            new_ver = _bump_semver(ver)
            console.print(
                f"  [yellow]⚠ Version conflict — auto-bumping " f"{ver} → {new_ver}[/yellow]"
            )
            result = _try_register(new_ver)
            if result.stdout:
                for line in result.stdout.strip().splitlines():
                    console.print(f"    {line}")
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    console.print(f"    [dim]{line}[/dim]")
            if result.returncode == 0:
                console.print(
                    f"  [green]✅ {component_name} registered" f" (bumped to {new_ver})[/green]\n"
                )
                return True, new_ver

    if result.returncode == 0:
        console.print(f"  [green]✅ {component_name} registered[/green]\n")
        return True, ver
    else:
        console.print(f"  [red]❌ {component_name} failed (exit {result.returncode})[/red]\n")
        return False, None


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
        help="Version tag override (default: from component.yaml, or $ML_PLAT_COMPONENT_VERSION)",
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image",
        "-i",
        help="Container image override (default: auto from component.yaml + versions.env)",
    ),
    component_path: Optional[str] = typer.Option(
        None,
        "--component-path",
        help="Root directory containing components (default: projects/components/components/)",
    ),
    skip_compat_check: bool = typer.Option(
        False,
        "--skip-compat-check",
        help="Skip backward-compatibility check before registration",
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
    comp_root = Path(component_path) if component_path else COMPONENTS_ROOT

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
        comps = _discover_components(comp_root)
        if not comps:
            console.print("[red]No component directories found.[/red]")
            raise typer.Exit(1)
        for c in comps:
            targets.append((c["path"], f"{c['category']}/{c['name']}"))
    elif name:
        # Try: category/component  (e.g. "training/lora_finetune")
        if "/" in name:
            candidate = comp_root / name
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
            cat_dir = comp_root / name
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
                comps = _discover_components(comp_root)
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
        ok, _ver = _register_one(
            comp_dir,
            comp_name,
            proj,
            dom,
            version,
            image,
            skip_compat_check=skip_compat_check,
        )
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


# ── gen-stubs ────────────────────────────────────────────────────────────────


@app.command("gen-stubs")
def generate_stubs(
    output: Path = typer.Option(
        _REPO_ROOT / "projects/components/sdk/ml_platform_sdk/components.py",
        "--output",
        "-o",
        help="Path to the generated components.py file",
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Flyte project (default: $FLYTE_PROJECT)"
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d", help="Flyte domain (default: $FLYTE_DOMAIN)"
    ),
):
    """🤖 Auto-generate ReferenceTask stubs from registered components.

    Queries FlyteAdmin for all tasks matching the 'components.*' prefix and
    rebuilds the SDK components.py file with correct type signatures and
    parameter defaults.

    Example:
      ml-plat component gen-stubs
    """
    proj = project or os.getenv("FLYTE_PROJECT", "flytesnacks")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    # Validate output path early, before making any remote calls
    if not output.exists():
        console.print(f"[red]Output file {output} does not exist. Cannot find header.[/red]")
        raise typer.Exit(1)

    console.print(f"Introspecting components from Flyte ([bold]{proj}/{dom}[/bold])...")

    try:
        task_ids = _list_task_ids(project=proj, domain=dom)
    except Exception as exc:
        console.print(f"[bold red]Failed to connect to Flyte:[/bold red] {exc}")
        raise typer.Exit(1)

    # Filter for components.* prefix and exclude __init__ tasks
    component_ids = [
        tid
        for tid in task_ids
        if tid.name.startswith("components.")
        and ".__init__." not in tid.name
        and "._task." not in tid.name
    ]

    if not component_ids:
        console.print("[yellow]No components found matching 'components.*' prefix.[/yellow]")
        raise typer.Exit(0)

    # Group tasks by category and keep only the latest version of each unique
    # task name.  Deduplicate using the fetched task's canonical Flyte name so
    # legacy/current aliases do not generate duplicate stubs.
    def _canonical_task_name(task: Any, fallback_name: str) -> str:
        try:
            return task.task.id.name  # type: ignore[union-attr]
        except AttributeError:
            return fallback_name

    unique_tasks: Dict[str, Any] = {}
    for tid in component_ids:
        task = _fetch_task(name=tid.name, project=proj, domain=dom)
        if not task:
            continue
        canonical = _canonical_task_name(task, tid.name)
        if canonical not in unique_tasks:
            unique_tasks[canonical] = task

    categories: Dict[str, List[Any]] = {
        "Data": [],
        "Training": [],
        "Evaluation": [],
        "Serving": [],
        "Other": [],
    }

    for name, task in sorted(unique_tasks.items()):
        parts = name.split(".")
        # components.<category>.<name>...
        cat_name = parts[1].capitalize() if len(parts) > 1 else "Other"
        if cat_name not in categories:
            categories[cat_name] = []
        categories[cat_name].append(task)

    # ── Code generation ─────────────────────────────────────────────────────

    output_lines = []

    # Read the existing header from components.py (everything before the stub list).
    # The real SDK file has section headers like "# ── Environment-driven config"
    # *above* the stubs, so we must target the stub-specific marker pattern
    # ("# ── <Category> Components") or the first _comp( assignment.
    existing_content = output.read_text()
    # Find the first stub-list section header (e.g. "# ── Data Components")
    import re as _re

    _stub_section = _re.search(r"\n# ── \w+ Components", existing_content)
    header_end = _stub_section.start() if _stub_section else -1
    if header_end == -1:
        # Fallback: first _comp( assignment
        _comp_assign = _re.search(r"\n\w+ = _comp\(", existing_content)
        header_end = _comp_assign.start() if _comp_assign else -1

    if header_end != -1:
        output_lines.append(existing_content[:header_end].strip())
    else:
        # Fallback: find the _comp helper definition and preserve
        # everything through the helper's return statement.
        comp_def_start = existing_content.find("def _comp(")
        if comp_def_start == -1:
            console.print(
                f"[red]Could not find a generated-content marker "
                f"or the _comp helper in {output}.[/red]"
            )
            raise typer.Exit(1)

        func_return = existing_content.find("return Component", comp_def_start)
        if func_return == -1:
            console.print(
                f"[red]Could not determine the end of the _comp helper in {output}; "
                "expected a 'return Component' line.[/red]"
            )
            raise typer.Exit(1)

        func_end = existing_content.find("\n", func_return)
        if func_end == -1:
            func_end = len(existing_content)

        output_lines.append(existing_content[:func_end].strip())

    for cat_name, tasks in categories.items():
        if not tasks:
            continue

        section = f"# ── {cat_name} Components"
        output_lines.append(f"\n\n{section} {'─' * (72 - len(section))}\n")

        for task in tasks:
            short_name = task.id.name.split(".")[-1]
            if short_name == "task":  # handle components.cat.name.task.func
                short_name = task.id.name.split(".")[-2]

            inputs = {k: _format_literal_type(v.type) for k, v in task.interface.inputs.items()}
            outputs = {k: _format_literal_type(v.type) for k, v in task.interface.outputs.items()}

            # Extract defaults from parameter map.  Use a sentinel to
            # distinguish "no default / unsupported literal" from an explicit
            # default of None.
            _no_default = object()
            defaults = {}
            if hasattr(task.interface, "inputs") and task.interface.inputs:
                for k, v in task.interface.inputs.items():
                    has_default = (
                        hasattr(v, "required")
                        and not v.required
                        and hasattr(v, "default")
                        and v.default is not None
                    )
                    if has_default:
                        val = _format_literal_value(v.default, _sentinel=_no_default)
                        if val is not _no_default:
                            defaults[k] = val

            # Format dictionaries for code
            def _fmt_dict(d: Dict[str, str]) -> str:
                if not d:
                    return "{}"
                # We want the values (types) as literals, not strings, if possible
                # e.g. "path": str  instead of  "path": "str"
                items = []
                for k, v in d.items():
                    items.append(f'"{k}": {v}')
                return "{" + ", ".join(items) + "}"

            input_str = _fmt_dict(inputs)
            output_str = _fmt_dict(outputs)

            line = f"{short_name} = _comp(\n"
            line += f'    "{task.id.name}",\n'
            line += f"    inputs={input_str},\n"
            line += f"    outputs={output_str},"
            if defaults:
                line += f"\n    defaults={repr(defaults)},"
            line += "\n)"
            output_lines.append(line)

    output.write_text("\n".join(output_lines) + "\n")
    total = sum(len(t) for t in categories.values())
    console.print(f"[green]✅ Generated {total} stubs in {output}[/green]")


# ── run ──────────────────────────────────────────────────────────────────────


def _parse_typed_inputs(raw_inputs: List[str], interface) -> Dict[str, Any]:
    """Parse ``key=value`` CLI args, casting types from the FlyteAdmin interface.

    For ``FlyteFile`` / ``FlyteDirectory`` inputs: accepts S3 URIs or local
    paths as-is (Flyte handles the upload).  Primitive types are cast from
    string using the ``LiteralType`` metadata from the remote ``TaskClosure``.
    """
    from flytekit.types.directory import FlyteDirectory
    from flytekit.types.file import FlyteFile

    result: Dict[str, Any] = {}
    for item in raw_inputs:
        if "=" not in item:
            console.print(f"[red]Invalid input format: {item}  (expected key=value)[/red]")
            raise typer.Exit(1)
        key, _, value = item.partition("=")

        if key not in interface:
            available = ", ".join(sorted(interface.keys()))
            console.print(f"[red]Unknown input: {key}[/red]\n" f"[dim]Available: {available}[/dim]")
            raise typer.Exit(1)

        var = interface[key]
        lt = var.type

        # Blob types → FlyteFile / FlyteDirectory
        if lt.blob is not None:
            if lt.blob.dimensionality == 0:  # SINGLE → FlyteFile
                result[key] = FlyteFile(path=value)
            else:  # MULTIPART → FlyteDirectory
                result[key] = FlyteDirectory(path=value)
        elif lt.simple is not None:
            simple = str(lt.simple)
            if "INTEGER" in simple:
                result[key] = int(value)
            elif "FLOAT" in simple:
                result[key] = float(value)
            elif "BOOLEAN" in simple:
                result[key] = value.lower() in ("true", "1", "yes")
            else:
                result[key] = value
        else:
            result[key] = value

    return result


@app.command("run")
def run_component(
    name: str = typer.Argument(
        ...,
        help="Component name (e.g. lora_finetune, hf_dataset_loader)",
    ),
    inputs: Optional[List[str]] = typer.Argument(
        None,
        help="Inputs as key=value pairs (e.g. base_model=llama epochs=3)",
    ),
    version: Optional[str] = typer.Option(
        None, "--version", "-v", help="Task version (default: latest)"
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Flyte project (default: $FLYTE_PROJECT)"
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d", help="Flyte domain (default: $FLYTE_DOMAIN)"
    ),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch execution after submitting"),
):
    """🚀 Run a registered component with the given inputs.

    Fetches the task's interface from FlyteAdmin to validate and cast inputs.
    FlyteFile/FlyteDirectory inputs accept S3 URIs or local paths.

    Examples:\n
      ml-plat component run lora_finetune \\
          base_model=meta-llama/Llama-3.1-8B \\
          train_data_path=s3://bucket/data.jsonl \\
          epochs=3 --watch\n
      ml-plat component run hf_dataset_loader \\
          dataset_name=tatsu-lab/alpaca\n
    """
    from cli.utils import flyte_console_url

    remote = _get_remote()

    # Fetch task from FlyteAdmin (gets interface with actual types)
    task = _fetch_task(name=name, version=version, project=project, domain=domain)
    if task is None:
        console.print(
            f"[bold red]Component not found:[/bold red] {name}\n"
            "Run [bold]ml-plat component list[/bold] to see available components."
        )
        raise typer.Exit(1)

    # Parse inputs using the remote interface
    typed_inputs = _parse_typed_inputs(inputs or [], task.interface.inputs)

    # Execute
    try:
        execution = remote.execute(task, inputs=typed_inputs, wait=False)
    except Exception as exc:
        console.print(f"[bold red]Failed to submit:[/bold red] {exc}")
        raise typer.Exit(1)

    exec_id = execution.id.name
    url = flyte_console_url(execution.id.project, execution.id.domain, exec_id)

    console.print(
        Panel.fit(
            f"[bold green]✅ Component submitted![/bold green]\n"
            f"  Execution ID : [bold]{exec_id}[/bold]\n"
            f"  Flyte URL    : {url}",
            border_style="green",
        )
    )

    if watch:
        try:
            from cli.commands.workflow import _watch_execution

            _watch_execution(exec_id, remote)
        except ImportError:
            console.print(f"\n[dim]Run:[/dim] ml-plat workflow watch {exec_id}")


# ── bump-image ───────────────────────────────────────────────────────────────


def _find_versions_env() -> Optional[Path]:
    """Locate versions.env file."""
    for c in [
        Path("projects/components/images/versions.env"),
        Path("images/versions.env"),
    ]:
        if c.exists():
            return c
    return None


def _bump_semver_tag(tag: str) -> str:
    """Increment the patch segment of a semver tag (e.g. 1.1.0 → 1.2.0)."""
    parts = tag.split(".")
    if len(parts) == 3:
        parts[1] = str(int(parts[1]) + 1)
        parts[2] = "0"
        return ".".join(parts)
    return tag + ".1"


@app.command("bump-image")
def bump_image(
    image_name: str = typer.Argument(
        ...,
        help="Image short name (e.g. 'data-cpu', 'ml-gpu')",
    ),
    set_version: Optional[str] = typer.Option(
        None,
        "--set",
        "-s",
        help="Set an explicit version instead of auto-incrementing",
    ),
):
    """🔄 Bump the per-image version tag in versions.env.

    Auto-increments the minor version for the given image, or sets an explicit
    version with --set.  Creates the per-image key if it doesn't exist yet.

    Examples:
      ml-plat component bump-image data-cpu          # 1.2.0 → 1.3.0
      ml-plat component bump-image ml-gpu --set 2.0.0
    """
    versions_env = _find_versions_env()
    if versions_env is None:
        console.print("[bold red]Cannot find versions.env[/bold red]")
        raise typer.Exit(1)

    env_key = f"IMAGE_TAG_{image_name.upper().replace('-', '_')}"
    text = versions_env.read_text()
    lines = text.splitlines()

    # Find the per-image line
    found = False
    old_tag = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(env_key):
            for sep in (":=", "="):
                if sep in stripped:
                    _, old_tag = stripped.split(sep, 1)
                    old_tag = old_tag.strip()
                    break
            new_tag = set_version or _bump_semver_tag(old_tag or "1.0.0")
            lines[i] = f"{env_key} := {new_tag}"
            found = True
            break

    if not found:
        # Add new per-image key after the last IMAGE_TAG_ line or after IMAGE_TAG
        insert_after = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("IMAGE_TAG"):
                insert_after = i
        old_tag = None
        new_tag = set_version or "1.1.0"
        if insert_after is not None:
            lines.insert(insert_after + 1, f"{env_key} := {new_tag}")
        else:
            lines.append(f"{env_key} := {new_tag}")

    versions_env.write_text("\n".join(lines) + "\n")

    if old_tag:
        console.print(
            f"[green]✅ {image_name}:[/green] {old_tag} → [bold]{new_tag}[/bold]  "
            f"({versions_env})"
        )
    else:
        console.print(
            f"[green]✅ {image_name}:[/green] [bold]{new_tag}[/bold] (new entry)  "
            f"({versions_env})"
        )
