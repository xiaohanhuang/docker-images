"""
workflow.py — CLI commands for ML workflow registration and management.

Commands:
  mlp workflow register — register a workflow with Flyte
  mlp workflow list     — list registered workflows
  mlp workflow info     — show workflow inputs, outputs, and metadata
  mlp workflow run      — submit the pipeline to Flyte
  mlp workflow watch    — live Mission Control dashboard
  mlp workflow status   — quick status of an execution
  mlp workflow history  — list recent executions with inputs
  mlp workflow compare  — compare two runs in MLflow
  mlp workflow promote  — promote model to Production
  mlp workflow query    — run live inference via the serve endpoint
  mlp workflow serve    — deploy inference server for a given run
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.utils import flyte_console_url, flyte_remote, platform_config

console = Console()
app = typer.Typer(help="ML workflow commands")


# ── Config helpers ────────────────────────────────────────────────────


def _mlflow_client():
    import mlflow
    from mlflow.tracking import MlflowClient

    cfg = platform_config()
    uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.get(
        "mlflow_tracking_uri", "http://localhost:5000"
    )
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def _resolve_default_workflow_image(workflow_dir: str) -> str:
    """Resolve the default workflow image from env or the central versions file."""
    from cli.utils import resolve_image

    return resolve_image("DATA_CPU")


# ── register ──────────────────────────────────────────────────────────
@app.command("register")
def register_workflow(
    path: str = typer.Argument(
        ".", help="Path to the workflow directory (must contain pipeline.py)"
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Registration version (default: git short SHA)",
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Flyte project (default: $FLYTE_PROJECT)"
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d", help="Flyte domain (default: $FLYTE_DOMAIN)"
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image",
        "-i",
        help="Container image (default: resolved ECR data-cpu semver tag)",
    ),
):
    """📦 Register a workflow with Flyte.

    Discovers tasks, workflows, and launch plans from pipeline.py and
    registers them via the Flytekit API (no pyflyte subprocess).

    Examples:\n
      mlp workflow register\n
      mlp workflow register projects/workflows/text2sql\n
      mlp workflow register . --version v2.0.0\n
            mlp workflow register . --image <ecr-uri>:1.0.0\n
    """
    import importlib.util
    import sys

    workflow_dir = os.path.abspath(path)
    pipeline_file = os.path.join(workflow_dir, "pipeline.py")
    if not os.path.isfile(pipeline_file):
        console.print(
            f"[bold red]No pipeline.py found in {workflow_dir}[/bold red]\n"
            "[dim]Point to a directory containing pipeline.py, "
            "or cd into one and run: mlp workflow register[/dim]"
        )
        raise typer.Exit(1)

    proj = project or os.getenv("FLYTE_PROJECT", "ml-platform")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    # Auto-detect version and image tag from workflow.yaml
    wf_yaml = os.path.join(workflow_dir, "workflow.yaml")
    wf_meta: dict = {}
    if os.path.isfile(wf_yaml):
        import yaml

        with open(wf_yaml) as f:
            wf_meta = yaml.safe_load(f) or {}

    if not version:
        version = wf_meta.get("version")
        if not version:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=workflow_dir,
                )
                version = result.stdout.strip() if result.returncode == 0 else "v1"
            except FileNotFoundError:
                version = "v1"

    # Build ImageConfig — workflow.yaml can override image_tag for
    # the default data-cpu / ml-gpu images.
    from flytekit.configuration import Image, ImageConfig

    if not image:
        image = _resolve_default_workflow_image(workflow_dir)
    if ":" in image:
        default_fqn, _ = image.rsplit(":", 1)
    else:
        default_fqn = image

    if "/ml-platform/" in default_fqn:
        base_registry = default_fqn.rsplit("/ml-platform/", 1)[0]
    else:
        base_registry = default_fqn.rsplit("/", 1)[0]
    cpu_fqn = f"{base_registry}/ml-platform/data-cpu"
    gpu_fqn = f"{base_registry}/ml-platform/training-llm"

    cpu_tag = "1.2.0"
    gpu_tag = "1.1.0"

    image_config = ImageConfig(
        default_image=Image(name="default", fqn=default_fqn, tag=cpu_tag),
        images=[
            Image(name="cpu", fqn=cpu_fqn, tag=cpu_tag),
            Image(name="gpu", fqn=gpu_fqn, tag=gpu_tag),
        ],
    )

    console.print(
        Panel.fit(
            f"[bold cyan]📦 Registering workflow[/bold cyan]\n"
            f"  Directory: {workflow_dir}\n"
            f"  Project:   {proj}\n"
            f"  Domain:    {dom}\n"
            f"  Version:   {version}\n"
            f"  Images:    cpu={cpu_fqn}:{cpu_tag}\n"
            f"             gpu={gpu_fqn}:{gpu_tag}",
            border_style="cyan",
        )
    )

    # Dynamically import pipeline modules from the workflow directory.
    # By default, imports "pipeline". If workflow.yaml has a "pipelines" list,
    # imports each named module so multiple pipeline files get registered.
    pipeline_modules: list[str] = wf_meta.get("pipelines", ["pipeline"])
    # Strip .py suffix if provided (e.g. "pipeline_large.py" → "pipeline_large")
    pipeline_modules = [m.removesuffix(".py") for m in pipeline_modules]

    orig_path = sys.path.copy()
    orig_cwd = os.getcwd()
    sys.path.insert(0, workflow_dir)
    os.chdir(workflow_dir)

    from flytekit import LaunchPlan
    from flytekit.core.task import PythonTask
    from flytekit.core.workflow import WorkflowBase

    entities = []
    try:
        # Remove stale cached modules so reimport picks up current code
        for key in list(sys.modules):
            if key == "pipeline" or key.startswith("pipeline."):
                del sys.modules[key]
            if key == "tasks" or key.startswith("tasks."):
                del sys.modules[key]
            if key == "config" or key.startswith("config."):
                del sys.modules[key]

        for mod_name in pipeline_modules:
            # Also clear this specific module
            for key in list(sys.modules):
                if key == mod_name or key.startswith(f"{mod_name}."):
                    del sys.modules[key]
            try:
                mod = importlib.import_module(mod_name)
            except Exception as exc:
                console.print(f"[bold red]Failed to load {mod_name}.py:[/bold red] {exc}")
                raise typer.Exit(1)

            for obj in vars(mod).values():
                if isinstance(obj, (PythonTask, WorkflowBase, LaunchPlan)):
                    entities.append(obj)
    finally:
        os.chdir(orig_cwd)
        sys.path = orig_path

    # Discover all flytekit entities (tasks, workflows, launch plans)
    if not entities:
        mods_str = ", ".join(f"{m}.py" for m in pipeline_modules)
        console.print(f"[bold red]No Flyte entities found in {mods_str}[/bold red]")
        raise typer.Exit(1)

    # Register each entity via FlyteRemote
    remote = flyte_remote()
    registered = 0
    for entity in entities:
        kind = type(entity).__name__
        name = getattr(entity, "name", str(entity))
        try:
            remote.register_script(
                entity,
                image_config=image_config,
                version=version,
                project=proj,
                domain=dom,
                source_path=workflow_dir,
            )
            console.print(f"  [green]✔[/green] {kind}: {name}")
            registered += 1
        except Exception as exc:
            console.print(f"  [red]✘[/red] {kind}: {name} — {exc}")

    if registered > 0:
        console.print(
            f"\n[bold green]✅ Registered {registered} entities "
            f"as version {version}[/bold green]"
        )
        # Infer the workflow name for the hint
        wf_names = [getattr(e, "name", "") for e in entities if isinstance(e, WorkflowBase)]
        if wf_names:
            console.print(
                f"[dim]Run with: mlp workflow run {wf_names[0]}" f" --version {version}[/dim]"
            )
    else:
        console.print("\n[bold red]❌ Registration failed.[/bold red]")
        raise typer.Exit(1)


# ── list ──────────────────────────────────────────────────────────────
@app.command("list")
def list_workflows(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Flyte project (default: $FLYTE_PROJECT)"
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d", help="Flyte domain (default: $FLYTE_DOMAIN)"
    ),
    limit: int = typer.Option(100, "--limit", "-n", help="Maximum number of workflows to show"),
):
    """📦 List all registered workflows in the Flyte registry.

    Examples:\n
      mlp workflow list\n
      mlp workflow list --project ml-platform --domain production\n
    """
    remote = flyte_remote()
    proj = project or os.getenv("FLYTE_PROJECT", "flytesnacks")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    try:
        all_ids = []
        token = None
        while True:
            page, token = remote.client.list_workflow_ids_paginated(
                project=proj,
                domain=dom,
                limit=limit,
                token=token,
            )
            all_ids.extend(page)
            if not token:
                break
    except Exception as exc:
        console.print(
            f"[bold red]Failed to connect to Flyte:[/bold red] {exc}\n"
            "Check FLYTE_ENDPOINT and cluster connectivity."
        )
        raise typer.Exit(1)

    if not all_ids:
        console.print("[dim]No workflows registered yet.[/dim]")
        raise typer.Exit(0)

    # Filter out archived workflows
    active_ids = []
    try:
        from flyteidl.admin.common_pb2 import (
            NamedEntityGetRequest,
            NamedEntityState,
        )
        from flyteidl.admin.common_pb2 import (
            NamedEntityIdentifier as PbNamedEntityIdentifier,
        )
        from flyteidl.core.identifier_pb2 import ResourceType

        stub = remote.client._stub
        for wid in all_ids:
            try:
                resp = stub.GetNamedEntity(
                    NamedEntityGetRequest(
                        resource_type=ResourceType.WORKFLOW,
                        id=PbNamedEntityIdentifier(
                            project=wid.project,
                            domain=wid.domain,
                            name=wid.name,
                        ),
                    )
                )
                if resp.metadata.state != NamedEntityState.NAMED_ENTITY_ARCHIVED:
                    active_ids.append(wid)
            except Exception:
                active_ids.append(wid)  # include on error
    except Exception:
        active_ids = list(all_ids)  # fallback: show all

    if not active_ids:
        console.print("[dim]No active workflows registered.[/dim]")
        raise typer.Exit(0)

    table = Table(
        title="[bold cyan]Registered Workflows[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Name", style="bold white", no_wrap=True)
    table.add_column("Project", style="yellow")
    table.add_column("Domain", style="dim")

    for wid in active_ids:
        table.add_row(wid.name, wid.project, wid.domain)

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Total: {len(active_ids)} workflow(s).  "
        "Run [bold]mlp workflow run <name>[/bold] to execute one.[/dim]\n"
    )


# ── info ──────────────────────────────────────────────────────────────


def _format_literal_default(lit) -> str:
    """Extract a human-readable string from a Flyte Literal default value."""
    if lit is None:
        return ""
    try:
        prim = lit.scalar.primitive
        return str(prim.value)
    except Exception:
        pass
    try:
        return str(lit)
    except Exception:
        return ""


def _format_literal_type(lt) -> str:
    """Convert a Flyte LiteralType to a human-readable Python type string."""
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
        inner = _format_literal_type(lt.collection_type)
        return f"list\\[{inner}]"
    if lt.map_value_type is not None:
        inner = _format_literal_type(lt.map_value_type)
        return f"dict\\[str, {inner}]"
    if lt.enum_type is not None:
        return "Enum"
    if lt.structured_dataset_type is not None:
        return "StructuredDataset"
    if lt.union_type is not None:
        variants = [_format_literal_type(v) for v in lt.union_type.variants]
        return " | ".join(variants)
    return str(lt)


@app.command("info")
def workflow_info(
    name: str = typer.Argument(..., help="Workflow name in Flyte"),
    version: Optional[str] = typer.Option(
        None, "--version", "-v", help="Workflow version (default: latest)"
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Flyte project (default: $FLYTE_PROJECT)"
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d", help="Flyte domain (default: $FLYTE_DOMAIN)"
    ),
):
    """🔍 Show detailed info about a workflow: inputs, outputs, nodes, and versions.

    Examples:\n
      mlp workflow info pipeline.text2sql_pipeline\n
      mlp workflow info recipe_openrlhf_llm_rlhf --version v1\n
    """
    remote = flyte_remote()
    proj = project or os.getenv("FLYTE_PROJECT", remote.default_project)
    dom = domain or os.getenv("FLYTE_DOMAIN", remote.default_domain)

    try:
        wf = remote.fetch_workflow(name=name, version=version or None, project=proj, domain=dom)
    except Exception as exc:
        console.print(f"[bold red]Workflow not found:[/bold red] {name}\n{exc}")
        raise typer.Exit(1)

    # Check if workflow is archived
    try:
        from flyteidl.admin.common_pb2 import (
            NamedEntityGetRequest,
            NamedEntityState,
        )
        from flyteidl.admin.common_pb2 import (
            NamedEntityIdentifier as PbNamedEntityIdentifier,
        )
        from flyteidl.core.identifier_pb2 import ResourceType

        resp = remote.client._stub.GetNamedEntity(
            NamedEntityGetRequest(
                resource_type=ResourceType.WORKFLOW,
                id=PbNamedEntityIdentifier(
                    project=proj,
                    domain=dom,
                    name=name,
                ),
            )
        )
        if resp.metadata.state == NamedEntityState.NAMED_ENTITY_ARCHIVED:
            console.print(
                f"[bold yellow]Workflow '{name}' is archived.[/bold yellow]\n"
                "[dim]Use [bold]mlp workflow list[/bold] to see active workflows.[/dim]"
            )
            raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception:
        pass  # best-effort check; continue if it fails

    wf_id = wf.id

    # ── Header ──────────────────────────────────────────────────────────
    header = (
        f"[bold white]{wf_id.name}[/bold white]  "
        f"[dim]{wf_id.version}[/dim]  "
        f"[yellow]{wf_id.project}/{wf_id.domain}[/yellow]"
    )

    console.print()
    console.print(Panel(header, title="[bold cyan]Workflow Info[/bold cyan]", border_style="cyan"))

    # ── Fetch launch plan defaults (best-effort) ──────────────────────
    defaults: dict[str, str] = {}
    try:
        lp = remote.fetch_launch_plan(name=name, version=wf_id.version, project=proj, domain=dom)
        if lp.default_inputs and lp.default_inputs.parameters:
            for pname, param in lp.default_inputs.parameters.items():
                defaults[pname] = _format_literal_default(param.default)
    except Exception:
        pass  # defaults are best-effort

    # ── Inputs table ────────────────────────────────────────────────────
    inputs_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    inputs_table.add_column("Parameter", style="cyan", no_wrap=True)
    inputs_table.add_column("Type", style="yellow")
    inputs_table.add_column("Default", style="green")

    if wf.interface and wf.interface.inputs:
        for var_name, var in wf.interface.inputs.items():
            type_str = _format_literal_type(var.type) if hasattr(var, "type") else str(var)
            inputs_table.add_row(var_name, type_str, defaults.get(var_name, ""))

    console.print("\n[bold]Inputs:[/bold]")
    console.print(inputs_table)

    # ── Outputs table ───────────────────────────────────────────────────
    outputs_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    outputs_table.add_column("Parameter", style="cyan", no_wrap=True)
    outputs_table.add_column("Type", style="yellow")

    if wf.interface and wf.interface.outputs:
        for var_name, var in wf.interface.outputs.items():
            type_str = _format_literal_type(var.type) if hasattr(var, "type") else str(var)
            outputs_table.add_row(var_name, type_str)

    console.print("[bold]Outputs:[/bold]")
    console.print(outputs_table)

    # ── Nodes (sub-tasks) ───────────────────────────────────────────────
    nodes = getattr(wf, "nodes", None)
    if nodes:
        nodes_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        nodes_table.add_column("Step", style="cyan", no_wrap=True)
        nodes_table.add_column("Task", style="white")

        for node in nodes:
            # Prefer human-readable metadata.name over internal id (n0, n1, ...)
            node_name = str(node.id) if hasattr(node, "id") else str(node)
            if hasattr(node, "metadata") and node.metadata and hasattr(node.metadata, "name"):
                node_name = node.metadata.name or node_name
            task_ref = ""
            if hasattr(node, "task_node") and node.task_node:
                ref = node.task_node.reference_id
                task_ref = f"{ref.name} ({ref.version})" if ref else ""
            nodes_table.add_row(node_name, task_ref)

        console.print("[bold]Nodes:[/bold]")
        console.print(nodes_table)

    # ── Versions list ───────────────────────────────────────────────────
    try:
        from flytekit.models.admin.common import Sort
        from flytekit.models.common import NamedEntityIdentifier

        latest_first = Sort(key="created_at", direction=Sort.Direction.DESCENDING)
        identifier = NamedEntityIdentifier(project=proj, domain=dom, name=name)
        wf_list, _ = remote.client.list_workflows_paginated(
            identifier, limit=10, sort_by=latest_first
        )
        if wf_list:
            # Build version → timestamp map from executions (workflows don't
            # store created_at, but executions do).
            version_ts: dict[str, str] = {}
            try:
                execs, _ = remote.client.list_executions_paginated(
                    project=proj,
                    domain=dom,
                    limit=100,
                    sort_by=Sort(key="created_at", direction=Sort.Direction.ASCENDING),
                )
                for ex in execs:
                    lp = ex.spec.launch_plan
                    if lp.name == name and lp.version not in version_ts:
                        ts = ex.closure.started_at
                        version_ts[lp.version] = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
            except Exception:
                pass  # best-effort timestamp lookup

            ver_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
            ver_table.add_column("Version", style="cyan", no_wrap=True)
            ver_table.add_column("First Run", style="dim")

            for w in wf_list:
                ver_table.add_row(w.id.version, version_ts.get(w.id.version, ""))

            console.print("[bold]Recent Versions:[/bold]")
            console.print(ver_table)
    except Exception:
        pass  # version listing is best-effort

    console.print()


# ── run ───────────────────────────────────────────────────────────────
@app.command("run")
def run_workflow(
    name: str = typer.Argument(..., help="Workflow name in Flyte"),
    inputs: Optional[list[str]] = typer.Argument(
        None, help="Inputs as key=value pairs (e.g. num_epochs=5 batch_size=32)"
    ),
    version: str = typer.Option("", help="Workflow version (default: latest)"),
    overwrite_cache: bool = typer.Option(
        False, "--overwrite-cache", "--owc", help="Overwrite cached task outputs"
    ),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch live after submitting"),
):
    """Submit a registered workflow to Flyte.

    Examples:\n
      mlp workflow run pipeline.text2sql_pipeline num_epochs=5 batch_size=32 --watch\n
      mlp workflow run pipeline.llm_sft_lora_pipeline base_model=llama-3.1-8B epochs=3\n
    """
    remote = flyte_remote()
    try:
        wf = remote.fetch_workflow(name=name, version=version or None)
    except Exception as exc:
        console.print(f"[bold red]Workflow not found:[/bold red] {name}\n{exc}")
        raise typer.Exit(1)

    # Parse key=value inputs, casting via the remote interface
    typed_inputs: dict = {}
    iface_inputs = wf.interface.inputs
    for item in inputs or []:
        if "=" not in item:
            console.print(f"[red]Invalid input: {item}  (expected key=value)[/red]")
            raise typer.Exit(1)
        key, _, value = item.partition("=")
        if key not in iface_inputs:
            valid_keys = ", ".join(sorted(iface_inputs.keys()))
            console.print(
                f"[red]Unknown input key:[/red] {key}\n"
                f"[dim]Valid inputs: {valid_keys or '(none)'}[/dim]"
            )
            raise typer.Exit(1)
        lt = iface_inputs[key].type
        if lt.blob is not None:
            from flytekit.types.directory import FlyteDirectory
            from flytekit.types.file import FlyteFile

            if lt.blob.dimensionality == 0:
                typed_inputs[key] = FlyteFile(path=value)
            else:
                typed_inputs[key] = FlyteDirectory(path=value)
        elif lt.simple is not None:
            if lt.simple == 1:  # INTEGER
                typed_inputs[key] = int(value)
            elif lt.simple == 2:  # FLOAT
                typed_inputs[key] = float(value)
            elif lt.simple == 4:  # BOOLEAN
                typed_inputs[key] = value.lower() in ("true", "1", "yes")
            else:
                typed_inputs[key] = value
        else:
            typed_inputs[key] = value

    console.print(
        Panel.fit(
            f"[bold cyan]🚀 ML Platform — {name}[/bold cyan]\n"
            f"  inputs: {typed_inputs or '(defaults)'}",
            border_style="cyan",
        )
    )

    # Sanitize execution name prefix (Flyte requires lowercase alphanumeric + hyphens)
    raw_prefix = name.split(".")[-1][:20]
    exec_prefix = re.sub(r"[^a-z0-9-]", "-", raw_prefix.lower()).strip("-")

    execution = remote.execute(
        wf,
        inputs=typed_inputs,
        wait=False,
        execution_name_prefix=exec_prefix,
        overwrite_cache=overwrite_cache,
    )

    exec_id = execution.id.name
    url = flyte_console_url(execution.id.project, execution.id.domain, exec_id)
    console.print("\n[bold green]✅ Pipeline submitted![/bold green]")
    console.print(f"   Execution ID : [bold]{exec_id}[/bold]")
    console.print(f"   Flyte URL    : {url}", soft_wrap=True)
    console.print(f"\n[dim]Run:[/dim] mlp workflow watch {exec_id}\n")

    if watch:
        _watch_execution(exec_id, remote)


# ── watch ─────────────────────────────────────────────────────────────
@app.command("watch")
def watch_execution(
    execution_id: str = typer.Argument(..., help="Flyte execution ID"),
    interval: int = typer.Option(10, help="Refresh interval (seconds)"),
):
    """🖥️  Live Mission Control dashboard — shows nodes, GPU, tasks, and MLflow metrics."""
    remote = flyte_remote()
    _watch_execution(execution_id, remote, interval=interval)


def _watch_execution(exec_id: str, remote, interval: int = 10):
    """Internal: renders the live dashboard."""
    import mlflow
    from flytekit.models.core.execution import WorkflowExecutionPhase

    phase_style_map = {
        "RUNNING": "bold green",
        "SUCCEEDED": "bold cyan",
        "FAILED": "bold red",
        "ABORTED": "yellow",
        "UNDEFINED": "dim",
        "QUEUED": "dim white",
    }

    def _phase_str(phase) -> tuple[str, str]:
        s = WorkflowExecutionPhase.enum_to_string(phase) if hasattr(phase, "real") else str(phase)
        if s == "SUCCEEDING":
            s = "SUCCEEDED"
        return s, phase_style_map.get(s, "white")

    def _build_dashboard(exec) -> Panel:
        # ── workflow header ───────────────────────────────────────────
        phase_str, phase_style = _phase_str(exec.closure.phase)
        elapsed = (
            int(time.time() - exec.closure.started_at.timestamp()) if exec.closure.started_at else 0
        )
        elapsed_str = f"{elapsed // 60}m {elapsed % 60}s" if elapsed else "—"

        header = (
            f"[bold]{exec_id}[/bold]   "
            f"Status: [{phase_style}]{phase_str}[/{phase_style}]   "
            f"Duration: [white]{elapsed_str}[/white]"
        )

        # ── task table ────────────────────────────────────────────────
        task_table = Table(
            box=box.SIMPLE, show_header=True, expand=True, header_style="bold magenta"
        )
        task_table.add_column("Task", style="cyan", no_wrap=True)
        task_table.add_column("Status", style="white", justify="left")

        task_emoji = {
            "RUNNING": "🔄",
            "SUCCEEDED": "✅",
            "FAILED": "❌",
            "QUEUED": "⏳",
            "UNDEFINED": "⏳",
        }
        node_map = {}
        try:
            synced = remote.sync(exec, sync_nodes=True)
            node_map = {n.id.node_id: n for n in (synced.node_executions or {}).values()}
        except Exception:
            pass

        # Filter out purely structural nodes
        named_nodes = {
            nid: n for nid, n in node_map.items() if not re.match(r"^(start|end)-node$", nid)
        }

        # Sort node keys: numeric for "n0", "n1", else alphabetic
        def _node_sort_key(nid: str):
            if re.match(r"^n\d+$", nid):
                return (0, int(nid[1:]))
            return (1, nid)

        sorted_nids = sorted(named_nodes.keys(), key=_node_sort_key)

        for nid in sorted_nids:
            node = named_nodes[nid]
            ps, _ = _phase_str(node.closure.phase)
            emoji = task_emoji.get(ps, "⏳")

            # Map default positional Flyte names (n0, n1) to
            # "task_0", "task_1" for slightly better display
            display_name = f"task_{nid[1:]}" if re.match(r"^n\d+$", nid) else nid.replace("-", "_")
            task_table.add_row(display_name, f"{emoji} {ps}")

        if not sorted_nids:
            task_table.add_row("Initializing...", "⏳ QUEUED")

        # ── GPU & node panel ──────────────────────────────────────────
        try:
            subprocess.check_output(
                ["kubectl", "describe", "node", "-l", "role=gpu-worker"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            gpu_text = Text("GPU node: g5.xlarge  ", style="green")
            gpu_text.append("nvidia.com/gpu: 1", style="bold")
        except Exception:
            gpu_text = Text("GPU node: not available", style="dim")

        # ── MLflow metrics panel ──────────────────────────────────────
        mlflow_text = Text()
        try:
            _cfg = platform_config()
            _mlflow_uri = os.getenv("MLFLOW_TRACKING_URI") or _cfg.get(
                "mlflow_tracking_uri", "http://localhost:5000"
            )
            mlflow.set_tracking_uri(_mlflow_uri)
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=client.get_experiment_by_name("text2sql").experiment_id,
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                r = runs[0]
                for k, v in r.data.metrics.items():
                    mlflow_text.append(f"  {k}: ", style="dim")
                    mlflow_text.append(f"{v:.4f}\n", style="bold white")
            else:
                mlflow_text.append("  Waiting for first eval...", style="dim")
        except Exception:
            mlflow_text.append("  MLflow not yet available", style="dim")

        # ── Assemble layout ───────────────────────────────────────────
        left = Panel(
            task_table,
            title="[bold]Pipeline Tasks[/bold]",
            border_style="cyan",
            expand=True,
        )
        right = Panel(
            mlflow_text,
            title="[bold]MLflow Metrics[/bold]",
            border_style="magenta",
            expand=True,
        )

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(left, right)

        return Panel(
            grid,
            title=f"[bold cyan]⚙️  ML Platform Mission Control[/bold cyan]  —  {header}",
            border_style="bright_blue",
            padding=(0, 1),
        )

    terminal_phases = {"SUCCEEDED", "FAILED", "ABORTED"}

    with Live(console=console, refresh_per_second=0.5, screen=False) as live:
        while True:
            try:
                exec_obj = remote.fetch_execution(name=exec_id)
                live.update(_build_dashboard(exec_obj))
                phase_str, _ = _phase_str(exec_obj.closure.phase)
                if phase_str in terminal_phases:
                    time.sleep(1)
                    break
            except Exception as e:
                live.update(Panel(f"[red]Error: {e}[/red]"))
            time.sleep(interval)

    if phase_str == "SUCCEEDED":
        console.print("\n[bold green]🎉 Pipeline SUCCEEDED![/bold green]")
        console.print("   Run: [bold]mlp workflow query[/bold]  to test inference")
    else:
        console.print(f"\n[bold red]❌ Pipeline {phase_str}[/bold red]")


# ── status ────────────────────────────────────────────────────────────
@app.command("status")
def status(execution_id: str = typer.Argument(..., help="Execution ID to inspect")):
    """Show status, inputs, and exact rerun command for a workflow execution."""

    remote = flyte_remote()
    try:
        ex = remote.fetch_execution(name=execution_id)
        remote.sync_execution(ex, sync_nodes=False)
    except Exception as exc:
        console.print(f"[red]Could not fetch execution '{execution_id}': {exc}[/red]")
        raise typer.Exit(1)

    from flytekit.models.core.execution import WorkflowExecutionPhase

    phase_int = ex.closure.phase
    phase_str = (
        WorkflowExecutionPhase.enum_to_string(phase_int)
        if hasattr(phase_int, "real")
        else str(phase_int)
    )
    if phase_str == "SUCCEEDING":
        phase_str = "SUCCEEDED"

    phase_style = {
        "SUCCEEDED": "cyan",
        "RUNNING": "bold green",
        "FAILED": "bold red",
        "ABORTED": "yellow",
    }
    style = phase_style.get(phase_str, "white")

    wf_name = ex.spec.launch_plan.name
    started = ex.closure.started_at
    started_str = started.strftime("%Y-%m-%d %H:%M UTC") if started else "—"
    ended = ex.closure.updated_at
    duration_str = "—"
    if started and ended:
        delta = int((ended - started).total_seconds())
        duration_str = f"{delta // 60}m {delta % 60}s" if delta >= 60 else f"{delta}s"

    console.print()
    from rich.panel import Panel

    console.print(
        Panel(
            f"[bold]{execution_id}[/bold]\n"
            f"Workflow : [yellow]{wf_name}[/yellow]\n"
            f"Status   : [{style}]{phase_str}[/{style}]\n"
            f"Started  : [dim]{started_str}[/dim]\n"
            f"Duration : [dim]{duration_str}[/dim]",
            title="[bold cyan]Execution[/bold cyan]",
            border_style="cyan",
        )
    )

    # Extract and display inputs
    input_parts = []
    try:
        inputs_lm = ex.spec.inputs
        if inputs_lm and inputs_lm.literals:
            for key in sorted(inputs_lm.literals):
                val = _extract_literal_value(inputs_lm.literals[key])
                input_parts.append((key, val))
    except Exception:
        pass

    if input_parts:
        from rich.table import Table

        t = Table(box=None, show_header=False, padding=(0, 2))
        t.add_column(style="dim")
        t.add_column(style="white")
        for k, v in input_parts:
            t.add_row(k, v)
        console.print("\n[bold]Inputs:[/bold]")
        console.print(t)
        kv_args = " ".join(f"{k}={v}" for k, v in input_parts)
        console.print("\n[bold]Rerun:[/bold]")
        console.print(f"  [bold green]mlp workflow run {wf_name} {kv_args}[/bold green]")
    else:
        console.print("\n[dim]Inputs: (defaults)[/dim]")
        console.print("\n[bold]Rerun:[/bold]")
        console.print(f"  [bold green]mlp workflow run {wf_name}[/bold green]")
    console.print()


# ── history ───────────────────────────────────────────────────────────


def _extract_literal_value(lit) -> str:
    """Extract a human-readable string from a Flyte Literal."""
    try:
        prim = lit.scalar.primitive
        # primitive has one of: integer, float_value, string_value, boolean
        for attr in ("string_value", "integer", "float_value", "boolean"):
            v = getattr(prim, attr, None)
            if v is not None:
                # boolean 0/False should still display
                if attr == "boolean":
                    return str(bool(v))
                if attr == "float_value" and v == 0.0:
                    return "0.0"
                if attr == "integer" and v == 0:
                    return "0"
                if v or attr == "string_value":
                    return str(v)
        return str(prim)
    except Exception:
        pass
    try:
        return str(lit)
    except Exception:
        return "?"


@app.command("history")
def history(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Flyte project (default: $FLYTE_PROJECT)"
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d", help="Flyte domain (default: $FLYTE_DOMAIN)"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of executions to show"),
    workflow_filter: Optional[str] = typer.Option(
        None, "--workflow", "-w", help="Filter by workflow name (substring match)"
    ),
    status_filter: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status: running, succeeded, failed"
    ),
):
    """📜 List recent workflow executions with inputs and rerun commands.

    Examples:\n
      mlp workflow history\n
      mlp workflow history -n 5\n
      mlp workflow history -w text2sql\n
      mlp workflow history -s failed\n
    """
    from datetime import datetime, timezone

    from flytekit.models.admin.common import Sort
    from flytekit.models.core.execution import WorkflowExecutionPhase

    remote = flyte_remote()
    proj = project or os.getenv("FLYTE_PROJECT", "ml-platform")
    dom = domain or os.getenv("FLYTE_DOMAIN", "development")

    phase_emoji = {
        "RUNNING": "🔄",
        "SUCCEEDED": "✅",
        "FAILED": "❌",
        "ABORTED": "⚠️",
        "QUEUED": "⏳",
        "UNDEFINED": "⏳",
    }
    phase_style = {
        "RUNNING": "bold green",
        "SUCCEEDED": "cyan",
        "FAILED": "bold red",
        "ABORTED": "yellow",
    }
    status_to_phase = {
        "running": "RUNNING",
        "succeeded": "SUCCEEDED",
        "failed": "FAILED",
        "aborted": "ABORTED",
    }

    try:
        latest_first = Sort(key="created_at", direction=Sort.Direction.DESCENDING)
        # Fetch more than limit to allow for client-side filtering
        fetch_limit = limit * 3 if (workflow_filter or status_filter) else limit
        execs, _ = remote.client.list_executions_paginated(
            project=proj,
            domain=dom,
            limit=fetch_limit,
            sort_by=latest_first,
        )
    except Exception as exc:
        console.print(
            f"[bold red]Failed to list executions:[/bold red] {exc}\n"
            "Check FLYTE_ENDPOINT and cluster connectivity."
        )
        raise typer.Exit(1)

    if not execs:
        console.print("[dim]No executions found.[/dim]")
        raise typer.Exit(0)

    # Apply filters
    filtered = []
    for ex in execs:
        wf_name = ex.spec.launch_plan.name
        phase_int = ex.closure.phase
        phase_str = (
            WorkflowExecutionPhase.enum_to_string(phase_int)
            if hasattr(phase_int, "real")
            else str(phase_int)
        )
        if phase_str == "SUCCEEDING":
            phase_str = "SUCCEEDED"

        if workflow_filter and workflow_filter.lower() not in wf_name.lower():
            continue
        if status_filter:
            target = status_to_phase.get(status_filter.lower())
            if target and phase_str != target:
                continue

        filtered.append((ex, wf_name, phase_str))
        if len(filtered) >= limit:
            break

    if not filtered:
        console.print("[dim]No executions match the filters.[/dim]")
        raise typer.Exit(0)

    # Build output
    table = Table(
        title=f"[bold cyan]Recent Executions[/bold cyan]  ({proj} / {dom})",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Execution ID", style="bold white", no_wrap=True)
    table.add_column("Workflow", style="yellow", max_width=35)
    table.add_column("Status", justify="center")
    table.add_column("Started", style="dim")
    table.add_column("Duration", style="dim", justify="right")

    now = datetime.now(timezone.utc)

    for ex, wf_name, phase_str in filtered:
        exec_id = ex.id.name
        emoji = phase_emoji.get(phase_str, "")
        style = phase_style.get(phase_str, "white")
        status_cell = f"{emoji} [{style}]{phase_str}[/{style}]"

        started = ex.closure.started_at
        started_str = started.strftime("%Y-%m-%d %H:%M") if started else "—"

        duration_str = "—"
        if started:
            ended = ex.closure.updated_at or (now if phase_str == "RUNNING" else None)
            if ended:
                delta = int((ended - started).total_seconds())
                if delta >= 3600:
                    duration_str = f"{delta // 3600}h {(delta % 3600) // 60}m"
                elif delta >= 60:
                    duration_str = f"{delta // 60}m {delta % 60}s"
                else:
                    duration_str = f"{delta}s"

        # Truncate long workflow names
        short_wf = wf_name if len(wf_name) <= 35 else "…" + wf_name[-(34):]

        table.add_row(exec_id, short_wf, status_cell, started_str, duration_str)

    console.print()
    console.print(table)

    # Print inputs and rerun command for each execution
    for ex, wf_name, phase_str in filtered:
        exec_id = ex.id.name
        console.print(f"\n[bold]{exec_id}[/bold]")

        # Extract inputs from execution spec
        input_parts = []
        try:
            inputs_lm = ex.spec.inputs
            if inputs_lm and inputs_lm.literals:
                for key in sorted(inputs_lm.literals):
                    val = _extract_literal_value(inputs_lm.literals[key])
                    input_parts.append((key, val))
        except Exception:
            pass

        if input_parts:
            kv_str = "  ".join(f"{k}={v}" for k, v in input_parts)
            console.print(f"  [dim]Inputs:[/dim]  {kv_str}")
            # Build rerun command
            kv_args = " ".join(f"{k}={v}" for k, v in input_parts)
            console.print(f"  [dim]Rerun:[/dim]   mlp workflow run {wf_name} {kv_args}")
        else:
            console.print("  [dim]Inputs:[/dim]  (defaults)")
            console.print(f"  [dim]Rerun:[/dim]   mlp workflow run {wf_name}")

    total = len(filtered)
    console.print(
        f"\n[dim]{total} execution(s) shown.  "
        "Run [bold]mlp workflow watch <ID>[/bold] to monitor one.[/dim]\n"
    )


# ── compare ───────────────────────────────────────────────────────────
@app.command("compare")
def compare(
    run_id_1: str = typer.Argument(..., help="First MLflow run ID"),
    run_id_2: str = typer.Argument(..., help="Second MLflow run ID"),
):
    """Compare two training runs side-by-side."""
    client = _mlflow_client()

    table = Table(title="Run Comparison", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column(f"Run 1 ({run_id_1[:8]})", justify="right")
    table.add_column(f"Run 2 ({run_id_2[:8]})", justify="right")

    r1 = client.get_run(run_id_1)
    r2 = client.get_run(run_id_2)

    all_keys = sorted(set(r1.data.metrics) | set(r2.data.metrics))
    for key in all_keys:
        v1 = r1.data.metrics.get(key, None)
        v2 = r2.data.metrics.get(key, None)
        s1, s2 = "—", "—"
        if v1 is not None and v2 is not None:
            s1 = f"{v1:.4f}"
            s2 = f"{v2:.4f}"
            if v2 > v1 and "accuracy" in key:
                s2 = f"[bold green]{s2}[/bold green]"
            elif v2 < v1 and "loss" in key:
                s2 = f"[bold green]{s2}[/bold green]"
        elif v1 is not None:
            s1 = f"{v1:.4f}"
        elif v2 is not None:
            s2 = f"{v2:.4f}"
        table.add_row(key, s1, s2)

    console.print(table)
    cfg = platform_config()
    mlflow_url = cfg.get("mlflow_tracking_uri", "http://localhost:5000")
    console.print(f"\n[dim]View in MLflow:[/dim] {mlflow_url}", soft_wrap=True)


# ── promote ───────────────────────────────────────────────────────────
@app.command("promote")
def promote(
    run_id: str = typer.Argument(..., help="MLflow run ID of the model to promote"),
    version: int = typer.Option(None, help="Model version (latest if omitted)"),
):
    """Promote a Text2SQL model version to Production in MLflow Registry."""
    client = _mlflow_client()

    model_name = "text2sql"
    if version is None:
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not versions:
            console.print("[red]No model found in Staging. Run the pipeline first.[/red]")
            raise typer.Exit(1)
        version = versions[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    console.print(
        f"[bold green]✅ Model [white]{model_name}[/white] v{version} → Production[/bold green]"
    )
    console.print(f"\n[dim]Deploy with:[/dim] mlp workflow serve {run_id}")


# ── serve ─────────────────────────────────────────────────────────────
@app.command("serve")
def deploy_serve(run_id: str = typer.Argument(..., help="MLflow run ID whose checkpoint to serve")):
    """Deploy the inference server for a trained model."""
    console.print(f"[cyan]Deploying inference server for run [bold]{run_id}[/bold]...[/cyan]")
    result = subprocess.run(
        ["make", "serve", f"RUN_ID={run_id}"],
        cwd=os.path.join(
            os.path.dirname(__file__), "..", "..", "projects", "workflows", "text2sql"
        ),
        capture_output=False,
    )
    if result.returncode == 0:
        console.print("[bold green]✅ Server deployed![/bold green]")
        console.print("[dim]Access with:[/dim] mlp workflow query")
    else:
        console.print("[bold red]❌ Deployment failed.[/bold red]")
        raise typer.Exit(1)


# ── query ─────────────────────────────────────────────────────────────
@app.command("query")
def query(
    question: str = typer.Argument(..., help="Natural language question"),
    context: str = typer.Option(
        "CREATE TABLE sales (id INT, customer TEXT, amount FLOAT, date TEXT);",
        help="CREATE TABLE schema context",
    ),
    port: int = typer.Option(8080, help="Local port for inference server"),
):
    """
    🔮  Translate a natural language question to SQL.

    Example:
      mlp workflow query "list customers who spent over 1000 dollars"
    """
    import requests

    try:
        url = f"http://localhost:{port}/predict"
        resp = requests.post(url, json={"question": question, "context": context}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        console.print(
            Panel.fit(
                f"[bold cyan]Question:[/bold cyan] {question}\n\n"
                f"[bold green]SQL:[/bold green]\n  [white]{data['sql']}[/white]",
                title="🤖 Text2SQL Inference",
                border_style="green",
            )
        )
    except requests.exceptions.ConnectionError:
        console.print(
            "[red]Inference server not running. " "Deploy with: mlp workflow serve <run_id>[/red]"
        )
        raise typer.Exit(1)
