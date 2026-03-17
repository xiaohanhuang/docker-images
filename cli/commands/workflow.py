"""
workflow.py — CLI commands for the Text2SQL ML workflow.

Commands:
  ml-plat workflow run      — submit the pipeline to Flyte
  ml-plat workflow watch    — live Mission Control dashboard
  ml-plat workflow status   — quick status of an execution
  ml-plat workflow compare  — compare two runs in MLflow
  ml-plat workflow promote  — promote model to Production
  ml-plat workflow query    — run live inference via the serve endpoint
  ml-plat workflow serve    — deploy inference server for a given run
"""

from __future__ import annotations

import os
import subprocess
import time

import requests
import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.utils import flyte_console_url, flyte_remote, platform_config

console = Console()
app = typer.Typer(help="Text2SQL ML workflow commands")


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


# ── run ───────────────────────────────────────────────────────────────
@app.command("run")
def run_workflow(
    epochs: int = typer.Option(3, help="Training epochs"),
    batch_size: int = typer.Option(16, help="Batch size per GPU"),
    learning_rate: float = typer.Option(5e-4, help="Peak learning rate"),
    version: str = typer.Option("", help="Workflow version (default: latest)"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch live after submitting"),
    teams_webhook: str = typer.Option(
        "", envvar="TEAMS_WEBHOOK_URL", help="Teams webhook URL for notification"
    ),
):
    """Submit the text2sql fine-tuning pipeline to Flyte."""
    console.print(
        Panel.fit(
            "[bold cyan]🚀 ML Platform — Text2SQL Pipeline[/bold cyan]\n"
            f"  epochs={epochs}  batch={batch_size}  lr={learning_rate}",
            border_style="cyan",
        )
    )

    remote = flyte_remote()
    wf = remote.fetch_workflow(
        name="pipeline.text2sql_pipeline",
        version=version or None,
    )
    execution = remote.execute(
        wf,
        inputs={
            "num_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        wait=False,
        execution_name_prefix="text2sql",
    )

    exec_id = execution.id.name
    url = flyte_console_url(execution.id.project, execution.id.domain, exec_id)
    console.print("\n[bold green]✅ Pipeline submitted![/bold green]")
    console.print(f"   Execution ID : [bold]{exec_id}[/bold]")
    console.print(f"   Flyte URL    : {url}")
    console.print(f"\n[dim]Run:[/dim] ml-plat workflow watch {exec_id}\n")

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

        task_order = [
            "ingest_data",
            "preprocess",
            "train",
            "evaluate",
            "register_model",
        ]
        task_emoji = {
            "RUNNING": "🔄",
            "SUCCEEDED": "✅",
            "FAILED": "❌",
            "QUEUED": "⏳",
            "UNDEFINED": "⏳",
        }
        node_map = {}
        try:
            remote.sync_execution(exec, sync_nodes=False)
            node_map = {n.id.node_id: n for n in (exec.node_executions or {}).values()}
        except Exception:
            pass

        for tname in task_order:
            found = None
            for nid, node in node_map.items():
                if tname in nid:
                    found = node
                    break
            if found:
                ps, _ = _phase_str(found.closure.phase)
                emoji = task_emoji.get(ps, "⏳")
                task_table.add_row(tname, f"{emoji} {ps}")
            else:
                task_table.add_row(tname, "⏳ QUEUED")

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
        console.print("   Run: [bold]ml-plat workflow query[/bold]  to test inference")
    else:
        console.print(f"\n[bold red]❌ Pipeline {phase_str}[/bold red]")


# ── status ────────────────────────────────────────────────────────────
@app.command("status")
def status(execution_id: str = typer.Argument(...)):
    """Quick status of a workflow execution."""
    remote = flyte_remote()
    ex = remote.fetch_execution(name=execution_id)
    remote.sync_execution(ex, sync_nodes=False)
    phase = str(ex.closure.phase)
    console.print(f"[bold]{execution_id}[/bold]: {phase}")


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
    console.print(f"\n[dim]View in MLflow:[/dim] {mlflow_url}")


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
    console.print(f"\n[dim]Deploy with:[/dim] ml-plat workflow serve {run_id}")


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
        console.print("[dim]Access with:[/dim] ml-plat workflow query")
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
      ml-plat workflow query "list customers who spent over 1000 dollars"
    """
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
            "[red]Inference server not running. "
            "Deploy with: ml-plat workflow serve <run_id>[/red]"
        )
        raise typer.Exit(1)
