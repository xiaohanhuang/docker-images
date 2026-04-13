import datetime
import os

import typer
from rich.console import Console
from rich.table import Table

from backend.pricing import get_cost_estimate

app = typer.Typer(help="Manage training jobs")
console = Console()


def _api():
    """Return a configured APIClient."""
    from cli.api_client import APIClient

    return APIClient()


# Endpoint resolution is handled by cli.utils.flyte_remote() which reads
# from ~/.ml-plat/config.yaml and environment variables.


@app.command("submit")
def submit_job(
    workflow_name: str = typer.Option(
        ...,
        help=(
            "Name of the workflow to run "
            "(e.g., workflows.llm_finetune.workflow.llm_finetune_workflow)"
        ),
    ),
    version: str = typer.Option("v1", help="Version of the workflow"),
    dataset: str = typer.Option(..., help="S3 path to dataset"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    profile: bool = typer.Option(
        False, "--profile", help="Enable PyTorch profiling (traces saved to EFS)"
    ),
    nsight: bool = typer.Option(
        False, "--nsight", help="Enable Nsight Systems profiling (nsys injected via init container)"
    ),
):
    """
    Submit a training job to the cluster.
    """
    try:
        env_overrides = {}
        if profile:
            env_overrides["ML_PLAT_PROFILE"] = "1"
            console.print(
                "[bold yellow]Profiling enabled[/bold yellow]"
                " — traces → /mnt/efs/profiles/<job_id>/"
            )
        if nsight:
            env_overrides["ML_PLAT_NSIGHT"] = "1"
            console.print(
                "[bold yellow]Nsight Systems enabled[/bold yellow]"
                " — nsys injected via init container"
            )

        with _api() as client:
            result = client.submit_job(
                workflow_name=workflow_name,
                version=version,
                inputs={"s3_dataset_path": dataset, "num_epochs": epochs},
                env_overrides=env_overrides or None,
            )

        typer.echo("Job submitted successfully!")
        typer.echo(f"Job ID: {result['job_id']}")
        if result.get("console_url"):
            typer.echo(f"View console: {result['console_url']}")

    except Exception as e:
        typer.secho(f"Failed to submit job: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command("status")
def get_status(job_id: str):
    """
    Get the status of a job, including instance type and estimated cost.
    """
    try:
        with _api() as client:
            result = client.get_job_status(job_id)

        status = result.get("status", "UNKNOWN")
        instance_type = result.get("instance_type", "unknown")
        duration_str = result.get("duration", "0:00:00")

        # Parse duration for cost estimate
        try:
            parts = duration_str.split(":")
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            seconds = int(parts[2].split(".")[0]) if len(parts) > 2 else 0
            duration = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        except Exception:
            duration = datetime.timedelta(0)

        cost = get_cost_estimate(instance_type, duration)

        phase_color = {
            "RUNNING": "green",
            "SUCCEEDED": "cyan",
            "FAILED": "red",
            "ABORTED": "yellow",
        }
        color = phase_color.get(status, "white")
        table = Table(title=f"Job Status: {job_id}", show_header=False, box=None)
        table.add_row("[bold]Job ID:[/bold]", job_id)
        table.add_row("[bold]Status:[/bold]", f"[{color}]{status}[/{color}]")
        table.add_row("[bold]Instance Type:[/bold]", instance_type)
        table.add_row("[bold]Uptime:[/bold]", str(duration).split(".")[0])
        table.add_row(
            "[bold]Current Estimated Run Cost:[/bold]",
            f"[bold green]${cost:.2f}[/bold green]",
        )

        console.print(table)

    except Exception as e:
        typer.secho(f"Error fetching status: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command("logs")
def get_logs(job_id: str):
    """
    Get logs for a job.
    """
    try:
        with _api() as client:
            logs = client.get_job_logs(job_id)
        if logs:
            typer.echo(logs)
        else:
            typer.echo("No logs available yet.")
    except Exception as e:
        typer.secho(f"Error fetching logs: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Profile trace management
# ---------------------------------------------------------------------------

S3_PROFILES_PREFIX = "profiles"


def _get_s3_bucket() -> str:
    """Return the platform S3 bucket from env."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        console.print("[red]S3_BUCKET not set. Run [bold]ml-plat init[/bold] first.[/red]")
        raise typer.Exit(1)
    return bucket


def _s3_client():
    """Return a boto3 S3 client (lazy import)."""
    import boto3

    return boto3.client("s3")


@app.command("profiles")
def profiles(
    job_id: str = typer.Argument(..., help="Job/execution ID to list or download traces for"),
    download: str = typer.Option(
        None, "--download", "-d", help="Local directory to download traces to"
    ),
):
    """
    List or download profiling traces (PyTorch / Nsight) for a job.

    By default, lists available trace files on S3.
    Use --download to copy them to a local directory.

    Examples:
        ml-plat job profiles abc-123
        ml-plat job profiles abc-123 --download ./traces
    """
    bucket = _get_s3_bucket()
    prefix = f"{S3_PROFILES_PREFIX}/{job_id}/"

    s3 = _s3_client()

    # List objects under the prefix
    try:
        paginator = s3.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                files.append(obj)
    except Exception as e:
        console.print(f"[red]Failed to list S3 objects: {e}[/red]")
        raise typer.Exit(1)

    if not files:
        console.print(
            f"[yellow]No profiling traces found for job {job_id}.[/yellow]\n"
            f"Expected at: s3://{bucket}/{prefix}\n"
            "Make sure the job was submitted with --profile or --nsight."
        )
        raise typer.Exit(0)

    if download is None:
        # List mode
        table = Table(title=f"Profiling Traces — {job_id}")
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Last Modified")

        total_size = 0
        for obj in files:
            key = obj["Key"]
            filename = key[len(prefix) :]
            size = obj["Size"]
            total_size += size
            modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S")

            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"

            table.add_row(filename, size_str, modified)

        console.print(table)
        console.print(
            f"\n[dim]{len(files)} file(s), " f"{total_size / 1_000_000:.1f} MB total[/dim]"
        )
        console.print(f"\n[dim]Download: ml-plat job profiles {job_id} --download ./traces[/dim]")
    else:
        # Download mode
        dest = os.path.abspath(download)
        os.makedirs(dest, exist_ok=True)

        console.print(f"Downloading {len(files)} file(s) to [bold]{dest}[/bold]...")

        for obj in files:
            key = obj["Key"]
            relative = key[len(prefix) :]
            local_path = os.path.join(dest, relative)

            # Create subdirectories if needed
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            s3.download_file(bucket, key, local_path)
            console.print(f"  ✓ {relative}")

        console.print(f"\n[bold green]✅ Downloaded {len(files)} file(s) to {dest}[/bold green]")

        # Suggest next steps
        has_torch = any(f["Key"].endswith(".json") for f in files)
        has_nsight = any(f["Key"].endswith(".nsys-rep") for f in files)
        if has_torch:
            console.print(f"\n[dim]View PyTorch traces: tensorboard --logdir {dest}[/dim]")
        if has_nsight:
            nsys_file = next(
                os.path.join(dest, f["Key"][len(prefix) :])
                for f in files
                if f["Key"].endswith(".nsys-rep")
            )
            console.print(f"\n[dim]Analyze Nsight: nsys stats {nsys_file}[/dim]")
