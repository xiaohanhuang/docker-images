import subprocess
import webbrowser

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="TensorBoard log management")


def _get_tb_url(namespace: str) -> str | None:
    """Discover TensorBoard URL from the Kubernetes ingress."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "ingress",
                "tensorboard",
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.loadBalancer.ingress[0].hostname}"
                "/{.metadata.annotations.alb\\.ingress\\.kubernetes\\.io\\/listen-ports}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        parts = result.stdout.strip()
        if not parts:
            return None
        # kubectl concatenates hostname + listen-ports without a separator;
        # detect HTTPS in the annotations to pick the scheme.
        scheme = "https" if "HTTPS" in parts else "http"
        hostname = parts.split("/")[0]
        if hostname:
            return f"{scheme}://{hostname}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


@app.command("open")
def open_tensorboard(
    execution_id: str | None = typer.Argument(None, help="Optional execution/run ID to filter to"),
    namespace: str = typer.Option("monitoring", help="TensorBoard namespace"),
):
    """Open the shared TensorBoard instance in your browser."""
    url = _get_tb_url(namespace)
    if not url:
        console.print(
            "[bold red]Could not discover TensorBoard URL.[/bold red]\n"
            "Check that the tensorboard ingress exists:\n"
            f"  kubectl get ingress tensorboard -n {namespace}"
        )
        raise typer.Exit(1)

    if execution_id:
        from urllib.parse import quote

        url = f"{url}/#scalars&regexInput={quote(execution_id)}"

    webbrowser.open(url)
    console.print(f"[bold green]✅ Opened TensorBoard: {url}[/bold green]", soft_wrap=True)


@app.command("list")
def list_runs(
    bucket: str | None = typer.Option(None, help="S3 bucket (defaults to $S3_BUCKET)"),
):
    """List available TensorBoard log directories in S3."""
    import os

    bucket = bucket or os.environ.get("S3_BUCKET")
    if not bucket:
        console.print("[bold red]S3_BUCKET not set. Pass --bucket or run mlp init.[/bold red]")
        raise typer.Exit(1)

    try:
        import boto3

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        prefixes: list[dict] = []
        for page in paginator.paginate(
            Bucket=bucket,
            Prefix="tensorboard/",
            Delimiter="/",
        ):
            prefixes.extend(page.get("CommonPrefixes", []))
    except Exception as e:
        console.print(f"[bold red]Failed to list S3: {e}[/bold red]")
        raise typer.Exit(1)
    if not prefixes:
        console.print("[dim]No TensorBoard logs found in S3.[/dim]")
        return

    table = Table(title="TensorBoard Runs")
    table.add_column("Execution ID", style="cyan")
    table.add_column("S3 Path", style="dim")

    for p in prefixes:
        prefix = p["Prefix"]
        exec_id = prefix.rstrip("/").split("/")[-1]
        table.add_row(exec_id, f"s3://{bucket}/{prefix}")

    console.print(table)


@app.command("serve")
def serve_local(
    namespace: str = typer.Option("monitoring", help="TensorBoard namespace"),
    port: int = typer.Option(6006, help="Local port"),
):
    """Port-forward to the cluster TensorBoard service."""
    console.print(f"[cyan]Forwarding TensorBoard to http://localhost:{port} ...[/cyan]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    try:
        subprocess.run(
            [
                "kubectl",
                "port-forward",
                "svc/tensorboard",
                f"{port}:6006",
                "-n",
                namespace,
            ],
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
