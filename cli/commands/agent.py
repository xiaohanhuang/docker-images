"""
CLI commands for managing the remote execution agent service.

⚠️  DEPRECATED: This command is deprecated. Use the new execution-service and
registry-service instead.

See projects/components/services/README.md for migration instructions.
"""

import subprocess

import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    help="[DEPRECATED] Manage the remote execution agent service. Use execution-service instead."
)


def _print_deprecation_warning():
    """Print deprecation notice."""
    console.print(
        "[yellow]⚠️  WARNING: The 'ml-plat agent' command is deprecated.[/yellow]\n"
        "[dim]The monolithic remote-agent has been split into:[/dim]\n"
        "  • [cyan]execution-service[/cyan] - Remote function execution\n"
        "  • [cyan]registry-service[/cyan] - Recipe registry\n\n"
        "[dim]See [bold]projects/components/services/README.md[/bold] for details.[/dim]\n"
    )


@app.command("deploy")
def deploy_agent():
    """
    [DEPRECATED] Deploy the remote execution agent service to the cluster.

    Use execution-service and registry-service instead.
    """
    _print_deprecation_warning()
    console.print(
        "[bold red]Error:[/bold red] The remote-agent service has been removed.\n"
        "\nTo deploy the new services:\n"
        "  [cyan]cd projects/components/services/execution-service && make deploy[/cyan]\n"
        "  [cyan]cd projects/components/services/registry-service && make deploy[/cyan]"
    )
    raise typer.Exit(1)


@app.command("status")
def agent_status(namespace: str = typer.Option("default", help="Kubernetes namespace")):
    """
    [DEPRECATED] Show the status of the remote execution agent service.
    """
    _print_deprecation_warning()
    console.print("[bold]Checking for new services...[/bold]\n")

    # Check for new services
    for service in ["execution-service", "registry-service"]:
        console.print(f"[dim]{service}:[/dim]")
        result = subprocess.run(
            ["kubectl", "get", "deployment", service, "-n", namespace],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print(result.stdout)
        else:
            console.print("  [yellow]Not deployed[/yellow]")
        console.print()


@app.command("logs")
def agent_logs(
    namespace: str = typer.Option("default", help="Kubernetes namespace"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow log output"),
    tail: int = typer.Option(50, "--tail", help="Number of lines to show"),
):
    """
    [DEPRECATED] Show logs from the remote execution agent service.
    """
    _print_deprecation_warning()
    console.print(
        "[bold red]Error:[/bold red] The remote-agent service no longer exists.\n"
        "\nTo view logs for the new services:\n"
        f"  [cyan]kubectl logs -l app=execution-service -n {namespace} --tail {tail}[/cyan]\n"
        f"  [cyan]kubectl logs -l app=registry-service -n {namespace} --tail {tail}[/cyan]"
    )
    raise typer.Exit(1)


@app.command("uninstall")
def uninstall_agent(
    namespace: str = typer.Option("default", help="Kubernetes namespace"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    [DEPRECATED] Uninstall the remote execution agent service from the cluster.
    """
    _print_deprecation_warning()
    console.print(
        "[bold red]Error:[/bold red] The remote-agent service no longer exists.\n"
        "\nTo uninstall the new services:\n"
        "  [cyan]kubectl delete -k projects/components/services/execution-service/k8s/[/cyan]\n"
        "  [cyan]kubectl delete -k projects/components/services/registry-service/k8s/[/cyan]"
    )
    raise typer.Exit(1)


@app.command("port-forward")
def port_forward_agent(
    local_port: int = typer.Option(8765, help="Local port to forward to"),
    namespace: str = typer.Option("default", help="Kubernetes namespace"),
):
    """
    [DEPRECATED] Port-forward the agent service to localhost for local development.
    """
    _print_deprecation_warning()
    console.print(
        "[bold red]Error:[/bold red] The remote-agent service no longer exists.\n"
        "\nTo port-forward the new services:\n"
        f"  [cyan]kubectl port-forward svc/execution-service 8765:8080 -n {namespace}[/cyan]\n"
        f"  [cyan]kubectl port-forward svc/registry-service 8766:8081 -n {namespace}[/cyan]"
    )
    raise typer.Exit(1)
