import subprocess
import webbrowser

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Manage Jupyter notebook sessions")


def _get_jupyter_url(namespace: str) -> str | None:
    """Discover JupyterHub URL from the Kubernetes ingress."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "ingress",
                "jupyterhub",
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.loadBalancer.ingress[0].hostname}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        hostname = result.stdout.strip()
        if hostname:
            return f"http://{hostname}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


@app.command("open")
def open_notebook(
    namespace: str = typer.Option("jupyter", help="JupyterHub namespace"),
):
    """
    Open JupyterHub in your browser.

    Discovers the JupyterHub URL from the cluster ingress and opens it.
    JupyterLab provides Jupyter notebooks, Marimo, VS Code, and terminal
    from the Launcher.
    """
    url = _get_jupyter_url(namespace)
    if not url:
        console.print(
            "[bold red]Could not discover JupyterHub URL.[/bold red]\n"
            "Check that the jupyterhub ingress exists:\n"
            f"  kubectl get ingress jupyterhub -n {namespace}"
        )
        raise typer.Exit(1)

    webbrowser.open(url)
    console.print(f"[bold green]✅ Opened JupyterHub: {url}[/bold green]", soft_wrap=True)


@app.command("status")
def notebook_status(
    namespace: str = typer.Option("jupyter", help="JupyterHub namespace"),
):
    """
    Show the status of all running notebook servers.
    """
    console.print("[cyan]Active notebook servers:[/cyan]")
    subprocess.run(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            "component=singleuser-server",
        ],
    )


@app.command("stop")
def stop_notebook(
    username: str = typer.Argument(..., help="Username whose server to stop"),
    namespace: str = typer.Option("jupyter", help="JupyterHub namespace"),
):
    """
    Stop a user's notebook server to free up resources.
    """
    pod_name = f"jupyter-{username}"
    console.print(f"Stopping notebook server for {username}...")
    subprocess.run(
        ["kubectl", "delete", "pod", pod_name, "-n", namespace],
    )
    console.print(f"[bold green]Notebook server for {username} stopped.[/bold green]")
