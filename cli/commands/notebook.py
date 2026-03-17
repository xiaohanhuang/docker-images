import socket
import subprocess
import time
import webbrowser

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Manage Jupyter notebook sessions")


@app.command("open")
def open_notebook(
    namespace: str = typer.Option("jupyter", help="JupyterHub namespace"),
    port: int = typer.Option(8080, help="Local port to forward to"),
    ide: str = typer.Option("jupyter", help="IDE to use: jupyter, marimo, or vscode"),
):
    """
    Open JupyterHub in your browser.
    Sets up port-forwarding and opens the URL automatically.
    """
    if ide not in ["jupyter", "marimo", "vscode"]:
        console.print(f"[bold red]Invalid IDE: {ide}[/bold red]")
        console.print("Valid options: jupyter, marimo, vscode")
        raise typer.Exit(1)

    ide_name = "JupyterHub" if ide == "jupyter" else ide.capitalize()
    console.print(f"[dim]Setting up port-forward to {ide_name}...[/dim]")

    cmd = [
        "kubectl",
        "port-forward",
        "svc/proxy-public",
        f"{port}:80",
        "-n",
        namespace,
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait until port is actually listening (up to 10 s)
        deadline = time.time() + 10
        while time.time() < deadline:
            if process.poll() is not None:
                console.print(
                    f"[bold red]Failed to start {ide_name} port-forward.[/bold red] "
                    "Check kubectl auth/context."
                )
                raise typer.Exit(1)
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            process.terminate()
            console.print(f"[bold red]Timed out waiting for {ide_name} on port {port}.[/bold red]")
            raise typer.Exit(1)

        url = f"http://localhost:{port}"
        console.print(f"\n[bold green]✅ {ide_name} available at: {url}[/bold green]")
        if ide == "marimo":
            console.print(
                "[dim]Note: Select a Marimo profile when spawning your server in JupyterHub[/dim]"
            )
        console.print("Press Ctrl+C to stop.\n")

        webbrowser.open(url)
        process.wait()

    except KeyboardInterrupt:
        console.print("\nStopping port-forward...")
        process.terminate()


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
