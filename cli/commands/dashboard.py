import os
import webbrowser

import typer
from rich.console import Console

app = typer.Typer(help="💻 Open the ML Platform Global Dashboard.")
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Open the ML Platform Global Dashboard in your browser."""
    if ctx.invoked_subcommand is None:
        # Load the configuration to check if there is a dashboard URL configured
        config_path = os.path.expanduser("~/.mlp/config.yaml")
        dashboard_url = (
            "http://k8s-mlplatfo-mlplatfo-b0f5b5dde2-1200669602.us-west-2.elb.amazonaws.com"
        )

        if os.path.exists(config_path):
            import yaml

            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                cluster = cfg.get("cluster", {})
                if "dashboard_url" in cluster:
                    dashboard_url = cluster["dashboard_url"]
            except Exception:
                pass

        # Override with env var if provided
        dashboard_url = os.environ.get("MLPLAT_DASHBOARD_URL", dashboard_url)

        console.print(f"[bold green]Opening Dashboard:[/bold green] {dashboard_url}")

        try:
            webbrowser.open(dashboard_url)
        except Exception as e:
            console.print(f"[yellow]Could not open browser automatically.[/yellow] {e}")
