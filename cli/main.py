import os
import sys

import typer
import yaml
from rich.console import Console

from cli.commands import (
    agent,
    component,
    cost,
    debug,
    init,
    job,
    notebook,
    pod,
    recipe,
    workflow,
)

_console = Console()


def load_config():
    config_path = os.path.expanduser("~/.ml-plat/config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        cluster = cfg.get("cluster", {})
        # Use setdefault so that values already in the environment (e.g. from
        # CI or test patches) are NOT overwritten by the config file.
        os.environ.setdefault(
            "FLYTE_ENDPOINT",
            cluster.get("flyte_endpoint", "flyteadmin.ml-platform.internal:80"),
        )
        os.environ.setdefault("FLYTE_PROJECT", cluster.get("flyte_project", "flytesnacks"))
        os.environ.setdefault("FLYTE_DOMAIN", cluster.get("flyte_domain", "development"))
        flyte_console_url = cluster.get("flyte_console_url")
        if flyte_console_url:
            os.environ.setdefault("FLYTE_CONSOLE_URL", flyte_console_url)
        os.environ.setdefault(
            "MLFLOW_TRACKING_URI",
            cluster.get("mlflow_tracking_uri", "http://mlflow.ml-platform.internal"),
        )
        s3_bucket = cluster.get("s3_bucket")
        if s3_bucket:
            os.environ.setdefault("S3_BUCKET", s3_bucket)
        # Also export vars used directly by the pyflyte CLI
        os.environ.setdefault("FLYTE_DEFAULT_PROJECT", os.environ["FLYTE_PROJECT"])
        os.environ.setdefault("FLYTE_DEFAULT_DOMAIN", os.environ["FLYTE_DOMAIN"])

        ecr = cfg.get("ecr", {})
        ecr_registry = ecr.get("registry")
        if ecr_registry:
            os.environ.setdefault("ECR_REGISTRY", ecr_registry)
        os.environ.setdefault("AWS_REGION", ecr.get("region", "us-west-2"))

        notifs = cfg.get("notifications", {})
        if notifs.get("teams_webhook_url"):
            os.environ.setdefault("TEAMS_WEBHOOK_URL", notifs["teams_webhook_url"])


app = typer.Typer(
    name="ml-plat",
    help="CLI for the ML Training Platform on AWS EKS",
)


@app.callback()
def _startup(ctx: typer.Context):
    """Load cluster config into environment variables before any subcommand."""
    load_config()


# Register the wizard as a top-level command (ml-plat wizard)
from cli.commands.onboard import wizard  # noqa: E402

app.command("wizard")(wizard)

app.add_typer(init.app, name="init")
app.add_typer(job.app, name="job")
app.add_typer(debug.app, name="debug")
app.add_typer(notebook.app, name="notebook")
app.add_typer(workflow.app, name="workflow")
app.add_typer(pod.app, name="pod")
app.add_typer(cost.app, name="cost")
app.add_typer(component.app, name="component")
app.add_typer(agent.app, name="agent")
app.add_typer(recipe.app, name="recipe")


def main() -> None:
    """Entry point: delegates to the Typer app, but intercepts unknown commands."""
    import typer.main as _typer_main

    args = sys.argv[1:]
    if args:
        # Materialise the Click group to get the set of registered names
        click_group = _typer_main.get_command(app)
        known = set(click_group.commands.keys())

        first = args[0]
        if not first.startswith("-") and first not in known:
            attempted = " ".join(args)

            # If it looks like the user is trying to launch the wizard, just do it.
            _wizard_words = {"wizard", "onboard", "start", "setup", "new", "begin", "help"}
            if _wizard_words.intersection(a.lower() for a in args):
                try:
                    wizard()
                except (SystemExit, KeyboardInterrupt):
                    pass
                return

            # Otherwise: intercept, store the command, let wizard explain
            os.environ["_MLPLAT_UNKNOWN_CMD"] = attempted
            _console.print(
                f"\n[yellow]I don't recognise '[bold]{first}[/bold]'.[/yellow] "
                "Let me help you find the right command.\n"
                "[dim](Ctrl+C or type 'done' at any time to exit)[/dim]\n"
            )
            try:
                wizard()
            except (SystemExit, KeyboardInterrupt):
                pass
            finally:
                os.environ.pop("_MLPLAT_UNKNOWN_CMD", None)
            return

    app()


if __name__ == "__main__":
    main()
