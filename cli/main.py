import os
import sys

# Suppress Flytekit's noisy "Unsupported Type typing.Any" warning.
# Flytekit uses its own Rich console handler (not Python warnings), so the
# only way to suppress it is via this env var.  Level 40 = ERROR.
# Without this, the warning breaks zsh tab-completion by injecting
# timestamps into the completion output.
os.environ.setdefault("FLYTE_SDK_LOGGING_LEVEL", "40")

import typer

from cli.commands import (
    component,
    cost,
    dashboard,
    debug,
    feature_store,
    init,
    job,
    notebook,
    pod,
    recipe,
    tensorboard,
    workflow,
)


def load_config():
    config_path = os.path.expanduser("~/.mlp/config.yaml")
    if os.path.exists(config_path):
        import yaml

        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return

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


# Use the actual binary name so --install-completion registers the right
# env var (_MLP_COMPLETE when invoked as mlp).
_invoked_name = os.path.basename(sys.argv[0]) if sys.argv else "mlp"
# Normalise: strip .py / .exe suffixes and default if empty
if "." in _invoked_name:
    _invoked_name = _invoked_name.rsplit(".", 1)[0]
_cli_name = _invoked_name if _invoked_name == "mlp" else "mlp"

app = typer.Typer(
    name=_cli_name,
    help="CLI for the ML Training Platform on AWS EKS",
)


@app.callback()
def _startup(ctx: typer.Context):
    """Load cluster config into environment variables before any subcommand."""
    load_config()


# Register the wizard as a top-level command (mlp wizard)
@app.command("wizard")
def _wizard_cmd():
    """Interactive onboarding wizard for new ML engineers."""
    from cli.commands.onboard import wizard

    wizard()


app.add_typer(init.app, name="init")
app.add_typer(job.app, name="job")
app.add_typer(debug.app, name="debug")
app.add_typer(notebook.app, name="notebook")
app.add_typer(workflow.app, name="workflow")
app.add_typer(pod.app, name="pod")
app.add_typer(cost.app, name="cost")
app.add_typer(component.app, name="component")

app.add_typer(recipe.app, name="recipe")
app.add_typer(feature_store.app, name="feature-store")
app.add_typer(tensorboard.app, name="tensorboard")
app.add_typer(dashboard.app, name="dashboard")


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
                    _wizard_cmd()
                except (SystemExit, KeyboardInterrupt):
                    pass
                return

            # Otherwise: intercept, store the command, let wizard explain
            os.environ["_MLPLAT_UNKNOWN_CMD"] = attempted
            from rich.console import Console

            _console = Console()
            _console.print(
                f"\n[yellow]I don't recognise '[bold]{first}[/bold]'.[/yellow] "
                "Let me help you find the right command.\n"
                "[dim](Ctrl+C or type 'done' at any time to exit)[/dim]\n"
            )
            try:
                _wizard_cmd()
            except (SystemExit, KeyboardInterrupt):
                pass
            finally:
                os.environ.pop("_MLPLAT_UNKNOWN_CMD", None)
            return

    app()


if __name__ == "__main__":
    main()
