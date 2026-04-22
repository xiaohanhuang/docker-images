import os
import subprocess

import typer
import yaml
from rich.console import Console

console = Console()

app = typer.Typer(help="Initialize ML Platform CLI")


@app.callback(invoke_without_command=True)
def init(
    teams_webhook: str = typer.Option(
        "",
        prompt="Enter Microsoft Teams Webhook URL for notifications (press Enter to skip)",
    ),
):
    """
    Initialize the mlp CLI by reading Terraform outputs and generating config.
    """
    config_dir = os.path.expanduser("~/.mlp")
    config_path = os.path.join(config_dir, "config.yaml")

    os.makedirs(config_dir, exist_ok=True)
    console.print("[cyan]🔍 Auto-detecting cluster configuration from Terraform...[/cyan]")

    try:
        # Assuming we are running this from near the sandbox dir
        eks_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "projects", "eks")
        )

        # We need to run terraform output
        def get_tf_output(name):
            try:
                subprocess.check_output(
                    ["aws", "sts", "get-caller-identity", "--profile", "adfs"],
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
            res = subprocess.check_output(
                ["terraform", "output", "-raw", name],
                env={**os.environ, "AWS_PROFILE": "adfs"},
                cwd=eks_dir,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return res

        s3_bucket = get_tf_output("s3_bucket_name")
        cluster_name = get_tf_output("cluster_name")

        if not s3_bucket or not cluster_name:
            console.print(
                "[yellow]\u26a0\ufe0f  Could not automatically fetch Terraform outputs. "
                "Using defaults.[/yellow]"
            )
            s3_bucket = "ml-platform-data-ml-platform-eks-805673386114"
            cluster_name = "ml-platform-eks"

    except Exception as e:
        console.print(f"[yellow]⚠️  Terraform detection failed: {e}. Using defaults.[/yellow]")
        s3_bucket = "ml-platform-data-ml-platform-eks-805673386114"
        cluster_name = "ml-platform-eks"

    config_data = {
        "cluster": {
            "name": cluster_name,
            "region": "us-west-2",
            "flyte_endpoint": "flyteadmin.ml-platform.internal:80",
            "flyte_console_url": "http://k8s-flyte-flytecon-a425d1f87c-1407955100.us-west-2.elb.amazonaws.com",
            "flyte_project": "ml-platform",
            "flyte_domain": "development",
            "mlflow_tracking_uri": "http://k8s-monitori-mlflow-0985830759-1985412867.us-west-2.elb.amazonaws.com",
            "dashboard_url": "http://k8s-mlplatfo-mlplatfo-b0f5b5dde2-1200669602.us-west-2.elb.amazonaws.com",
            "s3_bucket": s3_bucket,
        },
        "ecr": {
            "registry": "805673386114.dkr.ecr.us-west-2.amazonaws.com",
            "region": "us-west-2",
        },
        "notifications": {"teams_webhook_url": teams_webhook.strip()},
    }

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    console.print(f"[bold green]✅ Configuration saved to {config_path}[/bold green]")
    console.print(f"   Cluster : {cluster_name}")
    console.print(f"   Bucket  : {s3_bucket}")
    if teams_webhook:
        console.print("   Webhooks: Enabled")


if __name__ == "__main__":
    app()
