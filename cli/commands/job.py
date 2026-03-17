import datetime
import os

import typer
from flytekit.models.core.execution import WorkflowExecutionPhase
from rich.console import Console
from rich.table import Table

from cli.utils import flyte_console_url, flyte_remote

app = typer.Typer(help="Manage training jobs")
console = Console()


class FinOpsManager:
    # Cost per hour for common instance types (On-Demand pricing)
    COST_PER_HOUR = {
        "g5.xlarge": 1.006,
        "g5.2xlarge": 1.212,
        "g5.4xlarge": 1.624,
        "g5.8xlarge": 2.448,
        "g5.12xlarge": 5.672,
        "g5.16xlarge": 4.096,
        "g5.24xlarge": 8.144,
        "g5.48xlarge": 16.288,
        "m5.large": 0.096,
        "m5.xlarge": 0.192,
        "m5.2xlarge": 0.384,
        "m5.4xlarge": 0.768,
        "m5.8xlarge": 1.536,
        "p3.2xlarge": 3.06,
        "p3.8xlarge": 12.24,
    }

    @classmethod
    def get_cost_estimate(cls, instance_type: str, duration: datetime.timedelta) -> float:
        hourly_rate = cls.COST_PER_HOUR.get(instance_type, 0.0)
        total_hours = duration.total_seconds() / 3600.0
        return total_hours * hourly_rate


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
):
    """
    Submit a training job to the cluster.
    """
    try:
        remote = flyte_remote()
        typer.echo(f"Fetching workflow {workflow_name} version {version}...")

        # Fetch the registered workflow
        flyte_workflow = remote.fetch_workflow(name=workflow_name, version=version)

        # Execute it
        execution = remote.execute(
            flyte_workflow,
            inputs={"s3_dataset_path": dataset, "num_epochs": epochs},
            wait=False,
        )

        typer.echo("Job submitted successfully!")
        typer.echo(f"Job ID: {execution.id.name}")
        url = flyte_console_url(execution.id.project, execution.id.domain, execution.id.name)
        typer.echo(f"View console: {url}")

    except Exception as e:
        typer.secho(f"Failed to submit job: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command("status")
def get_status(job_id: str):
    """
    Get the status of a job, including instance type and estimated cost.
    """
    try:
        remote = flyte_remote()
        execution = remote.fetch_execution(name=job_id)
        remote.sync_execution(execution, sync_nodes=True)

        # 1. Determine Instance Type via K8s labels
        instance_type = "unknown"
        # Try to find instance type in node executions
        for node in execution.node_executions.values():
            if hasattr(node.closure, "task_node_metadata") and node.closure.task_node_metadata:
                pass

        # Fallback: Check if we can get it from K8s if the job is running
        if instance_type == "unknown":
            try:
                from kubernetes import client, config

                config.load_kube_config()
                v1 = client.CoreV1Api()
                _proj = os.getenv("FLYTE_PROJECT", "flytesnacks")
                _dom = os.getenv("FLYTE_DOMAIN", "development")
                target_namespace = os.getenv("FLYTE_NAMESPACE", f"{_proj}-{_dom}")
                pods = v1.list_namespaced_pod(
                    namespace=target_namespace,
                    label_selector=f"flyte-execution-id={job_id}",
                )
                if not pods.items:
                    pods = v1.list_namespaced_pod(
                        namespace=target_namespace,
                        label_selector=f"flyte.org/execution={job_id}",
                    )

                if pods.items and pods.items[0].spec.node_name:
                    node_info = v1.read_node(name=pods.items[0].spec.node_name)
                    instance_type = (
                        node_info.metadata.labels.get("node.kubernetes.io/instance-type")
                        or node_info.metadata.labels.get("karpenter.k8s.aws/instance-type")
                        or "unknown"
                    )
            except Exception:
                instance_type = "g5.xlarge"

        # 2. Calculate Duration
        started_at = execution.closure.started_at
        phase = execution.closure.phase

        phase_str = (
            WorkflowExecutionPhase.enum_to_string(phase)
            if hasattr(WorkflowExecutionPhase, "enum_to_string")
            else str(phase)
        )

        if not started_at:
            duration = datetime.timedelta(0)
        elif phase in [
            WorkflowExecutionPhase.SUCCEEDED,
            WorkflowExecutionPhase.FAILED,
            WorkflowExecutionPhase.ABORTED,
        ]:
            if execution.closure.duration:
                duration = execution.closure.duration
            else:
                duration = execution.closure.updated_at - started_at
        else:
            now = datetime.datetime.now(datetime.timezone.utc)
            duration = now - started_at

        # 3. Estimate Cost
        cost = FinOpsManager.get_cost_estimate(instance_type, duration)

        # 4. Display Status Table
        phase_color = {
            "RUNNING": "green",
            "SUCCEEDED": "cyan",
            "FAILED": "red",
            "ABORTED": "yellow",
        }
        color = phase_color.get(phase_str, "white")
        table = Table(title=f"Job Status: {job_id}", show_header=False, box=None)
        table.add_row("[bold]Job ID:[/bold]", job_id)
        table.add_row("[bold]Status:[/bold]", f"[{color}]{phase_str}[/{color}]")
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
        remote = flyte_remote()
        execution = remote.fetch_execution(name=job_id)
        remote.sync_execution(execution, sync_nodes=True)
        typer.echo(f"Fetching logs for {job_id}...")
        if not execution.node_executions:
            typer.echo("No node executions found yet.")
            return
        for node in execution.node_executions.values():
            phase_str = (
                WorkflowExecutionPhase.enum_to_string(node.closure.phase)
                if hasattr(node.closure.phase, "real")
                else str(node.closure.phase)
            )
            typer.echo(f"  Node: {node.id.node_id}, Status: {phase_str}")
            if hasattr(node.closure, "logs") and node.closure.logs:
                for log in node.closure.logs:
                    typer.echo(f"    Log: {log.uri}")
    except Exception as e:
        typer.secho(f"Error fetching logs: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)
