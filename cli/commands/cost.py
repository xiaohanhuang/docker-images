"""
cost.py — Real-time cost tracking for GPU workloads.

Commands:
  ml-plat cost              — Show live cost for all running GPU pods
  ml-plat cost summary      — Aggregated cost by user/namespace
  ml-plat cost estimate     — Estimate cost for a hypothetical job
"""

import datetime
import time
from collections import defaultdict
from typing import Optional

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
app = typer.Typer(help="Real-time GPU cost tracking")


# ── Pricing Data ─────────────────────────────────────────────────────
# On-Demand pricing for us-west-2 (Oregon) as of 2026
# Source: https://aws.amazon.com/ec2/pricing/on-demand/
INSTANCE_PRICING = {
    # GPU instances (g5 = NVIDIA A10G)
    "g5.xlarge": {"cost_hr": 1.006, "gpus": 1, "gpu_mem": 24, "vcpu": 4, "ram": 16},
    "g5.2xlarge": {"cost_hr": 1.212, "gpus": 1, "gpu_mem": 24, "vcpu": 8, "ram": 32},
    "g5.4xlarge": {"cost_hr": 1.624, "gpus": 1, "gpu_mem": 24, "vcpu": 16, "ram": 64},
    "g5.8xlarge": {"cost_hr": 2.448, "gpus": 1, "gpu_mem": 24, "vcpu": 32, "ram": 128},
    "g5.12xlarge": {"cost_hr": 5.672, "gpus": 4, "gpu_mem": 96, "vcpu": 48, "ram": 192},
    "g5.16xlarge": {"cost_hr": 4.096, "gpus": 1, "gpu_mem": 24, "vcpu": 64, "ram": 256},
    "g5.24xlarge": {"cost_hr": 8.144, "gpus": 4, "gpu_mem": 96, "vcpu": 96, "ram": 384},
    "g5.48xlarge": {
        "cost_hr": 16.288,
        "gpus": 8,
        "gpu_mem": 192,
        "vcpu": 192,
        "ram": 768,
    },
    # GPU instances (p4d = NVIDIA A100)
    "p4d.24xlarge": {
        "cost_hr": 32.77,
        "gpus": 8,
        "gpu_mem": 320,
        "vcpu": 96,
        "ram": 1152,
    },
    "p4de.24xlarge": {
        "cost_hr": 40.97,
        "gpus": 8,
        "gpu_mem": 640,
        "vcpu": 96,
        "ram": 1152,
    },
    # CPU instances
    "m5.large": {"cost_hr": 0.096, "gpus": 0, "gpu_mem": 0, "vcpu": 2, "ram": 8},
    "m5.xlarge": {"cost_hr": 0.192, "gpus": 0, "gpu_mem": 0, "vcpu": 4, "ram": 16},
    "m5.2xlarge": {"cost_hr": 0.384, "gpus": 0, "gpu_mem": 0, "vcpu": 8, "ram": 32},
    "m5.4xlarge": {"cost_hr": 0.768, "gpus": 0, "gpu_mem": 0, "vcpu": 16, "ram": 64},
    "m5.8xlarge": {"cost_hr": 1.536, "gpus": 0, "gpu_mem": 0, "vcpu": 32, "ram": 128},
    "m5a.xlarge": {"cost_hr": 0.172, "gpus": 0, "gpu_mem": 0, "vcpu": 4, "ram": 16},
    "m6i.xlarge": {"cost_hr": 0.192, "gpus": 0, "gpu_mem": 0, "vcpu": 4, "ram": 16},
    "m6a.xlarge": {"cost_hr": 0.173, "gpus": 0, "gpu_mem": 0, "vcpu": 4, "ram": 16},
}

# Spot discount estimates (actual spot pricing is dynamic)
SPOT_DISCOUNT = 0.60  # Spot is typically ~60% cheaper


def _load_kube():
    """Load kubernetes client, return (CoreV1Api, error_message)."""
    try:
        from kubernetes import client, config

        config.load_kube_config()
        return client.CoreV1Api(), None
    except Exception as e:
        return None, str(e)


_node_label_cache: dict[str, dict] = {}


def _get_node_labels(v1, node_name: str) -> dict:
    """Get and cache labels for a node to avoid duplicate API calls."""
    if node_name in _node_label_cache:
        return _node_label_cache[node_name]
    try:
        node = v1.read_node(name=node_name)
        labels = node.metadata.labels or {}
    except Exception:
        labels = {}
    _node_label_cache[node_name] = labels
    return labels


def _get_node_instance_type(v1, node_name: str) -> str:
    """Get instance type from a node's labels."""
    labels = _get_node_labels(v1, node_name)
    return (
        labels.get("node.kubernetes.io/instance-type")
        or labels.get("karpenter.k8s.aws/instance-type")
        or labels.get("beta.kubernetes.io/instance-type")
        or "unknown"
    )


def _get_capacity_type(v1, node_name: str) -> str:
    """Get capacity type (on-demand or spot) from node labels."""
    labels = _get_node_labels(v1, node_name)
    return labels.get("karpenter.sh/capacity-type", "on-demand")


def _get_cost_per_hour(instance_type: str, capacity_type: str = "on-demand") -> float:
    """Look up hourly cost for an instance type."""
    info = INSTANCE_PRICING.get(instance_type, {})
    base_cost = info.get("cost_hr", 0.0)
    if capacity_type == "spot":
        return base_cost * (1 - SPOT_DISCOUNT)
    return base_cost


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def _format_cost(cost: float) -> str:
    """Format cost with appropriate precision."""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.00:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def _get_gpu_pods(v1, namespace: Optional[str] = None):
    """
    Find all pods that request GPU resources.
    Returns a list of dicts with pod info.
    """
    pods = []

    if namespace:
        pod_list = v1.list_namespaced_pod(namespace=namespace)
    else:
        pod_list = v1.list_pod_for_all_namespaces()

    now = datetime.datetime.now(datetime.timezone.utc)

    for pod in pod_list.items:
        # Skip completed/failed pods
        if pod.status.phase not in ("Running", "Pending"):
            continue

        # Check if any container requests GPUs
        gpu_requested = 0
        for container in pod.spec.containers or []:
            limits = (container.resources.limits or {}) if container.resources else {}
            gpu_str = limits.get("nvidia.com/gpu", "0")
            gpu_requested += int(gpu_str)

        if gpu_requested == 0:
            continue

        # Get node info
        node_name = pod.spec.node_name or "pending"
        instance_type = "pending"
        capacity_type = "on-demand"
        if node_name != "pending":
            instance_type = _get_node_instance_type(v1, node_name)
            capacity_type = _get_capacity_type(v1, node_name)

        # Calculate run duration
        start_time = pod.status.start_time
        if start_time:
            duration_secs = (now - start_time).total_seconds()
        else:
            duration_secs = 0

        # Calculate cost
        cost_per_hour = _get_cost_per_hour(instance_type, capacity_type)
        total_cost = cost_per_hour * (duration_secs / 3600)

        # Get labels for categorization
        labels = pod.metadata.labels or {}
        user = (
            labels.get("user")
            or labels.get("app.kubernetes.io/managed-by")
            or labels.get("hub.jupyter.org/username")
            or "—"
        )
        workload_type = _classify_workload(labels, pod.metadata.name)

        pods.append(
            {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "phase": pod.status.phase,
                "node": node_name,
                "instance_type": instance_type,
                "capacity_type": capacity_type,
                "gpus": gpu_requested,
                "duration_secs": duration_secs,
                "cost_per_hour": cost_per_hour,
                "total_cost": total_cost,
                "user": user,
                "workload_type": workload_type,
                "start_time": start_time,
                "labels": labels,
            }
        )

    return pods


def _classify_workload(labels: dict, pod_name: str) -> str:
    """Classify pod into workload category."""
    if labels.get("ray.io/node-type") or "ray" in pod_name:
        return "🔮 Ray Training"
    if labels.get("hub.jupyter.org/username"):
        return "📓 Notebook"
    if labels.get("app") == "interactive-pod":
        return "🖥️  Interactive Pod"
    if "flyte" in labels.get("app", "") or "flyte" in pod_name:
        return "🔄 Flyte Task"
    if "train" in pod_name.lower():
        return "🏋️ Training"
    if "infer" in pod_name.lower() or "serve" in pod_name.lower():
        return "🚀 Inference"
    return "📦 Other"


def _build_cost_table(gpu_pods: list, title: str = "Running GPU Workloads") -> Table:
    """Build a Rich table of running GPU pods with costs."""
    table = Table(
        title=f"[bold]{title}[/bold]",
        box=box.ROUNDED,
        show_footer=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Pod", style="white", no_wrap=True, max_width=45)
    table.add_column("Type", style="yellow", no_wrap=True)
    table.add_column("GPUs", style="green", justify="center")
    table.add_column("Instance", style="magenta", no_wrap=True)
    table.add_column("Capacity", style="dim", no_wrap=True)
    table.add_column("Duration", style="white", justify="right")
    table.add_column("$/hr", style="yellow", justify="right")
    table.add_column("Cost", style="bold green", justify="right", footer_style="bold white")

    total_cost = 0.0
    total_gpus = 0
    total_cost_hr = 0.0

    for p in sorted(gpu_pods, key=lambda x: x["total_cost"], reverse=True):
        # Truncate long pod names
        name = p["name"]
        if len(name) > 45:
            name = name[:42] + "..."

        capacity_style = "green" if p["capacity_type"] == "spot" else "dim"
        capacity_label = "spot ⚡" if p["capacity_type"] == "spot" else "on-demand"

        table.add_row(
            name,
            p["workload_type"],
            str(p["gpus"]),
            p["instance_type"],
            f"[{capacity_style}]{capacity_label}[/{capacity_style}]",
            _format_duration(p["duration_secs"]),
            f"${p['cost_per_hour']:.3f}",
            _format_cost(p["total_cost"]),
        )
        total_cost += p["total_cost"]
        total_gpus += p["gpus"]
        total_cost_hr += p["cost_per_hour"]

    # Add footer with totals
    table.columns[0].footer = f"[bold]{len(gpu_pods)} pods[/bold]"
    table.columns[2].footer = f"[bold]{total_gpus}[/bold]"
    table.columns[6].footer = f"[bold]${total_cost_hr:.2f}[/bold]"
    table.columns[7].footer = f"[bold white]${total_cost:.2f}[/bold white]"

    return table


def _build_summary_table(gpu_pods: list) -> Table:
    """Build a summary table aggregated by user and workload type."""
    # Aggregate by user
    user_stats = defaultdict(lambda: {"pods": 0, "gpus": 0, "cost": 0.0, "cost_hr": 0.0})
    for p in gpu_pods:
        user = p["user"]
        user_stats[user]["pods"] += 1
        user_stats[user]["gpus"] += p["gpus"]
        user_stats[user]["cost"] += p["total_cost"]
        user_stats[user]["cost_hr"] += p["cost_per_hour"]

    table = Table(
        title="[bold]Cost by User[/bold]",
        box=box.ROUNDED,
        show_footer=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("User", style="white")
    table.add_column("Pods", justify="center")
    table.add_column("GPUs", style="green", justify="center")
    table.add_column("$/hr", style="yellow", justify="right")
    table.add_column("Total Cost", style="bold green", justify="right", footer_style="bold white")

    total = 0.0
    for user, stats in sorted(user_stats.items(), key=lambda x: x[1]["cost"], reverse=True):
        table.add_row(
            user,
            str(stats["pods"]),
            str(stats["gpus"]),
            f"${stats['cost_hr']:.2f}",
            _format_cost(stats["cost"]),
        )
        total += stats["cost"]

    table.columns[4].footer = f"[bold]${total:.2f}[/bold]"
    return table


def _build_workload_summary(gpu_pods: list) -> Table:
    """Build summary by workload type."""
    wl_stats = defaultdict(lambda: {"pods": 0, "gpus": 0, "cost": 0.0, "cost_hr": 0.0})
    for p in gpu_pods:
        wl = p["workload_type"]
        wl_stats[wl]["pods"] += 1
        wl_stats[wl]["gpus"] += p["gpus"]
        wl_stats[wl]["cost"] += p["total_cost"]
        wl_stats[wl]["cost_hr"] += p["cost_per_hour"]

    table = Table(
        title="[bold]Cost by Workload Type[/bold]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Workload", style="white")
    table.add_column("Pods", justify="center")
    table.add_column("GPUs", style="green", justify="center")
    table.add_column("$/hr", style="yellow", justify="right")
    table.add_column("Total", style="bold green", justify="right")

    for wl, stats in sorted(wl_stats.items(), key=lambda x: x[1]["cost"], reverse=True):
        table.add_row(
            wl,
            str(stats["pods"]),
            str(stats["gpus"]),
            f"${stats['cost_hr']:.2f}",
            _format_cost(stats["cost"]),
        )
    return table


# ── CLI Commands ─────────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def show_cost(
    ctx: typer.Context,
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Filter by namespace"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Live updating display"),
    interval: int = typer.Option(
        10, "--interval", "-i", help="Refresh interval in seconds (with --watch)"
    ),
):
    """
    💰 Show real-time GPU costs for all running workloads.

    Examples:
      ml-plat cost                      Show current costs
      ml-plat cost -w                   Live-updating cost ticker
      ml-plat cost -n ml-platform       Filter by namespace
    """
    # If a subcommand was invoked, skip this
    if ctx.invoked_subcommand is not None:
        return

    v1, err = _load_kube()
    if err:
        console.print(f"[bold red]❌ Could not connect to cluster:[/bold red] {err}")
        raise typer.Exit(1)

    if watch:
        _live_cost_display(v1, namespace, interval)
    else:
        _static_cost_display(v1, namespace)


def _static_cost_display(v1, namespace: Optional[str]):
    """One-shot cost display."""
    gpu_pods = _get_gpu_pods(v1, namespace)

    if not gpu_pods:
        console.print("[dim]No GPU pods are currently running.[/dim]")
        return

    # Main cost table
    console.print()
    console.print(_build_cost_table(gpu_pods))
    console.print()

    # Total burn rate
    total_cost_hr = sum(p["cost_per_hour"] for p in gpu_pods)
    total_cost = sum(p["total_cost"] for p in gpu_pods)
    total_gpus = sum(p["gpus"] for p in gpu_pods)

    console.print(
        Panel(
            f"[bold]🔥 Burn Rate:[/bold] [bold yellow]${total_cost_hr:.2f}/hr[/bold yellow]"
            f"   ({_format_cost(total_cost_hr / 60)}/min)\n"
            f"[bold]💰 Accrued Cost:[/bold] [bold green]{_format_cost(total_cost)}[/bold green]\n"
            f"[bold]🎮 Active GPUs:[/bold] [bold]{total_gpus}[/bold]    "
            f"[bold]📦 Running Pods:[/bold] [bold]{len(gpu_pods)}[/bold]\n"
            f"[dim]💡 Projected daily cost at current rate: "
            f"${total_cost_hr * 24:.2f}/day  |  "
            f"${total_cost_hr * 24 * 30:.0f}/month[/dim]",
            title="[bold]Cost Summary[/bold]",
            border_style="green",
        )
    )


def _live_cost_display(v1, namespace: Optional[str], interval: int):
    """Live-updating cost display that refreshes every N seconds."""

    def _build_live_panel():
        gpu_pods = _get_gpu_pods(v1, namespace)

        if not gpu_pods:
            return Panel("[dim]No GPU pods currently running.[/dim]", title="💰 GPU Cost Tracker")

        # Build tables
        cost_table = _build_cost_table(gpu_pods, title="Running GPU Workloads (live)")

        total_cost_hr = sum(p["cost_per_hour"] for p in gpu_pods)
        total_cost = sum(p["total_cost"] for p in gpu_pods)
        total_gpus = sum(p["gpus"] for p in gpu_pods)

        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        header = Text()
        header.append(f"  ⏱️  {now_str}", style="dim")
        header.append("    🔥 ", style="white")
        header.append(f"${total_cost_hr:.2f}/hr", style="bold yellow")
        header.append("    💰 ", style="white")
        header.append(_format_cost(total_cost), style="bold green")
        header.append(" accrued", style="dim")
        header.append(f"    🎮 {total_gpus} GPU{'s' if total_gpus != 1 else ''}", style="white")
        header.append(
            f"    📦 {len(gpu_pods)} pod{'s' if len(gpu_pods) != 1 else ''}",
            style="dim",
        )

        layout = Layout()
        layout.split_column(
            Layout(Panel(header, border_style="green", padding=(0, 1)), size=3),
            Layout(cost_table),
        )
        return layout

    try:
        with Live(_build_live_panel(), console=console, refresh_per_second=0.5) as live:
            while True:
                time.sleep(interval)
                live.update(_build_live_panel())
    except KeyboardInterrupt:
        console.print("\n[dim]Cost tracker stopped.[/dim]")


@app.command("summary")
def cost_summary(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Filter by namespace"),
):
    """
    📊 Show cost summary aggregated by user and workload type.

    Example:
      ml-plat cost summary
    """
    v1, err = _load_kube()
    if err:
        console.print(f"[bold red]❌ Could not connect to cluster:[/bold red] {err}")
        raise typer.Exit(1)

    gpu_pods = _get_gpu_pods(v1, namespace)

    if not gpu_pods:
        console.print("[dim]No GPU pods are currently running.[/dim]")
        return

    console.print()
    console.print(Columns([_build_summary_table(gpu_pods), _build_workload_summary(gpu_pods)]))
    console.print()


@app.command("estimate")
def cost_estimate(
    instance: str = typer.Option("g5.xlarge", "--instance", "-i", help="Instance type"),
    hours: float = typer.Option(1.0, "--hours", "-t", help="Estimated duration in hours"),
    spot: bool = typer.Option(False, "--spot", "-s", help="Use spot pricing estimate"),
    nodes: int = typer.Option(1, "--nodes", "-n", help="Number of nodes (for distributed)"),
):
    """
    🧮 Estimate cost for a hypothetical GPU job.

    Examples:
      ml-plat cost estimate                                  # 1x g5.xlarge for 1 hour
      ml-plat cost estimate -i g5.12xlarge -t 8              # 4-GPU for 8 hours
      ml-plat cost estimate -i p4d.24xlarge -t 24 -n 2       # Multi-node A100
      ml-plat cost estimate -i g5.xlarge -t 4 --spot         # Spot pricing
    """
    info = INSTANCE_PRICING.get(instance)
    if not info:
        console.print(f"[bold red]Unknown instance type:[/bold red] {instance}")
        console.print("[dim]Available types:[/dim]")
        for it in sorted(INSTANCE_PRICING.keys()):
            p = INSTANCE_PRICING[it]
            if p["gpus"] > 0:
                console.print(
                    f"  {it:20s}  {p['gpus']} GPU  {p['gpu_mem']}GB  ${p['cost_hr']:.3f}/hr"
                )
        raise typer.Exit(1)

    capacity = "spot" if spot else "on-demand"
    cost_hr = _get_cost_per_hour(instance, capacity)
    total_cost = cost_hr * hours * nodes

    table = Table(
        title="[bold]Cost Estimate[/bold]",
        box=box.ROUNDED,
        show_header=False,
        border_style="cyan",
    )
    table.add_column("Label", style="bold")
    table.add_column("Value", style="white")

    table.add_row("Instance Type", f"{instance}")
    table.add_row(
        "GPUs per Node",
        f"{info['gpus']} × NVIDIA {'A100' if 'p4' in instance else 'A10G'} ({info['gpu_mem']}GB)",
    )
    table.add_row("vCPU / RAM", f"{info['vcpu']} vCPU / {info['ram']} GB")
    table.add_row("Nodes", str(nodes))
    table.add_row("Duration", f"{hours}h")
    table.add_row(
        "Capacity Type",
        f"[{'green' if spot else 'white'}]{capacity}[/{'green' if spot else 'white'}]",
    )
    table.add_row(
        "Cost per Hour",
        f"${cost_hr:.3f}/hr" + (" [green](~60% savings)[/green]" if spot else ""),
    )
    table.add_row("", "")
    table.add_row(
        "[bold]Estimated Total Cost[/bold]",
        f"[bold green]${total_cost:.2f}[/bold green]",
    )

    if not spot:
        spot_cost = _get_cost_per_hour(instance, "spot") * hours * nodes
        table.add_row(
            "[dim]With Spot[/dim]",
            f"[dim green]${spot_cost:.2f} (save ${total_cost - spot_cost:.2f})[/dim green]",
        )

    console.print()
    console.print(table)
    console.print()


@app.command("pricing")
def show_pricing():
    """
    📋 Show available instance types and their pricing.
    """
    table = Table(
        title="[bold]GPU Instance Pricing (us-west-2 On-Demand)[/bold]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Instance", style="white", no_wrap=True)
    table.add_column("GPUs", style="green", justify="center")
    table.add_column("GPU Memory", style="yellow", justify="right")
    table.add_column("vCPU", style="dim", justify="center")
    table.add_column("RAM (GB)", style="dim", justify="right")
    table.add_column("On-Demand $/hr", style="bold green", justify="right")
    table.add_column("Spot $/hr (est)", style="bold yellow", justify="right")

    gpu_instances = {k: v for k, v in INSTANCE_PRICING.items() if v["gpus"] > 0}
    for it in sorted(gpu_instances.keys()):
        p = gpu_instances[it]
        spot_cost = p["cost_hr"] * (1 - SPOT_DISCOUNT)
        table.add_row(
            it,
            str(p["gpus"]),
            f"{p['gpu_mem']} GB",
            str(p["vcpu"]),
            str(p["ram"]),
            f"${p['cost_hr']:.3f}",
            f"${spot_cost:.3f}",
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        "[dim]Spot pricing is estimated at ~60% discount. "
        "Actual spot prices vary dynamically.[/dim]"
    )
    console.print("[dim]Prices are for us-west-2 (Oregon). Other regions may differ.[/dim]")
