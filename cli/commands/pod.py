import os
import subprocess
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cli.gpu_types import GpuType

console = Console()
app = typer.Typer(help="Manage RunPod-style interactive GPU pods")


def _api():
    """Return a configured APIClient."""
    from cli.api_client import APIClient

    return APIClient()


def _ssh_marker_start(pod_name: str) -> str:
    return f"### ML-PLAT {pod_name} START ###"


def _ssh_marker_end(pod_name: str) -> str:
    return f"### ML-PLAT {pod_name} END ###"


# Kept for backward-compat (used by tests)
SSH_CONFIG_MARKER_START = _ssh_marker_start("ml-pod")
SSH_CONFIG_MARKER_END = _ssh_marker_end("ml-pod")


def _find_free_port(start: int = 2222) -> int:
    """Return the first TCP port >= start that is not currently bound."""
    import socket as _socket

    port = start
    while True:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1


def _get_ssh_port_for_pod(pod_name: str) -> int | None:
    """Read the SSH port for a pod from the managed SSH config block."""
    import re as _re

    ssh_config = Path.home() / ".ssh" / "config"
    if not ssh_config.exists():
        return None
    content = ssh_config.read_text()
    marker_start = _ssh_marker_start(pod_name)
    marker_end = _ssh_marker_end(pod_name)
    pattern = _re.compile(f"{_re.escape(marker_start)}(.*?){_re.escape(marker_end)}", _re.DOTALL)
    m = pattern.search(content)
    if not m:
        return None
    port_match = _re.search(r"Port (\d+)", m.group(1))
    return int(port_match.group(1)) if port_match else None


def _prune_ssh_config(namespace: str = "default", all_namespaces: bool = False) -> None:
    """Remove SSH config entries for pods that no longer exist."""
    import re as _re

    ssh_config = Path.home() / ".ssh" / "config"
    if not ssh_config.exists():
        return

    content = ssh_config.read_text()
    managed = _re.findall(r"### ML-PLAT (interactive-pod-\S+) START ###", content)
    if not managed:
        return

    try:
        with _api() as client:
            data = client.list_pods(namespace=namespace, all_namespaces=all_namespaces)
            existing = {p["name"] for p in data.get("pods", [])}
    except Exception:
        return  # Don't prune if we can't reach the backend

    for pod_name in managed:
        if pod_name not in existing:
            _remove_ssh_config(pod_name)


def _check_prerequisites():
    """Check that required CLI tools are available."""
    import shutil

    missing = []
    if not shutil.which("kubectl"):
        missing.append("kubectl (required for port-forwarding)")
    if not shutil.which("ssh"):
        missing.append("ssh (required for connecting to pods)")
    if missing:
        console.print("[bold red]Missing required tools:[/bold red]")
        for tool in missing:
            console.print(f"  • {tool}")
        console.print("\n[dim]Install kubectl: https://kubernetes.io/docs/tasks/tools/[/dim]")
        raise typer.Exit(1)


def _setup_ssh_config(port: int, pod_name: str = "ml-pod"):
    """Add or update the per-pod SSH config block."""
    import re as _re

    ssh_dir = Path.home() / ".ssh"
    ssh_config = ssh_dir / "config"

    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    marker_start = _ssh_marker_start(pod_name)
    marker_end = _ssh_marker_end(pod_name)

    config_entry = (
        f"{marker_start}\n"
        f"Host {pod_name}\n"
        f"    HostName localhost\n"
        f"    User root\n"
        f"    Port {port}\n"
        f"    StrictHostKeyChecking no\n"
        f"    UserKnownHostsFile /dev/null\n"
        f"{marker_end}\n"
    )

    content = ""
    if ssh_config.exists():
        content = ssh_config.read_text()

    if marker_start in content:
        pattern = _re.compile(
            f"{_re.escape(marker_start)}.*?{_re.escape(marker_end)}\n?",
            _re.DOTALL,
        )
        new_content = pattern.sub(config_entry, content)
        ssh_config.write_text(new_content)
    else:
        if _re.search(rf"^Host\s+{_re.escape(pod_name)}\s*$", content, _re.MULTILINE):
            console.print(
                f"[yellow]Warning: existing 'Host {pod_name}' found in SSH config. "
                "The managed block is appended and may be shadowed.[/yellow]"
            )
        with open(ssh_config, "a") as f:
            f.write(f"\n{config_entry}")

    console.print(f"[dim]SSH config updated for VS Code: `ssh {pod_name}`[/dim]")


def _remove_ssh_config(pod_name: str = "ml-pod"):
    """Remove the per-pod managed SSH config block."""
    import re as _re

    ssh_config = Path.home() / ".ssh" / "config"
    if not ssh_config.exists():
        return

    marker_start = _ssh_marker_start(pod_name)
    marker_end = _ssh_marker_end(pod_name)

    content = ssh_config.read_text()
    if marker_start in content:
        pattern = _re.compile(
            f"\n?{_re.escape(marker_start)}.*?{_re.escape(marker_end)}\n?",
            _re.DOTALL,
        )
        new_content = pattern.sub("", content)
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"
        ssh_config.write_text(new_content)
        console.print(f"[dim]SSH config entry `{pod_name}` removed.[/dim]")


def _find_ssh_pub_key() -> Path | None:
    """Find the user's SSH public key."""
    for key_name in ["id_rsa.pub", "id_ed25519.pub", "id_ecdsa.pub"]:
        key_path = Path.home() / ".ssh" / key_name
        if key_path.exists():
            return key_path
    return None


def _copy_ssh_key_to_pod(pod_name: str, namespace: str):
    """Copy the user's SSH public key into a pod for passwordless auth."""
    ssh_pub_key = _find_ssh_pub_key()
    if ssh_pub_key:
        pub_key_content = ssh_pub_key.read_text().strip()
        subprocess.run(
            [
                "kubectl",
                "exec",
                pod_name,
                "-n",
                namespace,
                "--",
                "bash",
                "-c",
                f"mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
                f"echo '{pub_key_content}' > /root/.ssh/authorized_keys && "
                f"chmod 600 /root/.ssh/authorized_keys",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        console.print("[dim]SSH public key copied to pod.[/dim]")
    else:
        console.print(
            "[yellow]Warning: No SSH public key found (~/.ssh/id_*.pub). "
            "VS Code Remote-SSH may prompt for password (root:root).[/yellow]"
        )


def _start_port_forward(
    pod_name: str, namespace: str, ssh_port: int = 2222, jupyter_port: int = 8888
):
    """Start kubectl port-forward and return the process. Raises typer.Exit on failure."""
    pf_command = [
        "kubectl",
        "port-forward",
        pod_name,
        f"{ssh_port}:22",
        f"{jupyter_port}:8888",
        "-n",
        namespace,
    ]

    pf_process = subprocess.Popen(pf_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Verify port-forward started
    time.sleep(0.3)
    if pf_process.poll() is not None:
        console.print("[bold red]❌ Port-forward failed.[/bold red]")
        raise typer.Exit(1)

    console.print(
        f"[dim]Port-forwarding: localhost:{ssh_port} → 22, "
        f"localhost:{jupyter_port} → 8888[/dim]"
    )
    return pf_process


def _wait_for_port(port: int, timeout: int = 20) -> bool:
    """Poll localhost:port until it accepts connections or timeout (seconds)."""
    import socket as _socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with _socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _connect_ssh(port: int = 2222, retries: int = 3):
    """Open an interactive SSH session to the pod with retries."""
    console.print("[bold cyan]Connecting via SSH...[/bold cyan]")
    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-t",
        "-p",
        str(port),
        "root@localhost",
    ]
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(ssh_cmd)
            if result.returncode == 0:
                return  # Successful session
            if attempt < retries:
                msg = f"SSH connection failed (attempt {attempt}/{retries}), retrying..."
                console.print(f"[yellow]{msg}[/yellow]")
                time.sleep(1)
            else:
                console.print(
                    "[bold red]❌ SSH connection failed after all retries. "
                    "You can still connect via:[/bold red]\n"
                    f"  [cyan]kubectl exec -it <pod-name> -- bash[/cyan]\n"
                    f"  [cyan]ssh -p {port} root@localhost[/cyan]"
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting shell...[/yellow]")
            return


def _wait_for_pod_running(
    pod_name: str,
    namespace: str,
    timeout: int = 600,
) -> None:
    """Poll the backend until the pod containers are ready. Raises typer.Exit on failure/timeout."""
    start_time = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description="Waiting for pod to be ready...", total=None)
        while True:
            if time.time() - start_time > timeout:
                console.print(f"[bold red]❌ Timeout reached after {timeout} seconds.[/bold red]")
                raise typer.Exit(1)

            with _api() as client:
                data = client.list_pods(namespace=namespace)
            pods = data.get("pods", [])
            pod_info = next((p for p in pods if p["name"] == pod_name), None)
            status = pod_info["status"] if pod_info else "Pending"
            ready = pod_info.get("ready", False) if pod_info else False

            if status == "Running" and ready:
                break
            if status in ("Failed", "Succeeded"):
                console.print(f"[bold red]❌ Pod terminated with phase: {status}.[/bold red]")
                raise typer.Exit(1)

            # Show more granular status
            if status == "Running" and not ready:
                display_status = "ContainerCreating"
            else:
                display_status = status
            progress.update(task, description=f"Waiting for pod to be ready ({display_status})...")
            time.sleep(0.5)


@app.command("list")
def list_pods(
    namespace: str = typer.Option("default", help="Namespace to search"),
    all_namespaces: bool = typer.Option(False, "--all", "-A", help="Search all namespaces"),
):
    """
    List running interactive pods.
    """
    try:
        with _api() as client:
            data = client.list_pods(namespace=namespace, all_namespaces=all_namespaces)
    except Exception as e:
        console.print(f"[bold red]Error listing pods:[/bold red] {e}")
        raise typer.Exit(1)

    # Auto-clean SSH config entries for pods that have been deleted outside the CLI
    _prune_ssh_config(namespace=namespace, all_namespaces=all_namespaces)

    pods = data.get("pods", [])
    if not pods:
        console.print("[yellow]No interactive pods found.[/yellow]")
        return

    table = Table(title="Interactive Pods")
    table.add_column("Name", style="cyan")
    table.add_column("Namespace", style="dim")
    table.add_column("Status", style="green")
    table.add_column("GPU", style="yellow")
    table.add_column("Age")
    table.add_column("User", style="dim")

    from datetime import datetime, timezone

    for pod_info in pods:
        created = pod_info.get("created_at")
        if created:
            try:
                ts = datetime.fromisoformat(created)
                age_sec = int((datetime.now(timezone.utc) - ts).total_seconds())
            except Exception:
                age_sec = 0
        else:
            age_sec = 0
        if age_sec < 3600:
            age = f"{age_sec // 60}m"
        else:
            age = f"{age_sec // 3600}h{(age_sec % 3600) // 60}m"
        table.add_row(
            pod_info["name"],
            pod_info.get("namespace", "default"),
            pod_info.get("status", "Unknown"),
            pod_info.get("gpu", "0"),
            age,
            pod_info.get("user", "unknown"),
        )

    console.print(table)


@app.command("connect")
def connect_pod(
    pod_name: str = typer.Argument(None, help="Pod name to connect to (auto-detects if only one)"),
    namespace: str = typer.Option("default", help="Namespace of the pod"),
):
    """
    Connect to an existing interactive pod via SSH.

    Re-establishes port-forwarding and SSH config, then opens an SSH shell.
    """
    _check_prerequisites()

    try:
        with _api() as client:
            if not pod_name:
                data = client.list_pods(namespace=namespace)
                pods = [p for p in data.get("pods", []) if p["status"] == "Running"]
                if not pods:
                    console.print("[bold red]No running interactive pods found.[/bold red]")
                    console.print("Launch one with: [cyan]ml-plat pod launch[/cyan]")
                    raise typer.Exit(1)
                if len(pods) == 1:
                    pod_name = pods[0]["name"]
                    console.print(f"[dim]Auto-detected pod: {pod_name}[/dim]")
                else:
                    console.print("[yellow]Multiple pods found. Specify one:[/yellow]")
                    for p in pods:
                        console.print(f"  [cyan]{p['name']}[/cyan]")
                    raise typer.Exit(1)

            # Verify pod exists and is running via SSH info endpoint
            client.get_ssh_info(pod_name, namespace=namespace)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)

    # Copy SSH key (idempotent)
    _copy_ssh_key_to_pod(pod_name, namespace)

    # Reuse the port from an existing SSH config entry if available
    local_ssh_port = _get_ssh_port_for_pod(pod_name)

    # Check if a port-forward is already running for this pod
    pf_already_running = False
    if local_ssh_port is not None:
        try:
            ps_result = subprocess.run(
                ["pgrep", "-f", f"port-forward.*{pod_name}.*{local_ssh_port}"],
                capture_output=True,
                text=True,
            )
            if ps_result.returncode == 0:
                pf_already_running = True
                console.print("[dim]Port-forward already active, reusing.[/dim]")
        except Exception:
            pass

    if not pf_already_running:
        local_ssh_port = _find_free_port(local_ssh_port or 2222)
        _start_port_forward(pod_name, namespace, ssh_port=local_ssh_port)

    # Update SSH config
    _setup_ssh_config(local_ssh_port, pod_name)

    # Wait for the port-forward tunnel to be ready
    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]Waiting for SSH tunnel...[/dim]"),
        transient=True,
    ) as progress:
        progress.add_task("", total=None)
        ready = _wait_for_port(local_ssh_port, timeout=20)

    if not ready:
        console.print(
            f"[bold red]❌ SSH port {local_ssh_port} never became available. "
            "Check that the pod has sshd running.[/bold red]"
        )
        raise typer.Exit(1)

    console.print(f"\n[bold green]✅ Connected to {pod_name}[/bold green]")
    console.print(f"[dim]VS Code: Cmd+Shift+P → Remote-SSH: Connect to Host → {pod_name}[/dim]")

    # Open SSH shell
    _connect_ssh(local_ssh_port)


@app.command("stop")
def stop_pod(
    pod_name: str = typer.Argument(None, help="Pod name to stop (auto-detects if only one)"),
    namespace: str = typer.Option("default", help="Namespace of the pod"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """
    Stop and delete an interactive pod.

    Cleans up port-forwarding and SSH config automatically.
    """
    try:
        with _api() as client:
            if not pod_name:
                data = client.list_pods(namespace=namespace)
                running = [p for p in data.get("pods", []) if p["status"] == "Running"]
                if not running:
                    console.print("[yellow]No running interactive pods found.[/yellow]")
                    return
                if len(running) == 1:
                    pod_name = running[0]["name"]
                    console.print(f"[dim]Auto-detected pod: {pod_name}[/dim]")
                else:
                    console.print("[yellow]Multiple pods found. Specify one:[/yellow]")
                    for p in running:
                        console.print(f"  [cyan]{p['name']}[/cyan]")
                    raise typer.Exit(1)

            if not force:
                if not typer.confirm(f"Delete pod {pod_name}?"):
                    console.print("[dim]Cancelled.[/dim]")
                    return

            # Kill local port-forward for this pod
            subprocess.run(
                ["pkill", "-f", f"port-forward.*{pod_name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            client.delete_pod(pod_name, namespace=namespace)
            console.print(f"[bold green]✅ Pod {pod_name} deleted.[/bold green]")
    except typer.Exit:
        raise
    except Exception as e:
        if "404" in str(e):
            console.print(f"[yellow]Pod {pod_name} not found (already deleted?).[/yellow]")
        else:
            console.print(f"[bold red]Error deleting pod:[/bold red] {e}")
            raise typer.Exit(1)

    _remove_ssh_config(pod_name)


@app.command("launch")
def launch_pod(
    gpu: int = typer.Option(0, help="Number of GPUs to request"),
    shared: bool = typer.Option(
        False,
        "--shared-gpu",
        help=(
            "Use a time-sliced shared GPU node (1/4 GPU slice). "
            "Sets gpu=1 and targets the shared-gpu nodepool."
        ),
    ),
    gpu_type: GpuType = typer.Option(
        GpuType.any,
        "--gpu-type",
        help=(
            "GPU type to target: 't4' (g4dn, NVIDIA T4), "
            "'a10g' (g5, NVIDIA A10G), "
            "'a100' (p4d/p4de, NVIDIA A100), "
            "'any' (default, no preference)."
        ),
    ),
    image: str = typer.Option("ml-platform/base-cpu:latest", help="Docker image to use"),
    namespace: str = typer.Option("default", help="Namespace to deploy the pod"),
    pvc: str = typer.Option("efs-claim", help="Name of the EFS PVC to mount"),
    mount_path: str = typer.Option("/shared", help="Path to mount the EFS volume"),
    timeout: int = typer.Option(600, help="Timeout in seconds to wait for pod to be ready"),
):
    """
    Launch a RunPod-style interactive GPU pod with EFS storage and automatic access.

    Use --shared-gpu to land on the time-sliced shared GPU nodepool (1/4 GPU slice).
    Use --gpu N to request N GPUs on a dedicated GPU node.
    Use --gpu-type to target a specific GPU: t4 (g4dn), a10g (g5), or a100 (p4d/p4de).

    Examples:
      ml-plat pod launch --shared-gpu                        # 1/4 GPU slice
      ml-plat pod launch --gpu 1 --gpu-type t4             # NVIDIA T4 (g4dn)
      ml-plat pod launch --gpu 1 --gpu-type a10g           # full A10G
      ml-plat pod launch --gpu 8 --gpu-type a100           # 8x A100 (p4de)
    """
    _check_prerequisites()

    # Clean up SSH config entries for any pods deleted outside the CLI
    _prune_ssh_config(namespace)

    # Validate: --shared + --gpu-type a100 is not supported (no A100 shared nodepool)
    if shared and gpu_type == GpuType.a100:
        typer.secho(
            "Error: --shared-gpu is not supported with --gpu-type a100. "
            "No A100 time-sliced nodepool exists.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    if shared:
        gpu = 1
        console.print("[dim]Shared GPU mode: targeting time-sliced nodepool (1/4 GPU slice)[/dim]")
    elif gpu > 0 and gpu_type != GpuType.any:
        console.print(f"[dim]GPU type: {gpu_type.value}[/dim]")

    # Resolve full image path
    registry = os.getenv("ECR_REGISTRY", "805673386114.dkr.ecr.us-west-2.amazonaws.com")
    full_image = image
    if "/" not in image or image.startswith("ml-platform/"):
        full_image = f"{registry.rstrip('/')}/{image}"

    pod_name = f"interactive-pod-{int(time.time())}"
    pod_created = False

    try:
        console.print(f"[bold blue]🚀 Launching pod {pod_name}...[/bold blue]")

        with _api() as client:
            client.launch_pod(
                name=pod_name,
                image=full_image,
                gpu_type=gpu_type.value,
                gpu_count=gpu,
                namespace=namespace,
                shared=shared,
                pvc=pvc,
                mount_path=mount_path,
            )
        pod_created = True

        # Wait for pod to be running
        _wait_for_pod_running(pod_name, namespace, timeout=timeout)

        console.print("[bold green]✅ Pod is ready![/bold green]")

        # Copy SSH key and start port-forward
        _copy_ssh_key_to_pod(pod_name, namespace)

        local_ssh_port = _find_free_port(2222)
        local_jupyter_port = _find_free_port(8888)
        pf_process = _start_port_forward(
            pod_name,
            namespace,
            ssh_port=local_ssh_port,
            jupyter_port=local_jupyter_port,
        )

        # Automatically update SSH config for VS Code (per-pod entry)
        _setup_ssh_config(local_ssh_port, pod_name)

        console.print(
            f"\n[dim]VS Code: Cmd+Shift+P → Remote-SSH: Connect to Host → {pod_name}[/dim]"
        )

        # Wait for the port-forward tunnel to be ready before connecting
        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]Waiting for SSH tunnel...[/dim]"),
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            ready = _wait_for_port(local_ssh_port, timeout=30)

        if not ready:
            console.print(
                f"[bold red]❌ SSH port {local_ssh_port} never became available. "
                "Check that the pod image has openssh-server installed.[/bold red]"
            )
            raise typer.Exit(1)

        # Connect via SSH
        _connect_ssh(local_ssh_port)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
    finally:
        if pod_created:
            if typer.confirm("Would you like to delete the pod?"):
                try:
                    pf_process.terminate()
                except (NameError, UnboundLocalError):
                    pass  # Port-forward never started
                try:
                    with _api() as client:
                        client.delete_pod(pod_name, namespace=namespace)
                    console.print(f"Pod {pod_name} deleted.")
                except Exception:
                    console.print(f"[dim]Pod {pod_name} already gone.[/dim]")
                _remove_ssh_config(pod_name)
            else:
                console.print(
                    f"\n[bold green]Pod {pod_name} kept running.[/bold green]\n"
                    f"Reconnect: [cyan]ml-plat pod connect {pod_name}[/cyan]\n"
                    f"VS Code:   [dim]Cmd+Shift+P → Remote-SSH → {pod_name}[/dim]"
                )
