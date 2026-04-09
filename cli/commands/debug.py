import os
import subprocess
import time

import typer
from kubernetes import client, config
from rich.console import Console

console = Console()
app = typer.Typer(help="Debug tools")


@app.command("start")
def start_debug_session(
    gpu: int = typer.Option(0, help="Number of GPUs to request"),
    image: str = typer.Option("ml-platform/base-gpu:latest", help="Docker image to use"),
    namespace: str = typer.Option("default", help="Namespace to deploy debug pod"),
    pvc: str = typer.Option("efs-claim", help="Name of the EFS PVC to mount"),
    mount_path: str = typer.Option("/workspace", help="Path to mount the EFS volume"),
):
    """
    Launch an interactive debug pod with SSH enabled.
    """
    try:
        config.load_kube_config()
    except Exception:
        console.print(
            "[bold red]Error: Could not load kube config. "
            "Make sure you are connected to the cluster.[/bold red]"
        )
        raise typer.Exit(1)

    v1 = client.CoreV1Api()

    pod_name = f"debug-session-{int(time.time())}"

    # Resolve full image path using registry from environment
    registry = os.getenv("ECR_REGISTRY", "805673386114.dkr.ecr.us-west-2.amazonaws.com")
    full_image = image
    if "/" not in image or image.startswith("ml-platform/"):
        full_image = f"{registry.rstrip('/')}/{image}"

    tolerations = []
    if gpu > 0:
        tolerations.append(
            {
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        )

    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": namespace,
            "labels": {
                "app": "debug-session",
                "user": os.getenv("USER", "unknown"),
            },
        },
        "spec": {
            "containers": [
                {
                    "name": "debug-container",
                    "image": full_image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/usr/sbin/sshd", "-D"],  # Start SSH server
                    "resources": {"limits": {"nvidia.com/gpu": str(gpu)} if gpu > 0 else {}},
                    "volumeMounts": [{"name": "efs-storage", "mountPath": mount_path}],
                }
            ],
            "volumes": [{"name": "efs-storage", "persistentVolumeClaim": {"claimName": pvc}}],
            "tolerations": tolerations,
            "restartPolicy": "Never",
        },
    }

    console.print(f"[bold blue]🔧 Creating debug pod {pod_name}...[/bold blue]")
    pod_created = False

    try:
        v1.create_namespaced_pod(namespace=namespace, body=pod_manifest)
        pod_created = True

        # Wait for pod to be ready
        console.print("Waiting for pod to be Ready...")
        while True:
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            if pod.status.phase == "Running":
                break
            if pod.status.phase in ["Failed", "Succeeded"]:
                console.print("[bold red]Pod failed to start[/bold red]")
                raise typer.Exit(1)
            time.sleep(2)

        console.print("[bold green]✅ Debug Pod is Ready![/bold green]")

        # Port Forward
        local_port = 2222
        console.print(f"Setting up port forward on localhost:{local_port}...")

        pf_command = [
            "kubectl",
            "port-forward",
            pod_name,
            f"{local_port}:22",
            "-n",
            namespace,
        ]

        try:
            console.print("\n[bold cyan]To connect with VS Code Remote:[/bold cyan]")
            console.print(f"  ssh -p {local_port} root@localhost")
            console.print("  (Password is 'root')")
            console.print("\nPress Ctrl+C to stop the session.")

            subprocess.run(pf_command)

        except KeyboardInterrupt:
            console.print("\nStopping port forward...")

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
    finally:
        if pod_created:
            if typer.confirm("Delete debug pod?"):
                v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
                console.print(f"Pod {pod_name} deleted.")
            else:
                console.print(
                    f"Pod {pod_name} kept running. Reconnect with: "
                    f"kubectl exec -it {pod_name} -n {namespace} -- /bin/bash"
                )
