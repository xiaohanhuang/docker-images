# Sub-Project: Jupyter

Deploys JupyterHub on EKS, providing managed notebook servers with Jupyter, Marimo, and VS Code Server for data scientists and ML engineers.

## Deploy

```bash
cd projects/jupyter
make install
```

## Access

```bash
kubectl port-forward svc/proxy-public 8080:80 -n jupyter
# → http://localhost:8080
```

Or use the CLI:

```bash
ml-plat notebook open
# Opens JupyterHub in your browser
```

## VS Code Server

VS Code Server is available as a launcher icon inside JupyterLab (alongside
notebooks, terminals, and Marimo). Click the **VS Code** icon in the Launcher
after spawning any profile. See [VSCODE_SERVER.md](VSCODE_SERVER.md) for details.

## Available Profiles

| Profile | Resources | Use Case |
|---------|-----------|----------|
| CPU - Standard | 4 CPU, 8GB | Data exploration, quick experiments |
| GPU - Shared A10G | 1/4 GPU (time-sliced), 4GB RAM | Light GPU workloads, prototyping |
| GPU - Single A10G | 1 GPU, 24GB VRAM, 16GB RAM | Model prototyping and training |
| GPU - Single A10G High-Mem | 1 GPU, 24GB VRAM, 128GB RAM | Heavy data loading with GPU |
| GPU - 4x NVIDIA A10G | 4 GPUs, 96GB VRAM, 192GB RAM | Distributed training |

All profiles use the `desk-gpu` image which includes JupyterLab,
Marimo, and VS Code Server.

## Key Features

- **Platform SDK pre-installed** in all notebook environments
- **Submit jobs directly** from notebooks using `FlyteRemote`
- **Interactive GPU access** for prototyping before scaling
- **Persistent storage** (50GB per user on gp3 EBS + shared EFS)
- **VS Code Server** via JupyterLab Launcher icon
- **GPU idle monitoring** automatically shuts down inactive GPU servers

## Building Custom Images

```bash
# Jupyter + Marimo base images
make build-images

# With VS Code Server
make build-vscode
```

## Dependencies

- `projects/eks` (cluster must be running)
- `projects/components` (base images and SDK)
