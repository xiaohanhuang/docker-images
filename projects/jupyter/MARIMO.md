# Marimo Notebook Integration

This document describes the Marimo reactive notebook integration in the ML Platform.

## Overview

Marimo is a next-generation reactive Python notebook where:
- **Cells re-execute automatically** when their dependencies change
- **Outputs are always consistent** with the code (no hidden state issues)
- **Notebooks are stored as pure Python scripts** (`.py` files, Git-friendly)
- **Built-in UI widgets** for interactive exploration without Gradio
- **Apps mode** for exporting notebooks as interactive apps

## Architecture

Marimo runs as a JupyterHub named server profile with three resource tiers:

1. **Marimo - CPU**: 4 CPU, 8GB RAM, no GPU
2. **Marimo - GPU Shared A10G**: 1/4 time-sliced A10G, 4GB RAM
3. **Marimo - GPU Single A10G**: Dedicated A10G, 14GB RAM

All Marimo profiles:
- Use the custom `notebook-marimo` Docker image (based on PyTorch with CUDA)
- Mount the shared EFS volume at `/shared` for persistent storage
- Run `marimo edit --host 0.0.0.0 --port 8888 --no-token` as the entrypoint
- GPU profiles include the idle monitor sidecar with `MARIMO_MODE=true`

## Usage

### Opening Marimo via CLI

```bash
# Open JupyterHub with Marimo profiles available
ml-plat notebook open --ide marimo

# Or use the default (JupyterHub)
ml-plat notebook open
```

When you access JupyterHub:
1. Log in with your credentials
2. Select a **Marimo** profile from the profile list
3. Click "Start" to spawn your Marimo server

### Creating Notebooks

Marimo notebooks are pure Python scripts:

```python
import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell
def __():
    import torch
    return torch,


@app.cell
def __(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device,


if __name__ == "__main__":
    app.run()
```

Save notebooks to `/shared` (EFS) for persistence:
```bash
/shared/my_experiment.py
```

### Reactive Execution Example

```python
# Cell 1: Define a hyperparameter
learning_rate = 0.001

# Cell 2: Use it in training config
config = {"lr": learning_rate, "batch_size": 32}

# When you change learning_rate in Cell 1, Cell 2 automatically re-runs!
```

## GPU Idle Monitor Integration

The GPU idle monitor detects Marimo activity by:
1. Checking `/api/status` for active sessions (when `MARIMO_MODE=true`)
2. Monitoring SSH connections via `/proc/net/tcp`
3. Deleting the pod after 30 minutes (1800s) of inactivity

Environment variables for Marimo GPU profiles:
```yaml
- name: MARIMO_MODE
  value: "true"
- name: IDLE_THRESHOLD_SECONDS
  value: "1800"
- name: CHECK_INTERVAL_SECONDS
  value: "60"
```

## Docker Image

**Build:**
```bash
cd projects/jupyter
make build-marimo-image
```

**Image:** `805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/notebook-marimo:<sha>`

Tags are the Git commit SHA from the `docker-images` repo; pinned in `helm-values.yaml`.

**Base:** `quay.io/jupyter/pytorch-notebook:python-3.12`

**Additional packages:**
- `marimo` - Reactive notebook framework
- `flytekit==1.13.0` - Workflow orchestration SDK
- `ray[default]==2.54.0` - Distributed computing
- `mlflow==2.14.0` - Experiment tracking
- All standard ML libraries (transformers, torch, accelerate, etc.)

## Deployment

Marimo profiles are included in the JupyterHub Helm chart:

```bash
cd projects/jupyter
make upgrade  # Upgrade JupyterHub with new Marimo profiles
```

This adds the profiles to the user selection screen without disrupting existing Jupyter sessions.

## Differences from Jupyter

| Feature | Jupyter | Marimo |
|---------|---------|--------|
| **Cell execution** | Manual, order-dependent | Automatic, reactive |
| **Hidden state** | Possible (out-of-order execution) | Impossible (enforced DAG) |
| **File format** | `.ipynb` (JSON blob) | `.py` (pure Python) |
| **Git-friendly** | No (binary JSON) | Yes (text diffs work) |
| **Variable scope** | Notebook global | Cell-local with explicit returns |
| **UI widgets** | Requires ipywidgets | Built-in (`mo.ui.*`) |
| **Apps mode** | Requires Voilà or similar | Built-in (`marimo run`) |

## Example Workflow

1. **Start a Marimo session:**
   ```bash
   ml-plat notebook open --ide marimo
   # Select "Marimo - GPU Single A10G" profile
   ```

2. **Create a training notebook:**
   - New notebook in Marimo UI → `/shared/train_model.py`
   - Define cells for: data loading, model definition, training loop, evaluation
   - Cells auto-update when dependencies change

3. **Convert to Flyte workflow:**
   - Export functions from Marimo cells
   - Wrap in `@task` decorators
   - Submit via `ml-plat workflow submit`

4. **Share with team:**
   - Commit `/shared/train_model.py` to Git
   - Pure Python file → clean diffs, easy review
   - Others can `marimo edit train_model.py` to reproduce

## References

- [Marimo Documentation](https://docs.marimo.io/)
- [Marimo GitHub](https://github.com/marimo-team/marimo)
- [JupyterHub Named Servers](https://jupyterhub.readthedocs.io/en/stable/reference/api/named-servers.html)
- [Platform Notebook Command](../../cli/commands/notebook.py)
- [GPU Idle Monitor](gpu-idle-monitor/idle_monitor.py)
