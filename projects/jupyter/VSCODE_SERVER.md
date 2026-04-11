# VS Code Server Integration

VS Code Server (code-server) is integrated into the JupyterLab environment via
[jupyter-server-proxy](https://github.com/jupyterhub/jupyter-server-proxy). It
appears as a launcher icon alongside Jupyter notebooks, terminals, and Marimo —
no separate profile or CLI flag is needed.

## How It Works

1. **Open JupyterHub** — `ml-plat notebook open` or port-forward manually.
2. **Spawn any profile** (CPU or GPU).
3. **Click the VS Code icon** in the JupyterLab Launcher.

code-server starts on demand inside the same pod and is proxied through
JupyterHub's authentication layer.

## Architecture

### Image Hierarchy

```
quay.io/jupyter/pytorch-notebook:python-3.12           (base)
  └─ notebook-marimo:<sha>                              (adds Marimo + SDK)
       └─ desk-gpu:<sha>                  (adds code-server v4.111.0 + vscode_proxy)
```

Images are tagged with the Git commit SHA of the `docker-images` repo and pinned
in `helm-values.yaml` for reproducible deploys (never `:latest`).

`Dockerfile.marimo.vscode` installs code-server and a small `vscode_proxy`
package that registers the "VS Code" entry point for jupyter-server-proxy.

### vscode_proxy Package

Located at `images/vscode_proxy/`. It follows the same pattern as `marimo_proxy`:

- `pyproject.toml` — declares the `jupyter_serverproxy_servers` entry point.
- `vscode_proxy/__init__.py` — `setup_vscode()` returns the command, timeout,
  and launcher entry (title + icon).
- `vscode-icon.svg` — VS Code icon shown in the launcher.

## Building Images

```bash
cd projects/jupyter

# Build the combined image locally
make build-vscode

# Production builds happen via CI in the docker-images repo
```

The ECR image is `ml-platform/desk-gpu`.

## GPU Idle Monitoring

The idle monitor checks all activity sources (Jupyter kernels/terminals, VS Code
TCP connections, and SSH) regardless of which IDE is in use. Any active source
keeps the pod alive.

- VS Code detection: established TCP connections on the code-server port
  (configurable via `VSCODE_PORT`, default `8888`).
- Same 30-minute idle threshold as Jupyter sessions.

## Troubleshooting

### VS Code icon not visible in Launcher

Verify the proxy package is installed:

```bash
kubectl exec -it -n jupyter jupyter-<user> -- pip list | grep vscode
# Should show: vscode-proxy 0.1.0
```

Check that jupyter-server-proxy is enabled:

```bash
kubectl exec -it -n jupyter jupyter-<user> -- jupyter server extension list 2>&1 | grep server_proxy
```

### code-server fails to start

Check the notebook container logs:

```bash
kubectl logs -n jupyter jupyter-<user>
```

Verify code-server is installed:

```bash
kubectl exec -it -n jupyter jupyter-<user> -- code-server --version
```

## Security

- code-server runs with `--auth=none`; JupyterHub handles authentication.
- Each user gets an isolated pod — no cross-user access.
- All traffic is proxied through the `proxy-public` service.

## References

- [code-server](https://github.com/coder/code-server)
- [jupyter-server-proxy](https://github.com/jupyterhub/jupyter-server-proxy)
