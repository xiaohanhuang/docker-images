# jupyter-server-proxy configuration for VS Code (code-server)
# This registers VS Code as a launchable app in the JupyterLab launcher.
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def setup_vscode():
    return {
        "command": [
            "code-server",
            "--auth=none",
            "--disable-telemetry",
            "--disable-update-check",
            "--bind-addr=0.0.0.0:{port}",
        ],
        "timeout": 90,
        "new_browser_tab": True,
        "launcher_entry": {
            "title": "VS Code",
            "icon_path": os.path.join(HERE, "..", "vscode-icon.svg"),
        },
    }
