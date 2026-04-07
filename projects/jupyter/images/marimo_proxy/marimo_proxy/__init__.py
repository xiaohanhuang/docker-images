# jupyter-server-proxy configuration for Marimo
# This registers Marimo as a launchable app in the JupyterLab launcher.


def setup_marimo():
    return {
        "command": ["marimo", "edit", "--no-token", "--host=0.0.0.0", "--port={port}"],
        "timeout": 30,
        "launcher_entry": {
            "title": "Marimo",
            "icon_path": "/opt/marimo-proxy/marimo-icon.svg",
        },
    }
