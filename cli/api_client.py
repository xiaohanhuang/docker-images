"""API client for ml-platform backend."""

import os
from typing import Any

import httpx


def use_backend_api() -> bool:
    """Return True if ML_PLATFORM_API_URL is explicitly set."""
    return "ML_PLATFORM_API_URL" in os.environ


class APIClient:
    """Client for interacting with ml-platform backend API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        """Initialize API client.

        Args:
            base_url: Base URL of the backend API
            api_key: API key for authentication
        """
        self.base_url = base_url or os.getenv(
            "ML_PLATFORM_API_URL", "http://ml-platform-api.ml-platform.internal:8000"
        )
        self.api_key = api_key or os.getenv("ML_PLATFORM_API_KEY", "")
        self.user = os.getenv("USER", "unknown")

        headers = {"X-User": self.user}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.client.close()

    # Pod operations

    def launch_pod(
        self,
        name: str,
        image: str,
        gpu_type: str = "any",
        gpu_count: int = 1,
        namespace: str = "default",
        env_vars: dict[str, str] | None = None,
        cpu: str = "4",
        memory: str = "16Gi",
        shared: bool = False,
        pvc: str = "efs-claim",
        mount_path: str = "/shared",
    ) -> dict[str, Any]:
        """Launch an interactive pod."""
        response = self.client.post(
            "/pods",
            json={
                "name": name,
                "image": image,
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "namespace": namespace,
                "env_vars": env_vars or {},
                "cpu": cpu,
                "memory": memory,
                "shared": shared,
                "pvc": pvc,
                "mount_path": mount_path,
            },
        )
        response.raise_for_status()
        return response.json()

    def list_pods(self, namespace: str = "default", all_namespaces: bool = False) -> dict:
        """List pods."""
        params = {"namespace": namespace, "all_namespaces": all_namespaces}
        response = self.client.get("/pods", params=params)
        response.raise_for_status()
        return response.json()

    def delete_pod(self, pod_name: str, namespace: str = "default") -> dict:
        """Delete a pod."""
        params = {"namespace": namespace}
        response = self.client.delete(f"/pods/{pod_name}", params=params)
        response.raise_for_status()
        return response.json()

    def get_ssh_info(self, pod_name: str, namespace: str = "default") -> dict:
        """Get SSH connection info for a pod."""
        response = self.client.post(
            f"/pods/{pod_name}/ssh",
            json={"namespace": namespace},
        )
        response.raise_for_status()
        return response.json()

    # Job operations

    def submit_job(self, workflow_name: str, version: str, inputs: dict) -> dict:
        """Submit a training job."""
        response = self.client.post(
            "/jobs",
            json={
                "workflow_name": workflow_name,
                "version": version,
                "inputs": inputs,
            },
        )
        response.raise_for_status()
        return response.json()

    def list_jobs(
        self, limit: int = 50, project: str | None = None, domain: str | None = None
    ) -> dict:
        """List recent jobs."""
        params = {"limit": limit}
        if project:
            params["project"] = project
        if domain:
            params["domain"] = domain
        response = self.client.get("/jobs", params=params)
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> dict:
        """Get job status."""
        response = self.client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def get_job_logs(self, job_id: str) -> str:
        """Get job logs (returns text)."""
        response = self.client.get(f"/jobs/{job_id}/logs")
        response.raise_for_status()
        return response.text

    # Notebook operations

    def launch_notebook(self, namespace: str = "jupyter", port: int = 8080) -> dict:
        """Get notebook launch information."""
        response = self.client.post(
            "/notebooks",
            json={"namespace": namespace, "port": port},
        )
        response.raise_for_status()
        return response.json()

    def list_notebooks(self, namespace: str = "jupyter") -> dict:
        """List running notebooks."""
        params = {"namespace": namespace}
        response = self.client.get("/notebooks", params=params)
        response.raise_for_status()
        return response.json()

    def stop_notebook(self, username: str, namespace: str = "jupyter") -> dict:
        """Stop a notebook server."""
        params = {"namespace": namespace}
        response = self.client.delete(f"/notebooks/{username}", params=params)
        response.raise_for_status()
        return response.json()

    # Cost operations

    def get_cost_report(
        self, days: int = 7, project: str | None = None, domain: str | None = None
    ) -> dict:
        """Get cost report."""
        params = {"days": days}
        if project:
            params["project"] = project
        if domain:
            params["domain"] = domain
        response = self.client.get("/cost/report", params=params)
        response.raise_for_status()
        return response.json()
