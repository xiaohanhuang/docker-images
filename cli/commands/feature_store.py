"""
feature_store.py — CLI commands for the Feast Feature Store.

Commands:
  mlp feature-store init           — one-command setup (generate + apply + materialize)
  mlp feature-store generate-data  — generate sample feature data
  mlp feature-store apply          — apply feature definitions to the registry
  mlp feature-store list           — list registered feature views
  mlp feature-store materialize    — push features to the online store (Redis)
  mlp feature-store get            — fetch feature values for entity keys
  mlp feature-store status         — check health of the feature store stack
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Feast Feature Store management")


def _resolve_repo(config: str | None = None) -> Path:
    """Resolve a Feast repo directory from a config file or directory path.

    Resolution order:
    1. If *config* is a file, use its parent directory.
    2. If *config* is a directory, use it directly.
    3. If *config* is None, use ``./feature_store.yaml`` in the cwd.
    """
    if config is None:
        repo = Path.cwd()
        if not (repo / "feature_store.yaml").exists():
            console.print(
                "[red]No feature_store.yaml in the current directory.[/red]\n"
                "Either cd into a Feast repo or pass the path explicitly."
            )
            raise typer.Exit(code=1)
        return repo

    p = Path(config)
    if p.is_file():
        return p.parent
    if p.is_dir():
        return p
    console.print(f"[red]{config} does not exist.[/red]")
    raise typer.Exit(code=1)


def _feast_store(config: str | None = None):
    """Return a Feast FeatureStore instance (lazy import)."""
    try:
        from feast import FeatureStore
    except ImportError:
        console.print(
            "[red]feast is not installed.[/red]\n" "Install it with: pip install 'feast[redis,aws]'"
        )
        raise typer.Exit(code=1)
    return FeatureStore(repo_path=str(_resolve_repo(config)))


# ── generate-data ─────────────────────────────────────────────────────


def _do_generate_data(repo: Path, *, users: int = 100, days: int = 90) -> Path:
    """Core data-generation logic (no typer dependency)."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        console.print(
            "[red]Missing required packages.[/red]\n" "Install them with: pip install pandas numpy"
        )
        raise typer.Exit(code=1)

    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "user_features.parquet"

    console.print(f"Generating data for [cyan]{users}[/cyan] users over [cyan]{days}[/cyan] days …")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    user_ids = list(range(1000, 1000 + users))

    rows = []
    for user_id in user_ids:
        n_events = np.random.randint(5, 11)
        for _ in range(n_events):
            ts = start_date + timedelta(
                seconds=int(np.random.randint(0, int((end_date - start_date).total_seconds())))
            )
            rows.append(
                {
                    "user_id": user_id,
                    "event_timestamp": ts,
                    "created_timestamp": datetime.now(),
                    "daily_transactions": int(np.random.randint(0, 20)),
                    "total_spend": round(float(np.random.uniform(10, 1000)), 2),
                    "avg_order_value": round(float(np.random.uniform(5, 100)), 2),
                    "is_premium": int(np.random.choice([0, 1], p=[0.8, 0.2])),
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(str(output_path), index=False)
    console.print(
        f"[green]Generated {len(df)} rows → {output_path}[/green]\n"
        f"  User IDs: {user_ids[0]}–{user_ids[-1]}"
    )
    return output_path


@app.command("generate-data")
def generate_data(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to feature_store.yaml or its parent directory (default: cwd).",
    ),
    users: int = typer.Option(100, "--users", "-u", help="Number of users to generate."),
    days: int = typer.Option(90, "--days", "-d", help="Days of history to generate."),
    upload: bool = typer.Option(
        False,
        "--upload",
        help="Upload to S3 (requires FEAST_S3_BUCKET env var).",
    ),
):
    """Generate sample feature data (Parquet) for the demo."""
    import os

    repo = _resolve_repo(config)
    output_path = _do_generate_data(repo, users=users, days=days)

    if upload:
        bucket = os.getenv("FEAST_S3_BUCKET")
        if not bucket:
            console.print("[red]Set FEAST_S3_BUCKET to upload to S3.[/red]")
            raise typer.Exit(code=1)
        import boto3

        s3 = boto3.client("s3")
        key = "feast/user_features.parquet"
        console.print(f"Uploading to [cyan]s3://{bucket}/{key}[/cyan] …")
        s3.upload_file(str(output_path), bucket, key)
        console.print("[green]Upload complete.[/green]")


# ── init ──────────────────────────────────────────────────────────────


@app.command("init")
def init_store(
    config: Optional[str] = typer.Argument(
        default=None,
        help="Path to feature_store.yaml or its parent directory (default: cwd).",
    ),
    users: int = typer.Option(100, "--users", "-u", help="Number of users to generate."),
    sync: bool = typer.Option(
        False,
        "--sync",
        help="Also sync features to the online store after applying.",
    ),
):
    """Set up a feature store: generate demo data and apply definitions.

    Use --sync to also push features to the online store for serving.
    """
    repo = _resolve_repo(config)
    repo_str = str(repo)
    total_steps = 3 if sync else 2

    # 1. Generate data
    console.print(f"\n[bold]Step 1/{total_steps}: Generating demo data …[/bold]")
    _do_generate_data(repo, users=users, days=90)

    # 2. Apply
    console.print(f"\n[bold]Step 2/{total_steps}: Applying feature definitions …[/bold]")
    apply_definitions(config=repo_str)

    # 3. Sync (optional)
    if sync:
        console.print(f"\n[bold]Step 3/{total_steps}: Syncing to online store …[/bold]")
        _do_sync(repo, days=90, incremental=False)

    console.print("\n[green bold]Feature store is ready![/green bold]")
    console.print("  List features:  mlp feature-store list")
    if not sync:
        console.print("  Sync to online:  mlp feature-store sync")


# ── apply ─────────────────────────────────────────────────────────────


@app.command("apply")
def apply_definitions(
    config: Optional[str] = typer.Argument(
        default=None,
        help="Path to a Feast repo directory containing feature_store.yaml.",
    ),
):
    """Apply feature definitions to the Feast registry (S3)."""
    if config is not None:
        config_path = Path(config).expanduser()
        if config_path.is_file() and config_path.name not in {
            "feature_store.yaml",
            "feature_store.yml",
        }:
            console.print(
                f"[red]Feast only reads 'feature_store.yaml' from the repo directory.[/red]\n"
                f"Rename or copy {config_path.name} to feature_store.yaml, then apply."
            )
            raise typer.Exit(code=1)

    repo = _resolve_repo(config)
    if not (repo / "feature_store.yaml").exists():
        console.print(f"[red]No feature_store.yaml found in {repo}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Applying feature definitions from [cyan]{repo}[/cyan] …")
    try:
        result = subprocess.run(
            ["feast", "apply"],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
    except OSError:
        console.print(
            "[red]feast is not installed or not on PATH.[/red]\n"
            "Install it with: pip install 'feast[redis,aws]'"
        )
        raise typer.Exit(code=1)
    if result.returncode != 0:
        console.print(f"[red]feast apply failed:[/red]\n{result.stderr}")
        raise typer.Exit(code=1)
    console.print("[green]Feature definitions applied successfully.[/green]")
    if result.stdout.strip():
        console.print(result.stdout.strip())


# ── list ──────────────────────────────────────────────────────────────


@app.command("list")
def list_features(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to feature_store.yaml or its parent directory (default: cwd).",
    ),
):
    """List registered feature views in the Feast registry."""
    store = _feast_store(config)
    try:
        views = store.list_feature_views()
        on_demand = store.list_on_demand_feature_views()
    except Exception as e:
        console.print(f"[red]Failed to read registry:[/red] {e}")
        raise typer.Exit(code=1)

    if not views and not on_demand:
        console.print(
            "[yellow]No feature views registered." " Run 'mlp feature-store apply' first.[/yellow]"
        )
        raise typer.Exit(code=0)

    table = Table(title="Feature Views", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Entities")
    table.add_column("Features")
    table.add_column("Online")
    table.add_column("TTL")
    table.add_column("Tags")

    for fv in views:
        table.add_row(
            fv.name,
            "batch",
            ", ".join(fv.entities),
            str(len(fv.schema)),
            "✓" if fv.online else "✗",
            str(fv.ttl) if fv.ttl else "—",
            ", ".join(f"{k}={v}" for k, v in (fv.tags or {}).items()) or "—",
        )

    for fv in on_demand:
        table.add_row(
            fv.name,
            "on-demand",
            "—",
            str(len(fv.schema)),
            "✓",
            "—",
            ", ".join(f"{k}={v}" for k, v in (fv.tags or {}).items()) or "—",
        )

    console.print(table)
    total = len(views) + len(on_demand)
    console.print(f"\n[green]{total} feature view(s) registered.[/green]")


# ── info ──────────────────────────────────────────────────────────────


@app.command("info")
def info_feature_view(
    name: str = typer.Argument(..., help="Feature view name (e.g. 'user_stats')."),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to feature_store.yaml or its parent directory (default: cwd).",
    ),
):
    """Show detailed info about a feature view (schema, source, entities)."""
    store = _feast_store(config)

    try:
        fv = store.get_feature_view(name)
    except Exception as e:
        console.print(f"[red]Feature view '{name}' not found:[/red] {e}")
        raise typer.Exit(code=1)

    # Header
    console.print(f"\n[bold cyan]{fv.name}[/bold cyan]")
    console.print(f"  Online:   {'✓' if fv.online else '✗'}")
    console.print(f"  TTL:      {fv.ttl or '—'}")
    console.print(f"  Entities: {', '.join(fv.entities) or '—'}")
    if fv.tags:
        console.print(f"  Tags:     {', '.join(f'{k}={v}' for k, v in fv.tags.items())}")

    # Source
    source = fv.batch_source
    if source:
        console.print("\n[bold]Source[/bold]")
        source_type = type(source).__name__
        console.print(f"  Type:      {source_type}")
        if hasattr(source, "path"):
            console.print(f"  Path:      {source.path}")
        if hasattr(source, "timestamp_field"):
            console.print(f"  Timestamp: {source.timestamp_field}")

    # Schema
    entity_names = set(fv.entities)
    features = [f for f in fv.schema if f.name not in entity_names]

    table = Table(title="Schema", box=box.ROUNDED)
    table.add_column("Field", style="cyan")
    table.add_column("Type", style="green")

    for f in features:
        table.add_row(f.name, str(f.dtype))

    console.print()
    console.print(table)


# ── sync (aka materialize) ────────────────────────────────────────────


def _do_sync(repo: Path, *, days: int = 7, incremental: bool = True):
    """Core sync logic — push features from offline to online store."""
    store = _feast_store(str(repo))

    end_date = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=days)

    console.print(
        f"Syncing features to online store " f"({'incremental' if incremental else 'full'}) …"
    )
    console.print(
        f"  Window: [cyan]{start_date:%Y-%m-%d}[/cyan]" f" → [cyan]{end_date:%Y-%m-%d}[/cyan]"
    )

    try:
        if incremental:
            store.materialize_incremental(end_date=end_date)
        else:
            store.materialize(start_date=start_date, end_date=end_date)
    except Exception as e:
        console.print(f"[red]Sync failed:[/red] {e}")
        raise typer.Exit(code=1)

    console.print("[green]Sync complete.[/green]")


def _sync_command(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to feature_store.yaml or its parent directory (default: cwd).",
    ),
    days: int = typer.Option(
        7,
        "--days",
        "-d",
        help="Number of days to sync (backward from now).",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Incremental (since last run) or full sync.",
    ),
):
    """Push offline features to the online store for serving."""
    _do_sync(_resolve_repo(config), days=days, incremental=incremental)


# Register under both names
app.command("sync")(_sync_command)
app.command("materialize", hidden=True)(_sync_command)


# ── get ───────────────────────────────────────────────────────────────


@app.command("get")
def get_features(
    feature_view: str = typer.Argument(..., help="Feature view name (e.g. 'user_stats')."),
    entity_key: str = typer.Argument(..., help="Entity key value (e.g. '1001')."),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to feature_store.yaml or its parent directory (default: cwd).",
    ),
):
    """Fetch online features for a single entity key."""
    store = _feast_store(config)

    # Look up the feature view to get its feature names
    try:
        fv = store.get_feature_view(feature_view)
    except Exception as e:
        console.print(f"[red]Feature view '{feature_view}' not found:[/red] {e}")
        raise typer.Exit(code=1)

    # Determine the entity join key
    entity_name = fv.entities[0] if fv.entities else "entity_id"

    # Build feature refs, excluding entity columns
    feature_refs = [f"{feature_view}:{f.name}" for f in fv.schema if f.name != entity_name]

    try:
        result = store.get_online_features(
            features=feature_refs,
            entity_rows=[{entity_name: int(entity_key)}],
        ).to_dict()
    except ValueError:
        # entity key might be a string, not int
        result = store.get_online_features(
            features=feature_refs,
            entity_rows=[{entity_name: entity_key}],
        ).to_dict()
    except Exception as e:
        console.print(
            f"[red]Failed to fetch features:[/red] {e}\n"
            "[yellow]Have you run 'mlp feature-store materialize' first?[/yellow]"
        )
        raise typer.Exit(code=1)

    table = Table(title=f"Online Features — {feature_view} (entity={entity_key})", box=box.ROUNDED)
    table.add_column("Feature", style="cyan")
    table.add_column("Value", style="green")

    for key, values in result.items():
        if key == entity_name:
            continue
        val = values[0] if values else None
        table.add_row(key, str(val) if val is not None else "[dim]null[/dim]")

    console.print(table)


# ── status ────────────────────────────────────────────────────────────


@app.command("status")
def status(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to feature_store.yaml or its parent directory (default: cwd).",
    ),
    namespace: str = typer.Option(
        "ml-platform",
        "--namespace",
        "-n",
        help="Kubernetes namespace.",
    ),
):
    """Check health of the feature store stack (Redis, registry, server)."""
    checks: list[tuple[str, str, str]] = []  # (component, status, detail)

    # 1. Registry check
    try:
        store = _feast_store(config)
        views = store.list_feature_views()
        checks.append(("Registry (S3)", "[green]Healthy[/green]", f"{len(views)} feature views"))
    except Exception as e:
        checks.append(("Registry (S3)", "[red]Unreachable[/red]", str(e)[:80]))

    # 2. Redis check (via kubectl)
    redis_ok = _check_pod_ready(namespace, "app.kubernetes.io/name=redis")
    checks.append(
        (
            "Redis (online store)",
            "[green]Healthy[/green]" if redis_ok else "[red]Down[/red]",
            "Pod ready" if redis_ok else "Pod not ready",
        )
    )

    # 3. Feast server check
    server_ok = _check_pod_ready(namespace, "app=feast-server")
    checks.append(
        (
            "Feature Server",
            "[green]Healthy[/green]" if server_ok else "[red]Down[/red]",
            "Pod ready" if server_ok else "Pod not ready",
        )
    )

    table = Table(title="Feature Store Status", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")

    for name, st, detail in checks:
        table.add_row(name, st, detail)

    console.print(table)


def _check_pod_ready(namespace: str, label_selector: str) -> bool:
    """Check if at least one pod matching the label selector is Ready."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                label_selector,
                "-o",
                "jsonpath={.items[*].status.conditions[?(@.type=='Ready')].status}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "True" in result.stdout
    except Exception:
        return False
