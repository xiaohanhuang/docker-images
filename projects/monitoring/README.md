# Sub-Project: Monitoring

Full observability stack: Prometheus, Grafana, MLflow, and DCGM GPU Exporter.

## Deploy

```bash
cd projects/monitoring

# 1. Initialize MLflow database (first-time setup only)
make init-mlflow-db

# 2. Install all monitoring components
make install-all
```

**Prerequisites**:
- PostgreSQL must be running (from `projects/postgres`)
- MLflow database credentials Secret must exist (see [MLflow PostgreSQL Migration Guide](../../docs/mlflow-postgres-migration.md))

## Components

| Component | Purpose | Backend |
|-----------|---------|---------|
| kube-prometheus-stack | Cluster metrics + alerting | Prometheus TSDB |
| Grafana | Dashboards for GPU, CPU, memory | Embedded SQLite (default) |
| MLflow | Experiment tracking + model registry | PostgreSQL |
| DCGM Exporter | GPU utilization metrics to Prometheus | N/A |

**Note**: MLflow uses the shared PostgreSQL instance (same as Flyte) for concurrent write support and better reliability. See the [MLflow PostgreSQL Migration Guide](../../docs/mlflow-postgres-migration.md) for details.

## Access

```bash
# Grafana
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring
# MLflow
kubectl port-forward svc/mlflow 5000:80 -n monitoring
```

## Dependencies

- `projects/postgres` (PostgreSQL must be running for MLflow)
- `projects/eks` (cluster must be running)

## MLflow Database Setup

MLflow now uses PostgreSQL instead of SQLite for the backend store. This enables:
- Concurrent writes from multiple training jobs
- Horizontal scaling with multiple MLflow replicas
- Better reliability and no corruption risk

For first-time setup or troubleshooting, see the complete [MLflow PostgreSQL Migration Guide](../../docs/mlflow-postgres-migration.md).
