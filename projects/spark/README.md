# Sub-Project: Spark

Deploys the Spark-on-Kubernetes Operator for ETL/data preprocessing workloads.

## Deploy

```bash
cd projects/spark
make install
```

## Build Image

```bash
make build-image
```

## Key Components

| Component | Description |
|-----------|-------------|
| Spark Operator | Manages SparkApplication CRDs |
| `base-spark` image | PySpark + Hadoop S3 connector |
| SDK `@spark_task` | Decorator for ETL pipelines |

## Dependencies

- `projects/eks` (cluster must be running)
