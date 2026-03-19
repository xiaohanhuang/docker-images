# Sub-Project: Flyte

Deploys the Flyte orchestration engine with Ray and Spark plugin support.

## Deploy

```bash
cd projects/flyte
make install
```

## Key Configuration

- **Storage**: S3 (`ml-platform-flyte-data`)
- **Database**: Embedded Postgres (switch to RDS for production)
- **Plugins**: Ray and Spark enabled

## Dependencies

- `projects/eks` (cluster must be running)
