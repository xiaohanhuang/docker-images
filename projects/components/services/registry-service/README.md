# Registry Service

Stateless FastAPI service that provides recipe registry CRUD operations.

## Features

- List recipes with filtering (tags, verification status)
- Push recipe archives to S3 + register in PostgreSQL
- Pull recipe archives from S3
- Get/set verification status per recipe version + profile
- Delete recipe versions

## Architecture

- **Stateless**: All state in PostgreSQL + S3
- **Independent**: No dependencies on execution service
- **Scalable**: Can run multiple replicas (PostgreSQL handles concurrency)

## Endpoints

- `GET /` - Health check
- `GET /health` - Kubernetes readiness probe
- `GET /registry/list` - List recipes (with optional filters)
- `POST /registry/push` - Upload and register a recipe
- `GET /registry/pull/{recipe_name}` - Download a recipe archive
- `GET /registry/status/{recipe_name}/{version}` - Get verification status
- `POST /registry/status` - Update verification status
- `DELETE /registry/{recipe_name}/{version}` - Delete a recipe

## Configuration

Environment variables:
- `POSTGRES_HOST` - PostgreSQL host (default: `postgres.postgres.svc.cluster.local`)
- `POSTGRES_PORT` - PostgreSQL port (default: `5432`)
- `POSTGRES_DB` - Database name (default: `flyte`)
- `POSTGRES_USER` - Database user (default: `flyte`)
- `POSTGRES_PASSWORD` - Database password
- `S3_BUCKET` - S3 bucket for recipe archives
- `PORT` - HTTP server port (default: `8081`)

## Deployment

```bash
# Build image
docker build -t registry-service:latest .

# Run locally (requires PostgreSQL + S3 access)
docker run -p 8081:8081 \
  -e POSTGRES_HOST=localhost \
  -e POSTGRES_PASSWORD=mypassword \
  -e S3_BUCKET=my-bucket \
  registry-service:latest

# Deploy to Kubernetes
kubectl apply -k k8s/
```

## Database Schema

The service automatically creates the `recipe_registry` table on first connection:

```sql
CREATE TABLE recipe_registry (
    recipe_name          TEXT        NOT NULL,
    version              TEXT        NOT NULL,
    s3_key               TEXT        NOT NULL,
    archive_name         TEXT        NOT NULL,
    verification_status  TEXT        NOT NULL DEFAULT 'experimental',
    verified_profile     TEXT,
    verified_at          TIMESTAMPTZ,
    tags                 TEXT[]      DEFAULT '{}',
    pushed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata             JSONB       DEFAULT '{}',
    profile_statuses     JSONB       DEFAULT '{}',
    PRIMARY KEY (recipe_name, version)
);
```
