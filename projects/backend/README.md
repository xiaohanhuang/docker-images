# ML Platform Backend API

This directory contains the FastAPI backend for the ml-platform CLI.

## Overview

The backend API provides centralized pod/job/notebook management with:
- Audit logging of all operations
- RBAC and quota enforcement (future)
- Removes need for users to have direct cluster access

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pods` | POST | Launch an interactive pod |
| `/pods` | GET | List all pods |
| `/pods/{name}` | DELETE | Delete a pod |
| `/pods/{name}/ssh` | POST | Get SSH connection info |
| `/jobs` | POST | Submit a training job |
| `/jobs` | GET | List recent jobs |
| `/jobs/{id}` | GET | Get job status |
| `/jobs/{id}/logs` | GET | Stream job logs |
| `/notebooks` | POST | Launch notebook |
| `/notebooks` | GET | List notebooks |
| `/notebooks/{username}` | DELETE | Stop notebook |
| `/cost/report` | GET | Get cost report |

## Deployment

```bash
# Build and push image
make push

# Deploy to cluster
make deploy

# Or both
make install
```

## Configuration

The backend is configured via environment variables:

- `FLYTE_ENDPOINT`: Flyte admin endpoint (default: `flyteadmin.ml-platform.internal:80`)
- `FLYTE_PROJECT`: Default Flyte project (default: `flytesnacks`)
- `FLYTE_DOMAIN`: Default Flyte domain (default: `development`)
- `AUDIT_LOG_FILE`: Path to audit log file (default: `/var/log/ml-platform/audit.log`)

## Development

Run locally:
```bash
cd ../..
pip install -e ".[backend]"
python -m uvicorn backend.main:app --reload
```

Access API docs at http://localhost:8000/docs
