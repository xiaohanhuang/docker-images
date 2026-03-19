# ML Platform Services

This directory contains the backend services for the ML Platform.

## Services

### execution-service

Stateless service for remote GPU/CPU function execution.

- **Port:** 8080
- **Purpose:** Execute serialized Python functions on Kubernetes
- **Features:**
  - Remote function execution via `@remote` decorator
  - Pod pool for warm container reuse
  - TTL-based eviction for idle pods
  - K8s Jobs and Pods management
  - Log streaming

**Quick Start:**
```bash
cd execution-service
make deploy
kubectl port-forward svc/execution-service 8765:8080
```

### registry-service

Stateless service for recipe registry CRUD operations.

- **Port:** 8081
- **Purpose:** Manage recipe lifecycle (list/push/pull/status)
- **Features:**
  - Recipe metadata storage (PostgreSQL)
  - Recipe archive storage (S3)
  - Verification status management
  - Tag-based filtering

**Quick Start:**
```bash
cd registry-service
make deploy
kubectl port-forward svc/registry-service 8766:8081
```

## Migration from remote-agent

The monolithic `remote-agent` service has been removed in favor of the split architecture above. See [migration guide](../../docs/migration-service-split.md) for historical context.

## Architecture

```
┌──────────────────────┐   ┌──────────────────────┐
│  execution-service   │   │  registry-service    │
│  :8080               │   │  :8081               │
│                      │   │                      │
│  ┌────────────────┐  │   │  ┌────────────────┐  │
│  │ Remote Exec    │  │   │  │ Recipe CRUD    │  │
│  │ Pod Pool       │  │   │  │ PostgreSQL     │  │
│  │ K8s Jobs       │  │   │  │ S3 Storage     │  │
│  └────────────────┘  │   │  └────────────────┘  │
└──────────────────────┘   └──────────────────────┘
         ▲                          ▲
         │                          │
    ┌────┴────┐                ┌────┴────┐
    │ @remote │                │   CLI   │
    │decorator│                │ recipes │
    └─────────┘                └─────────┘
```

## Deployment Order

1. **execution-service** - No dependencies, can be deployed anytime
2. **registry-service** - Requires PostgreSQL and S3, deploy after those are ready

## Environment Variables

### execution-service
- `POD_POOL_ENABLED` - Enable/disable pod pool (default: `true`)
- `POD_POOL_TTL_SECONDS` - Idle pod TTL (default: `600`)
- `NAMESPACE` - K8s namespace (default: `default`)
- `ECR_REGISTRY` - ECR registry URL
- `PORT` - HTTP port (default: `8080`)

### registry-service
- `POSTGRES_HOST` - PostgreSQL host (default: `postgres.postgres.svc.cluster.local`)
- `POSTGRES_PORT` - PostgreSQL port (default: `5432`)
- `POSTGRES_DB` - Database name (default: `flyte`)
- `POSTGRES_USER` - Database user (default: `flyte`)
- `POSTGRES_PASSWORD` - Database password (from secret)
- `S3_BUCKET` - S3 bucket for recipes
- `PORT` - HTTP port (default: `8081`)

## Common Operations

### Deploy All Services
```bash
make -C execution-service deploy
make -C registry-service deploy
```

### Check Status
```bash
kubectl get pods -l 'app in (execution-service,registry-service)'
kubectl get svc execution-service registry-service
```

### View Logs
```bash
# Execution service
kubectl logs -l app=execution-service -f

# Registry service
kubectl logs -l app=registry-service -f
```

### Port-Forward for Local Development
```bash
# Terminal 1: Execution service
kubectl port-forward svc/execution-service 8765:8080

# Terminal 2: Registry service
kubectl port-forward svc/registry-service 8766:8081
```

## Security Considerations

### execution-service
⚠️  **WARNING:** Accepts arbitrary cloudpickle payloads (RCE by design for Phase 1)

Production deployment MUST implement:
1. Authentication (API tokens, mTLS, OAuth)
2. Authorization (per-user RBAC)
3. NetworkPolicy to restrict access
4. Consider moving deserialization into execution pods

### registry-service
- Requires PostgreSQL credentials (stored in K8s Secret)
- Requires IAM role for S3 access
- No public internet exposure required
- Access via service mesh or ingress only

## Monitoring

Both services expose:
- `GET /health` - Kubernetes health check
- Prometheus metrics (future)
- Structured logging to stdout

## Development

### Build Images
```bash
# Execution service
docker build -t execution-service:latest execution-service/

# Registry service
docker build -t registry-service:latest registry-service/
```

### Run Tests
```bash
# From repo root
pytest tests/
```

## Migration

See [Migration Guide](../../docs/migration-service-split.md) for detailed migration instructions from the old `remote-agent` service.
