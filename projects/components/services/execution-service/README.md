# Execution Service

Stateless FastAPI service that handles remote function execution on Kubernetes.

## Features

- Accepts serialized Python functions from `@remote` decorator
- Creates Kubernetes Jobs/Pods to execute functions on GPU/CPU nodes
- Streams logs back to client in real-time
- Pod pool support for warm container reuse (reduced cold-start latency)
- TTL-based eviction for idle pods
- Startup reconciliation to recover from restarts

## Architecture

- **Stateless**: No persistent state (pod pool is in-memory, can be Redis in future)
- **Independent**: No dependencies on registry or other services
- **Scalable**: Can run multiple replicas (pod pool coordination via Redis TBD)

## Endpoints

- `GET /` - Health check
- `GET /health` - Kubernetes readiness probe
- `POST /execute` - Execute a serialized function
- `GET /pool` - Get pod pool status
- `GET /pool/stats` - Get pod pool statistics

## Configuration

Environment variables:
- `POD_POOL_ENABLED` - Enable/disable pod pool (default: `true`)
- `POD_POOL_TTL_SECONDS` - Idle pod TTL in seconds (default: `600`)
- `NAMESPACE` - Kubernetes namespace (default: `default`)
- `ECR_REGISTRY` - ECR registry URL for images
- `PORT` - HTTP server port (default: `8080`)

## Deployment

```bash
# Build image
docker build -t execution-service:latest .

# Run locally
docker run -p 8080:8080 execution-service:latest

# Deploy to Kubernetes
kubectl apply -k k8s/
```

## Security Considerations

**WARNING**: This service accepts arbitrary cloudpickle payloads and executes them.
This is a remote code execution (RCE) vulnerability by design for Phase 1 MVP.

Production deployment MUST implement:
1. Authentication (API tokens, mTLS, or OAuth)
2. Authorization (per-user RBAC for job creation)
3. NetworkPolicy to restrict access to trusted clients only
4. Consider: move deserialization into execution pods (not service process)

For now, expose only via port-forward or restrict via NetworkPolicy.
