# Executor Pool Image

Long-lived HTTP server for Pod Pool warm container execution.

## Overview

This image runs a Flask HTTP server that accepts cloudpickle payloads, executes functions, and returns results. Unlike one-shot execution pods, these pods stay alive for reuse.

## Architecture

```
Executor Pod (this image)
    ↓ (listens on :8080)
    ↓ (GET /health - readiness check)
    ↓ (POST /execute - function execution)
    ↓ (stays alive between executions)
    ↓ (evicted after TTL seconds of idle time)
```

## Base Image

Built on top of `ml-platform/base-gpu:latest` which includes:
- Python 3.12
- CUDA 12.9.1 + cuDNN
- PyTorch 2.9.0
- Common ML packages: transformers, accelerate, datasets, peft, bitsandbytes
- cloudpickle

## Additional Packages

- Flask 3.0+ (HTTP server)

## Endpoints

### `GET /health`

Health check endpoint for Kubernetes readiness probe.

**Response:**
```json
{"status": "healthy"}
```

### `POST /execute`

Execute a cloudpickle-serialized function.

**Request:**
- Content-Type: `application/octet-stream`
- Body: cloudpickle bytes

**Response:**
```json
{
  "success": true,
  "result": "<base64-encoded cloudpickle result>"
}
```

Or on error:
```json
{
  "success": false,
  "result": "<base64-encoded cloudpickle error dict>"
}
```

## Building

```bash
make build
make push
```

## Running Locally

```bash
docker run -p 8080:8080 \
  805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/executor-pool:latest
```

Test with curl:
```bash
# Health check
curl http://localhost:8080/health

# Execute function (requires cloudpickle payload)
# See examples/pod_pool_demo.py for usage
```

## Environment Variables

- `PORT`: HTTP server port (default: 8080)

## Kubernetes Integration

Pods are created by the remote-agent service with:
- Readiness probe: `GET /health` every 5s
- Labels: `app=remote-execution-pool`, `config-hash=<hash>`
- Managed by: `remote-agent`

## Security

- Runs as root (required for pip install inside container)
- No privileged mode
- Network isolation via Kubernetes NetworkPolicy
- Only accessible from remote-agent service

## Package Installation

User-specified packages are installed on-demand using:
```python
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "--root-user-action=ignore"
] + packages)
```

Packages persist for the lifetime of the pod (until TTL eviction).
