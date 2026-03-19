# Sub-Project: Ray

Deploys the KubeRay operator and provides the Ray worker Docker image.

## Deploy

```bash
cd projects/ray
make install
```

## Build Image

```bash
make build-image
```

## Key Components

| Component | Description |
|-----------|-------------|
| KubeRay Operator | Manages RayCluster CRDs on K8s |
| `ray-worker` image | CUDA + PyTorch + Ray runtime |
| SDK `@ray_task` | Decorator for distributed training |

## Dependencies

- `projects/eks` (cluster must be running)
- `projects/components` (base-gpu image)
