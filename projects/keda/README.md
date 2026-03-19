# Sub-Project: KEDA

Deploys KEDA (Kubernetes Event-driven Autoscaling) to enable scale-to-zero and event-driven scaling for inference workloads.

## Purpose

KEDA allows GPU inference deployments to scale to zero replicas when idle, eliminating cost for unused GPU nodes. When traffic arrives, KEDA scales the deployment back up, and Karpenter provisions the required GPU nodes automatically.

## Deploy

```bash
cd projects/keda
make install
```

## Verify

```bash
make status
# Expected: keda-operator, keda-metrics-apiserver, and keda-admission-webhooks pods running
```

## How It Works with Karpenter

1. KEDA watches external metrics (e.g., Prometheus query rate, queue depth).
2. When metrics exceed a threshold, KEDA scales the target Deployment from 0 → N replicas.
3. The new pods are pending (no GPU node available yet).
4. Karpenter detects the pending pods with GPU resource requests and provisions GPU nodes.
5. When metrics drop to zero, KEDA scales back to 0 replicas.
6. Karpenter consolidates the now-empty GPU nodes.

## Dependencies

- EKS cluster (projects/eks)
- Karpenter (deployed as part of EKS setup)
- Prometheus (projects/monitoring) — for metrics-based scaling triggers
