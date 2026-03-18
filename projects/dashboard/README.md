# ML Platform Dashboard

Unified web dashboard for the ML training platform, aggregating data from Flyte, MLflow, Grafana, Ray, and Kubernetes.

## Features

- **Overview**: Active GPU pods, burn rate, recent executions
- **Experiments**: MLflow experiments and runs with comparison
- **Pipelines**: Flyte execution status and history
- **Ray**: Ray cluster status, job queue, actor utilization
- **Infrastructure**: GPU nodes, utilization, cost trends (Grafana embedded)
- **Kubernetes**: Pod/node status, events, logs viewer
- **Models**: Model registry with deployment status

## Architecture

The dashboard is a Next.js application that:
- Runs as a standalone Node.js server
- Calls the ML Platform backend API for aggregated data
- Embeds Grafana panels for GPU metrics
- Provides real-time updates via React Query

## Development

### Prerequisites

- Node.js 20+
- npm or yarn

### Local Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Open http://localhost:3000
```

### Environment Variables

```bash
NEXT_PUBLIC_API_URL=http://ml-platform-api.ml-platform-development.svc.cluster.local:8000
NEXT_PUBLIC_GRAFANA_URL=http://grafana.ml-platform.internal
```

## Deployment

### Build and Push Docker Image

```bash
make push
```

### Deploy to Kubernetes

```bash
make deploy
```

### Full Install

```bash
make install
```

### Check Status

```bash
make status
```

### View Logs

```bash
make logs
```

## API Integration

The dashboard consumes the following backend API endpoints:

### Dashboard Aggregation
- `GET /dashboard/overview` - Unified overview metrics
- `GET /dashboard/metrics/{metric}` - Prometheus metrics

### MLflow
- `GET /mlflow/experiments` - List experiments
- `GET /mlflow/experiments/{id}/runs` - List runs
- `GET /mlflow/models` - List registered models

### Ray
- `GET /ray/cluster` - Cluster status
- `GET /ray/jobs` - Active jobs

### Kubernetes
- `GET /kubernetes/nodes` - Node status
- `GET /kubernetes/events` - Cluster events
- `GET /kubernetes/pods/{name}/logs` - Pod logs

### Existing Endpoints
- `GET /pods` - List pods
- `GET /jobs` - List Flyte executions
- `GET /cost/report` - Cost analysis

## Technologies

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Data Fetching**: React Query (TanStack Query)
- **HTTP Client**: Axios
- **Icons**: Lucide React
- **Charts**: Recharts
- **Date Handling**: date-fns

## Project Structure

```
projects/dashboard/
├── app/                    # Next.js app directory
│   ├── overview/          # Overview page
│   ├── experiments/       # Experiments page
│   ├── pipelines/         # Pipelines page
│   ├── ray/              # Ray cluster page
│   ├── infrastructure/   # Infrastructure page
│   ├── kubernetes/       # Kubernetes page
│   ├── models/           # Models page
│   ├── layout.tsx        # Root layout
│   ├── page.tsx          # Home page (redirects to overview)
│   └── globals.css       # Global styles
├── components/           # Reusable React components
│   ├── Navigation.tsx
│   ├── Card.tsx
│   ├── StatCard.tsx
│   ├── StatusBadge.tsx
│   ├── LoadingSpinner.tsx
│   └── ErrorMessage.tsx
├── lib/                  # Utility libraries
│   ├── api.ts           # API client
│   └── utils.ts         # Helper functions
├── types/               # TypeScript type definitions
├── Dockerfile           # Multi-stage Docker build
├── deployment.yaml      # Kubernetes manifests
├── Makefile            # Build and deployment automation
└── package.json        # Dependencies
```

## Kubernetes Deployment

The dashboard is deployed as:
- **Namespace**: `ml-platform`
- **Replicas**: 2 for high availability
- **Service**: ClusterIP on port 80
- **Ingress**: ALB ingress for external access
- **RBAC**: Service account with read-only access to cluster resources

## Troubleshooting

### Dashboard not loading

1. Check pod status:
   ```bash
   kubectl get pods -n ml-platform -l app=ml-platform-dashboard
   ```

2. View logs:
   ```bash
   make logs
   ```

3. Verify ingress:
   ```bash
   kubectl get ingress -n ml-platform ml-platform-dashboard
   ```

### API connection errors

1. Verify backend API is running:
   ```bash
   kubectl get pods -n ml-platform-development -l app=ml-platform-api
   ```

2. Test backend connectivity:
   ```bash
   kubectl exec -n ml-platform <dashboard-pod> -- curl http://ml-platform-api.ml-platform-development.svc.cluster.local:8000/health
   ```

### Grafana panels not loading

1. Verify Grafana URL in environment variables
2. Check Grafana service:
   ```bash
   kubectl get svc -n monitoring kube-prometheus-stack-grafana
   ```

## Future Enhancements

- [ ] Authentication and authorization
- [ ] User-specific views
- [ ] Custom dashboard builder
- [ ] Real-time metrics streaming
- [ ] Export and reporting features
- [ ] Cost optimization recommendations
- [ ] Alerting and notifications
