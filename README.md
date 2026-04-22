# ML Training Platform

A production-grade ML training platform on AWS EKS, combining **Flyte** (orchestration), **Ray** (distributed training), and **Spark** (ETL) with full observability.

**🎉 NEW: All services now exposed via ALB Ingress with stable internal URLs** — no more fragile `kubectl port-forward`!
- http://flyte.ml-platform.internal
- http://grafana.ml-platform.internal
- http://mlflow.ml-platform.internal
- [See full networking guide →](docs/infrastructure/alb-ingress-guide.md)

## Sub-Projects

| Sub-Project | Description | Deploy Order |
|------------|-------------|:---:|
| [`projects/eks`](projects/eks/) | EKS cluster + VPC + GPU nodes | 1 |
| [`projects/alb-controller`](projects/alb-controller/) | AWS Load Balancer Controller + Ingress | 2 |
| [`projects/flyte`](projects/flyte/) | Flyte workflow orchestrator | 3 |
| [`projects/ray`](projects/ray/) | KubeRay operator + worker image | 4 |
| [`projects/spark`](projects/spark/) | Spark-on-K8s operator + image | 4 |
| [`projects/monitoring`](projects/monitoring/) | Prometheus, Grafana, MLflow, DCGM | 5 |
| [`projects/components`](projects/components/) | Shared images, SDK, example workflows | 6 |
| [`cli`](cli/) | Customer-facing `mlp` CLI | 7 |

## Quick Start

```bash
# 1. Provision EKS + ALB Controller
cd projects/eks && terraform init && terraform apply

# 2. Deploy ALB Controller
cd ../alb-controller && make install

# 3. Deploy platform components
cd ../flyte     && make install
cd ../ray       && make install
cd ../spark     && make install
cd ../monitoring && make install-all

# 4. Deploy Ingress resources
cd ../alb-controller && make deploy-ingress

# 5. Install the CLI
cd ../.. && pip install -e .

# 6. Configure environment for internal DNS
export FLYTE_ENDPOINT=flyteadmin.ml-platform.internal:80
export ML_PLAT_EXECUTION_URL=http://execution-service.ml-platform.internal
export MLFLOW_TRACKING_URI=http://mlflow.ml-platform.internal

# 7. Submit a training job
mlp job submit --workflow workflows/llm_finetune
```

**Or use the root Makefile**:

```bash
make deploy-all  # Deploys everything in correct order
```

## Developer Setup

After cloning, install the git hooks so that ruff, black, and pytest run automatically before every push (matching CI):

```bash
pip install -e ".[dev]"
make install-hooks
```

The pre-push hook blocks the push and prints the fix command if any check fails:

| Check | Auto-fix command |
|-------|-----------------|
| `ruff` (lint) | `ruff check --fix .` |
| `black` (format) | `black .` |
| `pytest` (tests) | — fix the failing test |

## Architecture

See [`docs/architecture/architecture.md`](docs/architecture/architecture.md) for the full architecture design, including comparisons with OpenConnect, Metaflow, and Michelangelo.