# Sub-Project: Components

Shared Docker images, the `ml_platform_sdk` Python package, component library, and example workflows.

## Structure

```
components/
├── images/
│   ├── base-gpu/Dockerfile    # CUDA + PyTorch + SSH + debug tools
│   ├── training-llm/Dockerfile  # Transformers + DeepSpeed + PEFT
│   ├── data-cpu/Dockerfile    # Lightweight CPU image for data tasks
│   ├── ml-gpu/Dockerfile      # GPU image for training/evaluation
│   └── genai-gpu/Dockerfile   # GenAI image (vLLM, LangChain)
├── components/
│   ├── data/                  # Data ingestion and preprocessing
│   ├── training/              # Model fine-tuning
│   ├── evaluation/            # Model evaluation and metrics
│   ├── serving/               # Inference deployment
│   ├── genai/                 # Embeddings, RAG, guardrails
│   └── ops/                   # Notifications, monitoring, cost
├── sdk/
│   └── ml_platform_sdk/       # Python SDK for task decorators
│       ├── tasks/data.py      # @data_task
│       ├── tasks/training.py  # @ray_task
│       ├── tasks/spark.py     # @spark_task
│       └── tasks/efs.py       # @efs_task / build_efs_pod_template
└── workflows/
    └── llm_finetune/          # Example: LLM fine-tuning pipeline
```

## Component Registry

Components are registered in **FlyteAdmin** — the Flyte registry is the single
source of truth.  Use `pyflyte register` (or `ml-plat component register`) to
push task modules, then browse them via the CLI.

### Browse available components

```bash
ml-plat component list
ml-plat component list --project ml-platform --domain production
```

### Get detailed info for a component

```bash
ml-plat component info components.training.finetune.finetune_lm
ml-plat component info download_dataset --version v1
```

### Register components to Flyte

```bash
ml-plat component register projects/components/components/data/ --image ml-platform/data-cpu:latest
ml-plat component register projects/components/components/training/finetune.py --image ml-platform/ml-gpu:latest
```

## Build Images

```bash
# Standalone CPU image (no GPU required)
docker build -t ml-platform/data-cpu:latest images/data-cpu/

# GPU images — pass the ECR base-gpu image as the build arg
BASE=805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/base-gpu:latest
docker build --build-arg BASE_IMAGE=$BASE -t ml-platform/ml-gpu:latest  images/ml-gpu/
docker build --build-arg BASE_IMAGE=$BASE -t ml-platform/genai-gpu:latest images/genai-gpu/

# Legacy images
docker build -t ml-platform/base-gpu:latest images/base-gpu/
docker build -t ml-platform/training-llm:latest images/training-llm/
```

Images are built and pushed to ECR automatically by the CI pipeline in
[xiaohanhuang/docker-images](https://github.com/xiaohanhuang/docker-images) on
every merge to `main`.

## Install SDK

```bash
pip install -e sdk/
```

## Dependencies

- `projects/eks` (for running on the cluster)
- `projects/ray` (for `@ray_task`)
- `projects/spark` (for `@spark_task`)
