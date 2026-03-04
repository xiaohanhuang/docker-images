# ============================================================
# projects/components/images/Makefile
#
# Builds and pushes all ml-platform container images in strict
# dependency order so that base layers are always available
# before child layers are built.
#
# Pre-pull workflow (cold cluster nodes):
#   make prepull-base   — pull the two L1 images (smallest)
#   make prepull-frame  — pull the L2 framework images
#   # workload images build on top of those cached layers
#
# Full release workflow:
#   make login          — ECR docker login
#   make build-all      — build every image in order
#   make push-all       — push every image in order
#   make release        — login + build-all + push-all
#
# Bump IMAGE_TAG in versions.env to cut a new release.
# ============================================================

include versions.env

REGISTRY := $(ECR_REGISTRY)
REPO     := $(ECR_REPO)
TAG      := $(IMAGE_TAG)

PLATFORM := linux/amd64

# ── Computed image tags ──────────────────────────────────────
# Format: <name>:<what's-inside>-<release-tag>
# The descriptive prefix lets you tell at a glance what CUDA /
# framework versions are baked in when you `docker images`.

BASE_CPU_TAG      := py$(PYTHON_VERSION)-$(TAG)
BASE_GPU_TAG      := cu$(CUDA_VERSION)-torch$(TORCH_VERSION)-$(TAG)
FLYTE_CPU_TAG     := fly$(FLYTEKIT_VERSION)-$(TAG)
FLYTE_GPU_TAG     := fly$(FLYTEKIT_VERSION)-cu$(CUDA_VERSION)-$(TAG)
RAY_WORKER_TAG    := ray$(RAY_VERSION)-cu$(CUDA_VERSION)-$(TAG)
DATA_CPU_TAG      := $(TAG)
ML_GPU_TAG        := $(TAG)
GENAI_GPU_TAG     := $(TAG)
TRAINING_LLM_TAG  := $(TAG)

# ── Full image URIs ──────────────────────────────────────────
BASE_CPU_IMAGE      := $(REGISTRY)/$(REPO)/base-cpu:$(BASE_CPU_TAG)
BASE_GPU_IMAGE      := $(REGISTRY)/$(REPO)/base-gpu:$(BASE_GPU_TAG)
FLYTE_CPU_IMAGE     := $(REGISTRY)/$(REPO)/flyte-cpu:$(FLYTE_CPU_TAG)
FLYTE_GPU_IMAGE     := $(REGISTRY)/$(REPO)/flyte-gpu:$(FLYTE_GPU_TAG)
RAY_WORKER_IMAGE    := $(REGISTRY)/$(REPO)/ray-worker:$(RAY_WORKER_TAG)
DATA_CPU_IMAGE      := $(REGISTRY)/$(REPO)/data-cpu:$(DATA_CPU_TAG)
ML_GPU_IMAGE        := $(REGISTRY)/$(REPO)/ml-gpu:$(ML_GPU_TAG)
GENAI_GPU_IMAGE     := $(REGISTRY)/$(REPO)/genai-gpu:$(GENAI_GPU_TAG)
TRAINING_LLM_IMAGE  := $(REGISTRY)/$(REPO)/training-llm:$(TRAINING_LLM_TAG)

# ── Helpers ──────────────────────────────────────────────────
# --load ensures the image is available in the local docker daemon after the
# build completes.  This is required when using a non-default buildx builder
# (e.g. as GitHub Actions creates) so that child-layer FROM instructions can
# resolve the locally-built parent image.
BUILDX := docker buildx build --platform $(PLATFORM) --load

.PHONY: login \
        build-base-cpu build-base-gpu \
        build-flyte-cpu build-flyte-gpu build-ray-worker \
        build-data-cpu build-ml-gpu build-genai-gpu build-training-llm \
        build-bases build-frameworks build-workloads build-all \
        push-base-cpu push-base-gpu \
        push-flyte-cpu push-flyte-gpu push-ray-worker \
        push-data-cpu push-ml-gpu push-genai-gpu push-training-llm \
        push-bases push-frameworks push-workloads push-all \
        prepull-bases prepull-frameworks \
        release tags

# ── ECR login ────────────────────────────────────────────────
# Local dev:  make login AWS_PROFILE=adfs
# CI (OIDC):  make login          (no profile needed; creds are in env)
login:
	$(if $(AWS_PROFILE),AWS_PROFILE=$(AWS_PROFILE) ,)aws ecr get-login-password \
	  --region us-west-2 \
	  | docker login --username AWS --password-stdin $(REGISTRY)

# ── Layer 1: base images ─────────────────────────────────────
build-base-cpu:
	$(BUILDX) \
	  --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
	  -t $(BASE_CPU_IMAGE) \
	  base-cpu/

build-base-gpu:
	$(BUILDX) \
	  --build-arg CUDA_VERSION=$(CUDA_VERSION) \
	  --build-arg TORCH_VERSION=$(TORCH_VERSION) \
	  --build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
	  --build-arg TORCHAUDIO_VERSION=$(TORCHAUDIO_VERSION) \
	  -t $(BASE_GPU_IMAGE) \
	  base-gpu/

build-bases: build-base-cpu build-base-gpu

# ── Layer 2: framework images ────────────────────────────────
build-flyte-cpu: build-base-cpu
	$(BUILDX) \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg REPO=$(REPO) \
	  --build-arg BASE_TAG=$(BASE_CPU_TAG) \
	  --build-arg FLYTEKIT_VERSION=$(FLYTEKIT_VERSION) \
	  --build-arg BOTO3_VERSION=$(BOTO3_VERSION) \
	  --build-arg S3FS_VERSION=$(S3FS_VERSION) \
	  --build-arg REQUESTS_VERSION=$(REQUESTS_VERSION) \
	  -t $(FLYTE_CPU_IMAGE) \
	  flyte-cpu/

build-flyte-gpu: build-base-gpu
	$(BUILDX) \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg REPO=$(REPO) \
	  --build-arg BASE_TAG=$(BASE_GPU_TAG) \
	  --build-arg FLYTEKIT_VERSION=$(FLYTEKIT_VERSION) \
	  --build-arg BOTO3_VERSION=$(BOTO3_VERSION) \
	  --build-arg S3FS_VERSION=$(S3FS_VERSION) \
	  --build-arg REQUESTS_VERSION=$(REQUESTS_VERSION) \
	  -t $(FLYTE_GPU_IMAGE) \
	  flyte-gpu/

build-ray-worker: build-base-gpu
	$(BUILDX) \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg REPO=$(REPO) \
	  --build-arg BASE_TAG=$(BASE_GPU_TAG) \
	  --build-arg RAY_VERSION=$(RAY_VERSION) \
	  -t $(RAY_WORKER_IMAGE) \
	  ray-worker/

build-frameworks: build-flyte-cpu build-flyte-gpu build-ray-worker

# ── Layer 3: workload images ─────────────────────────────────
build-data-cpu: build-flyte-cpu
	$(BUILDX) \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg REPO=$(REPO) \
	  --build-arg BASE_TAG=$(FLYTE_CPU_TAG) \
	  --build-arg PANDAS_VERSION=$(PANDAS_VERSION) \
	  --build-arg PYARROW_VERSION=$(PYARROW_VERSION) \
	  --build-arg SKLEARN_VERSION=$(SKLEARN_VERSION) \
	  --build-arg TIKTOKEN_VERSION=$(TIKTOKEN_VERSION) \
	  --build-arg PYPDF_VERSION=$(PYPDF_VERSION) \
	  --build-arg BS4_VERSION=$(BS4_VERSION) \
	  --build-arg LXML_VERSION=$(LXML_VERSION) \
	  --build-arg NLTK_VERSION=$(NLTK_VERSION) \
	  -t $(DATA_CPU_IMAGE) \
	  data-cpu/

build-ml-gpu: build-flyte-gpu
	$(BUILDX) \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg REPO=$(REPO) \
	  --build-arg BASE_TAG=$(FLYTE_GPU_TAG) \
	  --build-arg TRANSFORMERS_VERSION=$(TRANSFORMERS_VERSION) \
	  --build-arg ACCELERATE_VERSION=$(ACCELERATE_VERSION) \
	  --build-arg PEFT_VERSION=$(PEFT_VERSION) \
	  --build-arg DATASETS_VERSION=$(DATASETS_VERSION) \
	  --build-arg MLFLOW_VERSION=$(MLFLOW_VERSION) \
	  --build-arg SKLEARN_VERSION=$(SKLEARN_VERSION) \
	  --build-arg EVALUATE_VERSION=$(EVALUATE_VERSION) \
	  --build-arg ROUGE_VERSION=$(ROUGE_VERSION) \
	  --build-arg SENTENCEPIECE_VERSION=$(SENTENCEPIECE_VERSION) \
	  -t $(ML_GPU_IMAGE) \
	  ml-gpu/

build-genai-gpu: build-flyte-gpu
	$(BUILDX) \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg REPO=$(REPO) \
	  --build-arg BASE_TAG=$(FLYTE_GPU_TAG) \
	  --build-arg VLLM_VERSION=$(VLLM_VERSION) \
	  --build-arg LANGCHAIN_VERSION=$(LANGCHAIN_VERSION) \
	  --build-arg LANGCHAIN_COMMUNITY_VERSION=$(LANGCHAIN_COMMUNITY_VERSION) \
	  --build-arg SENTENCE_TRANSFORMERS_VERSION=$(SENTENCE_TRANSFORMERS_VERSION) \
	  --build-arg OPENAI_VERSION=$(OPENAI_VERSION) \
	  --build-arg TIKTOKEN_VERSION=$(TIKTOKEN_VERSION) \
	  --build-arg FAISS_VERSION=$(FAISS_VERSION) \
	  --build-arg CHROMADB_VERSION=$(CHROMADB_VERSION) \
	  --build-arg PSYCOPG2_VERSION=$(PSYCOPG2_VERSION) \
	  --build-arg PGVECTOR_VERSION=$(PGVECTOR_VERSION) \
	  -t $(GENAI_GPU_IMAGE) \
	  genai-gpu/

build-training-llm: build-ray-worker
	$(BUILDX) \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg REPO=$(REPO) \
	  --build-arg BASE_TAG=$(RAY_WORKER_TAG) \
	  --build-arg TRANSFORMERS_VERSION=$(TRANSFORMERS_VERSION) \
	  --build-arg ACCELERATE_VERSION=$(ACCELERATE_VERSION) \
	  --build-arg PEFT_VERSION=$(PEFT_VERSION) \
	  --build-arg DATASETS_VERSION=$(DATASETS_VERSION) \
	  --build-arg BITSANDBYTES_VERSION=$(BITSANDBYTES_VERSION) \
	  --build-arg DEEPSPEED_VERSION=$(DEEPSPEED_VERSION) \
	  --build-arg FLYTE_RAY_PLUGIN=$(FLYTE_RAY_PLUGIN) \
	  -t $(TRAINING_LLM_IMAGE) \
	  training-llm/

build-workloads: build-data-cpu build-ml-gpu build-genai-gpu build-training-llm

build-all: build-bases build-frameworks build-workloads

# ── Push targets ─────────────────────────────────────────────
push-base-cpu:
	docker push $(BASE_CPU_IMAGE)

push-base-gpu:
	docker push $(BASE_GPU_IMAGE)

push-flyte-cpu:
	docker push $(FLYTE_CPU_IMAGE)

push-flyte-gpu:
	docker push $(FLYTE_GPU_IMAGE)

push-ray-worker:
	docker push $(RAY_WORKER_IMAGE)

push-data-cpu:
	docker push $(DATA_CPU_IMAGE)

push-ml-gpu:
	docker push $(ML_GPU_IMAGE)

push-genai-gpu:
	docker push $(GENAI_GPU_IMAGE)

push-training-llm:
	docker push $(TRAINING_LLM_IMAGE)

push-bases:    push-base-cpu push-base-gpu
push-frameworks: push-flyte-cpu push-flyte-gpu push-ray-worker
push-workloads: push-data-cpu push-ml-gpu push-genai-gpu push-training-llm
push-all: push-bases push-frameworks push-workloads

# ── One-shot release ─────────────────────────────────────────
release: login build-all push-all
	@echo ""
	@echo "Released IMAGE_TAG=$(TAG) — all images pushed to ECR."

# ── Pre-pull helpers (for DaemonSet / cluster bootstrap) ─────
# Run these on nodes BEFORE deploying workloads so the heavy
# base layers are already in the node image cache.
prepull-bases:
	@echo "Pulling L1 base images..."
	docker pull $(BASE_CPU_IMAGE)
	docker pull $(BASE_GPU_IMAGE)

prepull-frameworks: prepull-bases
	@echo "Pulling L2 framework images..."
	docker pull $(FLYTE_CPU_IMAGE)
	docker pull $(FLYTE_GPU_IMAGE)
	docker pull $(RAY_WORKER_IMAGE)

# ── Print all resolved tags (useful for CI / release notes) ──
tags:
	@echo "base-cpu:      $(BASE_CPU_IMAGE)"
	@echo "base-gpu:      $(BASE_GPU_IMAGE)"
	@echo "flyte-cpu:     $(FLYTE_CPU_IMAGE)"
	@echo "flyte-gpu:     $(FLYTE_GPU_IMAGE)"
	@echo "ray-worker:    $(RAY_WORKER_IMAGE)"
	@echo "data-cpu:      $(DATA_CPU_IMAGE)"
	@echo "ml-gpu:        $(ML_GPU_IMAGE)"
	@echo "genai-gpu:     $(GENAI_GPU_IMAGE)"
	@echo "training-llm:  $(TRAINING_LLM_IMAGE)"
