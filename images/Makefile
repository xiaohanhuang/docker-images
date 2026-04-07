# ============================================================
# images/Makefile
#
# Builds and pushes all ml-platform container images in strict
# dependency order so that base layers are always available
# before child layers are built.
#
# Directory layout:
#   images/
#   ├── base/        ← L1: slowest-changing (CUDA, PyTorch)
#   ├── framework/   ← L2: Flyte, Ray
#   └── workload/    ← L3: ML libs, fastest-changing
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
BASE_CPU_TAG      := py$(PYTHON_VERSION)-$(TAG)
BASE_GPU_TAG      := cu$(CUDA_VERSION)-torch$(TORCH_VERSION)-$(TAG)
FLYTE_CPU_TAG     := fly$(FLYTEKIT_VERSION)-$(TAG)
FLYTE_GPU_TAG     := fly$(FLYTEKIT_VERSION)-cu$(CUDA_VERSION)-$(TAG)
RAY_GPU_TAG       := ray$(RAY_VERSION)-cu$(CUDA_VERSION)-$(TAG)
DATA_CPU_TAG      := $(TAG)
ML_GPU_TAG        := $(TAG)
GENAI_GPU_TAG     := $(TAG)
TRAINING_LLM_TAG  := $(TAG)

# ── Full image URIs ──────────────────────────────────────────
BASE_CPU_IMAGE      := $(REGISTRY)/$(REPO)/base-cpu:$(BASE_CPU_TAG)
BASE_GPU_IMAGE      := $(REGISTRY)/$(REPO)/base-gpu-torch:$(BASE_GPU_TAG)
FLYTE_CPU_IMAGE     := $(REGISTRY)/$(REPO)/flyte-cpu:$(FLYTE_CPU_TAG)
FLYTE_GPU_IMAGE     := $(REGISTRY)/$(REPO)/flyte-gpu:$(FLYTE_GPU_TAG)
RAY_GPU_IMAGE       := $(REGISTRY)/$(REPO)/ray-gpu:$(RAY_GPU_TAG)
DATA_CPU_IMAGE      := $(REGISTRY)/$(REPO)/data-cpu:$(DATA_CPU_TAG)
ML_GPU_IMAGE        := $(REGISTRY)/$(REPO)/ml-gpu:$(ML_GPU_TAG)
GENAI_GPU_IMAGE     := $(REGISTRY)/$(REPO)/genai-gpu:$(GENAI_GPU_TAG)
TRAINING_LLM_IMAGE  := $(REGISTRY)/$(REPO)/training-llm:$(TRAINING_LLM_TAG)

# ── Helpers ──────────────────────────────────────────────────
BUILDX := docker buildx build --platform $(PLATFORM) --load

.PHONY: login \
        build-base-cpu build-base-gpu \
        build-flyte-cpu build-flyte-gpu build-ray-gpu \
        build-data-cpu build-ml-gpu build-genai-gpu build-training-llm \
        build-bases build-frameworks build-workloads build-all \
        push-base-cpu push-base-gpu \
        push-flyte-cpu push-flyte-gpu push-ray-gpu \
        push-data-cpu push-ml-gpu push-genai-gpu push-training-llm \
        push-bases push-frameworks push-workloads push-all \
        prepull-bases prepull-frameworks \
        release tags

# ── ECR login ────────────────────────────────────────────────
login:
	$(if $(AWS_PROFILE),AWS_PROFILE=$(AWS_PROFILE) ,)aws ecr get-login-password \
	  --region us-west-2 \
	  | docker login --username AWS --password-stdin $(REGISTRY)

# ── Layer 1: base images ─────────────────────────────────────
build-base-cpu:
	$(BUILDX) \
	  --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
	  -t $(BASE_CPU_IMAGE) \
	  base/cpu/

build-base-gpu:
	$(BUILDX) \
	  --build-arg CUDA_VERSION=$(CUDA_VERSION) \
	  --build-arg TORCH_VERSION=$(TORCH_VERSION) \
	  --build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
	  --build-arg TORCHAUDIO_VERSION=$(TORCHAUDIO_VERSION) \
	  -t $(BASE_GPU_IMAGE) \
	  base/gpu-torch/

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
	  framework/flyte-cpu/

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
	  framework/flyte-gpu/

build-ray-gpu: build-base-gpu
	$(BUILDX) \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg REPO=$(REPO) \
	  --build-arg BASE_TAG=$(BASE_GPU_TAG) \
	  --build-arg RAY_VERSION=$(RAY_VERSION) \
	  -t $(RAY_GPU_IMAGE) \
	  framework/ray-gpu/

build-frameworks: build-flyte-cpu build-flyte-gpu build-ray-gpu

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
	  workload/data-cpu/

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
	  --build-arg ROUGE_VERSION=$(ROUGE_VERSION) \
	  --build-arg SENTENCEPIECE_VERSION=$(SENTENCEPIECE_VERSION) \
	  -t $(ML_GPU_IMAGE) \
	  workload/ml-gpu/

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
	  --build-arg ANTHROPIC_VERSION=$(ANTHROPIC_VERSION) \
	  --build-arg TIKTOKEN_VERSION=$(TIKTOKEN_VERSION) \
	  --build-arg FAISS_VERSION=$(FAISS_VERSION) \
	  --build-arg CHROMADB_VERSION=$(CHROMADB_VERSION) \
	  --build-arg PSYCOPG2_VERSION=$(PSYCOPG2_VERSION) \
	  --build-arg PGVECTOR_VERSION=$(PGVECTOR_VERSION) \
	  --build-arg MLFLOW_VERSION=$(MLFLOW_VERSION) \
	  --build-arg PANDAS_VERSION=$(PANDAS_VERSION) \
	  -t $(GENAI_GPU_IMAGE) \
	  workload/genai-gpu/

build-training-llm: build-ray-gpu
	$(BUILDX) \
	  --build-arg BASE_IMAGE=$(RAY_GPU_IMAGE) \
	  --build-arg TRANSFORMERS_VERSION=$(TRANSFORMERS_VERSION) \
	  --build-arg ACCELERATE_VERSION=$(ACCELERATE_VERSION) \
	  --build-arg PEFT_VERSION=$(PEFT_VERSION) \
	  --build-arg DATASETS_VERSION=$(DATASETS_VERSION) \
	  --build-arg BITSANDBYTES_VERSION=$(BITSANDBYTES_VERSION) \
	  --build-arg DEEPSPEED_VERSION=$(DEEPSPEED_VERSION) \
	  --build-arg FLYTE_RAY_PLUGIN=$(FLYTE_RAY_PLUGIN) \
	  --build-arg MLFLOW_VERSION=$(MLFLOW_VERSION) \
	  --build-arg OPENRLHF_VERSION=$(OPENRLHF_VERSION) \
	  --build-arg VLLM_VERSION=$(VLLM_VERSION) \
	  -t $(TRAINING_LLM_IMAGE) \
	  workload/training-llm/

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

push-ray-gpu:
	docker push $(RAY_GPU_IMAGE)

push-data-cpu:
	docker push $(DATA_CPU_IMAGE)

push-ml-gpu:
	docker push $(ML_GPU_IMAGE)

push-genai-gpu:
	docker push $(GENAI_GPU_IMAGE)

push-training-llm:
	docker push $(TRAINING_LLM_IMAGE)

push-bases:    push-base-cpu push-base-gpu
push-frameworks: push-flyte-cpu push-flyte-gpu push-ray-gpu
push-workloads: push-data-cpu push-ml-gpu push-genai-gpu push-training-llm
push-all: push-bases push-frameworks push-workloads

# ── One-shot release ─────────────────────────────────────────
release: login build-all push-all
	@echo ""
	@echo "Released IMAGE_TAG=$(TAG) — all images pushed to ECR."

# ── Pre-pull helpers (for DaemonSet / cluster bootstrap) ─────
prepull-bases:
	@echo "Pulling L1 base images..."
	docker pull $(BASE_CPU_IMAGE)
	docker pull $(BASE_GPU_IMAGE)

prepull-frameworks: prepull-bases
	@echo "Pulling L2 framework images..."
	docker pull $(FLYTE_CPU_IMAGE)
	docker pull $(FLYTE_GPU_IMAGE)
	docker pull $(RAY_GPU_IMAGE)

# ── Print all resolved tags (useful for CI / release notes) ──
tags:
	@echo "base-cpu:       $(BASE_CPU_IMAGE)"
	@echo "base-gpu-torch: $(BASE_GPU_IMAGE)"
	@echo "flyte-cpu:      $(FLYTE_CPU_IMAGE)"
	@echo "flyte-gpu:      $(FLYTE_GPU_IMAGE)"
	@echo "ray-gpu:        $(RAY_GPU_IMAGE)"
	@echo "data-cpu:       $(DATA_CPU_IMAGE)"
	@echo "ml-gpu:         $(ML_GPU_IMAGE)"
	@echo "genai-gpu:      $(GENAI_GPU_IMAGE)"
	@echo "training-llm:   $(TRAINING_LLM_IMAGE)"
