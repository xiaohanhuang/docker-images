# Makefile for LLM-SFT Workflow

# Configuration
PROJECT := ml-platform
DOMAIN := development
WORKFLOW_NAME := llm_sft_lora_pipeline

# AWS Configuration
AWS_REGION := us-west-2
AWS_ACCOUNT_ID := 805673386114
ECR_REGISTRY := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

# Image names
CPU_IMAGE := $(ECR_REGISTRY)/ml-platform/llm-sft-cpu:latest
GPU_IMAGE := $(ECR_REGISTRY)/ml-platform/llm-sft-gpu:latest

.PHONY: help
help:
	@echo "LLM-SFT Workflow Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  prepare          Copy shared components into build context"
	@echo "  build-cpu        Build CPU Docker image"
	@echo "  build-gpu        Build GPU Docker image"
	@echo "  build            Build both images"
	@echo "  push-cpu         Push CPU image to ECR"
	@echo "  push-gpu         Push GPU image to ECR"
	@echo "  push             Push both images to ECR"
	@echo "  register         Register workflow with Flyte"
	@echo "  deploy           Build, push, and register"
	@echo "  clean            Clean up local artifacts"

.PHONY: prepare
prepare:
	@echo "Copying shared components into build context..."
	rm -rf components
	cp -r ../../components/components components

.PHONY: ecr-login
ecr-login:
	@echo "Logging in to ECR..."
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_REGISTRY)

.PHONY: build-cpu
build-cpu: prepare
	@echo "Building CPU image..."
	docker build -f images/Dockerfile.cpu -t $(CPU_IMAGE) .

.PHONY: build-gpu
build-gpu: prepare
	@echo "Building GPU image..."
	docker build -f images/Dockerfile.gpu -t $(GPU_IMAGE) .

.PHONY: build
build: build-cpu build-gpu

.PHONY: push-cpu
push-cpu: ecr-login
	@echo "Pushing CPU image to ECR..."
	docker push $(CPU_IMAGE)

.PHONY: push-gpu
push-gpu: ecr-login
	@echo "Pushing GPU image to ECR..."
	docker push $(GPU_IMAGE)

.PHONY: push
push: push-cpu push-gpu

.PHONY: register
register: prepare
	@echo "Registering workflow with Flyte..."
	PYTHONPATH=.:$$PYTHONPATH pyflyte register \
		--project $(PROJECT) \
		--domain $(DOMAIN) \
		--image cpu=$(CPU_IMAGE) \
		--image gpu=$(GPU_IMAGE) \
		pipeline.py

.PHONY: deploy
deploy: build push register
	@echo "Deployment complete!"

.PHONY: test-local
test-local: prepare
	@echo "Running local smoke test..."
	PYTHONPATH=.:$$PYTHONPATH pyflyte run pipeline.py $(WORKFLOW_NAME) \
		--dataset "tatsu-lab/alpaca" \
		--base_model "gpt2" \
		--epochs 1

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf components __pycache__ tasks/__pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
