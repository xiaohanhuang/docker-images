# Makefile for ML Platform Registry Service
# NOTE: Docker image builds are handled by CI in xiaohanhuang/docker-images.
# This Makefile only manages Kubernetes deployment.

.PHONY: deploy install uninstall status logs port-forward

IMAGE_NAME := ml-platform/registry-service
VERSION ?= $(error VERSION must be set (e.g. make deploy VERSION=v0.1.0))
ECR_REGISTRY ?= $(error ECR_REGISTRY must be set)
FULL_IMAGE := $(ECR_REGISTRY)/$(IMAGE_NAME):$(VERSION)
NAMESPACE := default

deploy:
	@echo "Deploying registry service to Kubernetes..."
	cd k8s && kustomize edit set image $(IMAGE_NAME)=$(FULL_IMAGE)
	kubectl apply -k k8s/
	@echo "Waiting for rollout..."
	kubectl rollout status deployment/registry-service -n $(NAMESPACE)

install: deploy

uninstall:
	@echo "Uninstalling registry service..."
	kubectl delete -k k8s/

status:
	@echo "Registry service status:"
	kubectl get pods -l app=registry-service -n $(NAMESPACE)
	kubectl get svc registry-service -n $(NAMESPACE)

logs:
	@echo "Registry service logs (last 50 lines):"
	kubectl logs -l app=registry-service -n $(NAMESPACE) --tail=50 -f

port-forward:
	@echo "Port-forwarding registry service to localhost:8766..."
	kubectl port-forward svc/registry-service 8766:8081 -n $(NAMESPACE)
