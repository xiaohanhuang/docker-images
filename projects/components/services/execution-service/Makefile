# Makefile for ML Platform Execution Service
# NOTE: Docker image builds are handled by CI in xiaohanhuang/docker-images.
# This Makefile only manages Kubernetes deployment.

.PHONY: deploy install uninstall status logs port-forward

IMAGE_NAME := ml-platform/execution-service
VERSION ?= $(error VERSION must be set (e.g. make deploy VERSION=v0.1.0))
ECR_REGISTRY ?= $(error ECR_REGISTRY must be set)
FULL_IMAGE := $(ECR_REGISTRY)/$(IMAGE_NAME):$(VERSION)
NAMESPACE := default

deploy:
	@echo "Deploying execution service to Kubernetes..."
	cd k8s && kustomize edit set image $(IMAGE_NAME)=$(FULL_IMAGE)
	kubectl apply -k k8s/
	@echo "Waiting for rollout..."
	kubectl rollout status deployment/execution-service -n $(NAMESPACE)

install: deploy

uninstall:
	@echo "Uninstalling execution service..."
	kubectl delete -k k8s/

status:
	@echo "Execution service status:"
	kubectl get pods -l app=execution-service -n $(NAMESPACE)
	kubectl get svc execution-service -n $(NAMESPACE)

logs:
	@echo "Execution service logs (last 50 lines):"
	kubectl logs -l app=execution-service -n $(NAMESPACE) --tail=50 -f

port-forward:
	@echo "Port-forwarding execution service to localhost:8765..."
	kubectl port-forward svc/execution-service 8765:8080 -n $(NAMESPACE)
