# Makefile for ML Platform Dashboard
# NOTE: Docker image builds are handled by CI in xiaohanhuang/docker-images.
# Pushing to main auto-syncs projects/dashboard/ to that repo, which then
# builds and pushes ml-platform/dashboard:latest to ECR via GitHub Actions.
# This Makefile only manages local development and Kubernetes deployment.

.PHONY: dev deploy install clean logs status

ECR_REGISTRY ?= 805673386114.dkr.ecr.us-west-2.amazonaws.com
IMAGE_NAME = ml-platform/dashboard
IMAGE_TAG ?= latest

dev:
	@echo "Starting local dev server..."
	npm run dev

deploy:
	@echo "Deploying dashboard to Kubernetes..."
	kubectl apply -f deployment.yaml

install: deploy
	@echo "Dashboard deployed (image built by CI in xiaohanhuang/docker-images)"

clean:
	@echo "Cleaning up..."
	kubectl delete -f deployment.yaml --ignore-not-found=true

logs:
	@echo "Fetching dashboard logs..."
	kubectl logs -n ml-platform -l app=ml-platform-dashboard --tail=100 -f

status:
	@echo "Dashboard status:"
	kubectl get deployment,pod,service,ingress -n ml-platform -l app=ml-platform-dashboard
