.PHONY: build push deploy

IMAGE_NAME = ml-platform/backend-api
IMAGE_TAG ?= latest
ECR_REGISTRY ?= $(error ECR_REGISTRY must be set)

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile ../..

push: build
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(ECR_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(ECR_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

deploy:
	kubectl apply -f deployment.yaml

install: push deploy
	@echo "Backend API deployed successfully"

uninstall:
	kubectl delete -f deployment.yaml

logs:
	kubectl logs -n ml-platform -l app=ml-platform-api --tail=100 -f
