.PHONY: build push

IMAGE_NAME := ml-platform/executor-pool
ECR_REGISTRY := 805673386114.dkr.ecr.us-west-2.amazonaws.com
VERSION := latest

build:
	docker build -t $(IMAGE_NAME):$(VERSION) .
	docker tag $(IMAGE_NAME):$(VERSION) $(ECR_REGISTRY)/$(IMAGE_NAME):$(VERSION)

push: build
	docker push $(ECR_REGISTRY)/$(IMAGE_NAME):$(VERSION)
