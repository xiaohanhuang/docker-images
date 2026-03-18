.PHONY: build push deploy install clean

ECR_REGISTRY ?= 805673386114.dkr.ecr.us-west-2.amazonaws.com
IMAGE_NAME = ml-platform/dashboard
IMAGE_TAG ?= latest

build:
	@echo "Building dashboard Docker image..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f Dockerfile \
		../../

push: build
	@echo "Pushing dashboard image to ECR..."
	aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $(ECR_REGISTRY)
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(ECR_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(ECR_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

deploy:
	@echo "Deploying dashboard to Kubernetes..."
	kubectl apply -f deployment.yaml

install: push deploy
	@echo "Dashboard installed successfully"

clean:
	@echo "Cleaning up..."
	kubectl delete -f deployment.yaml --ignore-not-found=true

logs:
	@echo "Fetching dashboard logs..."
	kubectl logs -n ml-platform -l app=ml-platform-dashboard --tail=100 -f

status:
	@echo "Dashboard status:"
	kubectl get deployment,pod,service,ingress -n ml-platform -l app=ml-platform-dashboard
