NAMESPACE := spark-operator
CHART_REPO := https://kubeflow.github.io/spark-operator
RELEASE_NAME := spark-operator
VERSION := 1.1.27
IMAGE_NAME := ml-platform/base-spark
IMAGE_TAG := latest

.PHONY: install upgrade uninstall status build-image

install:
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	helm repo add spark-operator $(CHART_REPO)
	helm repo update
	helm upgrade --install $(RELEASE_NAME) spark-operator/spark-operator \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml

upgrade:
	helm upgrade $(RELEASE_NAME) spark-operator/spark-operator \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml

uninstall:
	helm uninstall $(RELEASE_NAME) --namespace $(NAMESPACE)

status:
	kubectl get pods -n $(NAMESPACE)
	kubectl get sparkapplications --all-namespaces

build-image:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f images/Dockerfile .
