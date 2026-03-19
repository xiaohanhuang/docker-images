NAMESPACE := kuberay
CHART_REPO := https://ray-project.github.io/kuberay-helm
RELEASE_NAME := kuberay-operator
VERSION := 1.0.0
IMAGE_NAME := ml-platform/ray-worker
IMAGE_TAG := latest

.PHONY: install upgrade uninstall status build-image

install:
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	helm repo add kuberay $(CHART_REPO)
	helm repo update
	helm upgrade --install $(RELEASE_NAME) kuberay/kuberay-operator \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml

upgrade:
	helm upgrade $(RELEASE_NAME) kuberay/kuberay-operator \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml

uninstall:
	helm uninstall $(RELEASE_NAME) --namespace $(NAMESPACE)

status:
	kubectl get pods -n $(NAMESPACE)
	kubectl get rayclusters --all-namespaces

build-image:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f images/Dockerfile ../components/images/base-gpu/
