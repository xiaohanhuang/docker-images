NAMESPACE := keda
CHART_REPO := https://kedacore.github.io/charts
RELEASE_NAME := keda
VERSION := 2.16.1

.PHONY: install upgrade uninstall status

install:
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	helm repo add kedacore $(CHART_REPO)
	helm repo update
	helm upgrade --install $(RELEASE_NAME) kedacore/keda \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml

upgrade:
	helm upgrade $(RELEASE_NAME) kedacore/keda \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml

uninstall:
	helm uninstall $(RELEASE_NAME) --namespace $(NAMESPACE)

status:
	kubectl get pods -n $(NAMESPACE)
	kubectl get scaledobjects --all-namespaces
	kubectl get scaledjobs --all-namespaces
