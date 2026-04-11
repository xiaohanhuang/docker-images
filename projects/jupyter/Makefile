NAMESPACE := jupyter
CHART_REPO := https://hub.jupyter.org/helm-chart/
RELEASE_NAME := jupyterhub
VERSION := 4.3.2

.PHONY: install upgrade uninstall status build-images build-monitor build-vscode create-secret

install:
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f ../eks/gp3-sc.yaml || true
	kubectl apply -f service-account.yaml
	$(MAKE) create-secret
	helm repo add jupyterhub $(CHART_REPO)
	helm repo update
	helm upgrade --install $(RELEASE_NAME) jupyterhub/jupyterhub \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml

upgrade:
	$(MAKE) create-secret
	helm upgrade $(RELEASE_NAME) jupyterhub/jupyterhub \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml

uninstall:
	helm uninstall $(RELEASE_NAME) --namespace $(NAMESPACE)

build-images:
	docker build -t ml-platform/notebook:1.0.0 -f images/Dockerfile ../../
	docker build -t ml-platform/notebook-gpu:1.0.0 -f images/Dockerfile.gpu ../../

build-vscode:
	docker build -t ml-platform/notebook-marimo-vscode:1.0.0 -f images/Dockerfile.marimo.vscode ../../

build-notebook-cpu:
	docker build -t ml-platform/notebook-cpu:1.0.0 -f images/Dockerfile.notebook-cpu ../../

build-monitor:
	docker build -t ml-platform/gpu-idle-monitor:1.0.0 gpu-idle-monitor/

status:
	kubectl get pods -n $(NAMESPACE)
	@echo "Access: kubectl port-forward svc/proxy-public 8080:80 -n $(NAMESPACE)"

create-secret:
	@if [ -z "$(PASSWORD)" ]; then \
		echo "ERROR: PASSWORD is required. Run: make create-secret PASSWORD=<your-password>"; \
		exit 1; \
	fi
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl create secret generic jupyterhub-auth-secret \
		--namespace $(NAMESPACE) \
		--from-literal=password="$(PASSWORD)" \
		--dry-run=client -o yaml | kubectl apply -f -
	@echo "Secret jupyterhub-auth-secret created/updated in namespace $(NAMESPACE)"
