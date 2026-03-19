NAMESPACE := monitoring
KUBECOST_VERSION := 2.3.0

# Resolve MLFLOW_PASSWORD from env, falling back to reading it from cluster Secret
MLFLOW_PASSWORD ?= $(shell kubectl get secret mlflow-db-credentials -n monitoring \
  -o jsonpath='{.data.POSTGRES_PASSWORD}' 2>/dev/null | base64 -d)

.PHONY: install-prometheus install-mlflow install-dcgm install-kubecost install-all status init-mlflow-db

install-prometheus:
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update
	helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
		--namespace $(NAMESPACE) \
		--version 56.0.0 \
		-f helm-values-prometheus.yaml

install-mlflow:
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@echo "Checking for MLflow database credentials Secret..."
	@if ! kubectl get secret mlflow-db-credentials -n $(NAMESPACE) >/dev/null 2>&1; then \
		echo "ERROR: Secret 'mlflow-db-credentials' not found in namespace '$(NAMESPACE)'."; \
		echo "Create it using:"; \
		echo "  kubectl create secret generic mlflow-db-credentials \\"; \
		echo "    -n $(NAMESPACE) \\"; \
		echo "    --from-literal=POSTGRES_PASSWORD='<your-mlflow-password>'"; \
		exit 1; \
	fi
	@if [ -z "$(MLFLOW_PASSWORD)" ]; then \
		echo "ERROR: MLFLOW_PASSWORD is empty. Ensure the Secret contains POSTGRES_PASSWORD key."; \
		exit 1; \
	fi
	helm repo add community-charts https://community-charts.github.io/helm-charts
	helm repo update
	helm upgrade --install mlflow community-charts/mlflow \
		--namespace $(NAMESPACE) \
		--version 1.8.1 \
		-f helm-values-mlflow.yaml \
		--set-string backendStore.postgres.password="$(MLFLOW_PASSWORD)"

init-mlflow-db:
	@echo "Initializing MLflow database on PostgreSQL..."
	@if ! kubectl get secret postgres-credentials -n postgres >/dev/null 2>&1; then \
		echo "ERROR: Secret 'postgres-credentials' not found in namespace 'postgres'."; \
		exit 1; \
	fi
	@if ! kubectl get secret mlflow-db-credentials -n postgres >/dev/null 2>&1; then \
		echo "ERROR: Secret 'mlflow-db-credentials' not found in namespace 'postgres'."; \
		echo "Create it using:"; \
		echo "  kubectl create secret generic mlflow-db-credentials \\"; \
		echo "    -n postgres \\"; \
		echo "    --from-literal=MLFLOW_PASSWORD='<your-mlflow-password>'"; \
		exit 1; \
	fi
	kubectl apply -f postgres-init-mlflow.yaml
	@echo "Waiting for init job to complete..."
	kubectl wait --for=condition=complete --timeout=120s job/postgres-init-mlflow -n postgres || \
		(kubectl logs -n postgres job/postgres-init-mlflow && exit 1)
	@echo "MLflow database initialized successfully"

install-dcgm:
	helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
	helm repo update
	helm upgrade --install dcgm-exporter gpu-helm-charts/dcgm-exporter \
		--namespace $(NAMESPACE) \
		--version 4.8.1 \
		--set serviceMonitor.enabled=true \
		--set tolerations[0].key=nvidia.com/gpu \
		--set tolerations[0].operator=Exists \
		--set tolerations[0].effect=NoSchedule \
		--set nodeSelector.role=gpu-worker

install-kubecost:
	helm repo add kubecost https://kubecost.github.io/cost-analyzer/
	helm repo update
	helm upgrade --install kubecost kubecost/cost-analyzer \
		--namespace kubecost --create-namespace \
		--version $(KUBECOST_VERSION) \
		-f kubecost-values.yaml
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap grafana-dashboard-kubecost \
		--from-file=kubecost-gpu-allocation.json=grafana-kubecost-gpu-dashboard.json \
		--namespace $(NAMESPACE) \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl label configmap grafana-dashboard-kubecost \
		grafana_dashboard=1 \
		--namespace $(NAMESPACE) \
		--overwrite

install-all: install-prometheus install-mlflow install-dcgm install-loki install-kubecost
	@echo "All monitoring components installed."

install-loki:
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	helm repo add grafana https://grafana.github.io/helm-charts
	helm repo update
	helm upgrade --install loki grafana/loki-stack \
		--namespace $(NAMESPACE) \
		--version 2.10.3 \
		-f helm-values-loki.yaml

status:
	kubectl get pods -n $(NAMESPACE)
