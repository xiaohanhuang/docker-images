NAMESPACE := flyte
CHART_REPO := https://flyteorg.github.io/flyte
CHART_NAME := flyte-binary
RELEASE_NAME := flyte-binary
VERSION := v1.10.0

# Resolve PG_PASSWORD from env, falling back to reading it from the cluster Secret.
# Export PG_PASSWORD before running make, or ensure the postgres-credentials
# Secret already exists in the cluster.
PG_PASSWORD ?= $(shell kubectl get secret postgres-credentials -n postgres \
	-o jsonpath='{.data.POSTGRES_PASSWORD}' 2>/dev/null | base64 -d)

.PHONY: install upgrade uninstall status

install:
	@test -n "$(PG_PASSWORD)" || \
		(echo "ERROR: PG_PASSWORD is not set. Export it or ensure postgres-credentials exists." && exit 1)
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	helm repo add flyte $(CHART_REPO)
	helm repo update
	helm upgrade --install $(RELEASE_NAME) flyte/$(CHART_NAME) \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml \
		--set configuration.database.password="$(PG_PASSWORD)"

upgrade:
	@test -n "$(PG_PASSWORD)" || \
		(echo "ERROR: PG_PASSWORD is not set. Export it or ensure postgres-credentials exists." && exit 1)
	helm upgrade $(RELEASE_NAME) flyte/$(CHART_NAME) \
		--namespace $(NAMESPACE) \
		--version $(VERSION) \
		-f helm-values.yaml \
		--set configuration.database.password="$(PG_PASSWORD)"

uninstall:
	helm uninstall $(RELEASE_NAME) --namespace $(NAMESPACE)

status:
	kubectl get pods -n $(NAMESPACE)
