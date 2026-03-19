REGISTRY   := 805673386114.dkr.ecr.us-west-2.amazonaws.com
REGION     := us-west-2
CPU_IMAGE  := $(REGISTRY)/ml-platform/workflow-cpu:latest
GPU_IMAGE  := $(REGISTRY)/ml-platform/workflow-gpu:latest
SERVE_IMAGE:= $(REGISTRY)/ml-platform/text2sql-serve:latest
PLATFORM   := linux/amd64

FLYTE_PROJECT := flytesnacks
FLYTE_DOMAIN  := development
FLYTE_VERSION := $(shell git rev-parse --short HEAD 2>/dev/null || echo "v1")

.PHONY: ecr-login build-cpu build-gpu build-serve build-all push-all \
        register run serve status

## ── Auth ──────────────────────────────────────────────────────────────
ecr-login:
	AWS_PROFILE=adfs aws ecr get-login-password --region $(REGION) \
	  | docker login --username AWS --password-stdin $(REGISTRY)

## ── Build ─────────────────────────────────────────────────────────────
build-cpu:
	docker buildx build --platform $(PLATFORM) \
	  -f images/Dockerfile.cpu \
	  -t $(CPU_IMAGE) . --load

build-gpu:
	docker buildx build --platform $(PLATFORM) \
	  -f images/Dockerfile.gpu \
	  -t $(GPU_IMAGE) . --load

build-serve:
	docker buildx build --platform $(PLATFORM) \
	  -f serve/Dockerfile \
	  -t $(SERVE_IMAGE) serve/ --load

build-all: build-cpu build-gpu build-serve

## ── Push ──────────────────────────────────────────────────────────────
push-all: ecr-login
	docker push $(CPU_IMAGE)
	docker push $(GPU_IMAGE)
	docker push $(SERVE_IMAGE)

## ── Flyte: create project + register workflow ─────────────────────────
flyte-project:
	@kubectl port-forward svc/flyte-binary-http 8088:8088 -n flyte &
	@sleep 2
	flytectl create project --id $(FLYTE_PROJECT) \
	  --name "ML Platform" --description "Text2SQL and ML Workflows" \
	  --config /tmp/flytectl.yaml 2>/dev/null || true

register:
	@echo "Registering workflow (version: $(FLYTE_VERSION))..."
	@kubectl port-forward svc/flyte-binary-grpc 8089:8089 -n flyte &
	@sleep 2
	pyflyte register \
	  --project $(FLYTE_PROJECT) \
	  --domain  $(FLYTE_DOMAIN) \
	  --version $(FLYTE_VERSION) \
	  --image   $(CPU_IMAGE) \
	  pipeline.py
	@echo "✅ Workflow registered as version $(FLYTE_VERSION)"

## ── Deploy inference server ───────────────────────────────────────────
serve:
	@test -n "$(RUN_ID)" || (echo "Usage: make serve RUN_ID=<mlflow-run-id>"; exit 1)
	kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text2sql-serve
  namespace: flyte
spec:
  replicas: 1
  selector:
    matchLabels:
      app: text2sql-serve
  template:
    metadata:
      labels:
        app: text2sql-serve
    spec:
      nodeSelector:
        role: system
      containers:
      - name: server
        image: $(SERVE_IMAGE)
        env:
        - name: S3_BUCKET
          value: ml-platform-data-ml-platform-eks-805673386114
        - name: MODEL_RUN_ID
          value: $(RUN_ID)
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: text2sql-serve
  namespace: flyte
spec:
  selector:
    app: text2sql-serve
  ports:
  - port: 8080
    targetPort: 8080
EOF

## ── Status ────────────────────────────────────────────────────────────
status:
	@echo "=== JupyterHub ==="
	@kubectl get pods -n jupyter
	@echo "\n=== Flyte ==="
	@kubectl get pods -n flyte
	@echo "\n=== Inference Server ==="
	@kubectl get pods -n flyte -l app=text2sql-serve 2>/dev/null || echo "  (not deployed)"
