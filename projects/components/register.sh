#!/bin/bash
set -e

# Ensure we are in the right directory
cd "$(dirname "$0")/../.."

# ── Version configuration ─────────────────────────────────────
# Source the central versions file so images and component versions
# stay in sync with what was actually built and pushed.
# shellcheck source=projects/components/images/versions.env
_versions_tmp=$(mktemp)
trap 'rm -f "$_versions_tmp"' EXIT
sed -E 's/^([[:space:]]*[A-Za-z_][A-Za-z0-9_]*)[[:space:]]*[:?]=[[:space:]]*(.*)/\1=\2/' projects/components/images/versions.env > "$_versions_tmp"
source "$_versions_tmp"
rm -f "$_versions_tmp"

# Component version — when set explicitly, all components share this version.
# When empty (default), each component gets a content-hash version so that
# unchanged components are automatically skipped by Flyte.
COMPONENT_VERSION="${COMPONENT_VERSION:-}"

# Resolved image URIs (mirrors the Makefile tag scheme)
DATA_CPU_IMAGE="${ECR_REGISTRY}/${ECR_REPO}/data-cpu:${IMAGE_TAG}"
ML_GPU_IMAGE="${ECR_REGISTRY}/${ECR_REPO}/ml-gpu:${IMAGE_TAG}"
GENAI_GPU_IMAGE="${ECR_REGISTRY}/${ECR_REPO}/genai-gpu:${IMAGE_TAG}"
TRAINING_LLM_IMAGE="${ECR_REGISTRY}/${ECR_REPO}/training-llm:${IMAGE_TAG}"

echo "Registering Shared Components to Flyte (In-Cluster)..."
echo "  IMAGE_TAG          = ${IMAGE_TAG}"
if [ -n "${COMPONENT_VERSION}" ]; then
    echo "  COMPONENT_VERSION  = ${COMPONENT_VERSION} (global override)"
else
    echo "  COMPONENT_VERSION  = <content-hash per component>"
fi
echo "  data-cpu image     = ${DATA_CPU_IMAGE}"
echo "  ml-gpu image       = ${ML_GPU_IMAGE}"
echo "  genai-gpu image    = ${GENAI_GPU_IMAGE}"
echo "  training-llm image = ${TRAINING_LLM_IMAGE}"

POD_NAME="registrar"
NAMESPACE="flyte"
IMAGE="python:3.12-slim"

# 1. Start a registrar pod if not exists
if ! kubectl get pod $POD_NAME -n $NAMESPACE > /dev/null 2>&1; then
    echo "Starting registrar pod..."
    kubectl run $POD_NAME -n $NAMESPACE --image=$IMAGE --restart=Never -- sleep 3600
    echo "Waiting for pod to be ready..."
    kubectl wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=60s
fi

# 2. Copy SDK and component source trees to pod
echo "Copying SDK code to pod..."
kubectl exec -n $NAMESPACE $POD_NAME -- mkdir -p /tmp/sdk /tmp/components
kubectl cp projects/components/sdk/ml_platform_sdk $NAMESPACE/$POD_NAME:/tmp/sdk/ml_platform_sdk

# Copy each category directory individually so /tmp/components/{category}/{comp}/ exists
for cat_dir in projects/components/components/*/; do
    [ -d "$cat_dir" ] || continue
    cat_name=$(basename "$cat_dir")
    # Skip __pycache__ and services
    [[ "$cat_name" == "__pycache__" || "$cat_name" == "services" ]] && continue
    kubectl cp "$cat_dir" "$NAMESPACE/$POD_NAME:/tmp/components/$cat_name"
done

# 3. Install dependencies and register each component with content-hash versioning
echo "Installing dependencies and registering..."
kubectl exec -n $NAMESPACE $POD_NAME -- /bin/bash -c "
    pip install -q flytekit==${FLYTEKIT_VERSION} && \
    mkdir -p ~/.flyte && \
    cat > ~/.flyte/config.yaml << 'EOF'
admin:
  endpoint: dns:///flyte-binary-grpc.flyte.svc.cluster.local:8089
  insecure: true
EOF

    # ── Per-component registration with content-hash versioning ──────────
    # Maps each image short-name to its full URI.
    declare -A IMAGE_MAP
    IMAGE_MAP[ml-gpu]='${ML_GPU_IMAGE}'
    IMAGE_MAP[data-cpu]='${DATA_CPU_IMAGE}'
    IMAGE_MAP[genai-gpu]='${GENAI_GPU_IMAGE}'
    IMAGE_MAP[training-llm]='${TRAINING_LLM_IMAGE}'

    SKIPPED=0
    REGISTERED=0
    FAILED=0

    # content_hash: compute a deterministic SHA-256 hash of all files in a directory
    content_hash() {
        find \"\$1\" -type f ! -path '*/__pycache__/*' | sort | while read -r f; do
            echo \"\$(basename \"\$f\")\"
            cat \"\$f\"
        done | sha256sum | cut -c1-12
    }

    for comp_dir in /tmp/components/*/; do
        # Each category dir (data, training, etc.) contains component sub-dirs
        for sub_dir in \"\${comp_dir}\"*/; do
            [ -d \"\$sub_dir\" ] || continue
            [ -f \"\${sub_dir}component.yaml\" ] || continue

            comp_name=\$(grep '^name:' \"\${sub_dir}component.yaml\" | head -1 | awk '{print \$2}')
            comp_image=\$(grep '^image:' \"\${sub_dir}component.yaml\" | head -1 | awk '{print \$2}')
            comp_image=\${comp_image:-ml-gpu}
            resolved_image=\${IMAGE_MAP[\$comp_image]:-\${IMAGE_MAP[ml-gpu]}}

            # Use explicit version or content-hash
            if [ -n '${COMPONENT_VERSION}' ]; then
                ver='${COMPONENT_VERSION}'
            else
                ver=\$(content_hash \"\$sub_dir\")
            fi

            category=\$(basename \"\$(dirname \"\$sub_dir\")\")
            echo \"Registering \${category}/\${comp_name} (version=\${ver}, image=\${comp_image})...\"

            if pyflyte register \
                --project ml-platform \
                --domain development \
                --image \"\$resolved_image\" \
                --version \"\$ver\" \
                \"\$sub_dir\" 2>&1; then
                REGISTERED=\$((REGISTERED + 1))
            else
                FAILED=\$((FAILED + 1))
            fi
        done
    done

    echo \"\"
    echo \"Registration summary: \${REGISTERED} registered, \${FAILED} failed\"
"

echo "✅ Registration Complete!"
echo "Cleaning up..."
kubectl delete pod $POD_NAME -n $NAMESPACE --force --grace-period=0

