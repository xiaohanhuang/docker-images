#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="$1"
DOCKERFILE="$2"
CONTEXT="$3"
shift 3
BUILD_ARGS=("$@")

# Hashing inputs
SHASUM=$(cat "$DOCKERFILE" images/versions.env 2>/dev/null | sha256sum | awk '{print $1}')
TAG="sha-${SHASUM:0:16}"

echo "Building $REPO_NAME (Digest: $TAG)"

# Check if image exists
if aws ecr describe-images --repository-name "ml-platform/$REPO_NAME" --image-ids imageTag="$TAG" --region us-west-2 &>/dev/null; then
  echo "✅ Image ml-platform/$REPO_NAME:$TAG already exists in ECR. Skipping build."
  
  # Just retag it as latest
  MANIFEST=$(aws ecr batch-get-image --repository-name "ml-platform/$REPO_NAME" --image-ids imageTag="$TAG" --region us-west-2 --query 'images[0].imageManifest' --output text)
  aws ecr put-image --repository-name "ml-platform/$REPO_NAME" --image-manifest "$MANIFEST" --image-tag latest --region us-west-2 >/dev/null 2>&1 || true
  
  # Also retag with branch if branch is passed via env
  if [[ -n "${TARGET_BRANCH:-}" && "${TARGET_BRANCH}" != "main" ]]; then
    aws ecr put-image --repository-name "ml-platform/$REPO_NAME" --image-manifest "$MANIFEST" --image-tag "branch-${TARGET_BRANCH}" --region us-west-2 >/dev/null 2>&1 || true
  fi
  
  echo "is_rebuilt=false" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "🔨 Building ml-platform/$REPO_NAME..."
# Extract just the tag arguments from BUILD_ARGS
TAG_ARGS="-t ${ECR_REGISTRY}/ml-platform/${REPO_NAME}:latest -t ${ECR_REGISTRY}/ml-platform/${REPO_NAME}:${TAG}"
if [[ -n "${TARGET_BRANCH:-}" && "${TARGET_BRANCH}" != "main" ]]; then
  TAG_ARGS="$TAG_ARGS -t ${ECR_REGISTRY}/ml-platform/${REPO_NAME}:branch-${TARGET_BRANCH}"
  # We still tag :latest so caching works, but in reality we shouldn't overwrite latest on branch pushes.
  # Let's fix that:
  if [[ "${TARGET_BRANCH}" != "main" ]]; then
    TAG_ARGS="-t ${ECR_REGISTRY}/ml-platform/${REPO_NAME}:branch-${TARGET_BRANCH} -t ${ECR_REGISTRY}/ml-platform/${REPO_NAME}:${TAG}"
  fi
fi

# We use buildx
docker buildx build \
  --push \
  --context "$CONTEXT" \
  --file "$DOCKERFILE" \
  --platform linux/amd64 \
  --cache-from type=gha \
  --cache-to type=gha,mode=max \
  $TAG_ARGS \
  "${BUILD_ARGS[@]}"

echo "is_rebuilt=true" >> "$GITHUB_OUTPUT"
