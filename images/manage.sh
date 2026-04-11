#!/usr/bin/env bash
# ============================================================
# images/manage.sh — Image management utility for platform team
#
# Usage:
#   ./images/manage.sh sync [--only ml-gpu,training-llm] [--force]
#   ./images/manage.sh clean-ecr [--dry-run] [--older-than 90] [--untagged-only]
#   ./images/manage.sh list
# ============================================================
set -euo pipefail

REPO="xiaohanhuang/docker-images"
ECR_REGISTRY="805673386114.dkr.ecr.us-west-2.amazonaws.com"
ECR_REPO="ml-platform"
REGION="us-west-2"

# All ECR image names managed by this tool
ALL_IMAGES=(
  base-cpu base-gpu
  flyte-cpu flyte-gpu ray-gpu
  data-cpu ml-gpu genai-gpu training-llm
  executor-pool executor-pool-cpu
  notebook-cpu notebook-marimo notebook-marimo-vscode
  gpu-idle-monitor spark-base
  backend-api dashboard
  execution-service registry-service
)

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  sync         Trigger a rebuild in the docker-images repo
  clean-ecr    Clean old/unused images from ECR
  list         List all ECR images with tags and ages

Sync options:
  --only <img,...>   Only rebuild specific images (comma-separated)
  --force            Force rebuild all images regardless of changes

Clean options:
  --dry-run          Show what would be deleted without deleting
  --older-than N     Delete images older than N days (default: 90)
  --untagged-only    Only delete untagged (dangling) manifests
EOF
  exit 1
}

# ── sync command ─────────────────────────────────────────────
cmd_sync() {
  local only="" force="false"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --only)   only="$2"; shift 2 ;;
      --force)  force="true"; shift ;;
      *)        echo "Unknown option: $1"; usage ;;
    esac
  done

  local branch
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

  echo "🔄 Triggering rebuild in $REPO..."
  echo "   Branch: $branch"
  [[ -n "$only" ]] && echo "   Only: $only"
  [[ "$force" == "true" ]] && echo "   Force: yes"

  gh workflow run publish-images-to-ecr.yml \
    --repo "$REPO" \
    --ref main \
    -f source_branch="$branch" \
    -f force="$force" \
    ${only:+-f only_images="$only"}

  echo "✅ Workflow dispatched. Monitor at: https://github.com/$REPO/actions"
}

# ── clean-ecr command ────────────────────────────────────────
cmd_clean_ecr() {
  local dry_run="false" older_than=90 untagged_only="false"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)        dry_run="true"; shift ;;
      --older-than)     older_than="$2"; shift 2 ;;
      --untagged-only)  untagged_only="true"; shift ;;
      *)                echo "Unknown option: $1"; usage ;;
    esac
  done

  local cutoff_date
  cutoff_date=$(date -v-"${older_than}"d +%Y-%m-%dT%H:%M:%S 2>/dev/null \
    || date -d "${older_than} days ago" +%Y-%m-%dT%H:%M:%S)

  echo "🧹 ECR cleanup (older than ${older_than} days, cutoff: $cutoff_date)"
  [[ "$dry_run" == "true" ]] && echo "   DRY RUN — nothing will be deleted"
  echo ""

  local total_deleted=0

  for img in "${ALL_IMAGES[@]}"; do
    local repo_name="$ECR_REPO/$img"

    # Check if repo exists
    if ! aws ecr describe-repositories --repository-names "$repo_name" \
        --region "$REGION" &>/dev/null; then
      continue
    fi

    # Get all image details
    local images_json
    images_json=$(aws ecr describe-images \
      --repository-name "$repo_name" \
      --region "$REGION" \
      --output json 2>/dev/null || echo '{"imageDetails":[]}')

    # Use associative array to deduplicate digests (reset each iteration)
    unset seen_digests
    declare -A seen_digests
    local to_delete=()

    while IFS= read -r line; do
      local digest pushed_at tags
      digest=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['imageDigest'])")
      pushed_at=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('imagePushedAt',''))")
      tags=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(','.join(d.get('imageTags',[])))")

      # Skip already-seen digests
      if [[ -n "${seen_digests[$digest]+x}" ]]; then
        continue
      fi

      # Never delete 'latest'
      if echo "$tags" | grep -q "latest"; then
        continue
      fi

      # If untagged-only mode, skip tagged images
      if [[ "$untagged_only" == "true" && -n "$tags" ]]; then
        continue
      fi

      # Skip branch tags younger than 14 days
      if echo "$tags" | grep -q "branch-"; then
        if python3 - "$pushed_at" <<'PY'
import sys
from datetime import datetime, timezone
if len(sys.argv) < 2 or not sys.argv[1]:
    sys.exit(1)
try:
    pushed_at = datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00"))
except ValueError:
    sys.exit(1)
age = datetime.now(timezone.utc) - pushed_at
sys.exit(0 if age.days < 14 else 1)
PY
        then
          continue
        fi
      fi

      local should_delete="false"

      # Check age (use Python for robust ISO 8601 comparison with timezone handling)
      if [[ -n "$pushed_at" ]] && python3 -c "
import sys
from datetime import datetime, timezone
try:
    pushed = datetime.fromisoformat(sys.argv[1].replace('Z', '+00:00'))
    cutoff = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
    sys.exit(0 if pushed < cutoff else 1)
except Exception:
    sys.exit(1)
" "$pushed_at" "$cutoff_date"; then
        should_delete="true"
      fi

      # Always delete untagged
      if [[ -z "$tags" ]]; then
        should_delete="true"
      fi

      if [[ "$should_delete" == "true" ]]; then
        seen_digests[$digest]=1
        to_delete+=("$digest")
        if [[ "$dry_run" == "true" ]]; then
          if [[ -z "$tags" ]]; then
            echo "  Would delete: $repo_name  <untagged>  pushed=$pushed_at"
          else
            echo "  Would delete: $repo_name  tags=[$tags]  pushed=$pushed_at"
          fi
        fi
      fi
    done < <(echo "$images_json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for img in data.get('imageDetails', []):
    print(json.dumps(img))
")

    if [[ ${#to_delete[@]} -gt 0 && "$dry_run" == "false" ]]; then
      local ids=""
      for d in "${to_delete[@]}"; do
        ids+="imageDigest=$d "
      done
      aws ecr batch-delete-image \
        --repository-name "$repo_name" \
        --image-ids $ids \
        --region "$REGION" \
        --output text
      echo "  Deleted ${#to_delete[@]} images from $repo_name"
    fi

    total_deleted=$(( total_deleted + ${#to_delete[@]} ))
  done

  echo ""
  if [[ "$dry_run" == "true" ]]; then
    echo "Would delete $total_deleted images total."
  else
    echo "Deleted $total_deleted images total."
  fi
}

# ── list command ─────────────────────────────────────────────
cmd_list() {
  echo "📦 ECR images in $ECR_REPO"
  echo ""
  printf "%-25s %-40s %s\n" "IMAGE" "TAG" "PUSHED"
  printf "%-25s %-40s %s\n" "-----" "---" "------"

  for img in "${ALL_IMAGES[@]}"; do
    local repo_name="$ECR_REPO/$img"
    if ! aws ecr describe-repositories --repository-names "$repo_name" \
        --region "$REGION" &>/dev/null; then
      continue
    fi

    aws ecr describe-images \
      --repository-name "$repo_name" \
      --region "$REGION" \
      --query 'sort_by(imageDetails,&imagePushedAt)[*].[imageTags[0],imagePushedAt]' \
      --output text 2>/dev/null | while read -r tag pushed; do
        [[ "$tag" == "None" ]] && tag="<untagged>"
        printf "%-25s %-40s %s\n" "$img" "$tag" "$pushed"
      done
  done
}

# ── main ─────────────────────────────────────────────────────
[[ $# -lt 1 ]] && usage

case "$1" in
  sync)       shift; cmd_sync "$@" ;;
  clean-ecr)  shift; cmd_clean_ecr "$@" ;;
  list)       shift; cmd_list "$@" ;;
  *)          usage ;;
esac
