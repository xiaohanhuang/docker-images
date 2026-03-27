#!/usr/bin/env bash
# sync_source.sh — Copies platform source code from the main ml-platform repo
# into the docker-images-repo for Docker build context.
#
# This script is run by CI before building Docker images.
# The synced directories are git-ignored and should NOT be committed.
#
# Usage:
#   ./sync_source.sh /path/to/ml-platform
#   ./sync_source.sh  # defaults to ../  (assumes repos are siblings)

set -euo pipefail

SOURCE_REPO="${1:-../}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Validate source repo
if [[ ! -d "$SOURCE_REPO/cli" || ! -d "$SOURCE_REPO/backend" ]]; then
    echo "❌ Error: Cannot find ml-platform source at '$SOURCE_REPO'"
    echo "   Expected to find cli/ and backend/ directories."
    echo "   Usage: $0 /path/to/ml-platform"
    exit 1
fi

echo "🔄 Syncing source code from: $SOURCE_REPO"
echo "   Into docker context: $SCRIPT_DIR"

# Sync top-level source dirs
rsync -a --delete "$SOURCE_REPO/cli/" "$SCRIPT_DIR/cli/"
rsync -a --delete "$SOURCE_REPO/backend/" "$SCRIPT_DIR/backend/"
rsync -a --delete "$SOURCE_REPO/pyproject.toml" "$SCRIPT_DIR/pyproject.toml"

# Sync project sub-components
rsync -a --delete "$SOURCE_REPO/projects/components/sdk/" "$SCRIPT_DIR/projects/components/sdk/"
rsync -a --delete "$SOURCE_REPO/projects/components/components/" "$SCRIPT_DIR/projects/components/components/"

# Sync services (source files only — Dockerfiles are already in this repo)
for svc in execution-service registry-service; do
    svc_src="$SOURCE_REPO/projects/components/services/$svc/"
    svc_dst="$SCRIPT_DIR/projects/components/services/$svc/"
    if [[ -d "$svc_src" ]]; then
        # Sync .py files and requirements.txt, preserve existing Dockerfiles
        rsync -a --include='*.py' --include='requirements.txt' --exclude='Dockerfile' --exclude='Makefile' --exclude='README.md' "$svc_src" "$svc_dst"
    fi
done

echo "✅ Source sync complete."
echo ""
echo "Synced directories:"
echo "  cli/                                 -> $(find "$SCRIPT_DIR/cli" -name '*.py' | wc -l | xargs) .py files"
echo "  backend/                             -> $(find "$SCRIPT_DIR/backend" -name '*.py' | wc -l | xargs) .py files"
echo "  projects/components/sdk/             -> $(find "$SCRIPT_DIR/projects/components/sdk" -name '*.py' | wc -l | xargs) .py files"
echo "  projects/components/components/      -> $(find "$SCRIPT_DIR/projects/components/components" -name '*.py' 2>/dev/null | wc -l | xargs) .py files"
echo "  projects/components/services/        -> $(find "$SCRIPT_DIR/projects/components/services" -name '*.py' | wc -l | xargs) .py files"
