#!/usr/bin/env bash
# Install local git hooks from scripts/hooks into .git/hooks.
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/scripts/hooks"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

if [ ! -d "$HOOKS_DIR" ]; then
  echo "❌ No scripts/hooks directory found at $HOOKS_DIR"
  exit 1
fi

mkdir -p "$GIT_HOOKS_DIR"

installed=""
for hook in pre-commit commit-msg; do
  src="$HOOKS_DIR/$hook"
  dest="$GIT_HOOKS_DIR/$hook"
  if [ -f "$src" ]; then
    cp "$src" "$dest"
    chmod +x "$dest"
    installed="$installed $hook"
  fi
done

if [ -n "$installed" ]; then
  echo "✅ Git hooks installed:$installed"
else
  echo "⚠️ No hooks found in $HOOKS_DIR"
fi
