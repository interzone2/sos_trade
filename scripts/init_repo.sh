#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/init_repo.sh <github-username> <repo-name>
#
# This will:
#  - initialize git
#  - create a new remote on GitHub (requires gh CLI: https://cli.github.com/)
#  - push main branch
#
# Example:
#   ./scripts/init_repo.sh yourname sos-trade

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <github-username> <repo-name>"
  exit 1
fi

USER="$1"
REPO="$2"

# init git if needed
git init -b main
git add .
git commit -m "Initial commit: SoS scaffolding"

# create repo on GitHub (private by default; change --private to --public if you prefer)
if ! command -v gh >/dev/null 2>&1; then
  echo "Missing 'gh' CLI. Install from https://cli.github.com/ or use the manual steps in README."
  exit 1
fi

gh repo create "$USER/$REPO" --private --source=. --remote=origin --push

echo "Done. Repo: https://github.com/$USER/$REPO"
