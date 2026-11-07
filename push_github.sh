#!/usr/bin/env bash
# Push this repo to GitHub with a token that has write access.
#
# The sandbox's GITHUB_TOKEN is read-only, so it cannot push. Create a
# fine-grained Personal Access Token with:
#     Repository access : sumondryxyz1/Quotex-bot
#     Permissions       : Contents = Read and write
# https://github.com/settings/personal-access-tokens/new
#
# Then either:
#   ./push_github.sh <TOKEN>
#   GITHUB_WRITE_TOKEN=<TOKEN> ./push_github.sh
#
# The token is used once and is never written to .git/config.
set -eu
cd "$(dirname "$0")"

TOKEN="${1:-${GITHUB_WRITE_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
  echo "usage: $0 <token>   (or set GITHUB_WRITE_TOKEN)" >&2
  exit 1
fi

REPO="${REPO:-sumondryxyz1/Quotex-bot}"
BRANCH="$(git branch --show-current)"

git remote set-url origin "https://${TOKEN}@github.com/${REPO}.git"
trap 'git remote set-url origin "https://github.com/${REPO}.git"' EXIT

echo "pushing ${BRANCH} -> ${REPO} ..."
git push -u origin "${BRANCH}"

echo
echo "done. open: https://github.com/${REPO}"
echo "next: deploy a permanent link -> see DEPLOY.md"
