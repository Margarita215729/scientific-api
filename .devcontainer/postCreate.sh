#!/usr/bin/env bash
set -euo pipefail

if [[ -f /etc/os-release ]]; then
  . /etc/os-release
else
  echo "Cannot detect OS (missing /etc/os-release)" >&2
  exit 1
fi

if [[ "${ID:-}" == "alpine" ]]; then
  apk add --no-cache git-lfs
else
  apt-get update && apt-get install -y git-lfs
fi

git lfs install
git lfs version