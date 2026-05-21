#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
notes="${1:-branch=${branch}; commit=${commit}}"

python benchmarks/benchmark_tts.py \
  --text "Hi there! How can I help you" \
  --notes "${notes}"
