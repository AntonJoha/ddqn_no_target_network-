#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python ddqn_lunar_lander.py \
  --env-id LunarLander-v3 \
  --activation tgelu \
  --target-network-countdown 25 \
  "$@"
