#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${NPROC_PER_NODE:-1}" == 1 ]]; then
  exec python train.py "$@"
fi
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py "$@"
