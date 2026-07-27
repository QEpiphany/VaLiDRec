#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-Luxury}"
GPU_ID="${GPU_ID:-0}"
STAGE1_MODEL="${STAGE1_MODEL:-meta-llama/Meta-Llama-3.1-8B}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"

DATASET_DIR="${PROJECT_ROOT}/data/processed/${DATASET}"
ITEM_PATH="${DATASET_DIR}/${DATASET}.item.json"
TRAIN_PATH="${DATASET_DIR}/${DATASET}.train.inter"

for required_file in "${ITEM_PATH}" "${TRAIN_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing Stage 1 input: ${required_file}" >&2
    exit 1
  fi
done

args=(
  -m stage1.run
  --dataset "${DATASET}"
  --root "${PROJECT_ROOT}/data/processed"
  --gpu_id "${GPU_ID}"
  --model_name_or_path "${STAGE1_MODEL}"
)

if [[ -n "${HF_CACHE_DIR}" ]]; then
  args+=(--model_cache_dir "${HF_CACHE_DIR}")
fi

"${PYTHON_BIN}" "${args[@]}" "$@"
