#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-Luxury}"
GPU_ID="${GPU_ID:-0}"
STAGE2_MODEL="${STAGE2_MODEL:-meta-llama/Llama-3.2-1B}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"

DATASET_DIR="${PROJECT_ROOT}/data/processed/${DATASET}"
SID_PATH="${PROJECT_ROOT}/data/sids/${DATASET}/${DATASET}.sid-set-token-weight-stage1.json"
LOG_PATH="${PROJECT_ROOT}/logs/LLMSetSID/${DATASET}/itemrank.log"
SAVE_PATH="${PROJECT_ROOT}/checkpoints/LLMSetSID/${DATASET}/itemrank"

for split in train valid test; do
  split_path="${DATASET_DIR}/${DATASET}.${split}.inter"
  if [[ ! -f "${split_path}" ]]; then
    echo "Missing Stage 2 input: ${split_path}" >&2
    exit 1
  fi
done

if [[ ! -f "${SID_PATH}" ]]; then
  echo "Missing Stage 1 SID artifact: ${SID_PATH}" >&2
  echo "Run scripts/run_stage1.sh first." >&2
  exit 1
fi

args=(
  -m stage2.run
  --dataset_path "${DATASET_DIR}"
  --train_file "${DATASET}.train.inter"
  --valid_file "${DATASET}.valid.inter"
  --test_file "${DATASET}.test.inter"
  --sid_json "${SID_PATH}"
  --model_name_or_path "${STAGE2_MODEL}"
  --device "cuda:${GPU_ID}"
  --log_path "${LOG_PATH}"
  --save_path "${SAVE_PATH}"
  --topm_tokens 0
)

if [[ -n "${HF_CACHE_DIR}" ]]; then
  args+=(--model_cache_dir "${HF_CACHE_DIR}")
fi

"${PYTHON_BIN}" "${args[@]}" "$@"
