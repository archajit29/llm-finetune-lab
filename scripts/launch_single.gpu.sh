#!/usr/bin/env bash
#
# launch_single_gpu.sh
#
# Runs src/train.py directly (no torchrun) for the single_gpu cluster config.
# Intended for local dev / debugging / smoke-testing on one GPU.
#
# Usage:
#   ./scripts/launch_single_gpu.sh \
#       --method configs/method/lora.yaml \
#       [--distributed configs/distributed/fsdp.yaml] \
#       [--model configs/model/llama3_8b.yaml] \
#       [--data configs/data.yaml]
#
# All args are optional; defaults below match a typical single-GPU LoRA run.
# distributed/fsdp.yaml is accepted for CLI consistency with the other launch
# scripts, but is a no-op in single-process mode.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/llama3_8b.yaml}"
METHOD_CONFIG="${METHOD_CONFIG:-configs/method/lora.yaml}"
DISTRIBUTED_CONFIG="${DISTRIBUTED_CONFIG:-configs/distributed/fsdp.yaml}"
CLUSTER_CONFIG="${CLUSTER_CONFIG:-configs/cluster/single_gpu.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.yaml}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_CONFIG="$2"; shift 2 ;;
        --method) METHOD_CONFIG="$2"; shift 2 ;;
        --distributed) DISTRIBUTED_CONFIG="$2"; shift 2 ;;
        --cluster) CLUSTER_CONFIG="$2"; shift 2 ;;
        --data) DATA_CONFIG="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Single-GPU Launch ==="
echo "model:       ${MODEL_CONFIG}"
echo "method:      ${METHOD_CONFIG}"
echo "distributed: ${DISTRIBUTED_CONFIG}"
echo "cluster:     ${CLUSTER_CONFIG}"
echo "data:        ${DATA_CONFIG}"
echo "========================="

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python src/train.py \
    --model_config "${MODEL_CONFIG}" \
    --method_config "${METHOD_CONFIG}" \
    --distributed_config "${DISTRIBUTED_CONFIG}" \
    --cluster_config "${CLUSTER_CONFIG}" \
    --data_config "${DATA_CONFIG}" \
    "${EXTRA_ARGS[@]:-}"