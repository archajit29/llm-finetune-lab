#!/usr/bin/env bash
#
# launch_multi_gpu.sh
#
# Launches src/train.py across multiple GPUs on a single node using
# `torchrun --standalone --nproc_per_node=$NUM_GPUS`, for the
# single_node cluster config.
#
# Usage:
#   ./scripts/launch_multi_gpu.sh \
#       --method configs/method/qlora.yaml \
#       --distributed configs/distributed/deepspeed_zero2.json \
#       [--num-gpus 8] [--model configs/model/qwen3_30b.yaml] [--data configs/data.yaml]
#
# NUM_GPUS defaults to all GPUs visible via nvidia-smi (fallback: 1).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

detect_num_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l
    else
        echo 1
    fi
}

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/llama3_8b.yaml}"
METHOD_CONFIG="${METHOD_CONFIG:-configs/method/qlora.yaml}"
DISTRIBUTED_CONFIG="${DISTRIBUTED_CONFIG:-configs/distributed/fsdp.yaml}"
CLUSTER_CONFIG="${CLUSTER_CONFIG:-configs/cluster/single_node.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.yaml}"
NUM_GPUS="${NUM_GPUS:-$(detect_num_gpus)}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_CONFIG="$2"; shift 2 ;;
        --method) METHOD_CONFIG="$2"; shift 2 ;;
        --distributed) DISTRIBUTED_CONFIG="$2"; shift 2 ;;
        --cluster) CLUSTER_CONFIG="$2"; shift 2 ;;
        --data) DATA_CONFIG="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -lt 1 ]]; then
    echo "ERROR: could not determine NUM_GPUS (got '${NUM_GPUS}'). Pass --num-gpus N." >&2
    exit 1
fi

echo "=== Multi-GPU (single node) Launch ==="
echo "num_gpus:    ${NUM_GPUS}"
echo "model:       ${MODEL_CONFIG}"
echo "method:      ${METHOD_CONFIG}"
echo "distributed: ${DISTRIBUTED_CONFIG}"
echo "cluster:     ${CLUSTER_CONFIG}"
echo "data:        ${DATA_CONFIG}"
echo "======================================="

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    src/train.py \
    --model_config "${MODEL_CONFIG}" \
    --method_config "${METHOD_CONFIG}" \
    --distributed_config "${DISTRIBUTED_CONFIG}" \
    --cluster_config "${CLUSTER_CONFIG}" \
    --data_config "${DATA_CONFIG}" \
    "${EXTRA_ARGS[@]:-}"