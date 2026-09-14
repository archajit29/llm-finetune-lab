#!/usr/bin/env bash
#
# nccl_test.sh
#
# Verifies multi-node GPU networking (NCCL all-reduce bandwidth + latency)
# BEFORE launching a real training job. Run this identically on every node
# (same command, only NODE_RANK differs) or via your cluster's job launcher.
#
# Usage:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#     NPROC_PER_NODE=8 ./scripts/nccl_test.sh
#
# Requires: torch installed, and a small all-reduce probe script
# (we generate one on the fly so this script has no extra repo dependency).

set -euo pipefail

# ---- Config (override via env vars) ----
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SIZE_MB="${SIZE_MB:-512}"        # tensor size per all-reduce, in MB
NUM_ITERS="${NUM_ITERS:-20}"     # number of timed iterations
WARMUP_ITERS="${WARMUP_ITERS:-5}"

echo "=== NCCL Multi-Node Network Test ==="
echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Tensor size: ${SIZE_MB} MB, iters=${NUM_ITERS} (+${WARMUP_ITERS} warmup)"
echo "====================================="

# Recommended NCCL debugging / networking env vars — override as needed
# for your fabric (EFA, InfiniBand, RoCE, etc).
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

TMP_SCRIPT="$(mktemp /tmp/nccl_allreduce_test.XXXXXX.py)"
trap 'rm -f "${TMP_SCRIPT}"' EXIT

cat > "${TMP_SCRIPT}" <<'PYEOF'
import os
import time
import torch
import torch.distributed as dist

def main():
    size_mb = float(os.environ.get("SIZE_MB", "512"))
    num_iters = int(os.environ.get("NUM_ITERS", "20"))
    warmup_iters = int(os.environ.get("WARMUP_ITERS", "5"))

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    numel = int(size_mb * 1024 * 1024 / 4)  # float32 elements
    tensor = torch.rand(numel, dtype=torch.float32, device=f"cuda:{local_rank}")

    # Warmup
    for _ in range(warmup_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    dist.barrier()
    start = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_s = elapsed / num_iters
    # Bus bandwidth formula for ring all-reduce:
    # busbw = size * 2*(N-1)/N / time
    data_bytes = numel * 4
    algbw_gbps = (data_bytes / avg_time_s) / 1e9
    busbw_gbps = algbw_gbps * (2 * (world_size - 1) / world_size)

    result = (
        f"[rank {rank}/{world_size} local_rank {local_rank}] "
        f"size={size_mb:.0f}MB avg_latency={avg_time_s*1000:.3f}ms "
        f"alg_bw={algbw_gbps:.2f}GB/s bus_bw={busbw_gbps:.2f}GB/s"
    )
    print(result, flush=True)

    dist.barrier()
    if rank == 0:
        print("\n=== SUMMARY (rank 0) ===")
        print(f"World size: {world_size}")
        print(f"Tensor size: {size_mb:.0f} MB, iters: {num_iters}")
        print(f"Avg latency: {avg_time_s*1000:.3f} ms")
        print(f"Algorithm bandwidth: {algbw_gbps:.2f} GB/s")
        print(f"Bus bandwidth: {busbw_gbps:.2f} GB/s")
        print("========================")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PYEOF

export SIZE_MB NUM_ITERS WARMUP_ITERS

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TMP_SCRIPT}"

echo "=== NCCL test complete on node_rank=${NODE_RANK} ==="