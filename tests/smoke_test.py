#!/usr/bin/env python
"""
tests/smoke_test.py

Fast sanity check meant to catch config/networking bugs BEFORE you touch the
real model. Builds a tiny 2-layer transformer (a few hundred KB of params),
a tiny fake batch of random token ids, wraps it in whichever distributed
strategy is requested (none / fsdp / deepspeed), and runs 2 forward+backward+
optimizer steps.

Asserts:
  - loss is finite at every step (no NaN/Inf)
  - loss on step 2 is <= loss on step 1 (or at least doesn't blow up)

Should complete in well under 30 seconds on a single GPU, and works
multi-process via torchrun for FSDP, or `deepspeed`/torchrun for DeepSpeed.

Usage:
    # no distributed wrapper, single process
    python tests/smoke_test.py --strategy none

    # FSDP, single or multi GPU
    torchrun --standalone --nproc_per_node=2 tests/smoke_test.py --strategy fsdp

    # DeepSpeed (zero2/zero3), single or multi GPU
    torchrun --standalone --nproc_per_node=2 tests/smoke_test.py \
        --strategy deepspeed --deepspeed_config configs/distributed/deepspeed_zero2.json
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Tiny fake model — NOT the real model. Just enough structure (embedding,
# 2 transformer blocks, LM head) to exercise forward/backward/optimizer and
# any distributed wrapper's sharding/communication logic.
# --------------------------------------------------------------------------
class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, max_seq_len: int = 16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TinyBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        # causal mask
        causal_mask = torch.triu(
            torch.full((t, t), float("-inf"), device=input_ids.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def make_fake_batch(batch_size: int, seq_len: int, vocab_size: int, device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g).to(device)
    # next-token labels: shift by one, wrap last token (fine for a smoke test)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    return input_ids, labels


# --------------------------------------------------------------------------
# Distributed setup helpers
# --------------------------------------------------------------------------
def setup_distributed_process_group(strategy: str):
    """Init torch.distributed if running under torchrun (WORLD_SIZE > 1) or
    if strategy explicitly requires it. No-op for single-process 'none'."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 or strategy in ("fsdp", "deepspeed"):
        if not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def wrap_fsdp(model, device):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    model = model.to(device)
    model = FSDP(model, device_id=device if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def wrap_deepspeed(model, args):
    import deepspeed

    ds_config = args.deepspeed_config
    if ds_config is None:
        # Minimal inline config if none provided, so the script is still
        # runnable standalone for a quick check.
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
            "zero_optimization": {"stage": 2},
            "gradient_accumulation_steps": 1,
        }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    return model_engine, optimizer


def wrap_none(model, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


# --------------------------------------------------------------------------
# Main smoke test
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tiny smoke test before real training.")
    parser.add_argument("--strategy", choices=["none", "fsdp", "deepspeed"], default="none",
                         help="Distributed wrapper to exercise.")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                         help="Path to a deepspeed JSON config (only used with --strategy deepspeed).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1,
                         help="Accepted for deepspeed launcher compatibility; unused directly.")
    args = parser.parse_args()

    t_start = time.perf_counter()
    torch.manual_seed(args.seed)

    device = setup_distributed_process_group(args.strategy)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    is_main = rank == 0

    if is_main:
        print(f"=== smoke_test.py: strategy={args.strategy} device={device} "
              f"world_size={torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1} ===")

    model = TinyTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )

    if args.strategy == "fsdp":
        model, optimizer = wrap_fsdp(model, device)
    elif args.strategy == "deepspeed":
        model, optimizer = wrap_deepspeed(model, args)
    else:
        model, optimizer = wrap_none(model, device)

    losses = []
    for step in range(args.steps):
        input_ids, labels = make_fake_batch(
            args.batch_size, args.seq_len, args.vocab_size, device, seed=args.seed + step
        )

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, args.vocab_size), labels.view(-1)
        )

        loss_value = loss.item()
        assert math.isfinite(loss_value), (
            f"[rank {rank}] Loss is not finite at step {step}: {loss_value} "
            f"(NaN/Inf indicates a config or precision bug)."
        )
        losses.append(loss_value)

        if args.strategy == "deepspeed":
            model.backward(loss)
            model.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if is_main:
            print(f"[step {step}] loss={loss_value:.6f}")

    # Final assertions
    assert all(math.isfinite(v) for v in losses), "One or more losses were non-finite."
    if len(losses) >= 2:
        assert losses[-1] <= losses[0] + 1e-3, (
            f"Loss did not decrease (or stay flat): {losses[0]:.6f} -> {losses[-1]:.6f}. "
            f"This may indicate a broken optimizer step or gradient flow issue."
        )

    elapsed = time.perf_counter() - t_start
    if is_main:
        print(f"=== smoke_test.py PASSED in {elapsed:.2f}s (strategy={args.strategy}) ===")

    if elapsed > 30 and is_main:
        print(f"WARNING: smoke test took {elapsed:.2f}s, exceeding the 30s target.", file=sys.stderr)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()