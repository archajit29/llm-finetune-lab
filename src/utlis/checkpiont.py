"""
Checkpoint save/load for FSDP-wrapped models.

The naive approach -- each rank calling `model.state_dict()` and writing its
own copy -- either produces N sharded files that are painful to consume
downstream, or (if you try to gather the full model on every rank) OOMs as
soon as the model is large enough that N copies of it don't fit in
CPU/GPU memory across the node.

This module avoids that by using FSDP's `FULL_STATE_DICT` mode with
`rank0_only=True` + `offload_to_cpu=True`:
    - Only rank 0 ever materializes the fully-gathered, consolidated
      state dict (all other ranks get an empty dict).
    - It's offloaded to CPU as it's gathered, so it never sits fully
      resident on the GPU.
    - Only rank 0 touches disk, so you get a single portable .pt file
      instead of a directory of shards.

On load, the same care is taken in reverse: rank 0 reads the file from
disk, the full state dict is broadcast in-memory to the other ranks
(cheaper than every rank re-reading a huge file from disk), and then each
rank shards its own portion via FSDP's state-dict machinery.

Works transparently with plain / DDP-wrapped models too (falls back to a
standard `.state_dict()` save/load on rank 0 or all ranks respectively),
so callers don't need an `if fsdp:` branch at the call site.
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )

    _FSDP_AVAILABLE = True
except ImportError:  # torch build without FSDP support
    _FSDP_AVAILABLE = False


def _is_fsdp_model(model: torch.nn.Module) -> bool:
    return _FSDP_AVAILABLE and isinstance(model, FSDP)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if _is_distributed():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    epoch: int = 0,
    step: int = 0,
    output_dir: str = "checkpoints",
    checkpoint_name: Optional[str] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Save a single, consolidated checkpoint file from an FSDP-wrapped model.

    Args:
        model: An FSDP-wrapped (or plain/DDP) model.
        optimizer: Optional optimizer whose state should be saved alongside.
        scheduler: Optional LR scheduler (anything with `.state_dict()`).
        epoch, step: Training progress markers, stored in the checkpoint.
        output_dir: Directory the checkpoint file is written into.
        checkpoint_name: Filename (without extension). Defaults to
            "checkpoint-step{step}".
        extra_state: Any additional picklable metadata to persist
            (e.g. RNG state, best_metric, run config).

    Returns:
        The checkpoint file path on rank 0, `None` on all other ranks
        (nothing was written there).
    """
    rank = get_rank()
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_name = checkpoint_name or f"checkpoint-step{step}"
    checkpoint_path = os.path.join(output_dir, checkpoint_name + ".pt")

    if _is_fsdp_model(model):
        # rank0_only=True: only rank 0 ends up with a populated dict; every
        # other rank gets {} instead of a full duplicate of the model.
        # offload_to_cpu=True: params are moved to CPU as they're gathered,
        # so rank 0 never needs (full model size) of *GPU* memory headroom.
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            full_state_dict_config,
            optim_state_dict_config,
        ):
            model_state_dict = model.state_dict()
            optim_state_dict = (
                FSDP.optim_state_dict(model, optimizer) if optimizer is not None else None
            )
    else:
        # Plain nn.Module or DDP: every rank already holds the same full
        # state dict, nothing to gather.
        model_state_dict = model.state_dict()
        optim_state_dict = optimizer.state_dict() if optimizer is not None else None

    checkpoint_written = None
    if rank == 0:
        payload = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optim_state_dict,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step": step,
            "extra_state": extra_state or {},
        }
        torch.save(payload, checkpoint_path)
        checkpoint_written = checkpoint_path

    # Prevent other ranks from racing ahead (e.g. deleting old checkpoints,
    # or starting the next step) while rank 0 is still writing to disk.
    if _is_distributed():
        dist.barrier()

    return checkpoint_written


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint produced by `save_checkpoint` back into an
    FSDP-wrapped model.

    Only rank 0 reads the file from disk (avoids every rank hitting the
    filesystem for a potentially huge file simultaneously); the resulting
    full state dict is then broadcast in-memory to the other ranks, and each
    rank shards its own portion locally via FSDP.

    Returns:
        A dict with the restored "epoch", "step", and "extra_state" fields
        (identical on every rank).
    """
    rank = get_rank()
    distributed = _is_distributed()

    if rank == 0:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: '{checkpoint_path}'")
        payload = torch.load(checkpoint_path, map_location=map_location)
    else:
        payload = None

    if distributed:
        # Broadcast the full (CPU-offloaded) state dict from rank 0 to every
        # other rank in-memory, rather than each rank re-reading the file.
        obj_list = [payload]
        dist.broadcast_object_list(obj_list, src=0)
        payload = obj_list[0]

    if payload is None:
        raise RuntimeError(
            "No checkpoint payload available on this rank -- this should "
            "only happen on rank 0 with a missing file, which is already "
            "raised above; on other ranks it implies the broadcast failed."
        )

    if _is_fsdp_model(model):
        # rank0_only=False here: at this point every rank already has the
        # full (broadcast) state dict in hand, so each can independently
        # shard its own portion via FSDP's load path.
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            full_state_dict_config,
            optim_state_dict_config,
        ):
            model.load_state_dict(payload["model_state_dict"])
            if optimizer is not None and payload.get("optimizer_state_dict") is not None:
                sharded_optim_state_dict = FSDP.optim_state_dict_to_load(
                    model, optimizer, payload["optimizer_state_dict"]
                )
                optimizer.load_state_dict(sharded_optim_state_dict)
    else:
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    if distributed:
        dist.barrier()

    return {
        "epoch": payload.get("epoch", 0),
        "step": payload.get("step", 0),
        "extra_state": payload.get("extra_state", {}),
    }