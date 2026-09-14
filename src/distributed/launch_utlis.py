"""Launch utilities.

Helpers for reading the distributed training environment variables that
torchrun sets automatically for each launched process (see
configs/cluster/*.yaml for the corresponding topology configs).
"""

import os
from dataclasses import dataclass


@dataclass
class DistEnv:
    """Distributed environment info, with single-GPU-friendly defaults."""

    rank: int
    local_rank: int
    world_size: int
    master_addr: str
    master_port: int


def get_dist_env():
    """Read distributed environment variables set by torchrun.

    Falls back to sane single-GPU defaults (rank 0, world size 1, localhost)
    when the variables are not set, e.g. when running a script directly
    without torchrun.

    Returns:
        A DistEnv dataclass instance with fields: rank, local_rank,
        world_size, master_addr, master_port.
    """
    return DistEnv(
        rank=int(os.environ.get("RANK", 0)),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        master_port=int(os.environ.get("MASTER_PORT", 29500)),
    )


def is_main_process():
    """Return True only for the global main process (RANK == 0).

    Useful for gating logging, checkpoint saving, and other actions that
    should happen exactly once across all distributed processes.
    """
    return int(os.environ.get("RANK", 0)) == 0