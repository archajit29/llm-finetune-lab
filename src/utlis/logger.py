"""
Rank-aware logging for single-GPU, single-node, and multi-node training runs.

- Console output is emitted only on rank 0 (the "main process"), so multi-GPU
  and multi-node runs don't spam N copies of every log line.
- Every message is formatted with a timestamp and the process's rank, which
  is invaluable for debugging even on rank 0 alone.
- Optionally writes to a per-run log file. By default *every* rank writes to
  its own file (useful for tracking down a hang/crash on a specific rank),
  while only rank 0 writes to the console.

Rank detection order:
    1. torch.distributed, if initialized
    2. RANK / LOCAL_RANK environment variables (set by torchrun / DeepSpeed launcher)
    3. Fallback to rank 0 (single-process / non-distributed run)

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__, log_dir="logs", run_name="qlora_llama3_8b")
    logger.info("Starting training")   # printed on rank 0 only, written to file on every rank
    logger.warning("OOM risk high")
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def get_rank() -> int:
    """Best-effort resolution of this process's global rank."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except ImportError:
        pass

    for var in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass

    return 0


def is_main_process() -> bool:
    return get_rank() == 0


class _RankFilter(logging.Filter):
    """Injects the process rank into every log record so the formatter can use it."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return True


_LOG_FORMAT = "[%(asctime)s] [rank %(rank)d] [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "train",
    log_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    level: int = logging.INFO,
    console_on_all_ranks: bool = False,
    file_on_all_ranks: bool = True,
) -> logging.Logger:
    """
    Build (or fetch, if already configured) a rank-aware logger.

    Args:
        name: Logger name, typically `__name__` of the calling module.
        log_dir: If set, a log file is created under this directory. The
            directory is created if it doesn't exist.
        run_name: Used to build the log filename as "{run_name}.log". If a
            per-rank file is written (see `file_on_all_ranks`), the rank is
            appended, e.g. "{run_name}_rank0.log". Defaults to a timestamp if
            not provided.
        level: Logging level (default INFO).
        console_on_all_ranks: If True, every rank prints to console instead
            of just rank 0. Off by default -- this is the whole point of the
            "only rank 0 prints" requirement.
        file_on_all_ranks: If True (default), every rank gets its own log
            file. If False, only rank 0 writes a log file.

    Returns:
        A configured `logging.Logger`. Safe to call repeatedly with the same
        `name` -- handlers are not duplicated on repeated calls.
    """
    rank = get_rank()
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger() is called again for the
    # same logger name (e.g. from multiple modules).
    if getattr(logger, "_rank_aware_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    rank_filter = _RankFilter(rank)

    # Console handler: rank 0 only, unless explicitly overridden.
    if rank == 0 or console_on_all_ranks:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(rank_filter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # Optional file handler, one log file per run.
    if log_dir is not None:
        write_file = file_on_all_ranks or rank == 0
        if write_file:
            os.makedirs(log_dir, exist_ok=True)
            base_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
            filename = f"{base_name}_rank{rank}.log" if file_on_all_ranks else f"{base_name}.log"
            file_path = os.path.join(log_dir, filename)

            file_handler = logging.FileHandler(file_path, mode="a")
            file_handler.setFormatter(formatter)
            file_handler.addFilter(rank_filter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    logger._rank_aware_configured = True
    return logger