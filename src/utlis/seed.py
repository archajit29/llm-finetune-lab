"""
Sets random seeds across every RNG source touched during training, so a run
can be reproduced (data shuffling, dropout, weight init, augmentation, etc.).

Note: this makes RNG behavior deterministic but does NOT alone guarantee
bit-for-bit reproducible results on GPU -- cuDNN algorithm selection and some
CUDA ops are non-deterministic by default. Pass `deterministic=True` if you
need that extra guarantee (at a performance cost).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices).

    Args:
        seed: The seed value to use everywhere.
        deterministic: If True, also configures cuDNN and PyTorch to favor
            deterministic algorithms over speed. Off by default since it can
            noticeably slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keeps hash-based randomization (e.g. dict/set ordering in subprocesses)
    # consistent across runs and worker processes.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)