"""Global seed control, used identically across vision, market, and fusion training."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
