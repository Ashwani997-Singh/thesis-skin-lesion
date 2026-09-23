import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Fix every source of randomness. Call at the start of every run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed