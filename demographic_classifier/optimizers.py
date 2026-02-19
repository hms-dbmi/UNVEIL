from typing import Dict, Iterable
from torch.optim import Adam, SGD, Optimizer
import torch.nn as nn

OPTIMIZER_MAP = {
    "Adam": Adam,
    "SGD": SGD
}

def setup_optimizer(
    optimizer_type: str,
    optimizer_init_args: Dict,
    params: Iterable[nn.Parameter]
) -> Optimizer:

    if optimizer_type not in OPTIMIZER_MAP:
        raise ValueError(f"{optimizer_type} opt not founds. Select from {list(OPTIMIZER_MAP.keys())}")
    
    return OPTIMIZER_MAP[optimizer_type](params=params, **optimizer_init_args)
