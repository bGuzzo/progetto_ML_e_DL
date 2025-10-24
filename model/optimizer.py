"""
This module provides a factory function to get the desired optimizer for training the model.

It defines an enumeration of the supported optimizers and a function to get an instance of the chosen optimizer from the `torch.optim` module.
"""
from enum import Enum

import torch.optim


class OptimizerAlg(Enum):
    """
    An enumeration of the different optimizers that can be used for training the model.

    Attributes:
        ADAM: The Adam optimizer.
        SGD: The Stochastic Gradient Descent optimizer.
        LBFGS: The L-BFGS optimizer.
        RMS_PROP: The RMSprop optimizer.
        ADAGRAD: The Adagrad optimizer.
        ADADELTA: The Adadelta optimizer.
        ADAMW: The AdamW optimizer.
    """
    ADAM = 0
    SGD = 1
    LBFGS = 2
    RMS_PROP = 3
    ADAGRAD = 4
    ADADELTA = 5
    ADAMW = 6


def get_optimizer(opt_enum_str: str, params, lr):
    """
    Factory function to get a PyTorch optimizer instance based on a string identifier.

    Args:
        opt_enum_str (str): The string identifier of the optimizer (e.g., "ADAM", "SGD").
        params: The parameters of the model to be optimized.
        lr (float): The learning rate.

    Returns:
        torch.optim.Optimizer: An instance of the chosen PyTorch optimizer.
    """
    opt_enum = OptimizerAlg[opt_enum_str]
    match opt_enum:
        case OptimizerAlg.SGD:
            # Stochastic Gradient Descent optimizer.
            return torch.optim.SGD(params, lr)
        case OptimizerAlg.RMS_PROP:
            # RMSprop optimizer.
            return torch.optim.RMSprop(params, lr)
        case OptimizerAlg.ADADELTA:
            # Adadelta optimizer.
            return torch.optim.Adadelta(params, lr)
        case OptimizerAlg.ADAMW:
            # AdamW optimizer, a variant of Adam with decoupled weight decay.
            return torch.optim.AdamW(params, lr)
        case _:
            # Default to the Adam optimizer.
            return torch.optim.Adam(params, lr)