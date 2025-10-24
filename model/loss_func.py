"""
This module provides a factory function to get the desired loss function for training the model.

It defines an enumeration of the supported loss functions and a function to get an instance of the chosen loss function from the `torch.nn` module.
"""
from enum import Enum

from torch import nn


class LossFunc(Enum):
    """
    An enumeration of the different loss functions that can be used for training the model.

    Attributes:
        L1_LOSS: The L1 loss (Mean Absolute Error).
        MSE_LOSS: The Mean Squared Error loss.
        CROSS_ENTROPY: The Cross-Entropy loss.
        KL_DIV_LOSS: The Kullback-Leibler divergence loss.
    """
    L1_LOSS = 0,
    MSE_LOSS = 1
    CROSS_ENTROPY = 2,
    KL_DIV_LOSS = 3


def get_loss_func(loss_func_str: str):
    """
    Factory function to get a PyTorch loss function instance based on a string identifier.

    Args:
        loss_func_str (str): The string identifier of the loss function (e.g., "L1_LOSS", "MSE_LOSS").

    Returns:
        nn.Module: An instance of the chosen PyTorch loss function.
    """
    loss_func_enum = LossFunc[loss_func_str]
    match loss_func_enum:
        case LossFunc.L1_LOSS:
            # L1 Loss (Mean Absolute Error).
            return nn.L1Loss()
        case LossFunc.CROSS_ENTROPY:
            # Cross-Entropy Loss.
            return nn.CrossEntropyLoss()
        case LossFunc.KL_DIV_LOSS:
            # Kullback-Leibler Divergence Loss.
            return nn.KLDivLoss()
        case _:
            # Default to Mean Squared Error Loss.
            return nn.MSELoss()