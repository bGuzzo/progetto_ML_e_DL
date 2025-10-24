"""
This module defines the kernel functions used to compute the prior-association in the Anomaly-Attention mechanism.

The prior-association is a key component of the Anomaly Transformer, as it provides a baseline for the attention distribution based on the relative distance between time points. This module provides different kernel functions that can be used to model this prior, including a Gaussian kernel, a sigmoid kernel, and an option to have the prior as a directly learned parameter.
"""
import math
from enum import Enum

import torch
from torch import Tensor


def _no_kernel(_, sigma):
    """
    A kernel function where the prior-association is a learned parameter.

    In this case, the `sigma` parameter is directly used as the prior-association, after applying a softmax function to ensure it is a valid distribution.

    Args:
        _ (Tensor): The distance matrix (not used).
        sigma (Tensor): The learned parameter for the prior-association.

    Returns:
        Tensor: The prior-association tensor.
    """
    # The prior-association is a learned parameter, so we apply softmax to it directly.
    return torch.softmax(sigma, dim=-1)


def _gaussian_kernel(dist, sigma):
    """
    Applies a Gaussian kernel to the distance matrix.

    The Gaussian kernel assigns higher weights to closer time points, with the spread of the kernel controlled by the `sigma` parameter.

    Args:
        dist (Tensor): The distance matrix.
        sigma (Tensor): The standard deviation of the Gaussian kernel.

    Returns:
        Tensor: The prior-association tensor based on the Gaussian kernel.
    """
    return 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-dist ** 2 / 2 / (sigma ** 2))


def _sigmoid_kernel(dist, sigma):
    """
    Applies a sigmoid kernel to the distance matrix.

    The sigmoid kernel is another option for modeling the prior-association, which can capture different types of relationships between time points.

    Args:
        dist (Tensor): The distance matrix.
        sigma (Tensor): The parameter controlling the shape of the sigmoid function.

    Returns:
        Tensor: The prior-association tensor based on the sigmoid kernel.
    """
    return torch.tanh(dist * sigma)


class KernelType(Enum):
    """
    An enumeration of the different kernel types that can be used for the prior-association.

    Attributes:
        NO: The prior-association is a learned parameter.
        GAUSSIAN: A Gaussian kernel is used for the prior-association.
        SIGMOID: A sigmoid kernel is used for the prior-association.
    """
    NO = 0
    GAUSSIAN = 1
    SIGMOID = 2


def apply_kernel(kernel_type: KernelType, dist: Tensor, sigma: Tensor) -> Tensor:
    """
    Applies a specified kernel function to compute the prior-association.

    This function acts as a factory that selects the appropriate kernel function based on the `kernel_type` and applies it to the distance matrix and the learned sigma parameter.

    Args:
        kernel_type (KernelType): The type of kernel to apply.
        dist (Tensor): The distance matrix between time points.
        sigma (Tensor): The learned parameter for the kernel.

    Returns:
        Tensor: The computed prior-association tensor.
    """
    match kernel_type:
        case KernelType.NO:
            # Use the learned parameter directly as the prior-association.
            return _no_kernel(dist, sigma)
        case KernelType.GAUSSIAN:
            # Apply the Gaussian kernel.
            return _gaussian_kernel(dist, sigma)
        case KernelType.SIGMOID:
            # Apply the sigmoid kernel.
            return _sigmoid_kernel(dist, sigma)
        case _:
            # Default to the Gaussian kernel if the kernel type is not recognized.
            return _gaussian_kernel(dist, sigma)