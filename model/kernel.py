import math
from enum import Enum

import torch
from torch import Tensor


def _no_kernel(_, sigma):
    # Prior are learned param
    return torch.softmax(sigma, dim=-1)


def _gaussian_kernel(dist, sigma):
    return 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-dist ** 2 / 2 / (sigma ** 2))


def _sigmoid_kernel(dist, sigma):
    return torch.tanh(dist * sigma)


class KernelType(Enum):
    NO = 0
    GAUSSIAN = 1
    SIGMOID = 2


def apply_kernel(kernel_type: KernelType, dist: Tensor, sigma: Tensor) -> Tensor:
    match kernel_type:
        case KernelType.NO:
            return _no_kernel(dist, sigma)
        case KernelType.GAUSSIAN:
            return _gaussian_kernel(dist, sigma)
        case KernelType.SIGMOID:
            return _sigmoid_kernel(dist, sigma)
        case _:
            return _gaussian_kernel(dist, sigma)
