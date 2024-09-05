from enum import Enum

from torch import nn


class LossFunc(Enum):
    L1_LOSS = 0,
    MSE_LOSS = 1
    CROSS_ENTROPY = 2,
    KL_DIV_LOSS = 3


def get_loss_func(loss_func_str: str):
    loss_func_enum = LossFunc[loss_func_str]
    match loss_func_enum:
        case LossFunc.L1_LOSS:
            return nn.L1Loss()
        case LossFunc.CROSS_ENTROPY:
            return nn.CrossEntropyLoss()
        case LossFunc.KL_DIV_LOSS:
            return nn.KLDivLoss()
        case _:
            return nn.MSELoss()
