from enum import Enum

import torch.optim


class OptimizerAlg(Enum):
    ADAM = 0
    SGD = 1
    LBFGS = 2
    RMS_PROP = 3
    ADAGRAD = 4
    ADADELTA = 5
    ADAMW = 6


def get_optimizer(opt_enum_str: str, params, lr):
    opt_enum = OptimizerAlg[opt_enum_str]
    match opt_enum:
        case OptimizerAlg.SGD:
            return torch.optim.SGD(params, lr)
        case OptimizerAlg.RMS_PROP:
            return torch.optim.RMSprop(params, lr)
        case OptimizerAlg.ADADELTA:
            return torch.optim.Adadelta(params, lr)
        case OptimizerAlg.ADAMW:
            return torch.optim.AdamW(params, lr)
        case _:
            return torch.optim.Adam(params, lr)
