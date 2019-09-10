import torch
from torch.autograd import Variable

from typing import Callable

from . import functional as F


class SuperSpike(torch.nn.Module):
    """
    Full module instantiation of F._SuperSpike and F.super_spike

    :param thresh: Initialization of spike threshold
    """

    def __init__(self, thresh: float = 0.0):
        super().__init__()

    def forward(self, v):
        return F.super_spike(v)
