import torch
from torch.autograd import Variable

from typing import Callable

from . import functional as F


class SuperSpike(torch.nn.Module):
    """
    Full module instantiation of F._SuperSpike and F.super_spike

    :param thresh: Initialization of spike threshold
    """

    def __init__(self, scale: float = 100.0):
        super().__init__()

        self.scale = torch.tensor(scale, dtype=torch.float)

    def forward(self, v):
        return F.super_spike(v, self.scale)
