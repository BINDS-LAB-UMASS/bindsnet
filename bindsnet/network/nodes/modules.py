from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Iterable, Optional, Union

import torch
from torch.nn import Parameter

from . import functional as F


class Nodes(torch.nn.Module):
    # language=rst
    """
    Abstract base class for groups of neurons.
    """

    def __init__(
        self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, **kwargs
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        """
        super().__init__()

        self.dt = None
        self.register_buffer("s", torch.FloatTensor())  # Spikes

        if n is None and shape is None:
            return

        if n is None:
            self.n = reduce(mul, shape)  # No. of neurons product of shape.
        else:
            self.n = n  # No. of neurons provided.

        if shape is None:
            self.shape = [self.n]  # Shape is equal to the size of the layer.
        else:
            self.shape = shape  # Shape is passed in as an argument.

        assert self.n == reduce(
            mul, self.shape
        ), "No. of neurons and shape do not match"

    def set_dt(self, dt: float) -> None:
        self.dt = dt

    def reset_(self, shape: Optional[Iterable[int]] = None) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables.
        """
        if shape is None:
            return
        else:
            self.shape = shape

        self.s = torch.zeros(*self.shape, dtype=self.s.dtype, device=self.s.device)


class IFNodes(Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        thresh: Union[float, torch.Tensor] = -52.0,
        reset: Union[float, torch.Tensor] = -65.0,
        lbound: Union[float, torch.Tensor] = -1e5,
        scale: Union[float, torch.Tensor] = 100.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n=n, shape=shape)

        self.reset = Parameter(
            torch.tensor(reset, dtype=torch.float), requires_grad=True
        )  # reset voltage
        self.thresh = Parameter(
            torch.tensor(thresh, dtype=torch.float), requires_grad=True
        )  # spike threshold
        self.lbound = Parameter(
            torch.tensor(lbound, dtype=torch.float), requires_grad=True
        )  # Lower bound of voltage.

        self.scale = torch.tensor(scale, requires_grad=False)

        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.

        :return: Layer spikes
        """

        self.v = F.if_update(x, self.v)
        self.s, self.v = F.super_spike_update(
            self.v, self.thresh, self.reset, self.lbound, self.scale
        )

        return self.s

    def reset_(self, shape: Optional[Iterable[int]] = None) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_(shape)

        self.v = self.reset * torch.ones_like(self.s)


class LIFNodes(Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        thresh: Union[float, torch.Tensor] = -52.0,
        reset: Union[float, torch.Tensor] = -65.0,
        lbound: Union[float, torch.Tensor] = -1e5,
        rest: Union[float, torch.Tensor] = -65.0,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        scale: Union[float, torch.Tensor] = 100.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param lbound: Lower bound of the voltage.

        :param rest: Resting membrane voltage.
        :param tc_decay: Time constant of neuron voltage decay.
        """
        super().__init__(n=n, shape=shape)

        self.reset = Parameter(torch.tensor(reset, dtype=torch.float))
        self.thresh = Parameter(torch.tensor(thresh, dtype=torch.float))
        self.lbound = Parameter(torch.tensor(lbound, dtype=torch.float))
        self.rest = Parameter(torch.tensor(rest, dtype=torch.float))
        self.tc_decay = Parameter(torch.tensor(tc_decay, dtype=torch.float))

        self.scale = torch.tensor(scale, requires_grad=False)

        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.

        :return: Layer spikes
        """

        self.v = F.lif_update(x, self.v, self.tc_decay, self.rest)
        self.s, self.v = F.super_spike_update(
            self.v, self.thresh, self.reset, self.lbound, self.scale
        )

        return self.s

    def reset_(self, shape: Optional[Iterable[int]] = None) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_(shape)

        self.v = self.reset * torch.ones_like(self.s)
