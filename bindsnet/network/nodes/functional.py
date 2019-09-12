import torch

from typing import Tuple


class _SuperSpike(torch.autograd.Function):
    """
    Zenke & Ganguli 2018

    Many thanks to the inspiration from:

    https://github.com/fzenke/spytorch
    """

    @staticmethod
    def forward(ctx, v, scale):
        assert scale.requires_grad == False, "No gradient to scale"

        ctx.save_for_backward(v, scale)

        out = torch.zeros_like(v)
        out[v >= 0.0] = 1.0

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, scale) = ctx.saved_tensors

        grad = grad_output / (scale * torch.abs(input) + 1.0) ** 2

        return grad, None


super_spike = _SuperSpike.apply


def super_spike_update(
    v: torch.Tensor,
    thresh: torch.Tensor,
    reset: torch.Tensor,
    lbound: torch.Tensor,
    scale: torch.Tensor,
):
    """
    Spike update of a neuron
    """

    # check for v >= thresh
    s = super_spike(v - thresh, scale)
    s_ = s.byte()

    # reset voltage upon spikes
    v = torch.where(s_, reset.expand(*v.shape), v)

    # bound voltage
    v = torch.where(v < lbound, lbound.expand(*v.shape), v)

    # Return the states that were updated
    return s, v


@torch.jit.script
def if_update(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Update of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.

    :param x: Input voltage into the membrane
    :param v: Membrane potential
    :return: New membrane potential
    """

    v = v + x

    return v


@torch.jit.script
def lif_update(
    x: torch.Tensor, v: torch.Tensor, decay: torch.Tensor, rest: torch.Tensor
) -> torch.Tensor:
    """
    Layer of `leaky integrate-and-fire (LIF) neurons
    <http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02311000000000000000>`_.
    """

    v = decay * (v - rest) + rest + x

    return v
